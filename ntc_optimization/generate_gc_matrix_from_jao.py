from public_api_extractor.hdl_fetch import HorizonDataHandler
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning
from alive_progress import alive_bar

import requests
import warnings
import json
import time
import pandas as pd

def get_jao(start, end):
    jao = HorizonDataHandler().get_jao(start, end)
    jao['dateTimeUtc'] = jao['dateTimeUtc'].dt.tz_convert('Europe/Oslo')
    jao.rename(columns={'dateTimeUtc': 'DatetimeCET'}, inplace=True)
    return jao


def get_shadow_price(start, end):
    sp = HorizonDataHandler().get_shadow_price(start, end)
    sp.rename(columns={'dateTime': 'DatetimeCET'}, inplace=True)
    return sp


def get_jao_api(start_date:datetime, end_date:datetime, token):
    url = f"https://publicationtool.jao.eu/nordic/api/data/fbDomainShadowPrice?Filter=%7B%7D&\
        Skip=0&Take=10000000&FromUtc={start_date}T00%3A00%3A00.000Z&ToUtc={end_date}T23%3A00%3A00.000Z"
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    session = requests.Session()
    session.verify = False
    content = requests.get(url, headers=headers, verify=False).text #noqa
    soup = BeautifulSoup(content, 'html.parser')
    soup = str(soup)
    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
    return soup


def create_dataframe_from_jao_data(start_date, end_date, token):
    dfs = []
    delta = end_date - start_date
    num_days = delta.days
    num_batches = (num_days // 20) + 1
    with alive_bar(num_batches) as bar:
        time.sleep(0.005)
        for batch in range(num_batches):
            batch_start_date = start_date + timedelta(days=batch * 20)
            batch_end_date = min(start_date + timedelta(days=(batch + 1) * 20), end_date)
            batch_start_date_str = batch_start_date.strftime('%Y-%m-%d')
            batch_end_date_str = batch_end_date.strftime('%Y-%m-%d')
            soup = get_jao_api(batch_start_date_str, batch_end_date_str, token)
            data = json.loads(soup)
            data_list = data.get("data", [])
            df = pd.DataFrame(data_list)
            dfs.append(df)
            print(f"Dataframe created for {batch_start_date_str} to {batch_end_date_str}")
            bar()
    result_df = pd.concat(dfs, ignore_index=True)
    result_df['dateTimeUtc'] = pd.to_datetime(result_df['dateTimeUtc'])
    result_df['dateTimeUtc'] = result_df['dateTimeUtc'].dt.tz_convert('Europe/Oslo')
    result_df.rename(columns={'dateTimeUtc': 'DatetimeCET'}, inplace=True)
    result_df['FAAC_FB'] = result_df['flowFb'] - result_df['fall']
    return result_df


def merge_jao_and_sp(jao, sp):
    merged = jao.merge(sp, on=["cnecName", "DatetimeCET"], how="left")
    merged['FAAC_FB'] = merged['marketFlow'] - merged['fall']
    merged['DatetimeCET'] = merged['DatetimeCET'].dt.strftime('%Y-%m-%d %H:%M:%S.000')
    return merged


def change_columns_names(df):
    df.rename(columns={'cnecName': 'JAO_CNEC_Name', 'cneName': 'JAO_CNE_Name', 
                       'contName': 'JAO_Contin_Name', 'nonRedundant': 'Non_Redundant', 
                       'significant': 'Significant', 'shadowPrice': 'SHADOWPRICE',
                       'ram': 'RAM_FB', 'biddingZoneFrom': 'BIDDINGAREA_FROM',
                       'biddingZoneTo': 'BIDDINGAREA_TO'}, inplace=True)
    return df


def calculate_z2z_ptdfs(df, border_list):
    df.columns = df.columns.str.replace('ptdf_', '', regex=False)
    for border in border_list:
        area_from, area_to = border.split('-')
        df[f'z2z_{border}'] = df[f'{area_from}'] - df[f'{area_to}']
    return df
