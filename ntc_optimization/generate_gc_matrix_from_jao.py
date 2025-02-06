from public_api_extractor.hdl_fetch import HorizonDataHandler
from datetime import date

def get_jao(start, end):
    jao = HorizonDataHandler().get_jao(start, end)
    jao['dateTimeUtc'] = jao['dateTimeUtc'].dt.tz_convert('Europe/Oslo')
    jao.rename(columns={'dateTimeUtc': 'DatetimeCET'}, inplace=True)
    return jao


def get_shadow_price(start, end):
    sp = HorizonDataHandler().get_shadow_price(start, end)
    sp.rename(columns={'dateTime': 'DatetimeCET'}, inplace=True)
    return sp


def merge_jao_and_sp(jao, sp):
    merged = jao.merge(sp, on=["cnecName", "DatetimeCET"], how="left")
    return merged


def change_columns_names(df):
    df.rename(columns={'cnecName': 'JAO_CNEC_Name', 'cneName': 'JAO_CNE_Name', 
                       'contName': 'JAO_Contin_Name', 'nonRedunant': 'Non_Redundant', 
                       'significant': 'Significant', 'shadowPrice': 'SHADOWPRICE',
                       'ram': 'RAM_FB', 'biddingZoneFrom': 'BIDDINGAREA_FROM',
                       'biddingZoneTo': 'BIDDINGAREA_TO'}, inplace=True)
    return df