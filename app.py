from datetime import date, datetime, timedelta
import pytz
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import tempfile
import zipfile
import re
from zipfile import ZipFile
from pathlib import Path
from io import BytesIO
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pytz import timezone
from contextlib import closing
from itertools import combinations

from ntc_optimization.dataclasses_and_config import OptimizationParameters
from ntc_optimization.prepare_data_from_gc_matrix import load_gc_matrix_with_ptdfs_and_aac

from ntc_optimization.solve_ntc_optimzation import ConstraintOptions, OptimizationConfig, compute_optimal_ntcs_for_mtu

atce_border_name_dict_hvdc = {
    "DK1A-SE3A":"DK1-DK1_KS",
    "SE3A-DK1A":"DK1_KS-DK1",
    "NO2-DK1A":"DK1_SK-DK1",
    "SE3-DK1A":"SE3-SE3_KS",
    "DK1A-SE3":"SE3_KS-SE3",
    "DK1A-NO2":"NO2_SK-NO2",
    "DK2-DK1":"DK2-DK2_SB",
    "DK1-DK2":"DK2_SB-DK2",
    "SE3_SWL-SE4_SWL": "DC_SE3-SE4",
    "SE4_SWL-SE3_SWL": "DC_SE4-SE3"
}

atce_border_name_dict_ac_conn = {
    "DK1A-SE3A":"DK1-SE3",
    "SE3A-DK1A":"SE3-DK1",
    "NO2-DK1A":"NO2-DK1",
    "DK1A-NO2":"DK1-NO2",
    "DK2-DK1":"DK2-DK1",
    "DK1-DK2":"DK1-DK2",
    "SE3_SWL-SE4_SWL": "DC_SE3-SE4",
    "SE4_SWL-SE3_SWL": "DC_SE4-SE3"
}

output_csv_translation_names = {
    "SE3-SE3_KS":"SE3-DK1A",
    "SE3_KS-SE3":"DK1A-SE3",
    "DC_SE3-SE4":"SE3_SWL-SE4_SWL",
    "DK2-DK2_SB":"DK2-DK1",
    "DC_SE4-SE3":"SE4_SWL-SE3_SWL",
    "DK1-DK1_SB":"DK1-DK2",
    "DK1_KS-DK1":"SE3A-DK1A",
    "NO2_SK-NO2":"DK1-NO2",
    }

list_of_borders_without_NTC_initial = ['SE3-DK1A','DK1A-SE3','SE3A-NO1', 'NO2-DK1', 'DK2-DK1', 
                                       'NO2A-NO2', 'DK1A-SE3A', 'DK1-SE3', 'DK1-DK1A', 'SE3B-SE3A',
                                       'DK1A-DK1', 'DK1-DK2', 'DK1A-NO2', 'SE3A-DK1A', 
                                       'SE4_ACDC-SE3_ACDC', 'NO2-DK1A', 'NO1-SE3A',
                                       'SE3_ACDC-SE4_ACDC', 'SE3A-SE3B','DK1-NO2', 'NO2-NO2A']

list_of_borders_to_remove = ['DK1_SB-DK1', 'DK1_SK-DK1', 'DK2_SB-DK2']


@st.cache_data(persist=True)
def get_matrix_from_url(url: str) -> pd.DataFrame:
    matrix = pd.read_csv(url, encoding="unicode_escape", sep=";")
    return matrix


@st.cache_data(persist=True)
def download_and_concatenate_csv(url: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with closing(requests.get(url, stream=True, verify=False)) as response:
            if response.status_code != 200:
                raise ValueError(f"Failed to download file: {response.status_code}")

            zip_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with open(zip_file_path.name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(zip_file_path.name, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            csv_files = [file for file in zip_ref.namelist() if file.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV files found in the zip archive")

            dfs = []
            for csv_file in csv_files:
                csv_path = f"{tmp_dir}/{csv_file}"
                df = pd.read_csv(csv_path, header=[0, 1])
                dfs.append(df)

            concatenated_df = pd.concat(dfs, axis=0)
            return concatenated_df


@st.cache_data(persist=True)
def extract_zip_data(zip_data: UploadedFile) -> pd.DataFrame | None:
    all_dfs = []
    with ZipFile(BytesIO(zip_data)) as zfile:
        for file_name in zfile.namelist():
            if file_name.endswith(".csv"):
                with zfile.open(file_name) as myfile:
                    df = pd.read_csv(myfile, header=[0, 1])
                    all_dfs.append(df)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return None


def top_three_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Apply along the rows to find the top 3 column names for each row
    def get_top_columns(row: pd.Series) -> pd.Series:
        # Sort the row in descending order and get the top 3 indices (column names)
        top_indices = row.sort_values(ascending=False).head(3).index
        return pd.Series(top_indices, index=["Largest PTDF", "Second Largest PTDF", "Third Largest PTDF"])

    # Apply the function across the DataFrame rows
    new_df = df.apply(get_top_columns, axis=1)
    return new_df


def add_column_as_sum_of_others(df: pd.DataFrame, list_of_borders: list[str], new_border_name:str):
    new_rows = []

    unique_borders = df['BorderName'].unique()
    if not all([border in unique_borders for border in list_of_borders]):
        raise ValueError(f'Not all of {list_of_borders} in BorderName col')
    
    subset = df[df['BorderName'].isin(list_of_borders)]
    new_aac = subset[["MTU", 'AAC']].groupby('MTU').sum().rename({0: 'AAC'})
    new_ntc_final = subset[["MTU", 'NTC_final']].groupby('MTU').sum().rename({0: 'NTC_final'})
    new_rows = new_aac.merge(new_ntc_final, 'inner', left_index=True, right_index=True)
    new_rows['BorderName'] = new_border_name
    new_rows = new_rows.reset_index(names='MTU')
    new_rows = new_rows.merge(df[['MTU', 'DatetimeUTC', 'ProblemStatus']], on='MTU', how='left')

    new_df = pd.concat([df, new_rows], axis=0)
    new_df = new_df.drop('index', axis=1, errors='ignore').drop_duplicates().reset_index(drop=True)
    return new_df


def convert_atce_to_csv(df: pd.DataFrame):
    df['BorderName'] = df['BorderName'].str.replace('z2z_', '')
    df = df.rename({'NTC': 'NTC_final'}, axis=1)
    df = add_column_as_sum_of_others(df, ['SE3_KS-SE3', 'NO2_SK-NO2'], 'DK1-DK1A')
    df = add_column_as_sum_of_others(df, ['DK1_KS-DK1', 'DK1_SK-DK1'], 'DK1A-DK1')
    df = add_column_as_sum_of_others(df, ['NO2_ND-NO2', 'NO2_NK-NO2'], 'NO2A-NO2') 
    df = add_column_as_sum_of_others(df, ['NO2-NO2_ND', 'NO2-NO2_NK'], 'NO2-NO2A')
    df = add_column_as_sum_of_others(df, ['SE3-NO1', 'DK1_KS-DK1'], 'SE3A-SE3B') 
    df = add_column_as_sum_of_others(df, ['NO1-SE3', 'DK1-DK1_KS'], 'SE3B-SE3A')
    df = add_column_as_sum_of_others(df, ['SE3-SE4', 'DC_SE3-SE4'], 'SE3_ACDC-SE4_ACDC')
    df = add_column_as_sum_of_others(df, ['SE4-SE3', 'DC_SE4-SE3'], 'SE4_ACDC-SE3_ACDC')
    df = add_column_as_sum_of_others(df, ['NO2_SK-NO2'], 'DK1A-NO2')
    df = add_column_as_sum_of_others(df, ['DK1_SK-DK1'], 'NO2-DK1A')
    df = add_column_as_sum_of_others(df, ['DK1-DK1_KS'], 'DK1-SE3')
    df = add_column_as_sum_of_others(df, ['DK1_SK-DK1'], 'NO2-DK1')
    df = add_column_as_sum_of_others(df, ['DK1-DK1_KS'], 'DK1A-SE3A')
    df = add_column_as_sum_of_others(df, ['NO1-SE3'], 'NO1-SE3A')
    df = add_column_as_sum_of_others(df, ['SE3-NO1'], 'SE3A-NO1')
    #Not included in ATCE output file:
    #df = add_column_as_sum_of_others(df, ['NO2-NO1', 'NO5-NO1'], 'NO1A-NO1') 
    #df = add_column_as_sum_of_others(df, ['NO1-NO2', 'NO1-NO5'], 'NO1-NO1A')
    #df = add_column_as_sum_of_others(df, ['SE3-SE3_KS', 'SE3-NO1'], 'SE3-SE3A')
    #df = add_column_as_sum_of_others(df, ['SE3_KS-SE3', 'NO1-SE3'], 'SE3A-SE3') 

    df['ATC'] = df['NTC_final'] - df['AAC']
    df['NTC_initial'] = df['NTC_final']
    pivot_df = pd.pivot_table(df, values=['NTC_initial', 'NTC_final', 'AAC', 'ATC'], index='MTU', columns='BorderName')
    pivot_df = pivot_df.swaplevel(axis=1)
    main_cols = sorted(set(col[0] for col in pivot_df.columns))
    sub_cols = sorted(set(col[1] for col in pivot_df.columns))
    new_columns = []
    for main in main_cols:
        new_columns.append((main, 'NTC_initial'))
        new_columns.append((main, 'NTC_final'))
        new_columns.append((main, 'AAC'))
        new_columns.append((main, 'ATC'))
    pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_tuples(new_columns))
    pivot_df.columns = [(output_csv_translation_names[col[0]], col[1]) if col[0] in output_csv_translation_names else col for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    levels = pivot_df.columns.to_list()
    levels[0] = ('MTU', 'MTU')
    pivot_df.columns = pd.MultiIndex.from_tuples(levels)
    #pivot_df.reset_index(level=0, drop=True, inplace=True)
    pivot_df = remove_ntc_initial_for_some_borders(pivot_df, list_of_borders_without_NTC_initial)
    pivot_df = pivot_df.drop(columns=list_of_borders_to_remove)
    pivot_df.insert(1, ('Backup', 'Backup'), 'False')
    pivot_df[('MTU', 'MTU')] = pd.to_datetime(pivot_df[('MTU', 'MTU')]).dt.strftime('%Y-%m-%dT%H:%MZ')

    return pivot_df.to_csv(index=False).encode('utf-8')

def remove_ntc_initial_for_some_borders(df: pd.DataFrame, list_of_borders: list[str]):
    columns = pd.MultiIndex.from_tuples(df.columns)
    border_columns = [(border, 'NTC_initial') for border in list_of_borders]
    border_indices = [columns.get_loc(border) for border in border_columns]
    df = df.drop(df.columns[border_indices], axis=1)    
    return df


# mtu = "2024-02-07 05:00:00.000"
eliminate_internal_hvdc_ptdfs = False
which_hvdcs_to_eliminate = []

# https://nordic-rcc.net/wp-content/uploads/2023/10/SF_GC_Matrix_W39_2023_Stakeholder_V2.csv
# https://nordic-rcc.net/wp-content/uploads/2024/01/SF_GC_Matrix_W48_2023_Stakeholder.csv

st.set_page_config(layout="wide")
st.title("ATC Optimization toolkit")

st.header("Input data configuration - see the Nordic RCC website for links to data")
st.subheader("Grid Constraint Matrix")
gc_url = st.text_input("URL to get GC matrix from")
st.subheader("Industrial tool output for comparison")
file_option = st.radio("Choose file option for ATCE results from GE/RCC:", ("URL (nordic-rcc.net)", "Local File"))
if file_option == "URL (nordic-rcc.net)":
    atce_url = st.text_input("URL to get ATCE results from Nordic RCC")
    uploaded_file = None
elif file_option == "Local File":
    uploaded_file = st.file_uploader("Upload a zip file:", type=["zip"])
    atce_url = None

st.header("Optimization problem configuration")
col1, col2, _, _ = st.columns([1, 1, 1, 1])
with col1:
    eliminate_internal_hvdc_ptdfs = col1.toggle("Represent SWL and FS as AC corridors", value=True)
with col2:
    eliminate_all_nordic_hvdc_ptdfs = col2.toggle(
        "Represent all Nordic HVDCs as AC corridors", value=False, disabled=not eliminate_internal_hvdc_ptdfs
    )
    eliminate_all_nordic_hvdc_ptdfs = eliminate_all_nordic_hvdc_ptdfs if eliminate_internal_hvdc_ptdfs else False

if eliminate_all_nordic_hvdc_ptdfs:
    which_constraints_to_enforce = st.multiselect(
        "Which Constraints to enforce",
        [
            "RAM",
            "MIN_AAC",
        ],
        ["RAM", "MIN_AAC"],
    )
else:
    which_constraints_to_enforce = st.multiselect(
        "Which Constraints to enforce",
        ["RAM", "MIN_AAC", "HVDC_MATCHING"],
        ["RAM", "MIN_AAC", "HVDC_MATCHING"],
    )


st.subheader("Flags for known modelling errors")
(
    col1,
    col2,
) = st.columns([1, 1])

with col1:
    avoid_double_count_hvdcs = col1.toggle(
        "Avoid double counting HVDCs by eliminating one side from the optimization",
        help="If enabled, multiplies one of the corridors in a virtual bidding zone pair with zero. one of the corridors in a virtual bidding zone pair with zero in the optimization. ",
        disabled="HVDC_MATCHING" not in which_constraints_to_enforce,
    )
    if "HVDC_MATCHING" not in which_constraints_to_enforce:
        avoid_double_count_hvdcs = False
with col2:
    eliminate_non_id_corridors = col2.toggle(
        "Remove corridors that should not participate in ID from optimzation",
        help="If enabled adds a constraint for the selected corridors below that the ATCs should be equal to the AAC +- a small value. Their corridors are also multiplied by zero in the preparation of the optimization",
    )
    which_hvdcs_to_eliminate = col2.multiselect(
        "Which HVDCs to eliminate from the ID market",
        ["SE4_BC", "FI-NO4", "NO4-FI"],
        disabled=not eliminate_non_id_corridors,
    )

which_hvdcs_to_eliminate = which_hvdcs_to_eliminate if eliminate_non_id_corridors else []
if eliminate_non_id_corridors:
    which_constraints_to_enforce.append("MAX_AAC_FOR_NON_ID")

disable_id_corridor_removal = "MAX_AAC_FOR_NON_ID" not in which_constraints_to_enforce
if not eliminate_all_nordic_hvdc_ptdfs and avoid_double_count_hvdcs:
    which_hvdcs_to_eliminate += ["NO2_SK", "DK1_SB", "SE3_KS"]
if not eliminate_internal_hvdc_ptdfs and avoid_double_count_hvdcs:
    which_hvdcs_to_eliminate += ["SE3_SWL", "FI_FS"]

st.header("Feasibility Options")
st.subheader("Toggels to help feasibility of the problem")
(
    col1,
    col2,
    col3,
    col4,
    col5
) = st.columns([1, 1, 1, 1, 1])
with col1:
    allow_forced_viability_of_aac = col1.toggle(
        "Allow forced viability of HVDC-matching at Market Point",
        value="HVDC_MATCHING" in which_constraints_to_enforce,
        help="Adjusts the  RAM values on grid constraints to allow the flow at the market point - when enforcing the HVDC matching constraint",
        disabled="HVDC_MATCHING" not in which_constraints_to_enforce,
    )

with col2:
    error_on_infeasible_constraints = col2.toggle(
        "Throw error on failed Constraint Satisfaction Problem",
        help="We first try to find at least 1 vector that satisfies all constraints - this option toggles errors if no such vector exists.",
    )

with col3:
    allow_hail_mary = col3.toggle(
        "Allow hail-mary solutions ",
        help="If no convergence is found for MTU, find and eliminate distruptive corridors from optimization function",
    )
with col4:
    simplified_hvdc = col4.toggle(
        "Allow simplified HVDC matching on failure ",
        value=False,
        help="If no convergence is found for MTU with HVDC matching, set the ATC to the lesser of the two in the pair to force matching",
        disabled="HVDC_MATCHING" not in which_constraints_to_enforce,
    )
with col5:
    translate_ptcs = col5.toggle(
        "Translate Statnett PTCs to CNECs for old data",
        help="When running GC-matrices from before 8, toggle this button to translate single-line CNECs defined as PTCs to CNECs",
    )

st.subheader("Problem relaxation parameters")
(
    col1,
    col2,
    col3,
) = st.columns([1, 1, 1])
with col1:
    cnec_ram = col1.number_input("How much CNEC ram relaxation", 0.0, 100.0, 10.0)
with col2:
    ptc_ram = col2.number_input("How much PTC ram relaxation", 0.0, 100.0, 0.01)
with col3:
    hvdc_ram = col3.number_input("How much HVDC ram relaxation", 0.0, 100.0, 0.01)

aac_relax = st.number_input("How much AAC relaxation", 0.0, 10.0, 0.001)

col1, col2 = st.columns([1, 1])
with col1:
    ptdf_relaxation_threshold = col1.number_input(
        "Threshold for PTDF relaxation",
        0.0,
        1.0,
        0.02,
        help="A value of 1 results in all PTDFs being relaxed, a value of 0 results in the inverse",
    )
with col2:
    ptdf_relaxation_multiplier = col2.number_input(
        "Multiplier for PTDF relaxation",
        0.0,
        1.0,
        0.0,
        help="The PTDFs smaller than the threshold is multiplied with this number - 0 is equivalent to eliminating the PTDF values",
    )


matrix = None
atce_df = None
mtus = []
atce_mtus = []
target_mtu = None
result = None
solutions_frame = None
flow_frame = None
base_ptdf_columns = None
all_parameters = None
time_range = [datetime(1970, 1, 1), datetime(2100, 1, 1)]

if gc_url:
    st.subheader("Runtime Options")
    matrix = get_matrix_from_url(gc_url)
    mtus = matrix["DatetimeCET"].unique()
    mtus.sort()

if matrix is not None:

    mtu_datetimes: list[datetime] = []
    # HACK: Transition from winter to summer time is not handled
    summertime_date: None | date = None
    for iso_datetime_str in mtus:
        if "24:00:00" in iso_datetime_str:
            iso_datetime_str = iso_datetime_str.replace("24:00:00", "00:00:00")
            base_date = datetime.strptime(iso_datetime_str, "%Y-%m-%d %H:%M:%S.%f")
            summertime_date = base_date
            corrected_datetime = base_date + timedelta(days=1)
        else:
            corrected_datetime = datetime.strptime(iso_datetime_str, "%Y-%m-%d %H:%M:%S.%f")

        cet_timezone = pytz.timezone('Europe/Berlin')
        cet_datetime = cet_timezone.localize(corrected_datetime)
        utc_datetime = cet_datetime.astimezone(pytz.utc)
        mtu_datetimes.append(utc_datetime)

    correct_mtu_datetimes = []
    for mtu in mtu_datetimes:
        if summertime_date is not None and mtu.day == summertime_date.day:
            correct_mtu_datetimes.append(mtu - timedelta(hours=1))
        else:
            correct_mtu_datetimes.append(mtu)
    
    mtu_datetimes = correct_mtu_datetimes
    time_range = st.slider(
        "Which MTUs to extract capacities for: ",
        min_value=min(mtu_datetimes),
        max_value=min(mtu_datetimes),
        value=(min(mtu_datetimes), max(mtu_datetimes)),
        step=timedelta(hours=1),
        format="MM/DD/YY - hh:mm",
    )

if matrix is not None and st.button("Run Computation"):
    config = OptimizationConfig(
        eliminate_internal_hvdc_ptdfs,
        eliminate_all_nordic_hvdc_ptdfs,
        which_hvdcs_to_eliminate,
        cnec_ram,
        ptc_ram,
        hvdc_ram,
        aac_relax,
        ptdf_relaxation_threshold,
        ptdf_relaxation_multiplier,
        allow_hail_mary,
        allow_forced_viability_of_aac,
    )

    constraint_opts = ConstraintOptions(
        "RAM" in which_constraints_to_enforce,
        "HVDC_MATCHING" in which_constraints_to_enforce,
        "MIN_AAC" in which_constraints_to_enforce,
        "MAX_AAC_FOR_NON_ID" in which_constraints_to_enforce,
        error_on_infeasible_constraints,
        simplified_hvdc,
    )

    progress = st.progress(
        0,
    )
    datafame_of_solutions = pd.DataFrame()
    dataframe_of_flows = pd.DataFrame()
    non_relaxed_parameters_for_mtu: dict[datetime, OptimizationParameters] = {}
    total_hours = (time_range[1] - time_range[0]).total_seconds() / 3600

    borders = matrix[matrix["Border"]]
    bz_names = borders['BIDDINGAREA_TO'].unique()
    
    seen_corridors = set()
    corridor_names = []

    for pair in list(combinations(bz_names, 2)):
        row = borders[(borders["BIDDINGAREA_FROM"] == pair[0]) & (borders["BIDDINGAREA_TO"] == pair[1])]
        if row.empty or (pair in seen_corridors):
            continue
        
        corridor_names.append(tuple(pair))
        row = borders[(borders["BIDDINGAREA_TO"] == pair[0]) & (borders["BIDDINGAREA_FROM"] == pair[1])]
        if row.empty:
            st.error(f'No corridor {pair[1]} {pair[0]}')
        
        corridor_names.append(tuple((pair[1], pair[0])))
        
        seen_corridors.add(tuple((pair[0], pair[1])))
        seen_corridors.add(tuple((pair[1], pair[0])))
        

    optimized_mtus = 0
    for i, mtu in enumerate(mtu_datetimes):
        if (mtu < time_range[0]) or (mtu > time_range[1]):
            continue

        progress.progress(min(optimized_mtus / total_hours, 1), f"Solving for {mtu}")

        mtu_cet_string = mtus[i]
        matrix_for_mtu = matrix[matrix["DatetimeCET"] == mtu_cet_string]
        parameters: OptimizationParameters = load_gc_matrix_with_ptdfs_and_aac(matrix_for_mtu, corridor_names, mtu, config)
        non_relaxed_parameters: OptimizationParameters = load_gc_matrix_with_ptdfs_and_aac(matrix_for_mtu, corridor_names, mtu, config, False)

        result, flows_at_mtu, prob = compute_optimal_ntcs_for_mtu(parameters, config, constraint_opts)
        dataframe_solutions_for_mtu = pd.DataFrame(result.__dict__)
        dataframe_solutions_for_mtu["DatetimeUTC"] = mtu
        dataframe_solutions_for_mtu["ProblemStatus"] = prob.status

        if prob.status != "optimal":
            st.error(f"No solution for {mtu}")

        dataframe_flows_for_mtu = pd.DataFrame(flows_at_mtu.__dict__)
        dataframe_flows_for_mtu["DatetimeUTC"] = mtu

        optimized_mtus += 1
        datafame_of_solutions = pd.concat([datafame_of_solutions, dataframe_solutions_for_mtu])
        dataframe_of_flows = pd.concat([dataframe_of_flows, dataframe_flows_for_mtu])
        non_relaxed_parameters_for_mtu[mtu] = non_relaxed_parameters

    if not datafame_of_solutions.empty:
        st.session_state.dataframe_of_solutions = datafame_of_solutions.reset_index()
        st.session_state.dataframe_of_flows = dataframe_of_flows.reset_index()
        st.session_state.non_relaxed_parameters_for_mtu = non_relaxed_parameters_for_mtu


if "dataframe_of_solutions" in st.session_state:
    solutions_frame = st.session_state.dataframe_of_solutions
    flow_frame = st.session_state.dataframe_of_flows
    all_non_relaxed_parameters_for_mtu = st.session_state.non_relaxed_parameters_for_mtu

if solutions_frame is not None:
    csv = convert_atce_to_csv(solutions_frame)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with ZipFile(Path(tmp_dir) / f"SN_tool_ATCE_results_RAM1_{cnec_ram}_RAM2_{ptc_ram}_PTDF{ptdf_relaxation_threshold}.zip", "w") as zip_file: 
            zip_file.writestr(f"SN_tool_ATCE_results_RAM1_{cnec_ram}_RAM2_{ptc_ram}_PTDF{ptdf_relaxation_threshold}.csv", csv)
        with open(Path(tmp_dir) / f"SN_tool_ATCE_results_RAM1_{cnec_ram}_RAM2_{ptc_ram}_PTDF{ptdf_relaxation_threshold}.zip", 'rb') as outfile:
            st.download_button(
        label="Download ATCE results as zipped CSV",
        data=outfile,
        file_name=f'SN_tool_ATCE_results_RAM1_{cnec_ram}_RAM2_{ptc_ram}_PTDF{ptdf_relaxation_threshold}.zip',
        mime='application/zip')

if atce_url:
    try:
        atce_df = download_and_concatenate_csv(atce_url)
    except Exception as e:
        st.error(f"Error occurred: {e}")

if uploaded_file:
    try:
        zip_data = uploaded_file.read()
        atce_df = extract_zip_data(zip_data)
    except Exception as e:
        st.error(f"Error occurred: {e}")

if atce_df is not None:
    atce_df[('MTU', 'MTU')] = pd.to_datetime(atce_df[('MTU', 'MTU')])
    atce_df.drop(('Backup', 'Backup'), axis=1, inplace=True, errors='ignore')
    atce_df.set_index(('MTU', 'MTU'), inplace=True)
    atce_df.index.names = ['MTU']
    if eliminate_all_nordic_hvdc_ptdfs:
        atce_df.rename(columns=atce_border_name_dict_ac_conn, inplace=True)
    else:
        atce_df.rename(columns=atce_border_name_dict_hvdc, inplace=True)
    atce_df = atce_df.loc[:,~atce_df.columns.duplicated()]
    atce_df.drop(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), axis=1, inplace=True, errors='ignore')
    cet_tz = timezone('CET')
    if atce_df.index.tz is None:
        atce_df.index = atce_df.index.tz_localize('UTC')
    date_rng_cet = atce_df.index.tz_convert('UTC')
    atce_df.index = date_rng_cet
    time_range_mask = (atce_df.index >= time_range[0]) & (atce_df.index <= time_range[-1])
    atce_df = atce_df[time_range_mask]
    first_row_values = list(set(col[0] for col in atce_df.columns))


st.header('Resulting capacity plots')

if (
    solutions_frame is not None
    and flow_frame is not None
    and all_non_relaxed_parameters_for_mtu is not None
    and matrix is not None
):
    col1, col2 = st.columns([1, 1])
    with col1:
        mtu_0 = solutions_frame["MTU"].min()
        corridor_names = solutions_frame[solutions_frame["MTU"] == mtu_0]["BorderName"]
        corridors = np.array(corridor_names).reshape((len(corridor_names) // 2, 2))
        border_to_plot_for = col1.selectbox("Select border to plot for", corridor_names)
        pair = [pair for pair in corridors if border_to_plot_for in pair][0]

        plot_subset = solutions_frame[solutions_frame["BorderName"].isin(pair)]

        aac = solutions_frame[solutions_frame["BorderName"] == border_to_plot_for]["AAC"]
        aac_mtus = solutions_frame[solutions_frame["BorderName"] == border_to_plot_for]["MTU"]

        fig = go.Figure()
        plot_mtus = plot_subset[plot_subset["BorderName"] == border_to_plot_for]["MTU"]
        ntc_over = plot_subset[plot_subset["BorderName"] == border_to_plot_for]["NTC"]
        ntc_under = -1 * plot_subset[plot_subset["BorderName"] != border_to_plot_for]["NTC"]
        fig.add_trace(
            go.Scatter(x=plot_mtus, y=ntc_over, mode="lines", showlegend=False, line=dict(color="rgba(255,255,255,0)"))
        )
        fig.add_trace(
            go.Scatter(
                x=plot_mtus,
                y=ntc_under,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(135, 206, 235, 0.4)",
                line=dict(color="rgba(255,255,255,0)"),
                name="NTC",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=aac_mtus,
                y=aac,
                mode="markers",
                marker_symbol="x",
                marker_line_color="black",
                marker_color="red",
                marker_line_width=2,
                marker_size=10,
                name="AAC",
            )
        )

        if atce_df is not None:
            rcc_border = border_to_plot_for.replace("z2z_", "")

            match = re.search(r'DC_(\D*)(\d+)-(\D+)?(\d+)?$', rcc_border)
            if match:
                pair_a = (match.group(1) if match.group(1) else '') + match.group(2)
                pair_a = pair_a.replace("DC_","")
                pair_b = (match.group(3) if match.group(3) else '') + (match.group(4) if match.group(4) else '')
                columns_to_select = [
                (f"DC_{pair_a}-{pair_b}", "AAC"),
                (f"DC_{pair_a}-{pair_b}", "NTC_final"),
                (f"DC_{pair_b}-{pair_a}", "AAC"),
                (f"DC_{pair_b}-{pair_a}", "NTC_final"),
            ]
            else:
                pair_parts = rcc_border.split("-")
                if len(pair_parts) == 2:
                    pair_a, pair_b = pair_parts[0], pair_parts[1]
                else:
                    pair_a, pair_b = pair_parts[0], pair_parts[1].split('-')[1]
                columns_to_select = [
                    (f"{pair_a}-{pair_b}", "AAC"),
                    (f"{pair_a}-{pair_b}", "NTC_final"),
                    (f"{pair_b}-{pair_a}", "AAC"),
                    (f"{pair_b}-{pair_a}", "NTC_final"),
                ]

            filtered_df = atce_df.loc[:, columns_to_select]
            filtered_df.columns = [
                f"AAC_{pair_a}-{pair_b}",
                f"NTC_final_{pair_a}-{pair_b}",
                f"AAC_{pair_b}-{pair_a}",
                f"NTC_final_{pair_b}-{pair_a}",
            ]

            fig.add_trace(
                go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df[f"NTC_final_{pair_a}-{pair_b}"],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="rgba(120,120,120,0)"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=filtered_df.index,
                    y=-filtered_df[f"NTC_final_{pair_b}-{pair_a}"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(119, 50, 50, 0.4)",
                    line=dict(color="rgba(120,120,120,0)"),
                    name=f"NTC from RCC calculation",
                )
            )

        failed_mtus = solutions_frame[solutions_frame["ProblemStatus"] != "optimal"]
        for failed_mtu in failed_mtus["MTU"]:
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=failed_mtu,
                y0=0,
                x1=failed_mtu,
                y1=1,
                line=dict(color="Red", width=2),
                name="Failed MTU",
            )

        col1.plotly_chart(fig)
    with col2:
        zones_in_border = [name.replace('z2z_', '').split('-') for name in corridor_names]
        zones = sorted(list(set([zone for zones in zones_in_border for zone in zones])))
        selected_zones = col2.multiselect('Select bidding zone(s) to plot min/max NP for', zones, )
        zone_corridors: dict[str, tuple[list[str], list[str]]] = {}

        for selected_zone in selected_zones:
            export_ntcs: list[str]= []
            import_ntcs: list[str]= []

            seen_corridors: set[tuple[str, str]] = set()
            for name in corridor_names:
                zone1, zone2 = name.replace('z2z_', '').split('-')
                pair = (zone1, zone2)
                if selected_zone not in pair:
                    continue
                
                if pair in seen_corridors:
                    continue
                
                if zone1 == selected_zone:
                    export_ntcs.append(name)
                else:
                    import_ntcs.append(name)

                seen_corridors.add(pair)

            zone_corridors[selected_zone] = (export_ntcs, import_ntcs)

        fig = go.Figure() 
        for selected_zone in selected_zones:
            export_subset = solutions_frame[solutions_frame["BorderName"].isin(zone_corridors[selected_zone][0])]
            import_subset = solutions_frame[solutions_frame["BorderName"].isin(zone_corridors[selected_zone][1])]

            export_sum = export_subset[['DatetimeUTC', 'NTC']].groupby('DatetimeUTC').sum()
            import_sum = import_subset[['DatetimeUTC', 'NTC']].groupby('DatetimeUTC').sum()

            fig.add_trace(
                go.Scatter(
                    x=export_sum.index,
                    y=export_sum['NTC'],
                    name=f'Max Net Position for {selected_zone}'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=import_sum.index,
                    y=import_sum['NTC'],
                    name=f'Minimum Net Position for {selected_zone}'
                )
            )

        
        if atce_df is not None:
            for selected_zone in selected_zones:
                
                import_rcc_borders = [border.replace("z2z_", "") for border in zone_corridors[selected_zone][0]]
                export_rcc_borders = [border.replace("z2z_", "") for border in zone_corridors[selected_zone][1]]

                import_pair_parts = [pair.split("-") for pair in import_rcc_borders]
                export_pair_parts = [pair.split("-") for pair in export_rcc_borders]

                import_to_select = [
                    (f"{pair[0]}-{pair[1]}", "NTC_final") for pair in import_pair_parts
                ]
                export_to_select = [
                    (f"{pair[0]}-{pair[1]}", "NTC_final") for pair in export_pair_parts
                ]

                import_filtered_df = atce_df.loc[:, [col for col in import_to_select if col in atce_df.columns]]
                export_filtered_df = atce_df.loc[:, [col for col in export_to_select if col in atce_df.columns]]
                fig.add_trace(
                    go.Scatter(
                        x=export_filtered_df.index,
                        y=export_filtered_df.sum(axis=1),
                        name=f'Alternative / Minimum Net Position for {selected_zone}'
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=import_filtered_df.index,
                        y=import_filtered_df.sum(axis=1),
                        name=f'Alternative / Maximum Net Position for {selected_zone}'
                    )
                )

        col2.plotly_chart(fig)
        
        # corridor_names = solutions_frame[solutions_frame["MTU"] == mtu_0]["BorderName"]

st.header('Information on binding grid constraints')

if solutions_frame is not None and flow_frame is not None and all_non_relaxed_parameters_for_mtu is not None:

    col1, col2 = st.columns([1, 1])
    with col2:
        only_binding = col2.toggle("Only show objects that are binding")
        only_cnecs_on_current_border = col2.toggle(f"Only show CNECs that have a PTDF on {border_to_plot_for}")

        selected_mtu: str = col2.select_slider(
            "Show constraining CNECs for MTU", solutions_frame["DatetimeUTC"].unique()
        )
        data_for_mtu: pd.DataFrame = solutions_frame[solutions_frame["DatetimeUTC"] == selected_mtu]
        ntcs_for_mtu = data_for_mtu["NTC"].to_numpy()
        parameters = all_non_relaxed_parameters_for_mtu[selected_mtu]

        base_ptdf_columns = parameters.ptdf_columns
        mtu_nonredundant_matrix = parameters.gc_matrix[parameters.gc_matrix["Non_Redundant"]]
        ptdfs = pd.DataFrame(parameters.ptdfs, columns=parameters.ptdf_columns)
        mtu_nonredundant_matrix = pd.concat(
            [ptdfs.reset_index(drop=True), mtu_nonredundant_matrix[["JAO_CNEC_Name", "RAM_FB"]].reset_index(drop=True)],
            axis=1,
        )

        largest_ptdfs = top_three_columns(ptdfs)
        ram_remainder = mtu_nonredundant_matrix["RAM_FB"].to_numpy() - np.einsum(
            "ij, kj", ntcs_for_mtu.reshape((1, ntcs_for_mtu.shape[0])), parameters.ptdfs
        )

        overload_frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Remaining RAM on CNEC": np.squeeze(ram_remainder),
                        "CNEC": mtu_nonredundant_matrix["JAO_CNEC_Name"],
                    }
                ),
                largest_ptdfs,
            ],
            axis=1,
        )
        overload_frame = overload_frame.sort_values("Remaining RAM on CNEC", ascending=True).reset_index(drop=True)

        if only_cnecs_on_current_border:
            allowed_cnecs = mtu_nonredundant_matrix[mtu_nonredundant_matrix[border_to_plot_for] > 0]["JAO_CNEC_Name"]
            overload_frame = overload_frame[overload_frame["CNEC"].isin(allowed_cnecs)]

        if only_binding:
            overload_frame = overload_frame[overload_frame["Remaining RAM on CNEC"] < 1e-2]

        col2.dataframe(overload_frame)

    with col1:
        cnec_to_plot_for = col1.selectbox(
            "Select CNEC/PTC to plot flow and RAM on for period", flow_frame["CnecName"].unique()
        )
        flow_frame_for_cnec = flow_frame[flow_frame["CnecName"] == cnec_to_plot_for]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=flow_frame_for_cnec["MTU"],
                y=flow_frame_for_cnec["FLOW"],
                mode="lines+markers",
                name="Flow on object",
                line=dict(width=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=flow_frame_for_cnec["MTU"], y=flow_frame_for_cnec["RAM"], mode="lines+markers", name="RAM on object"
            )
        )
        col1.plotly_chart(fig)

