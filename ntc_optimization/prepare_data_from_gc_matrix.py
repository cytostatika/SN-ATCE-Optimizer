from datetime import datetime

import logging
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from ntc_optimization.dataclasses_and_config import OptimizationConfig, OptimizationParameters

from ntc_optimization.optimization_functions import ram_constraint


def extract_corridor_names_and_aac(matrix: pd.DataFrame, corridor_names: list[tuple[str, str]]):
    seen_opposite_pairs = set()
    borders = matrix[matrix["Border"]]
    non_existing_borders_in_matrix = []
    variabe_initial_values = []

    for pair in corridor_names:
        if tuple(pair) in seen_opposite_pairs:
            continue

        row = borders[(borders["BIDDINGAREA_FROM"] == pair[0]) & (borders["BIDDINGAREA_TO"] == pair[1])]
        if row.empty:
            logging.getLogger().warning(f'No data in GC Matrix for corridor {pair} on MTU {matrix["DatetimeCET"].iloc[0]} ')
            row = pd.DataFrame({
                'BIDDINGAREA_FROM': [pair[0]],
                'BIDDINGAREA_TO': [pair[1]],
                'FAAC_FB': [0],
            })
            non_existing_borders_in_matrix.append(pair)
            
        variabe_initial_values.append(float(row["FAAC_FB"].iloc[0]))

        pair = [pair[1], pair[0]]
        seen_opposite_pairs.add(tuple(pair))
        row = borders[(borders["BIDDINGAREA_FROM"] == pair[0]) & (borders["BIDDINGAREA_TO"] == pair[1])]
        if row.empty:
            row = pd.DataFrame({
                'BIDDINGAREA_FROM': [pair[1]],
                'BIDDINGAREA_TO': [pair[0]],
                'FAAC_FB': [0],
            })
        variabe_initial_values.append(float(row["FAAC_FB"].iloc[0]))

    #return corridor_names, variabe_initial_values
    return variabe_initial_values, non_existing_borders_in_matrix


def extract_ptdfs_from_gc_matrix(
    matrix: pd.DataFrame, corridor_names: list[tuple[str, str]]
) -> tuple[list[str], NDArray, NDArray]:
    ptdf_columns = [f"z2z_{pair[0]}-{pair[1]}" for pair in corridor_names]
    ptdfs = matrix[ptdf_columns].map(lambda x: float(x) if isinstance(x, str) else x).to_numpy()
    ram = matrix["RAM_FB"].apply(lambda x: float(x)).to_numpy()[matrix["Non_Redundant"]]
    ptdfs = ptdfs[matrix["Non_Redundant"]]

    return ptdf_columns, ptdfs, ram


def calculate_new_ptdfs_to_eliminate_virtual_zones(
    aac: NDArray, ptdfs: NDArray, ptdf_columns: list[str], eliminate_all_nordic_vbzs: bool
):
    synch_hvcdc_new_ptdf_names = [
        "z2z_SE3-FI",
        "z2z_FI-SE3",
        "z2z_DC_SE3-SE4",
        "z2z_DC_SE4-SE3",
    ]
    synch_area_hvdcs: list[tuple[str, str]] = [
        ("z2z_SE3-SE3_FS", "z2z_FI_FS-FI"),  # Fennoscan
        ("z2z_FI-FI_FS", "z2z_SE3_FS-SE3"),
        ("z2z_SE3-SE3_SWL", "z2z_SE4_SWL-SE4"),  # South west link
        ("z2z_SE4-SE4_SWL", "z2z_SE3_SWL-SE3"),
    ]

    if eliminate_all_nordic_vbzs:
        synch_area_hvdcs += [
            ("z2z_DK1-DK1_SK", "z2z_NO2_SK-NO2"),  # Skagerrak
            ("z2z_NO2-NO2_SK", "z2z_DK1_SK-DK1"),
            ("z2z_DK2-DK2_SB", "z2z_DK1_SB-DK1"),  # Storebelt
            ("z2z_DK1-DK1_SB", "z2z_DK2_SB-DK2"),
            ("z2z_DK1-DK1_KS", "z2z_SE3_KS-SE3"),  # Kontiskan
            ("z2z_SE3-SE3_KS", "z2z_DK1_KS-DK1"),
        ]
        synch_hvcdc_new_ptdf_names += [
            "z2z_DK1-NO2",
            "z2z_NO2-DK1",
            "z2z_DK2-DK1",
            "z2z_DK1-DK2",
            "z2z_DK1-SE3",
            "z2z_SE3-DK1",
        ]

    synch_hvdc_index_pairs: list[tuple[int, ...]] = []

    for pair in synch_area_hvdcs:
        index_pair: list[int] = []
        for element in pair:
            for i, col in enumerate(ptdf_columns):
                if col == element:
                    index_pair.append(i)
        synch_hvdc_index_pairs.append(tuple(index_pair))

    new_ptdf_colnames = []
    new_ptdfs = np.zeros((ptdfs.shape[0], ptdfs.shape[1] - len(synch_hvdc_index_pairs)))
    new_aac = np.zeros(new_ptdfs.shape[1])
    synch_hvdc_indices = [index for pair in synch_hvdc_index_pairs for index in pair]

    new_col_counter = 0
    for col in range(ptdfs.shape[1]):
        if not col in synch_hvdc_indices:
            new_ptdfs[:, new_col_counter] = ptdfs[:, col]
            new_ptdf_colnames.append(ptdf_columns[col])
            new_aac[new_col_counter] = aac[col]
            new_col_counter += 1

    for col_add, pair in enumerate(synch_hvdc_index_pairs):
        new_ptdfs[:, new_col_counter + col_add] = ptdfs[:, pair[0]] + ptdfs[:, pair[1]]
        new_ptdf_colnames.append(synch_hvcdc_new_ptdf_names[col_add])
        if aac[pair[0]] > 0:
            new_aac[new_col_counter + col_add] = aac[list(pair)].max()
        else:
            new_aac[new_col_counter + col_add] = aac[list(pair)].min()
    return new_ptdf_colnames, new_ptdfs, new_aac


def load_gc_matrix_with_ptdfs_and_aac(
    matrix: pd.DataFrame, corridor_names: list[str], mtu: datetime, config: OptimizationConfig, use_relaxations: bool = True
):
    mtus_in_matrix = matrix["DatetimeCET"].unique()
    if len(mtus_in_matrix) != 1:
        raise ValueError(f"{len(mtus_in_matrix)} unique MTUs in GC matrix, expected 1")
    aac_list, non_existing_borders_in_gcmatrix  = extract_corridor_names_and_aac(matrix, corridor_names)

    aac = np.array(aac_list)
    ptdf_columns, ptdfs, ram = extract_ptdfs_from_gc_matrix(matrix, corridor_names)
    if config.eliminate_internal_hvdc_vbzs:
        (
            new_ptdf_colnames,
            new_ptdfs,
            new_aac,
        ) = calculate_new_ptdfs_to_eliminate_virtual_zones(
            aac,
            ptdfs,
            ptdf_columns,
            config.eliminate_all_nordic_vbzs,
        )
        ptdf_columns = new_ptdf_colnames
        ptdfs = new_ptdfs
        aac = new_aac
        if config.eliminate_all_nordic_vbzs:
            hvdc_pairs = []
            hvdc_index_pairs = []
            hvdc_ilf = []
        else:
            hvdc_pairs, hvdc_index_pairs = make_hvdc_pairs_without_internal_connections(ptdf_columns)
            hvdc_ilf = [0.029, 0.029, 0, 0, 0, 0]
    else:
        hvdc_pairs, hvdc_index_pairs = make_hvdc_pairs_with_internal_connections(ptdf_columns)
        hvdc_ilf = [0.029, 0.029, 0, 0, 0, 0, 0, 0, 0, 0]

    ptdfs[ptdfs < 0] = 0

    hvdc_names = [
        "DK1_SK",
        "NO2_SK",  # Skagerrak
        "DK2_SB",
        "DK1_SB",  # Storebelt
        "SE3_KS",
        "DK1_KS",  # Kontiscan
        "SE3_FS",
        "FI_FS",  # Fennoscan
        "SE3_SWL",
        "SE4_SWL",  # South west link
        "FI_EL",
        "NO2_NK",
        "NO2_ND",
        "SE4_NB",
        "SE4_BC",
        "SE4_SP",
    ]
    nonredundant_matrix = matrix[matrix["Non_Redundant"]]
    is_cnec = nonredundant_matrix["JAO_Contin_Name"] != "BASECASE"
    is_hvdc = nonredundant_matrix["JAO_CNEC_Name"].str.contains("|".join(hvdc_names)) | nonredundant_matrix[
        "JAO_CNEC_Name"
    ].str.contains("|".join(["IMP", "EXP"]))
    is_ptc = (~is_cnec) & (~is_hvdc)

    if use_relaxations:
        ram[is_cnec] += config.cnec_ram_relaxation
        ram[is_hvdc] += config.hvdc_ram_relaxation
        ram[is_ptc] += config.ptc_ram_relaxation
        aac -= config.aac_relaxation
        ptdfs[ptdfs < config.ptdf_relaxation_threshold] *= config.ptdf_relaxation_multiplier

    epsilon = 1
    new_ram = compute_new_ram_inside_domain(aac, ptdfs, ram, epsilon, matrix["JAO_CNEC_Name"].to_numpy())

    if config.allow_enforced_viability_of_hvdc_matching:
        new_ram = enforce_viability_of_hvdc_matching_at_market_point(
            hvdc_index_pairs, hvdc_ilf, ptdfs, new_ram, aac, matrix["JAO_CNEC_Name"].to_numpy()
        )

    ram = new_ram
    return OptimizationParameters(mtu, matrix, ptdf_columns, hvdc_index_pairs, hvdc_pairs, hvdc_ilf, ram, aac, ptdfs)


def compute_new_ram_inside_domain(aac: NDArray, ptdfs: NDArray, ram: NDArray, epsilon: float, cnec_names: NDArray):
    new_ram = []
    for i, ram_val in enumerate(ram):
        ram_constr_val = ram_constraint(aac, ram_val, ptdfs[i, :], epsilon=0)
        if ram_constr_val < 0:
            new_ram_val = np.dot(aac, ptdfs[i, :]) + epsilon
            logging.getLogger().warning(
                f"{cnec_names[i]} gets new ram {new_ram_val}. Was {ram_val} which got exceeded with {ram_constr_val}"
            )
            new_ram.append(new_ram_val)
        else:
            new_ram.append(ram_val)
    return np.array(new_ram)


def enforce_viability_of_hvdc_matching_at_market_point(
    hvdc_pairs: list[tuple[int, int]],
    hvdc_ilf: list[float],
    ptdfs: NDArray,
    ram: NDArray,
    aac: NDArray,
    cnec_names: NDArray,
):
    new_ram = ram
    for which_hvdc, pair in enumerate(hvdc_pairs):
        i, j = pair
        flow_with_loss_at_j = (1 - hvdc_ilf[which_hvdc]) * aac[i]
        adjusted_aac = aac.copy()
        adjusted_aac[j] = flow_with_loss_at_j
        new_ram = compute_new_ram_inside_domain(adjusted_aac, ptdfs, new_ram, 1, cnec_names)

    return new_ram


def make_hvdc_pairs_with_internal_connections(ptdf_columns):
    hvdc_pairs = [
        ("z2z_DK1-DK1_SK", "z2z_NO2_SK-NO2"),  # Skagerrak
        ("z2z_NO2-NO2_SK", "z2z_DK1_SK-DK1"),
        ("z2z_DK2-DK2_SB", "z2z_DK1_SB-DK1"),  # Storebelt
        ("z2z_DK1-DK1_SB", "z2z_DK2_SB-DK2"),
        ("z2z_DK1-DK1_KS", "z2z_SE3_KS-SE3"),  # Kontiskan
        ("z2z_SE3-SE3_KS", "z2z_DK1_KS-DK1"),
        ("z2z_SE3-SE3_FS", "z2z_FI_FS-FI"),  # Fennoscan
        ("z2z_FI-FI_FS", "z2z_SE3_FS-SE3"),
        ("z2z_SE3-SE3_SWL", "z2z_SE4_SWL-SE4"),  # South west link
        ("z2z_SE4-SE4_SWL", "z2z_SE3_SWL-SE3"),
    ]

    hvdc_index_pairs = []

    for pair in hvdc_pairs:
        index_pair = []
        for element in pair:
            for i, col in enumerate(ptdf_columns):
                if col == element:
                    index_pair.append(i)
        hvdc_index_pairs.append(tuple(index_pair))
    return hvdc_pairs, hvdc_index_pairs


def make_hvdc_pairs_without_internal_connections(ptdf_columns):
    hvdc_pairs = [
        ("z2z_DK1-DK1_SK", "z2z_NO2_SK-NO2"),  # Skagerrak
        ("z2z_NO2-NO2_SK", "z2z_DK1_SK-DK1"),
        ("z2z_DK2-DK2_SB", "z2z_DK1_SB-DK1"),  # Storebelt
        ("z2z_DK1-DK1_SB", "z2z_DK2_SB-DK2"),
        ("z2z_DK1-DK1_KS", "z2z_SE3_KS-SE3"),  # Kontiskan
        ("z2z_SE3-SE3_KS", "z2z_DK1_KS-DK1"),
    ]

    hvdc_index_pairs = []

    for pair in hvdc_pairs:
        index_pair = []
        for element in pair:
            for i, col in enumerate(ptdf_columns):
                if col == element:
                    index_pair.append(i)
        hvdc_index_pairs.append(tuple(index_pair))
    return hvdc_pairs, hvdc_index_pairs
