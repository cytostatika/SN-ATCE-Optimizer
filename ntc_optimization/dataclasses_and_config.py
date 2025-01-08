from dataclasses import dataclass
import pandas as pd
import datetime
from numpy.typing import NDArray


@dataclass
class NTCSolutionForMTU:
    BorderName: list[str]
    MTU: list[datetime.datetime]
    NTC: list[float] | NDArray
    AAC: list[float] | NDArray


@dataclass
class ConstraintOptions:
    ram_constraints: bool
    hvdc_matching: bool
    minimum_aac: bool
    maximum_aac_for_non_id: bool
    error_on_feasability: bool
    allow_simplified_hvdc: bool


@dataclass
class OptimizationConfig:
    eliminate_internal_hvdc_vbzs: bool
    eliminate_all_nordic_vbzs: bool
    codes_to_eliminate_from_id: list[str]
    cnec_ram_relaxation: float
    ptc_ram_relaxation: float
    hvdc_ram_relaxation: float
    aac_relaxation: float
    ptdf_relaxation_threshold: float
    ptdf_relaxation_multiplier: float
    allow_hail_mary: bool
    allow_enforced_viability_of_hvdc_matching: bool


@dataclass
class OptimizationParameters:
    mtu: datetime.datetime
    gc_matrix: pd.DataFrame
    ptdf_columns: list[str]
    hvdc_index_pairs: list[tuple[int, int]]
    hvdc_pairs: list[tuple[str, str]]
    hvdc_ilf: list[float]
    ram: NDArray
    aac: NDArray
    ptdfs: NDArray


@dataclass
class FlowSituationforMTU:
    CnecName: list[str]
    MTU: list[datetime.datetime]
    RAM: NDArray
    FLOW: NDArray
