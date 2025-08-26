from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True, order=True)
class RegionKey:
    """
    Sorting key to get deterministic region ids.
    """
    country_code: str
    country_name: str
    zipcode: str = ""  


@dataclass
class RateBlock:
    title: str
    header_row: int
    weight_values: List[float]
    zone_cols: Dict[str, int]  # zone label -> dataframe column index
    data_rows: List[int]       # row indices containing weights
