from utils.helper import *
from utils.logger import get_logger

import re
import json
import logging

import pandas as pd

from constants import (
    UNITS_HEADER_RE
)

# Logging
logger = get_logger('NTR to Standard')

def validate_workbook(regions_df: pd.DataFrame, tariffs_df: pd.DataFrame, surcharges_df: pd.DataFrame) -> None:
    """
    Perform validation checks per spec. Raise RuntimeError on hard failures.
    """
    # Required sheets minimal columns
    req_regions_cols = {"id", "client", "carrier", "country", "zipcode", "identifier_string"}
    req_tariffs_cols = {"id", "client", "carrier", "route", "currency", "service_type"}

    missing_r = req_regions_cols - set(regions_df.columns)
    missing_t = req_tariffs_cols - set(tariffs_df.columns)
    if missing_r:
        raise RuntimeError(f"Regions missing required columns: {missing_r}")
    if missing_t:
        raise RuntimeError(f"Tariffs missing required columns: {missing_t}")

    # Regions duplicate check (country+zipcode+identifier_string)
    dup_mask = regions_df.duplicated(subset=["country", "zipcode", "identifier_string"], keep=False)
    if dup_mask.any():
        dup_rows = regions_df[dup_mask]
        raise RuntimeError(f"Duplicate Regions rows detected:\n{dup_rows}")

    # Route references existing Regions.id
    id_set = set(regions_df["id"].astype(int).tolist())
    bad_routes = []
    for i, r in tariffs_df.iterrows():
        try:
            route = json.loads(r["route"])
        except Exception:
            bad_routes.append((i, r["route"]))
            continue
        # route can be list of lists; flatten
        ids = []
        for seg in route:
            if isinstance(seg, list):
                ids.extend(seg)
            elif isinstance(seg, int):
                ids.append(seg)
        if not all(int(x) in id_set for x in ids):
            bad_routes.append((i, route))
    if bad_routes:
        raise RuntimeError(f"Tariffs route references unknown Regions ids: {bad_routes}")

    # Units/limits header syntax
    rate_cols = [c for c in tariffs_df.columns if re.search(r"_up_to_.*\[", c)]
    for c in rate_cols:
        if not UNITS_HEADER_RE.match(c):
            raise RuntimeError(f"Invalid rate column header syntax: {c}")

    logger.info("Validation passed.")
