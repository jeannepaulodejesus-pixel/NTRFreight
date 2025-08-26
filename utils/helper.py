from typing import List, Dict, Tuple, Optional, Any
from constants import (
    UNITS_WEIGHT,
    UNITS_PRICE
)

import re
import pandas as pd
import numpy as np



def read_vendor_workbook(path: str) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name, header=None) for name in xls.sheet_names}

def find_currency(sheet_df: pd.DataFrame) -> str:
    """
    Detect currency text like 'All Rates stated here are in CHF. 
    """
    mask = sheet_df.applymap(lambda v: isinstance(v, str) and "All Rates stated here are in " in v)
    if mask.any().any():
        row, col = np.argwhere(mask.values)[0]
        text = str(sheet_df.iloc[row, col])
        m = re.search(r"All Rates stated here are in\s+([A-Z]{3})", text)
        if m:
            return m.group(1)
    # fallback: token search
    text_all = " ".join([str(v) for v in sheet_df.values.flatten() if isinstance(v, str)])
    for cur in ("CHF", "EUR", "USD"):
        if cur in text_all:
            return cur
    return ""


def extract_country_code(label: str) -> Tuple[str, str]:
    """
    Given 'Switzerland (CH)' return ('Switzerland', 'CH').
    """
    m = re.match(r"^(.*)\(([A-Z]{2})\)\s*$", label)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return label.strip(), ""

def strip_float(x: float) -> str:
    s = f"{x}".rstrip("0").rstrip(".")
    return s if s else "0"


def merge_band_columns(all_headers: List[str]) -> List[str]:
    """
    Deterministic ordering of band headers by numeric low/high ascending.
    """
    def parse_header(h: str) -> Tuple[float, float]:
        m = re.match(r"^([0-9.]+)_up_to_([0-9.]+)\[", h)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        return (float("inf"), float("inf"))
    uniq = sorted(set(all_headers), key=parse_header)
    return uniq


def try_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
        return float(x)
    except Exception:
        return None

def block_lower_start(title: str, default_lower: float = 0.0) -> float:
    """
    Extract starting lower bound from title like 'Non-documents from 0.5 KG & Documents from 2.5 KG'.
    """
    if not isinstance(title, str):
        return default_lower
    m = re.search(r"from\s+(\d+(?:\.\d+)?)\s*KG", title, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return default_lower

def build_band_headers(edges: List[float], lower_start: float, step: float = 5.0) -> List[Tuple[Tuple[float, float], str]]:
    """
    Build weight bands in fixed increments (default = 5 kg).
    For example: 0→5, 5→10, 10→15 ...
    """
    if not edges:
        return []

    max_edge = max(edges)
    bands: List[Tuple[Tuple[float, float], str]] = []
    lo = lower_start

    # Generate fixed-step bands until we cover the max edge
    while lo < max_edge:
        hi = lo + step
        header = f"{strip_float(lo)}_up_to_{strip_float(hi)}[{UNITS_WEIGHT}][{UNITS_PRICE}]"
        bands.append(((lo, hi), header))
        lo = hi

    return bands