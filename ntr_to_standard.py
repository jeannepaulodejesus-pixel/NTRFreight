from utils.packages import install_packages
from utils.arg_parser import get_args
from utils.helper import *
from utils.classes import RegionKey, RateBlock
from utils.validation import validate_workbook
from utils.logger import get_logger

# Disable FututeWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Install dependencies first
install_packages()

# Get commandline arguments "input and output filenames"
filenames=get_args()


# Import Packages
import re
import json
import math
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

# Logging
logger = get_logger('Main')

# -------------------------------
# Constants
# -------------------------------
from constants import (
    CLIENT_NAME,
    CARRIER_NAME,
    SERVICE_EXP_WW,
    SERVICE_IMP_WW,
    SERVICE_DOMESTIC_3RD,
    SERVICE_ECONOMY,
    SHEET_EXP_WW,
    SHEET_IMP_WW,
    SHEET_DOM_3RD,
    SHEET_ECON_EXP,
    SHEET_ZONES_EXP_IMP,
    SHEET_ZONES_DOM_3RD,
    SHEET_ZONES_ECON_EXP
)


def parse_zone_tables(zones_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Parse zone mapping sheets which often contain repeated sections.

    Returns list of (country_label, zone_label) where country_label includes code like 'Switzerland (CH)'.
    """
    pairs: List[Tuple[str, str]] = []
    # Find header rows that contain 'Country'
    header_rows = zones_df.applymap(lambda v: isinstance(v, str) and v.strip().lower() == "country").any(axis=1)
    if not header_rows.any():
        header_idx = zones_df[zones_df.iloc[:, 0].astype(str).str.strip().str.lower() == "country"].index
    else:
        header_idx = zones_df[header_rows].index

    if len(header_idx) == 0:
        logger.warning("Zone sheet: couldn't find 'Country' header. Returning empty mapping.")
        return pairs

    header_row = int(header_idx[0])
    header = zones_df.iloc[header_row]

    # Build list of (country_col, zone_col) index pairs across the row
    col_pairs = []
    for i, val in enumerate(header):
        if isinstance(val, str) and val.strip().lower() == "country":
            # Find the nearest subsequent 'Zone'
            for j in range(i + 1, zones_df.shape[1]):
                if isinstance(zones_df.iloc[header_row, j], str) and zones_df.iloc[header_row, j].strip().lower() == "zone":
                    col_pairs.append((i, j))
                    break

    # Iterate rows below header row collecting mappings
    r = header_row + 1
    while r < zones_df.shape[0]:
        row = zones_df.iloc[r]
        empty_row = True
        for (c_col, z_col) in col_pairs:
            country = row[c_col]
            zone = row[z_col] if z_col < len(row) else None
            if isinstance(country, str) and country.strip():
                empty_row = False
                zone_label = str(zone).strip() if not (zone is None or (isinstance(zone, float) and math.isnan(zone))) else ""
                if zone_label == "" and z_col + 1 < len(row):
                    z2 = row[z_col + 1]
                    if isinstance(z2, (str, int, float)):
                        zone_label = str(z2).strip()
                if zone_label:
                    pairs.append((country.strip(), zone_label))
        if empty_row:
            break
        r += 1

    return pairs


def build_regions_from_zone_maps(zone_pairs_lists: List[List[Tuple[str, str]]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build Regions sheet and return (regions_df, countrycode->region_id mapping).
    Deduplicate across all zone maps.
    """
    region_keys: Dict[RegionKey, None] = {}

    # Always include Switzerland region explicitly
    region_keys[RegionKey("CH", "Switzerland", "")] = None

    for pairs in zone_pairs_lists:
        for country_label, _zone in pairs:
            name, code = extract_country_code(country_label)
            if not code:
                code_match = re.search(r"\(([A-Z]{2})\)$", country_label)
                if code_match:
                    code = code_match.group(1)
            region_keys[RegionKey(code or name, name, "")] = None

    # Deterministic ordering by RegionKey dataclass (country_code, name, zipcode)
    sorted_keys = sorted(region_keys.keys())

    rows = []
    for idx, key in enumerate(sorted_keys, start=1):
        rows.append({
            "id": idx,
            "client": CLIENT_NAME,
            "carrier": CARRIER_NAME,
            "country": key.country_code,
            "zipcode": "",   # vendor provides no zip prefixes
            "city": "",
            "airport": "",
            "seaport": "",
            "identifier_string": ""
        })
    regions_df = pd.DataFrame(rows)

    code_to_id = {}
    for row in rows:
        code_to_id[row["identifier_string"]] = row["id"]
        # map ISO code, if present
        if len(row["identifier_string"]) == 2 and row["identifier_string"].isalpha():
            code_to_id[row["identifier_string"]] = row["id"]
        # also map country name
        code_to_id[row["country"]] = row["id"]

    return regions_df, code_to_id


def parse_rate_blocks(df: pd.DataFrame) -> List[RateBlock]:
    """
    Parse a tariff sheet into one or more blocks.
    A block is identified by a header row where first cell is 'KG' and subsequent columns contain 'Zone'.
    Title for the block is the text in the previous non-empty row's first cell.
    """
    blocks: List[RateBlock] = []
    nrows, ncols = df.shape

    # Find candidate header rows
    for r in range(nrows):
        first = df.iat[r, 0]
        if isinstance(first, str) and first.strip().upper() == "KG":
            # collect zone columns
            zone_cols: Dict[str, int] = {}
            for c in range(1, ncols):
                val = df.iat[r, c]
                if isinstance(val, str) and "zone" in val.lower():
                    zone_label = val.strip()
                    zone_cols[zone_label] = c
            if zone_cols:
                # find title on previous non-empty row
                title = ""
                rr = r - 1
                while rr >= 0:
                    prev = df.iat[rr, 0]
                    if isinstance(prev, str) and prev.strip():
                        title = prev.strip()
                        break
                    rr -= 1
                # collect subsequent numeric rows until spacer/blank
                weights: List[float] = []
                data_rows: List[int] = []
                rr = r + 1
                while rr < nrows:
                    w = try_float(df.iat[rr, 0])
                    if w is None:
                        break
                    weights.append(float(w))
                    data_rows.append(rr)
                    rr += 1
                if weights:
                    blocks.append(RateBlock(title=title, header_row=r, weight_values=weights, zone_cols=zone_cols, data_rows=data_rows))

    if not blocks:
        logger.warning("No rate blocks detected in tariff sheet.")
    return blocks



def build_zone_mappings(vendor: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns dict:
      {
        'EXP_IMP': {'Zone 1': [ 'CH','DE',... ], ...},
        'DOM_3RD': {'Zone 1': [ 'DE', ... ], ...},
        'ECON_EXP': {'Zone 1': [ 'DE', ... ], ...},
      }
    Values are country ISO codes when available else country names.
    """
    def map_pairs(pairs: List[Tuple[str, str]], prefix: str = "Zone ") -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for country_label, z in pairs:
            zlabel = f"{prefix}{z}" if not str(z).startswith(prefix) else str(z)
            name, code = extract_country_code(country_label)
            key = code or name
            out.setdefault(zlabel, []).append(key)
        return out

    # Parse three zone sheets
    exp_imp_pairs = parse_zone_tables(vendor[SHEET_ZONES_EXP_IMP])
    dom_3rd_pairs = parse_zone_tables(vendor[SHEET_ZONES_DOM_3RD])
    econ_pairs = parse_zone_tables(vendor[SHEET_ZONES_ECON_EXP])

    return {
        "EXP_IMP": map_pairs(exp_imp_pairs, "Zone "),
        "DOM_3RD": map_pairs(dom_3rd_pairs, "Zone "),
        "ECON_EXP": map_pairs(econ_pairs, "Zone "),
    }


def build_regions(vendor: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build Regions sheet and mapping (country code/name) -> region_id."""
    # Collect all pair lists for deduplication
    all_pairs_lists = [
        parse_zone_tables(vendor[SHEET_ZONES_EXP_IMP]),
        parse_zone_tables(vendor[SHEET_ZONES_DOM_3RD]),
        parse_zone_tables(vendor[SHEET_ZONES_ECON_EXP]),
    ]
    regions_df, code_to_id = build_regions_from_zone_maps(all_pairs_lists)
    return regions_df, code_to_id

def build_tariffs_for_service(
    vendor: Dict[str, pd.DataFrame],
    regions_map: Dict[str, int],
    zone_map: Dict[str, List[str]],
    sheet_name: str,
    service_type: str,
    route_mode: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    route_mode:
      - 'EXPORT': route = [CH, dest]
      - 'IMPORT': route = [src, CH]
      - 'DOM_3RD': route = [ctry, ctry]  # domestic in third country
    """
    df = vendor[sheet_name]
    currency = find_currency(df)

    blocks = parse_rate_blocks(df)
    if not blocks:
        logger.warning(f"No rate blocks found in sheet '{sheet_name}'. Tariffs will be empty.")
        return pd.DataFrame(), []

    zone_band_prices: Dict[str, Dict[str, float]] = {}
    all_headers: List[str] = []

    for blk in blocks:
        lower_start = block_lower_start(blk.title, default_lower=0.0)
        bands = build_band_headers(blk.weight_values, lower_start)

        for _rng, header in bands:
            all_headers.append(header)

        for zone_label, col_idx in blk.zone_cols.items():
            for (row_idx, (rng, header)) in zip(blk.data_rows, bands):
                price = try_float(df.iat[row_idx, col_idx])
                if price is None:
                    continue
                zone_band_prices.setdefault(zone_label, {})
                if header not in zone_band_prices[zone_label]:
                    zone_band_prices[zone_label][header] = float(price)

    band_headers = merge_band_columns(all_headers)

    rows = []
    ch_id = regions_map.get("CH") or regions_map.get("Switzerland")
    if not ch_id:
        raise RuntimeError("Switzerland region not found in Regions map.")

    for zone_label, band_map in zone_band_prices.items():
        dest_codes = zone_map.get(zone_label, [])
        if not dest_codes:
            m = re.search(r"Zone\s+([A-Za-z0-9]+)", zone_label, flags=re.IGNORECASE)
            key = f"Zone {m.group(1)}" if m else zone_label
            dest_codes = zone_map.get(key, [])

        for code in dest_codes:
            dest_id = regions_map.get(code) or regions_map.get(str(code)) or regions_map.get(code)
            if not dest_id:
                logger.warning(f"Unknown country code/name '{code}' for {zone_label}; skipping.")
                continue

            # One row per route
            if route_mode == "EXPORT":
                route = [ch_id, dest_id]
            elif route_mode == "IMPORT":
                route = [dest_id, ch_id]
            elif route_mode == "DOM_3RD":
                route = [dest_id, dest_id]
            else:
                route = [ch_id, dest_id]

            row = {
                "start_date": "",
                "end_date": "",
                "client": CLIENT_NAME,
                "carrier": CARRIER_NAME,
                "route": json.dumps(route, ensure_ascii=False),
                "currency": currency,
                "service_type": service_type,
                "ldm_conversion": np.nan,
                "cbm_conversion": np.nan,
                "min_price": np.nan,
                "max_price": np.nan,
            }
            for h in band_headers:
                row[h] = band_map.get(h, np.nan)
            rows.append(row)

    tariffs_df = pd.DataFrame(rows)
    if not tariffs_df.empty:
        tariffs_df.insert(0, "id", range(1, len(tariffs_df) + 1))

    return tariffs_df, band_headers


#------------------------------------------
# Orchestrator
#------------------------------------------
def main(in_path: str, out_path: str) -> None:
    vendor = read_vendor_workbook(in_path)

    # Regions (deduplicate across all relevant zone maps)
    regions_df, code_to_id = build_regions(vendor)

    # Build zone label -> country-code lists per service
    zone_maps = build_zone_mappings(vendor)

    # Tariffs per service type
    tdfs: List[pd.DataFrame] = []
    all_band_cols: List[str] = []

    # Switzerland Export — Express Worldwide
    t_exp, bands_exp = build_tariffs_for_service(
        vendor, code_to_id, zone_maps["EXP_IMP"], SHEET_EXP_WW, SERVICE_EXP_WW, "EXPORT"
    )
    if not t_exp.empty:
        tdfs.append(t_exp)
        all_band_cols.extend(bands_exp)

    # Switzerland Import — Express Worldwide
    t_imp, bands_imp = build_tariffs_for_service(
        vendor, code_to_id, zone_maps["EXP_IMP"], SHEET_IMP_WW, SERVICE_IMP_WW, "IMPORT"
    )
    if not t_imp.empty:
        # offset ids for determinism
        t_imp["id"] = range(int(tdfs[-1]["id"].max()) + 1 if tdfs else 1,
                            int(tdfs[-1]["id"].max()) + 1 + len(t_imp) if tdfs else 1 + len(t_imp))
        tdfs.append(t_imp)
        all_band_cols.extend(bands_imp)

    # Domestic Express Third Country
    t_dom, bands_dom = build_tariffs_for_service(
        vendor, code_to_id, zone_maps["DOM_3RD"], SHEET_DOM_3RD, SERVICE_DOMESTIC_3RD, "DOM_3RD"
    )
    if not t_dom.empty:
        t_dom["id"] = range(int(tdfs[-1]["id"].max()) + 1 if tdfs else 1,
                            int(tdfs[-1]["id"].max()) + 1 + len(t_dom) if tdfs else 1 + len(t_dom))
        tdfs.append(t_dom)
        all_band_cols.extend(bands_dom)

    # Switzerland Export — Economy Select
    t_econ, bands_econ = build_tariffs_for_service(
        vendor, code_to_id, zone_maps["ECON_EXP"], SHEET_ECON_EXP, SERVICE_ECONOMY, "EXPORT"
    )
    if not t_econ.empty:
        t_econ["id"] = range(int(tdfs[-1]["id"].max()) + 1 if tdfs else 1,
                             int(tdfs[-1]["id"].max()) + 1 + len(t_econ) if tdfs else 1 + len(t_econ))
        tdfs.append(t_econ)
        all_band_cols.extend(bands_econ)

    # Combine tariffs (align columns)
    if tdfs:
        tariffs_df = pd.concat(tdfs, ignore_index=True).fillna(np.nan)
        # Ensure deterministic column order: metadata + all bands sorted
        meta_cols = ["id","start_date","end_date","client","carrier",
                     "route","currency","service_type","ldm_conversion","cbm_conversion","min_price","max_price"]
        band_cols = merge_band_columns([c for c in tariffs_df.columns if c not in meta_cols])
        tariffs_df = tariffs_df.reindex(columns=meta_cols + band_cols)
    else:
        # Empty but with headers
        tariffs_df = pd.DataFrame(columns=[
            "id","start_date","end_date","client","carrier","route",
            "currency","service_type","ldm_conversion","cbm_conversion","min_price","max_price"
        ])

    # Create empty Surcharges with just headers
    surcharges_cols = ["id","start_date","end_date","client","carrier",
                       "route","currency","service_type","ldm_conversion","cbm_conversion",
                       "min_price","max_price","tariff_ids","dyn_function","category"]
    surcharges_df = pd.DataFrame(columns=surcharges_cols)

    # Validation
    validate_workbook(regions_df, tariffs_df, surcharges_df)

    # Write output
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        regions_df.to_excel(writer, sheet_name="Regions", index=False)
        tariffs_df.to_excel(writer, sheet_name="Tariffs", index=False)
        surcharges_df.to_excel(writer, sheet_name="Surcharges", index=False)

    logger.info(f"Wrote standardized workbook: {out_path}")


if __name__ == "__main__":
    main(filenames.input_file, filenames.output_file)
