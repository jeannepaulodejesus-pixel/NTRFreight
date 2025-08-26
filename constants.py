"""
Constants used across the NTR Freight tariff processing scripts.
"""

import re

# Client & Carrier Info
CLIENT_NAME = "Intellyse bePro"
CARRIER_NAME = "NTR Freight"

# Matching strategy
MATCHING_STRATEGY = "zip_prefix_match"  # mirrors the example; we keep country granularity

# Service Types
SERVICE_EXP_WW = "Express Worldwide"
SERVICE_IMP_WW = "Express Worldwide"
SERVICE_DOMESTIC_3RD = "Domestic Express Third Country"
SERVICE_ECONOMY = "Economy Select"

# Vendor sheet names
SHEET_EXP_WW = "CH TD Exp WW"
SHEET_IMP_WW = "CH TD Imp WW"
SHEET_DOM_3RD = "CH TD 3rdCty Domestic"
SHEET_ECON_EXP = "CH DD Exp Economy"
SHEET_ZONES_EXP_IMP = "ZH Zones TDI Exp+Imp"
SHEET_ZONES_DOM_3RD = "CH Zones TD 3rdCty Domestic"
SHEET_ZONES_ECON_EXP = "CH Zones DDI Export"

# Units
UNITS_WEIGHT = "kg"
UNITS_PRICE = "flat"

# Regex patterns
UNITS_HEADER_RE = re.compile(r"^\d+(\.\d+)?_up_to_\d+(\.\d+)?\[[^\]]+\]\[[^\]]+\]$")
