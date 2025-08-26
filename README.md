# NTRFreight
Translate NTR vendor tariffs into the standard tariff workbook with sheets: Regions, Tariffs, Surcharges.

Usage:
```bash
  python ntr_to_standard.py --in "NTR-Freight-Tariffs.xlsx" --out "out_standard_tariffs.xlsx"
```
Notes:
- Built to the case study spec and aligned with the provided example workbook.
- Handles messy Excel (merged headers, spacer rows, "Unnamed:" columns, title blocks).
- Deterministic IDs and robust parsing.
"""