# -*- coding: utf-8 -*-
"""Shared paths/constants for the Piedmont Healthcare (Synthea) analysis."""

from pathlib import Path

DATA_PATH = Path(r"C:\Users\ashut\OneDrive\Documents\Ecommerce analysis and Healthcare overlap\Synthea")
OUTPUT_PATH = Path(__file__).parent / "piedmont_output"
OUTPUT_PATH.mkdir(exist_ok=True)

# RFM reference date: anchor "recency" to the day after the last encounter in the
# data (not today's date) since this is a static historical extract, not a live feed.
RFM_REFERENCE_DATE = None  # resolved at runtime in piedmont_rfm.py from the data itself
