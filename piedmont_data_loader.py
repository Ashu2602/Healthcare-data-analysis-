# -*- coding: utf-8 -*-
"""Load the real Synthea CSV extract (no synthetic/fabricated records)."""

import pandas as pd

from piedmont_config import DATA_PATH


def load_data() -> dict[str, pd.DataFrame]:
    """Load core tables with correct dtypes so downstream code never guesses column names."""
    patients = pd.read_csv(
        DATA_PATH / "patients.csv",
        parse_dates=["BIRTHDATE", "DEATHDATE"],
    )
    encounters = pd.read_csv(
        DATA_PATH / "encounters.csv",
        parse_dates=["START", "STOP"],
    )
    conditions = pd.read_csv(
        DATA_PATH / "conditions.csv",
        parse_dates=["START", "STOP"],
    )
    medications = pd.read_csv(
        DATA_PATH / "medications.csv",
        parse_dates=["START", "STOP"],
    )
    procedures = pd.read_csv(
        DATA_PATH / "procedures.csv",
        parse_dates=["DATE"],
    )

    # Synthea timestamps are UTC ('...Z'); drop the tz so they compare cleanly
    # against tz-naive reference dates used later in the RFM calculation.
    for df, cols in [
        (encounters, ["START", "STOP"]),
        (conditions, ["START", "STOP"]),
        (medications, ["START", "STOP"]),
        (procedures, ["DATE"]),
    ]:
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

    return {
        "patients": patients,
        "encounters": encounters,
        "conditions": conditions,
        "medications": medications,
        "procedures": procedures,
    }


if __name__ == "__main__":
    data = load_data()
    print("=" * 70)
    print("LOADED SYNTHEA DATA (real records only)")
    print("=" * 70)
    for name, df in data.items():
        print(f"  {name:12s} {len(df):>8,} rows  x  {len(df.columns)} cols")
