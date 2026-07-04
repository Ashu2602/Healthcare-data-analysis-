# -*- coding: utf-8 -*-
"""Exploratory analysis on real Synthea data: demographics, encounters, conditions, medications."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from piedmont_config import OUTPUT_PATH
from piedmont_data_loader import load_data

sns.set_style("whitegrid")


def add_age(patients: pd.DataFrame) -> pd.DataFrame:
    # Use DEATHDATE where present so age reflects age-at-death, not a phantom current age.
    as_of = patients["DEATHDATE"].fillna(pd.Timestamp.today())
    patients = patients.copy()
    patients["AGE"] = ((as_of - patients["BIRTHDATE"]).dt.days // 365).astype(int)
    return patients


def print_demographics(patients: pd.DataFrame) -> None:
    print("=" * 70)
    print("PATIENT DEMOGRAPHICS")
    print("=" * 70)
    print(f"Total patients: {len(patients):,}")
    print(f"Age: min={patients['AGE'].min()}, max={patients['AGE'].max()}, "
          f"mean={patients['AGE'].mean():.1f}, median={patients['AGE'].median():.1f}")

    for col in ["GENDER", "RACE", "ETHNICITY"]:
        print(f"\n{col} distribution:")
        for val, count in patients[col].value_counts().items():
            print(f"  {val:12s} {count:5,} ({count / len(patients):.1%})")


def print_encounters(encounters: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ENCOUNTERS")
    print("=" * 70)
    visits_per_patient = encounters.groupby("PATIENT").size()
    print(f"Total encounters: {len(encounters):,}")
    print(f"Unique patients with encounters: {encounters['PATIENT'].nunique():,}")
    print(f"Visits per patient: mean={visits_per_patient.mean():.1f}, "
          f"median={visits_per_patient.median():.0f}, max={visits_per_patient.max()}")
    print(f"Date range: {encounters['START'].min().date()} to {encounters['START'].max().date()}")

    print("\nEncounter class distribution:")
    for enc_type, count in encounters["ENCOUNTERCLASS"].value_counts().items():
        print(f"  {enc_type:12s} {count:6,} ({count / len(encounters):.1%})")

    cost = encounters["TOTAL_CLAIM_COST"]
    print(f"\nTOTAL_CLAIM_COST: total=${cost.sum():,.0f}, mean=${cost.mean():,.2f}, "
          f"median=${cost.median():,.2f}")


def print_conditions(conditions: pd.DataFrame, patients: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CONDITIONS")
    print("=" * 70)
    print(f"Total diagnoses: {len(conditions):,}")
    print(f"Unique condition types: {conditions['DESCRIPTION'].nunique()}")
    print(f"Patients with >=1 condition: {conditions['PATIENT'].nunique():,} "
          f"of {len(patients):,}")

    # Report *distinct patients affected*, not raw diagnosis counts — conditions like
    # sinusitis recur, so a naive count/patients ratio can exceed 100%.
    print("\nTop 15 conditions (by distinct patients affected):")
    patients_per_condition = conditions.groupby("DESCRIPTION")["PATIENT"].nunique().sort_values(ascending=False)
    for i, (cond, patient_count) in enumerate(patients_per_condition.head(15).items(), 1):
        print(f"  {i:2d}. {cond:45s} {patient_count:5,} patients ({patient_count / len(patients):.1%})")


def print_medications(medications: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("MEDICATIONS")
    print("=" * 70)
    print(f"Total medication records: {len(medications):,}")
    print(f"Unique medications: {medications['DESCRIPTION'].nunique()}")
    print(f"Patients on medication: {medications['PATIENT'].nunique():,}")
    cost = medications["TOTALCOST"]
    print(f"TOTALCOST: total=${cost.sum():,.0f}, mean=${cost.mean():,.2f}")

    print("\nTop 10 medications:")
    for i, (med, count) in enumerate(medications["DESCRIPTION"].value_counts().head(10).items(), 1):
        print(f"  {i:2d}. {med:55s} {count:5,}")


def make_charts(patients: pd.DataFrame, encounters: pd.DataFrame, conditions: pd.DataFrame,
                 medications: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Piedmont Healthcare (Synthea) — Exploratory Data Analysis", fontsize=15, fontweight="bold")

    axes[0, 0].hist(patients["AGE"], bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("Patient Age Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Count")

    visits_per_patient = encounters.groupby("PATIENT").size()
    axes[0, 1].hist(visits_per_patient, bins=40, color="seagreen", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title("Encounters per Patient", fontweight="bold")
    axes[0, 1].set_xlabel("Number of Encounters")
    axes[0, 1].set_ylabel("Count")

    conditions["DESCRIPTION"].value_counts().head(12).plot(
        kind="barh", ax=axes[1, 0], color="coral", edgecolor="black"
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Top 12 Conditions", fontweight="bold")
    axes[1, 0].set_xlabel("Count")

    encounters["ENCOUNTERCLASS"].value_counts().plot(
        kind="bar", ax=axes[1, 1], color="mediumpurple", edgecolor="black"
    )
    axes[1, 1].set_title("Encounter Types", fontweight="bold")
    axes[1, 1].set_xlabel("Encounter Class")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out_file = OUTPUT_PATH / "01_eda_summary.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved chart: {out_file}")


def run() -> dict[str, pd.DataFrame]:
    data = load_data()
    data["patients"] = add_age(data["patients"])

    print_demographics(data["patients"])
    print_encounters(data["encounters"])
    print_conditions(data["conditions"], data["patients"])
    print_medications(data["medications"])
    make_charts(data["patients"], data["encounters"], data["conditions"], data["medications"])

    return data


if __name__ == "__main__":
    run()
