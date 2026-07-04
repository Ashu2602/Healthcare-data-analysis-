# -*- coding: utf-8 -*-
"""RFM (Recency / Frequency / Monetary) segmentation on real Synthea patients."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from piedmont_config import OUTPUT_PATH

sns.set_style("whitegrid")


def build_rfm_table(patients: pd.DataFrame, encounters: pd.DataFrame) -> pd.DataFrame:
    # Anchor "today" to the day after the last encounter in the extract, not the real
    # current date — otherwise every patient looks inactive since this is a static 2020 dump.
    reference_date = encounters["START"].max() + pd.Timedelta(days=1)

    grouped = encounters.groupby("PATIENT").agg(
        RECENCY=("START", lambda s: (reference_date - s.max()).days),
        FREQUENCY=("Id", "count"),
        MONETARY=("TOTAL_CLAIM_COST", "sum"),
    )
    rfm = grouped.reindex(patients["Id"]).reset_index(names="PATIENT_ID")

    # Patients with zero encounters would break qcut and don't have a meaningful RFM
    # profile anyway, so drop them here rather than papering over with fabricated defaults.
    rfm = rfm.dropna(subset=["RECENCY", "FREQUENCY", "MONETARY"])
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    # Lower recency (more recent) => higher score; use rank-based qcut so ties don't
    # collapse the bins, which plain qcut on a low-cardinality column tends to do.
    rfm["R_SCORE"] = pd.qcut(rfm["RECENCY"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_SCORE"] = pd.qcut(rfm["FREQUENCY"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_SCORE"] = pd.qcut(rfm["MONETARY"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_SCORE"] = rfm["R_SCORE"].astype(str) + rfm["F_SCORE"].astype(str) + rfm["M_SCORE"].astype(str)
    return rfm


def classify_segment(row: pd.Series) -> str:
    r, f, m = row["R_SCORE"], row["F_SCORE"], row["M_SCORE"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if f >= 4 and m >= 3:
        return "Loyal"
    if f >= 3 and m >= 3:
        return "Potential"
    if r <= 2 and f >= 2:
        return "At Risk"
    if r <= 2:
        return "Lost"
    return "Other"


def print_report(rfm: pd.DataFrame) -> None:
    print("=" * 70)
    print("RFM SEGMENTATION (real patients only)")
    print("=" * 70)
    print(f"Patients scored: {len(rfm):,}")
    print(f"\nRECENCY (days since last visit): mean={rfm['RECENCY'].mean():.1f}, "
          f"median={rfm['RECENCY'].median():.0f}")
    print(f"FREQUENCY (visits): mean={rfm['FREQUENCY'].mean():.1f}, "
          f"median={rfm['FREQUENCY'].median():.0f}")
    print(f"MONETARY (total claim cost): mean=${rfm['MONETARY'].mean():,.2f}, "
          f"median=${rfm['MONETARY'].median():,.2f}")

    print("\nSegment distribution:")
    segment_counts = rfm["SEGMENT"].value_counts()
    total_value = rfm["MONETARY"].sum()
    for segment, count in segment_counts.items():
        seg_value = rfm.loc[rfm["SEGMENT"] == segment, "MONETARY"].sum()
        print(f"  {segment:12s} {count:4,} patients ({count / len(rfm):5.1%})  |  "
              f"${seg_value:>12,.0f} claim value ({seg_value / total_value:5.1%})")


def make_chart(rfm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Patient RFM Segmentation", fontsize=15, fontweight="bold")

    segment_order = rfm["SEGMENT"].value_counts().index
    rfm["SEGMENT"].value_counts().loc[segment_order].plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="black"
    )
    axes[0].set_title("Patients per Segment", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    seg_value = rfm.groupby("SEGMENT")["MONETARY"].sum().loc[segment_order]
    seg_value.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
    axes[1].set_title("Total Claim Value per Segment", fontweight="bold")
    axes[1].set_ylabel("Total Claim Cost ($)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_file = OUTPUT_PATH / "02_rfm_segmentation.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved chart: {out_file}")


def run(patients: pd.DataFrame, encounters: pd.DataFrame) -> pd.DataFrame:
    rfm = build_rfm_table(patients, encounters)
    rfm = score_rfm(rfm)
    rfm["SEGMENT"] = rfm.apply(classify_segment, axis=1)

    print_report(rfm)
    make_chart(rfm)
    return rfm


if __name__ == "__main__":
    from piedmont_data_loader import load_data

    data = load_data()
    run(data["patients"], data["encounters"])
