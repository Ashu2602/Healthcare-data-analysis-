# -*- coding: utf-8 -*-
"""Classify patients as high-healthcare-expense (top 20% of HEALTHCARE_EXPENSES).

Features are utilization/clinical signals only (encounter counts, condition flags,
medication/procedure counts) -- never dollar amounts -- so the model can't just
echo the cost-based label back to itself.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from piedmont_config import OUTPUT_PATH
from piedmont_data_loader import load_data
from piedmont_eda import add_age

sns.set_style("whitegrid")

# A handful of clinically meaningful chronic/high-burden conditions worth their own
# flag, rather than one-hot-encoding all 129 condition types (most are one-off acute visits).
CHRONIC_CONDITIONS = [
    "Hypertension",
    "Prediabetes",
    "Body mass index 30+ - obesity (finding)",
    "Anemia (disorder)",
    "Hyperlipidemia",
    "Chronic sinusitis (disorder)",
]

CATEGORICAL_COLS = ["GENDER", "RACE", "ETHNICITY"]


def build_features(data: dict) -> pd.DataFrame:
    patients = add_age(data["patients"])[
        ["Id", "AGE", "GENDER", "RACE", "ETHNICITY", "HEALTHCARE_EXPENSES"]
    ].copy()
    encounters, conditions = data["encounters"], data["conditions"]
    medications, procedures = data["medications"], data["procedures"]

    total_encounters = encounters.groupby("PATIENT").size().rename("total_encounters")

    # Care-mix ratios (not raw counts) so patients are compared on *how* they use care,
    # independent of how much overall -- total_encounters already captures volume.
    enc_type_counts = encounters.groupby(["PATIENT", "ENCOUNTERCLASS"]).size().unstack(fill_value=0)
    enc_type_ratio = enc_type_counts.div(enc_type_counts.sum(axis=1), axis=0).add_prefix("pct_")

    distinct_conditions = conditions.groupby("PATIENT")["DESCRIPTION"].nunique().rename("distinct_conditions")

    chronic_flags = (
        conditions[conditions["DESCRIPTION"].isin(CHRONIC_CONDITIONS)]
        .assign(flag=1)
        .pivot_table(index="PATIENT", columns="DESCRIPTION", values="flag", aggfunc="max", fill_value=0)
    )
    chronic_flags.columns = [f"has_{c.split(' (')[0].strip().replace(' ', '_').lower()}" for c in chronic_flags.columns]

    distinct_medications = medications.groupby("PATIENT")["DESCRIPTION"].nunique().rename("distinct_medications")
    total_medication_records = medications.groupby("PATIENT").size().rename("total_medication_records")
    total_procedures = procedures.groupby("PATIENT").size().rename("total_procedures")

    features = (
        patients.set_index("Id")
        .join([total_encounters, enc_type_ratio, distinct_conditions, chronic_flags,
               distinct_medications, total_medication_records, total_procedures])
    )
    # NaN here only means "zero records in that table for this patient" (e.g. never
    # saw a doctor for a condition) -- 0 is the correct value, not a missing one.
    fill_cols = [c for c in features.columns if c not in CATEGORICAL_COLS + ["AGE", "HEALTHCARE_EXPENSES"]]
    features[fill_cols] = features[fill_cols].fillna(0)
    return features


def train_and_evaluate(features: pd.DataFrame):
    threshold = features["HEALTHCARE_EXPENSES"].quantile(0.80)
    y = (features["HEALTHCARE_EXPENSES"] >= threshold).astype(int)
    X = features.drop(columns=["HEALTHCARE_EXPENSES"])
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", "passthrough", numeric_cols),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", min_samples_leaf=3, random_state=42
        )),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("=" * 70)
    print("HIGH-COST PATIENT CLASSIFICATION")
    print("=" * 70)
    print(f"Label: top 20% of HEALTHCARE_EXPENSES (threshold = ${threshold:,.0f})")
    print(f"Train: {len(X_train):,} patients | Test: {len(X_test):,} patients")
    print(f"Test set positive rate: {y_test.mean():.1%}\n")

    print("Classification report:")
    print(classification_report(y_test, preds, target_names=["Not high-cost", "High-cost"]))

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}  (baseline PR-AUC at this class balance = {y_test.mean():.3f})")

    return model, X_test, y_test, proba, preds, roc_auc, pr_auc


def make_charts(model, y_test, proba, preds) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("High-Cost Patient Classifier", fontsize=15, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0, 0].plot(fpr, tpr, color="steelblue", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 0].set_title("ROC Curve", fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")

    precision, recall, _ = precision_recall_curve(y_test, proba)
    axes[0, 1].plot(recall, precision, color="coral", linewidth=2)
    axes[0, 1].axhline(y_test.mean(), linestyle="--", color="gray", label="baseline")
    axes[0, 1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["Not high-cost", "High-cost"], yticklabels=["Not high-cost", "High-cost"])
    axes[1, 0].set_title("Confusion Matrix", fontweight="bold")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("Actual")

    feature_names = model.named_steps["prep"].get_feature_names_out()
    importances = model.named_steps["clf"].feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)
    top.iloc[::-1].plot(kind="barh", ax=axes[1, 1], color="mediumseagreen", edgecolor="black")
    axes[1, 1].set_title("Top 12 Feature Importances", fontweight="bold")
    axes[1, 1].set_xlabel("Importance")

    plt.tight_layout()
    out_file = OUTPUT_PATH / "03_highcost_classifier.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved chart: {out_file}")


def run() -> None:
    data = load_data()
    features = build_features(data)
    model, X_test, y_test, proba, preds, roc_auc, pr_auc = train_and_evaluate(features)
    make_charts(model, y_test, proba, preds)


if __name__ == "__main__":
    run()
