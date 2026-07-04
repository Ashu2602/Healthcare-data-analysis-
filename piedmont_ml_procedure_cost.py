# -*- coding: utf-8 -*-
"""Regress procedure cost (BASE_COST) from procedure type + patient context.

Benchmarked against a procedure-type-only baseline (train-set group means) so we can
honestly report whether patient demographics add real signal, or whether cost in this
Synthea extract is really just "procedure type + random noise" -- split by PATIENT
(GroupShuffleSplit) so a patient's own procedures never leak across train/test.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from piedmont_config import OUTPUT_PATH
from piedmont_data_loader import load_data
from piedmont_eda import add_age

TOP_N_PROCEDURES = 20
CATEGORICAL_COLS = ["PROC_TYPE", "GENDER", "RACE", "ETHNICITY"]
NUMERIC_COLS = ["AGE_AT_PROCEDURE", "has_reason"]


def build_dataset(data: dict) -> pd.DataFrame:
    procedures = data["procedures"].copy()
    patients = data["patients"][["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]]

    df = procedures.merge(patients, left_on="PATIENT", right_on="Id", how="left")
    # Age *at the time of the procedure*, not current age -- avoids nonsensical
    # negative/implausible ages for procedures that happened decades ago.
    df["AGE_AT_PROCEDURE"] = ((df["DATE"] - df["BIRTHDATE"]).dt.days // 365).clip(lower=0)
    df["has_reason"] = df["REASONCODE"].notna().astype(int)

    # Bucket rare procedure types into "Other" -- one-hotting all 144 types would let
    # the model memorize singleton procedures instead of learning a general pattern.
    top_procs = df["DESCRIPTION"].value_counts().head(TOP_N_PROCEDURES).index
    df["PROC_TYPE"] = df["DESCRIPTION"].where(df["DESCRIPTION"].isin(top_procs), "Other")
    return df


def split_data(df: pd.DataFrame):
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df["BASE_COST"]
    groups = df["PATIENT"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_model(X_train, y_train) -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("reg", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42)),
    ])
    # Cost is heavily right-skewed ($262-$187,968); fit in log space so the huge-value
    # outliers don't dominate the split criterion, then invert for dollar-scale metrics.
    model.fit(X_train, np.log1p(y_train))
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    preds = np.expm1(model.predict(X_test))

    # Baseline: predict using each procedure type's TRAIN-set mean cost -- i.e. "what if
    # we ignored the patient entirely and just used the procedure type?"
    train_means = y_train.groupby(X_train["PROC_TYPE"]).mean()
    baseline_preds = X_test["PROC_TYPE"].map(train_means).fillna(y_train.mean())

    model_mae, model_r2 = mean_absolute_error(y_test, preds), r2_score(y_test, preds)
    base_mae, base_r2 = mean_absolute_error(y_test, baseline_preds), r2_score(y_test, baseline_preds)

    print("=" * 70)
    print("PROCEDURE COST REGRESSION")
    print("=" * 70)
    print(f"Train: {len(X_train):,} procedures | Test: {len(X_test):,} procedures "
          f"(split by patient, not by row)\n")

    print(f"{'Model':30s} {'MAE':>12s} {'R2':>8s}")
    print(f"{'Procedure-type-only baseline':30s} ${base_mae:>10,.0f} {base_r2:>8.3f}")
    print(f"{'RF + patient features':30s} ${model_mae:>10,.0f} {model_r2:>8.3f}")

    mae_lift = (base_mae - model_mae) / base_mae
    r2_lift = model_r2 - base_r2
    print(f"\nMAE lift: {mae_lift:+.1%}   |   R2 lift: {r2_lift:+.3f}")
    if mae_lift > 0.03 and r2_lift < -0.01:
        print("-> Mixed signal: patient features reduce *typical* error (better MAE) but the "
              "model fits rare high-cost outlier procedures worse than the type-only baseline "
              "(worse R2, which is squared-error weighted). Read as 'patient context helps the "
              "common case, not the extreme case' rather than a clean win either way.")
    elif mae_lift > 0.03 and r2_lift > -0.01:
        print("-> Patient features improve both typical-case and outlier-fit accuracy: real signal.")
    else:
        print("-> Patient features barely move either metric: cost here is driven almost entirely "
              "by procedure type, not patient demographics -- reporting this honestly rather than "
              "overselling the model.")

    return preds, baseline_preds, model_mae, base_mae


def make_charts(model, X_test, y_test, preds, model_mae, base_mae) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Procedure Cost Regression", fontsize=15, fontweight="bold")

    axes[0, 0].scatter(y_test, preds, alpha=0.15, color="steelblue", s=10)
    lims = [0, max(y_test.max(), preds.max())]
    axes[0, 0].plot(lims, lims, "--", color="gray")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Actual vs. Predicted Cost (log scale)", fontweight="bold")
    axes[0, 0].set_xlabel("Actual BASE_COST ($)")
    axes[0, 0].set_ylabel("Predicted BASE_COST ($)")

    residuals = y_test.values - preds
    axes[0, 1].hist(residuals, bins=50, color="coral", edgecolor="black", alpha=0.8)
    axes[0, 1].axvline(0, color="black", linewidth=1)
    axes[0, 1].set_title("Residuals (Actual - Predicted)", fontweight="bold")
    axes[0, 1].set_xlabel("Residual ($)")
    axes[0, 1].set_ylabel("Count")

    feature_names = model.named_steps["prep"].get_feature_names_out()
    importances = model.named_steps["reg"].feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)
    top.iloc[::-1].plot(kind="barh", ax=axes[1, 0], color="mediumseagreen", edgecolor="black")
    axes[1, 0].set_title("Top 12 Feature Importances", fontweight="bold")
    axes[1, 0].set_xlabel("Importance")

    pd.Series({"Procedure-type\nbaseline": base_mae, "RF + patient\nfeatures": model_mae}).plot(
        kind="bar", ax=axes[1, 1], color=["gray", "steelblue"], edgecolor="black"
    )
    axes[1, 1].set_title("Mean Absolute Error: Baseline vs. Model", fontweight="bold")
    axes[1, 1].set_ylabel("MAE ($)")
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    out_file = OUTPUT_PATH / "04_procedure_cost_model.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved chart: {out_file}")


def run() -> None:
    data = load_data()
    df = build_dataset(data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    preds, baseline_preds, model_mae, base_mae = evaluate(model, X_train, X_test, y_train, y_test)
    make_charts(model, X_test, y_test, preds, model_mae, base_mae)


if __name__ == "__main__":
    run()
