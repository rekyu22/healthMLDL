import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from health_mldl.config import RAW_DATA_DIR, TABLES_DIR
from health_mldl.features.schema import PATIENT_ID_COL


def _safe_bin(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < 2:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index)
    bins = pd.qcut(series, q=min(n_bins, valid.nunique()), duplicates="drop")
    return bins.astype(str)


def subgroup_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    work = df[[group_col, "error"]].copy()
    work["abs_error"] = work["error"].abs()
    work["sq_error"] = work["error"] ** 2
    out = (
        work.groupby(group_col, dropna=False)
        .agg(
            n=("error", "size"),
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    out["rmse"] = np.sqrt(out["mse"])
    out = out.drop(columns=["mse"])
    return out.sort_values("mae", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis by clinical subgroups.")
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="",
        help="Optional path to predictions CSV.",
    )
    parser.add_argument(
        "--modalities-root",
        type=str,
        default=str(RAW_DATA_DIR / "modalities"),
    )
    args = parser.parse_args()

    suffix = f"{args.dataset_id}__{args.target_col}".replace("/", "_").replace(" ", "_")
    pred_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else TABLES_DIR / f"predictions_{suffix}.csv"
    )

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions introuvables: {pred_path}")

    pred_df = pd.read_csv(pred_path)

    clinical_path = Path(args.modalities_root) / args.dataset_id / "clinical.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"clinical.csv introuvable: {clinical_path}")

    clinical = pd.read_csv(clinical_path)
    merged = pred_df.merge(clinical, on=PATIENT_ID_COL, how="left")

    if "error" not in merged.columns:
        merged["error"] = merged["y_pred"] - merged["y_true"]

    out_base = TABLES_DIR
    out_base.mkdir(parents=True, exist_ok=True)

    by_sex = pd.DataFrame()
    if "sex" in merged.columns:
        by_sex = subgroup_metrics(merged, "sex")
        by_sex.to_csv(out_base / f"error_by_sex_{suffix}.csv", index=False)

    by_age = pd.DataFrame()
    if "age" in merged.columns:
        merged["age_bin"] = _safe_bin(pd.to_numeric(merged["age"], errors="coerce"), 5, "age")
        by_age = subgroup_metrics(merged, "age_bin")
        by_age.to_csv(out_base / f"error_by_age_bin_{suffix}.csv", index=False)

    by_bmi = pd.DataFrame()
    if "bmi" in merged.columns:
        merged["bmi_bin"] = _safe_bin(pd.to_numeric(merged["bmi"], errors="coerce"), 5, "bmi")
        by_bmi = subgroup_metrics(merged, "bmi_bin")
        by_bmi.to_csv(out_base / f"error_by_bmi_bin_{suffix}.csv", index=False)

    print("Error analysis complete")
    print(f"Input predictions: {pred_path}")
    if not by_sex.empty:
        print(f"Saved: {out_base / f'error_by_sex_{suffix}.csv'}")
    if not by_age.empty:
        print(f"Saved: {out_base / f'error_by_age_bin_{suffix}.csv'}")
    if not by_bmi.empty:
        print(f"Saved: {out_base / f'error_by_bmi_bin_{suffix}.csv'}")


if __name__ == "__main__":
    main()
