import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/interim/.mplconfig")).resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from sklearn.model_selection import train_test_split

from health_mldl.config import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR, TABLES_DIR
from health_mldl.data.merge_modalities import load_dataset_from_modalities
from health_mldl.data.quality import run_quality_checks
from health_mldl.evaluation.cv import run_regression_cv
from health_mldl.evaluation.metrics import regression_metrics
from health_mldl.features.build_features import add_simple_interactions
from health_mldl.features.schema import MODALITY_BLOCKS, PATIENT_ID_COL, TARGET_COL
from health_mldl.modeling.model_zoo import (
    build_elastic_net_pipeline,
    build_gradient_boosting_pipeline,
    build_random_forest_pipeline,
)
from health_mldl.modeling.multimodal_stacking import build_multimodal_stacking_pipeline
from health_mldl.utils.serialization import save_joblib, save_json

def split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' absente de la table fusionnee")

    x = df.drop(columns=[target_col, PATIENT_ID_COL], errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")

    mask = y.notna()
    x = x.loc[mask].copy()
    y = y.loc[mask].copy()

    if x.empty:
        raise ValueError("Aucune ligne exploitable apres filtrage de la cible")
    return x, y


def generic_cleaning(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    data = df.copy().drop_duplicates()
    if target_col not in data.columns:
        raise ValueError(f"Target '{target_col}' absente du dataset")
    data = data.dropna(subset=[target_col])

    num_cols = data.select_dtypes(include=["number"]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    cat_cols = data.select_dtypes(exclude=["number"]).columns
    for col in cat_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode().iloc[0])

    return data


def _fit_eval_single(name: str, pipeline, x_train, y_train, x_test, y_test) -> dict:
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    return {
        "model": name,
        **regression_metrics(y_test, pred),
        **run_regression_cv(pipeline, x_train, y_train),
    }


def has_full_multimodal_columns(x: pd.DataFrame) -> bool:
    required = (
        MODALITY_BLOCKS["clinical"]
        + MODALITY_BLOCKS["ultrasound"]
        + MODALITY_BLOCKS["mri"]
        + MODALITY_BLOCKS["dexa"]
        + MODALITY_BLOCKS["microwave"]
    )
    return all(col in x.columns for col in required)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate models from modality-separated CSVs using dataset-id."
    )
    parser.add_argument("--dataset-id", type=str, required=True, help="Ex: cohort_v1, nhanes_2017")
    parser.add_argument(
        "--modalities-root",
        type=str,
        default=str(RAW_DATA_DIR / "modalities"),
        help="Racine des CSV de modalites.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=TARGET_COL,
        help=f"Cible de regression (defaut: {TARGET_COL}).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    modalities_dir = Path(args.modalities_root) / args.dataset_id
    if not modalities_dir.exists():
        raise FileNotFoundError(f"Dossier dataset introuvable: {modalities_dir}")

    merged = load_dataset_from_modalities(modalities_dir, patient_id_col=PATIENT_ID_COL)
    quality_report = run_quality_checks(
        merged,
        patient_id_col=PATIENT_ID_COL,
        target_col=args.target_col,
    )
    clean_df = generic_cleaning(merged, target_col=args.target_col)
    feat_df = add_simple_interactions(clean_df)

    x, y = split_xy(feat_df, target_col=args.target_col)

    if len(x) < 100:
        print(f"Warning: dataset petit pour benchmark robuste (n={len(x)})")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state
    )

    benchmarks: list[dict] = []

    elastic = build_elastic_net_pipeline(x_train, random_state=args.random_state)
    benchmarks.append(_fit_eval_single("elastic_net", elastic, x_train, y_train, x_test, y_test))

    rf = build_random_forest_pipeline(x_train, random_state=args.random_state)
    benchmarks.append(_fit_eval_single("random_forest", rf, x_train, y_train, x_test, y_test))

    gbdt = build_gradient_boosting_pipeline(x_train, random_state=args.random_state)
    benchmarks.append(_fit_eval_single("gradient_boosting", gbdt, x_train, y_train, x_test, y_test))

    multimodal = None
    if has_full_multimodal_columns(x_train):
        multimodal = build_multimodal_stacking_pipeline(x_train, random_state=args.random_state)
        multimodal.pipeline.fit(x_train, y_train)
        multimodal_pred = multimodal.pipeline.predict(x_test)
        benchmarks.append(
            {
                "model": "multimodal_stacking",
                **regression_metrics(y_test, multimodal_pred),
                **run_regression_cv(multimodal.pipeline, x_train, y_train),
                "modalities": ",".join(multimodal.base_modalities),
            }
        )
    else:
        print("Info: modalites incompletes -> multimodal_stacking ignore")

    results_df = pd.DataFrame(benchmarks).sort_values("cv_rmse_mean")

    suffix = f"{args.dataset_id}__{args.target_col}"
    safe_suffix = suffix.replace("/", "_").replace(" ", "_")

    table_out = TABLES_DIR / f"benchmark_results_{safe_suffix}.csv"
    quality_out = TABLES_DIR / f"quality_report_{safe_suffix}.json"
    table_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_out, index=False)
    save_json(quality_report.to_dict(), quality_out)

    best_name = results_df.iloc[0]["model"]
    if best_name == "multimodal_stacking" and multimodal is not None:
        best_model = multimodal.pipeline
    elif best_name == "gradient_boosting":
        best_model = gbdt
    elif best_name == "random_forest":
        best_model = rf
    else:
        best_model = elastic

    model_out = MODELS_DIR / f"best_model_{safe_suffix}.joblib"
    summary_out = REPORTS_DIR / f"benchmark_summary_{safe_suffix}.json"

    save_joblib(best_model, model_out)
    save_json(
        {
            "dataset_id": args.dataset_id,
            "target_col": args.target_col,
            "n_rows": int(len(x)),
            "n_features": int(x.shape[1]),
            "best_model": best_name,
            "ranking": results_df.to_dict(orient="records"),
        },
        summary_out,
    )

    print("Training from dataset-id complete")
    print(f"Dataset: {args.dataset_id}")
    print(f"Target: {args.target_col}")
    print(results_df[["model", "cv_rmse_mean", "cv_r2_mean", "rmse", "r2"]])
    print(f"Saved: {table_out}")
    print(f"Saved: {quality_out}")
    print(f"Saved: {model_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
