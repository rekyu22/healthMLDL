import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/interim/.mplconfig")).resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

from health_mldl.config import ML_ARTIFACTS_DIR, RAW_DATA_DIR, ML_REPORTS_DIR, ML_TABLES_DIR
from health_mldl.data.io import load_csv, save_csv
from health_mldl.data.preprocess import basic_cleaning, split_xy
from health_mldl.data.validation import summarize_missingness, validate_required_columns
from health_mldl.evaluation.cv import run_regression_cv
from health_mldl.evaluation.metrics import regression_metrics
from health_mldl.features.build_features import add_simple_interactions
from health_mldl.ml_core.model_zoo import (
    build_elastic_net_pipeline,
    build_gradient_boosting_pipeline,
    build_random_forest_pipeline,
)
from health_mldl.ml_core.multimodal_stacking import build_multimodal_stacking_pipeline
from health_mldl.utils.serialization import save_joblib, save_json


def _fit_eval_single(name: str, pipeline, x_train, y_train, x_test, y_test) -> dict:
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    test_metrics = regression_metrics(y_test, pred)
    cv_metrics = run_regression_cv(pipeline, x_train, y_train)
    result = {"model": name, **test_metrics, **cv_metrics}
    return result


def main() -> None:
    raw_path = RAW_DATA_DIR / "synthetic_muscle_multimodal.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {raw_path}. Lance d'abord tasks/ingest/generate_synthetic_dataset.py"
        )

    df = load_csv(raw_path)
    validate_required_columns(df)

    missingness = summarize_missingness(df)
    missingness.to_csv(ML_TABLES_DIR / "missingness_summary.csv", index=False)

    clean_df = basic_cleaning(df)
    feat_df = add_simple_interactions(clean_df)
    save_csv(feat_df, Path("data/processed/training_table.csv"))

    x, y = split_xy(feat_df)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    benchmarks = []

    elastic = build_elastic_net_pipeline(x_train)
    benchmarks.append(
        _fit_eval_single("elastic_net", elastic, x_train, y_train, x_test, y_test)
    )

    rf = build_random_forest_pipeline(x_train)
    benchmarks.append(
        _fit_eval_single("random_forest", rf, x_train, y_train, x_test, y_test)
    )

    gbdt = build_gradient_boosting_pipeline(x_train)
    benchmarks.append(
        _fit_eval_single("gradient_boosting", gbdt, x_train, y_train, x_test, y_test)
    )

    multimodal = build_multimodal_stacking_pipeline(x_train)
    multimodal.pipeline.fit(x_train, y_train)
    multimodal_pred = multimodal.pipeline.predict(x_test)
    multimodal_test = regression_metrics(y_test, multimodal_pred)
    multimodal_cv = run_regression_cv(multimodal.pipeline, x_train, y_train)
    benchmarks.append(
        {
            "model": "multimodal_stacking",
            **multimodal_test,
            **multimodal_cv,
            "modalities": ",".join(multimodal.base_modalities),
        }
    )

    results_df = pd.DataFrame(benchmarks).sort_values("cv_rmse_mean")
    results_path = ML_TABLES_DIR / "benchmark_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    best_name = results_df.iloc[0]["model"]
    if best_name == "multimodal_stacking":
        best_model = multimodal.pipeline
    elif best_name == "gradient_boosting":
        best_model = gbdt
    elif best_name == "random_forest":
        best_model = rf
    else:
        best_model = elastic

    save_joblib(best_model, ML_ARTIFACTS_DIR / "best_model.joblib")
    save_json(
        {
            "best_model": best_name,
            "ranking": results_df.to_dict(orient="records"),
        },
        ML_REPORTS_DIR / "benchmark_summary.json",
    )

    print("Benchmark complete")
    print(results_df[["model", "cv_rmse_mean", "cv_r2_mean", "rmse", "r2"]])


if __name__ == "__main__":
    main()
