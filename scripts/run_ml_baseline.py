import os
from pathlib import Path

# Evite les problemes de cache matplotlib sur environnements restreints
os.environ.setdefault("MPLCONFIGDIR", str((Path("data/interim/.mplconfig")).resolve()))

from health_mldl.config import FIGURES_DIR, MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from health_mldl.data.io import load_csv, save_csv
from health_mldl.data.preprocess import basic_cleaning, split_xy
from health_mldl.data.split import train_val_test_split
from health_mldl.data.validation import validate_required_columns
from health_mldl.evaluation.metrics import regression_metrics
from health_mldl.features.build_features import add_simple_interactions
from health_mldl.modeling.model_zoo import build_random_forest_pipeline
from health_mldl.utils.serialization import save_joblib, save_json
from health_mldl.visualization.eda import save_correlation_heatmap


def main() -> None:
    raw_path = RAW_DATA_DIR / "synthetic_muscle_multimodal.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {raw_path}. Lance d'abord generate_synthetic_dataset.py"
        )

    df = load_csv(raw_path)
    validate_required_columns(df)

    clean_df = basic_cleaning(df)
    feat_df = add_simple_interactions(clean_df)

    save_csv(feat_df, Path("data/processed/training_table.csv"))
    save_correlation_heatmap(feat_df, FIGURES_DIR / "correlation_heatmap.png")

    x, y = split_xy(feat_df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    baseline = build_random_forest_pipeline(x_train)
    baseline.fit(x_train, y_train)

    val_pred = baseline.predict(x_val)
    test_pred = baseline.predict(x_test)

    metrics = {
        "validation": regression_metrics(y_val, val_pred),
        "test": regression_metrics(y_test, test_pred),
    }

    save_joblib(baseline, MODELS_DIR / "baseline_random_forest.joblib")
    save_json(metrics, REPORTS_DIR / "baseline_metrics.json")

    print("Baseline training complete")
    print(metrics)


if __name__ == "__main__":
    main()
