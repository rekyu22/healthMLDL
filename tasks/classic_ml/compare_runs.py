import argparse
import json
from pathlib import Path

import pandas as pd

from health_mldl.config import ML_REPORTS_DIR, ML_TABLES_DIR


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_ids_from_filename(path: Path) -> tuple[str | None, str | None]:
    name = path.name
    prefix = "benchmark_summary_"
    suffix = ".json"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None, None
    core = name[len(prefix) : -len(suffix)]
    if "__" not in core:
        return None, None
    dataset_id, target_col = core.split("__", 1)
    return dataset_id or None, target_col or None


def parse_summary(path: Path) -> tuple[dict, list[dict]]:
    payload = load_summary(path)

    file_dataset, file_target = infer_ids_from_filename(path)
    dataset_id = payload.get("dataset_id") or file_dataset or "unknown"
    target_col = payload.get("target_col") or file_target or "unknown"
    best_model = payload.get("best_model", "unknown")
    n_rows = payload.get("n_rows")
    n_features = payload.get("n_features")
    stratify_age = payload.get("stratify_age")

    ranking = payload.get("ranking", []) or []
    run_id = f"{dataset_id}__{target_col}"

    model_rows: list[dict] = []
    for rank_idx, row in enumerate(ranking, start=1):
        item = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "target_col": target_col,
            "stratify_age": stratify_age,
            "n_rows": n_rows,
            "n_features": n_features,
            "best_model": best_model,
            "rank_in_run": rank_idx,
            **row,
            "source_file": path.name,
        }
        model_rows.append(item)

    run_row = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "target_col": target_col,
        "stratify_age": stratify_age,
        "n_rows": n_rows,
        "n_features": n_features,
        "best_model": best_model,
        "n_models": len(ranking),
        "source_file": path.name,
    }

    if ranking:
        best_row = ranking[0]
        run_row.update(
            {
                "best_cv_rmse_mean": best_row.get("cv_rmse_mean"),
                "best_cv_r2_mean": best_row.get("cv_r2_mean"),
                "best_test_rmse": best_row.get("rmse"),
                "best_test_r2": best_row.get("r2"),
            }
        )

    return run_row, model_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark summaries across runs.")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(ML_REPORTS_DIR),
        help="Folder containing benchmark_summary*.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ML_TABLES_DIR),
        help="Output folder for comparison CSV files.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Inclut les summaries legacy sans dataset_id/target explicites.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(reports_dir.glob("benchmark_summary*.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No benchmark_summary*.json found in {reports_dir}")

    run_rows: list[dict] = []
    model_rows: list[dict] = []

    for path in summary_paths:
        run_row, rows = parse_summary(path)
        if not args.include_legacy and (
            run_row["dataset_id"] == "unknown" or run_row["target_col"] == "unknown"
        ):
            continue
        run_rows.append(run_row)
        model_rows.extend(rows)

    if not run_rows:
        raise ValueError("No comparable runs found. Use --include-legacy to include legacy files.")

    runs_df = pd.DataFrame(run_rows)
    if "best_cv_rmse_mean" in runs_df.columns:
        runs_df = runs_df.sort_values(
            by=["best_cv_rmse_mean", "best_cv_r2_mean"],
            ascending=[True, False],
            na_position="last",
        )

    models_df = pd.DataFrame(model_rows)
    if not models_df.empty and "cv_rmse_mean" in models_df.columns:
        models_df = models_df.sort_values(
            by=["cv_rmse_mean", "cv_r2_mean"],
            ascending=[True, False],
            na_position="last",
        )

    runs_out = out_dir / "run_comparison.csv"
    models_out = out_dir / "model_comparison.csv"

    runs_df.to_csv(runs_out, index=False)
    models_df.to_csv(models_out, index=False)

    print("Comparison complete")
    print(f"Runs: {runs_out}")
    print(f"Models: {models_out}")
    print("Top runs:")
    print(runs_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
