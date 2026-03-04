import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/interim/.mplconfig")).resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from health_mldl.config import DL_ARTIFACTS_DIR, RAW_DATA_DIR, DL_REPORTS_DIR, DL_TABLES_DIR
from health_mldl.data.merge_modalities import load_dataset_from_modalities
from health_mldl.data.quality import run_quality_checks
from health_mldl.evaluation.metrics import regression_metrics
from health_mldl.features.build_features import add_simple_interactions
from health_mldl.features.schema import PATIENT_ID_COL, TARGET_COL
from health_mldl.utils.serialization import save_json


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


def split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = df.drop(columns=[target_col, PATIENT_ID_COL], errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")
    pid = df[PATIENT_ID_COL] if PATIENT_ID_COL in df.columns else pd.Series(range(len(df)))
    mask = y.notna()
    return x.loc[mask].copy(), y.loc[mask].copy(), pid.loc[mask].copy()


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def main() -> None:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyTorch non installe. Lance: pip install torch (ou pip install -r requirements.txt)"
        ) from exc

    from health_mldl.dl_core.dl_tabular import (
        TabularMLPRegressor,
        evaluate_mse,
        make_regression_loader,
        predict,
        train_one_epoch,
    )

    parser = argparse.ArgumentParser(description="Train DL tabular model from modality dataset-id.")
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--target-col", type=str, default=TARGET_COL)
    parser.add_argument("--modalities-root", type=str, default=str(RAW_DATA_DIR / "modalities"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    modalities_dir = Path(args.modalities_root) / args.dataset_id
    merged = load_dataset_from_modalities(modalities_dir, patient_id_col=PATIENT_ID_COL)
    quality = run_quality_checks(merged, patient_id_col=PATIENT_ID_COL, target_col=args.target_col)

    clean = generic_cleaning(merged, target_col=args.target_col)
    feat = add_simple_interactions(clean)
    x, y, pid = split_xy(feat, target_col=args.target_col)

    x_train_full, x_test, y_train_full, y_test, pid_train_full, pid_test = train_test_split(
        x, y, pid, test_size=args.test_size, random_state=args.random_state
    )

    val_ratio = args.val_size / (1 - args.test_size)
    x_train, x_val, y_train, y_val, _pid_train, _pid_val = train_test_split(
        x_train_full,
        y_train_full,
        pid_train_full,
        test_size=val_ratio,
        random_state=args.random_state,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_arr = preprocessor.fit_transform(x_train)
    x_val_arr = preprocessor.transform(x_val)
    x_test_arr = preprocessor.transform(x_test)

    y_train_arr = y_train.to_numpy(dtype=np.float32)
    y_val_arr = y_val.to_numpy(dtype=np.float32)
    y_test_arr = y_test.to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLPRegressor(input_dim=x_train_arr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = make_regression_loader(x_train_arr, y_train_arr, batch_size=args.batch_size, shuffle=True)
    val_loader = make_regression_loader(x_val_arr, y_val_arr, batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_mse = evaluate_mse(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred = predict(model, x_test_arr, device=device)
    metrics = regression_metrics(y_test_arr, y_pred)

    suffix = f"{args.dataset_id}__{args.target_col}".replace("/", "_").replace(" ", "_")
    model_path = DL_ARTIFACTS_DIR / f"best_dl_model_{suffix}.pt"
    preproc_path = DL_ARTIFACTS_DIR / f"dl_preprocessor_{suffix}.joblib"
    summary_path = DL_REPORTS_DIR / f"dl_summary_{suffix}.json"
    history_path = DL_TABLES_DIR / f"dl_history_{suffix}.csv"
    pred_path = DL_TABLES_DIR / f"dl_predictions_{suffix}.csv"
    quality_path = DL_TABLES_DIR / f"dl_quality_report_{suffix}.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    import joblib

    joblib.dump(preprocessor, preproc_path)

    pd.DataFrame(history).to_csv(history_path, index=False)
    pd.DataFrame(
        {
            PATIENT_ID_COL: pid_test.values,
            "y_true": y_test_arr,
            "y_pred": y_pred,
            "error": y_pred - y_test_arr,
            "abs_error": np.abs(y_pred - y_test_arr),
        }
    ).sort_values("abs_error", ascending=False).to_csv(pred_path, index=False)

    save_json(quality.to_dict(), quality_path)
    save_json(
        {
            "dataset_id": args.dataset_id,
            "target_col": args.target_col,
            "framework": "pytorch",
            "model": "TabularMLPRegressor",
            "device": str(device),
            "n_rows": int(len(x)),
            "n_features_input": int(x.shape[1]),
            "n_features_after_encoding": int(x_train_arr.shape[1]),
            "best_val_mse": float(best_val),
            "test_metrics": metrics,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        summary_path,
    )

    print("DL training complete")
    print(f"Dataset: {args.dataset_id}")
    print(f"Target: {args.target_col}")
    print(metrics)
    print(f"Saved: {model_path}")
    print(f"Saved: {preproc_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {quality_path}")


if __name__ == "__main__":
    main()
