"""Cross-validation utilities for robust comparison."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import StratifiedKFold

from health_mldl.evaluation.metrics import regression_metrics


def run_regression_cv(model, x, y, n_splits: int = 5, random_state: int = 42) -> dict[str, float]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=1)

    return {
        "cv_mae_mean": float((-scores["test_mae"]).mean()),
        "cv_mae_std": float((-scores["test_mae"]).std(ddof=1)),
        "cv_rmse_mean": float((-scores["test_rmse"]).mean()),
        "cv_rmse_std": float((-scores["test_rmse"]).std(ddof=1)),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std(ddof=1)),
        "fit_time_mean": float(np.mean(scores["fit_time"])),
    }


def run_regression_cv_age_stratified(
    model,
    x: pd.DataFrame,
    y: pd.Series,
    age_col: str = "age",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    if age_col not in x.columns:
        raise ValueError(f"Colonne d'age absente: {age_col}")

    age_bins = pd.qcut(x[age_col], q=min(5, x[age_col].nunique()), duplicates="drop")
    if age_bins.nunique() < 2:
        return run_regression_cv(model, x, y, n_splits=n_splits, random_state=random_state)
    age_labels = age_bins.astype(str).fillna("missing")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    fit_times: list[float] = []

    for train_idx, test_idx in skf.split(x, age_labels):
        est = clone(model)
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]

        start = time.perf_counter()
        est.fit(x_train, y_train)
        fit_times.append(time.perf_counter() - start)

        pred = est.predict(x_test)
        m = regression_metrics(y_test, pred)
        maes.append(m["mae"])
        rmses.append(m["rmse"])
        r2s.append(m["r2"])

    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes, ddof=1)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses, ddof=1)),
        "cv_r2_mean": float(np.mean(r2s)),
        "cv_r2_std": float(np.std(r2s, ddof=1)),
        "fit_time_mean": float(np.mean(fit_times)),
    }
