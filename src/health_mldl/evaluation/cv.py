"""Cross-validation utilities for robust comparison."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_validate


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
