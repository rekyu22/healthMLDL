"""Shared model utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_tabular_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def make_pipeline(model, x: pd.DataFrame) -> Pipeline:
    preprocessor = build_tabular_preprocessor(x)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
