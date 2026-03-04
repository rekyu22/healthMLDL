"""Legacy compatibility wrappers.

Prefer using:
- health_mldl.ml_core.model_zoo
- health_mldl.ml_core.multimodal_stacking
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from health_mldl.ml_core.model_zoo import build_random_forest_pipeline
from health_mldl.utils.serialization import save_joblib


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_columns: list[str]


def train_baseline_regressor(x_train, y_train) -> TrainedModel:
    pipeline = build_random_forest_pipeline(x_train)
    pipeline.fit(x_train, y_train)
    return TrainedModel(pipeline=pipeline, feature_columns=x_train.columns.tolist())


def save_model(trained: TrainedModel, path: str) -> None:
    save_joblib(trained, path)
