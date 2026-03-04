"""Multimodal stacking regressor with modality-specific base learners."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from health_mldl.features.schema import MODALITY_BLOCKS
from health_mldl.ml_core.model_zoo import (
    build_elastic_net_pipeline,
    build_gradient_boosting_pipeline,
    build_random_forest_pipeline,
)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.columns]


@dataclass
class MultimodalModel:
    pipeline: Pipeline
    base_modalities: list[str]


def _modality_pipeline(modality_cols: list[str], reg_pipeline_builder, x: pd.DataFrame) -> Pipeline:
    subset = x[modality_cols].copy()
    base_pipe = reg_pipeline_builder(subset)
    return Pipeline(steps=[("select", ColumnSelector(modality_cols)), ("base", base_pipe)])


def build_multimodal_stacking_pipeline(x: pd.DataFrame, random_state: int = 42) -> MultimodalModel:
    blocks = {
        "clinical": MODALITY_BLOCKS["clinical"],
        "ultrasound": MODALITY_BLOCKS["ultrasound"],
        "mri": MODALITY_BLOCKS["mri"],
        "dexa": MODALITY_BLOCKS["dexa"],
        "microwave": MODALITY_BLOCKS["microwave"],
    }

    estimators = [
        (
            "clinical_enet",
            _modality_pipeline(blocks["clinical"], build_elastic_net_pipeline, x),
        ),
        (
            "ultrasound_rf",
            _modality_pipeline(blocks["ultrasound"], build_random_forest_pipeline, x),
        ),
        (
            "mri_gbdt",
            _modality_pipeline(blocks["mri"], build_gradient_boosting_pipeline, x),
        ),
        (
            "dexa_rf",
            _modality_pipeline(blocks["dexa"], build_random_forest_pipeline, x),
        ),
        (
            "microwave_gbdt",
            _modality_pipeline(blocks["microwave"], build_gradient_boosting_pipeline, x),
        ),
    ]

    final_estimator = ElasticNet(alpha=0.02, l1_ratio=0.2, random_state=random_state, max_iter=5000)

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        passthrough=False,
        n_jobs=1,
    )

    return MultimodalModel(pipeline=stack, base_modalities=list(blocks.keys()))
