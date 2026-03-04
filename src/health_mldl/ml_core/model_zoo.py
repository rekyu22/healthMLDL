"""Factory for recommended regression models."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from health_mldl.ml_core.common import make_pipeline


def build_elastic_net_pipeline(x, random_state: int = 42) -> Pipeline:
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=5000)
    return make_pipeline(model, x)


def build_random_forest_pipeline(x, random_state: int = 42) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    return make_pipeline(model, x)


def build_gradient_boosting_pipeline(x, random_state: int = 42) -> Pipeline:
    # Portable boosting fallback compatible with restricted environments.
    model = GradientBoostingRegressor(
        learning_rate=0.05,
        n_estimators=350,
        max_depth=3,
        random_state=random_state,
    )
    return make_pipeline(model, x)
