"""Data validation helpers."""

from __future__ import annotations

import pandas as pd

from health_mldl.features.schema import BASE_REQUIRED_COLUMNS


def validate_required_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    required_cols = required or BASE_REQUIRED_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
        }
    )
    return summary.sort_values("missing_pct", ascending=False)
