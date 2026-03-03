"""Dataset quality and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_rows: int
    n_columns: int
    duplicate_patient_ids: int
    duplicate_rows: int
    constant_columns: list[str]
    high_missing_columns: dict[str, float]
    suspicious_feature_names: list[str]
    suspicious_target_corr: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_patient_ids": self.duplicate_patient_ids,
            "duplicate_rows": self.duplicate_rows,
            "constant_columns": self.constant_columns,
            "high_missing_columns": self.high_missing_columns,
            "suspicious_feature_names": self.suspicious_feature_names,
            "suspicious_target_corr": self.suspicious_target_corr,
        }


def run_quality_checks(
    df: pd.DataFrame,
    patient_id_col: str,
    target_col: str,
    missing_threshold: float = 0.4,
    corr_threshold: float = 0.995,
) -> QualityReport:
    n_rows, n_cols = df.shape

    duplicate_patient_ids = 0
    if patient_id_col in df.columns:
        duplicate_patient_ids = int(df.duplicated(subset=[patient_id_col]).sum())

    duplicate_rows = int(df.duplicated().sum())

    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    missing_pct = df.isna().mean().to_dict()
    high_missing_columns = {
        col: float(round(pct, 4))
        for col, pct in missing_pct.items()
        if pct >= missing_threshold
    }

    lowered_target = target_col.lower()
    suspicious_feature_names = [
        col
        for col in df.columns
        if col != target_col and lowered_target in col.lower()
    ]

    suspicious_target_corr: dict[str, float] = {}
    if target_col in df.columns:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if target_col in numeric_df.columns:
            corr = numeric_df.corr(numeric_only=True)[target_col].drop(labels=[target_col])
            suspicious_target_corr = {
                col: float(round(val, 6))
                for col, val in corr.items()
                if pd.notna(val) and abs(val) >= corr_threshold
            }

    return QualityReport(
        n_rows=n_rows,
        n_columns=n_cols,
        duplicate_patient_ids=duplicate_patient_ids,
        duplicate_rows=duplicate_rows,
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        suspicious_feature_names=suspicious_feature_names,
        suspicious_target_corr=suspicious_target_corr,
    )
