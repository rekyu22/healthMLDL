"""Utilities to merge modality-separated CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset_from_modalities(modalities_dir: Path, patient_id_col: str) -> pd.DataFrame:
    csv_files = sorted([p for p in modalities_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouve dans: {modalities_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        if patient_id_col not in df.columns:
            raise ValueError(f"{path.name}: colonne '{patient_id_col}' manquante")
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        overlap = [c for c in df.columns if c in merged.columns and c != patient_id_col]
        if overlap:
            df = df.drop(columns=overlap)
        merged = merged.merge(df, on=patient_id_col, how="inner")

    return merged
