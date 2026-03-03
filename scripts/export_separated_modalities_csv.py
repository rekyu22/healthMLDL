import argparse
from pathlib import Path

import pandas as pd

from health_mldl.config import RAW_DATA_DIR
from health_mldl.features.schema import (
    CLINICAL_COLS,
    DEXA_COLS,
    MICROWAVE_COLS,
    MRI_COLS,
    PATIENT_ID_COL,
    TARGET_COL,
    ULTRASOUND_COLS,
)


def ensure_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if PATIENT_ID_COL not in data.columns:
        data[PATIENT_ID_COL] = [f"P{idx:05d}" for idx in range(1, len(data) + 1)]
        print("Info: patient_id absent, identifiants auto-generes.")
    return data


def export_modalities(df: pd.DataFrame, out_dir: Path, strict: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "clinical.csv": [PATIENT_ID_COL] + CLINICAL_COLS,
        "ultrasound.csv": [PATIENT_ID_COL] + ULTRASOUND_COLS,
        "mri.csv": [PATIENT_ID_COL] + MRI_COLS,
        "dexa.csv": [PATIENT_ID_COL] + DEXA_COLS,
        "microwave.csv": [PATIENT_ID_COL] + MICROWAVE_COLS,
        "target.csv": [PATIENT_ID_COL, TARGET_COL],
    }

    missing_by_file: dict[str, list[str]] = {}

    for filename, required_cols in mapping.items():
        present_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if strict and missing_cols:
            missing_by_file[filename] = missing_cols
            continue

        # Keep only files containing at least patient_id + one signal/target column.
        if len(present_cols) < 2:
            print(f"Skip: {filename} (colonnes insuffisantes)")
            continue

        df.loc[:, present_cols].to_csv(out_dir / filename, index=False)
        if missing_cols:
            print(f"Warn: {filename} colonnes manquantes ignorees: {missing_cols}")

    if strict and missing_by_file:
        details = "; ".join(f"{k}: {v}" for k, v in missing_by_file.items())
        raise ValueError(f"Mode strict: colonnes manquantes -> {details}")


def resolve_output_dir(out_root: Path, dataset_id: str | None) -> Path:
    if dataset_id:
        return out_root / dataset_id
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporte des CSV separes par modalite a partir d'un CSV tabulaire."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(RAW_DATA_DIR / "synthetic_muscle_multimodal.csv"),
        help="CSV source (defaut: dataset synthetique).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="",
        help="Sous-dossier de sortie optionnel (ex: cohort_v1).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(RAW_DATA_DIR / "modalities"),
        help="Dossier racine des CSV separes.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Echoue si une colonne attendue manque.",
    )
    args = parser.parse_args()

    src = Path(args.input_csv)
    if not src.exists():
        raise FileNotFoundError(f"Dataset introuvable: {src}")

    df = pd.read_csv(src)
    df = ensure_patient_id(df)

    out_dir = resolve_output_dir(Path(args.out_root), args.dataset_id.strip())
    export_modalities(df, out_dir=out_dir, strict=args.strict)

    print(f"CSV separes exportes dans: {out_dir}")
    print("Fichiers:")
    for path in sorted(out_dir.glob("*.csv")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
