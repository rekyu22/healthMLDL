import argparse
from pathlib import Path

import pandas as pd

from health_mldl.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from health_mldl.features.schema import PATIENT_ID_COL


def read_modality_csv(base_dir: Path, filename: str) -> pd.DataFrame:
    path = base_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant: {path}")
    return pd.read_csv(path)


def merge_modalities(modalities_dir: Path) -> pd.DataFrame:
    clinical = read_modality_csv(modalities_dir, "clinical.csv")
    ultrasound = read_modality_csv(modalities_dir, "ultrasound.csv")
    mri = read_modality_csv(modalities_dir, "mri.csv")
    dexa = read_modality_csv(modalities_dir, "dexa.csv")
    microwave = read_modality_csv(modalities_dir, "microwave.csv")
    target = read_modality_csv(modalities_dir, "target.csv")

    df = clinical.merge(ultrasound, on=PATIENT_ID_COL, how="inner")
    df = df.merge(mri, on=PATIENT_ID_COL, how="inner")
    df = df.merge(dexa, on=PATIENT_ID_COL, how="inner")
    df = df.merge(microwave, on=PATIENT_ID_COL, how="inner")
    df = df.merge(target, on=PATIENT_ID_COL, how="inner")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruit une table d'entrainement depuis des CSV separes par modalite."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="",
        help="Sous-dossier dataset dans data/raw/modalities (ex: cohort_v1).",
    )
    parser.add_argument(
        "--modalities-root",
        type=str,
        default=str(RAW_DATA_DIR / "modalities"),
        help="Dossier racine contenant les CSV de modalites.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Chemin de sortie CSV. Defaut: data/processed/training_table_<dataset_id|default>.csv",
    )
    args = parser.parse_args()

    modalities_root = Path(args.modalities_root)
    modalities_dir = modalities_root / args.dataset_id if args.dataset_id else modalities_root

    if not modalities_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {modalities_dir}")

    merged = merge_modalities(modalities_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = args.dataset_id if args.dataset_id else "default"
        output_path = PROCESSED_DATA_DIR / f"training_table_{suffix}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Table fusionnee ecrite: {output_path}")
    print(f"Shape: {merged.shape}")


if __name__ == "__main__":
    main()
