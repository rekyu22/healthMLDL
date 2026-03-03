import argparse
from pathlib import Path

from health_mldl.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from health_mldl.data.merge_modalities import load_dataset_from_modalities
from health_mldl.features.schema import PATIENT_ID_COL


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

    merged = load_dataset_from_modalities(modalities_dir, patient_id_col=PATIENT_ID_COL)

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
