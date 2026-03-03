import argparse
from pathlib import Path

import pandas as pd

from health_mldl.config import RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Importe un CSV dans data/raw")
    parser.add_argument("input_csv", type=str, help="Chemin du fichier CSV source")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom de sortie (ex: my_data.csv). Par défaut: nom original.",
    )
    args = parser.parse_args()

    src = Path(args.input_csv)
    if not src.exists():
        raise FileNotFoundError(f"Fichier introuvable: {src}")

    out_name = args.name or src.name
    dst = RAW_DATA_DIR / out_name
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df.to_csv(dst, index=False)

    print(f"Dataset importé vers: {dst}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
