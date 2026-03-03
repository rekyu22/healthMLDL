from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_correlation_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    corr = df.select_dtypes(include=["number"]).corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
