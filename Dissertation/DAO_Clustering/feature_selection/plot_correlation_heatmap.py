"""Generate a correlation heatmap from correlation_matrix_reduced.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "correlation_matrix_reduced.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: same folder as input, name heatmap_reduced.png)",
    )
    args = parser.parse_args()

    inp = args.input
    out = args.output or inp.with_name("heatmap_correlation_reduced.png")

    df = pd.read_csv(inp, index_col=0)
    # Ensure numeric square matrix
    df = df.apply(pd.to_numeric, errors="coerce")

    plt.figure(figsize=(max(10, len(df.columns) * 0.45), max(8, len(df.columns) * 0.45)))
    sns.heatmap(
        df,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
    )
    plt.title("Reduced feature correlation matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
