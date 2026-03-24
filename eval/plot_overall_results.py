"""Plot mean_rmse vs frac_interpretability_tests_passed from overall_results.csv.

Usage:
    uv run python eval/plot_overall_results.py
    uv run python eval/plot_overall_results.py --input results/overall_results.csv --output results/overall_results_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "results" / "overall_results.csv"
    default_output = repo_root / "results" / "overall_results_scatter.png"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input, help="Path to overall_results.csv")
    parser.add_argument("--output", type=Path, default=default_output, help="Path to save the plot PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    required_cols = {"mean_rmse", "frac_interpretability_tests_passed", "description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Drop rows with missing values in plotting columns.
    df = df.dropna(subset=["mean_rmse", "frac_interpretability_tests_passed", "description"]).copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        df["frac_interpretability_tests_passed"],
        df["mean_rmse"],
        alpha=0.85,
        s=40,
    )

    for _, row in df.iterrows():
        ax.annotate(
            str(row["description"]),
            (row["frac_interpretability_tests_passed"], row["mean_rmse"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.9,
        )

    ax.set_xlabel("frac_interpretability_tests_passed")
    ax.set_ylabel("mean_rmse")
    ax.set_title("mean_rmse vs frac_interpretability_tests_passed")
    ax.grid(True, linestyle="--", alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
