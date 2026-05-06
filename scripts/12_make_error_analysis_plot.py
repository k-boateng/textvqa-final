import os
import sys
sys.path.append(".")

import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir


def main():
    input_path = "outputs/tables/error_analysis_lora10000_sample_labeled.csv"
    plots_dir = "outputs/plots"
    tables_dir = "outputs/tables"

    ensure_dir(plots_dir)
    ensure_dir(tables_dir)

    df = pd.read_csv(input_path)

    counts = (
        df["error_type"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["error_type", "count"]
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100

    output_table = os.path.join(tables_dir, "error_analysis_summary.csv")
    counts.to_csv(output_table, index=False)

    print(counts)

    plt.figure(figsize=(8, 5))
    plt.bar(counts["error_type"], counts["count"])
    plt.title("Error Type Distribution for LoRA-10k Failures")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_plot = os.path.join(plots_dir, "error_distribution_lora10000.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"Saved table to {output_table}")
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()