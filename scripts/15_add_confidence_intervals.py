import math
import sys
sys.path.append(".")

import pandas as pd

from src.utils import ensure_dir


def accuracy_ci(p, n, z=1.96):
    if n <= 0:
        return 0.0

    return z * math.sqrt(p * (1.0 - p) / n)


def main():
    ensure_dir("outputs/tables")

    input_path = "outputs/tables/final_results_table.csv"
    output_path = "outputs/tables/final_results_table_with_ci.csv"

    df = pd.read_csv(input_path)

    ci_lowers = []
    ci_uppers = []
    ci_margins = []

    for _, row in df.iterrows():
        p = float(row["Accuracy"])
        n = int(row["N"])

        margin = accuracy_ci(p, n)
        ci_margins.append(margin)
        ci_lowers.append(max(0.0, p - margin))
        ci_uppers.append(min(1.0, p + margin))

    df["Accuracy CI Margin"] = ci_margins
    df["Accuracy CI Low"] = ci_lowers
    df["Accuracy CI High"] = ci_uppers

    for col in ["Accuracy CI Margin", "Accuracy CI Low", "Accuracy CI High"]:
        df[col] = df[col].round(4)

    df.to_csv(output_path, index=False)

    print(df)
    print(f"\nSaved table with confidence intervals to {output_path}")


if __name__ == "__main__":
    main()