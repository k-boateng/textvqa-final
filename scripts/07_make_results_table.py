import json
import os
import sys
sys.path.append(".")

import pandas as pd

from src.utils import ensure_dir


def load_metric(path):
    with open(path, "r") as f:
        return json.load(f)


def add_row(rows, method, train_size, ocr, prompt, path):
    metrics = load_metric(path)

    rows.append(
        {
            "Method": method,
            "Train Size": train_size,
            "OCR": ocr,
            "Prompt": prompt,
            "Accuracy": metrics.get("accuracy"),
            "Token F1": metrics.get("token_f1"),
            "ROUGE-L": metrics.get("rouge_l"),
            "BLEU": metrics.get("bleu"),
            "N": metrics.get("n"),
        }
    )


def main():
    ensure_dir("outputs/tables")

    rows = []

    # Prompt / zero-shot results
    add_row(
        rows,
        method="Zero-shot naive",
        train_size=0,
        ocr="No",
        prompt="naive",
        path="outputs/metrics/prompt_naive_no_ocr_n2000.json",
    )

    add_row(
        rows,
        method="Zero-shot naive + OCR",
        train_size=0,
        ocr="Yes",
        prompt="naive",
        path="outputs/metrics/prompt_naive_with_ocr_n2000.json",
    )

    add_row(
        rows,
        method="OCR-aware prompt",
        train_size=0,
        ocr="Yes",
        prompt="ocr_aware",
        path="outputs/metrics/prompt_ocr_aware_with_ocr_n2000.json",
    )

    add_row(
        rows,
        method="Constrained prompt",
        train_size=0,
        ocr="No",
        prompt="constrained",
        path="outputs/metrics/prompt_constrained_no_ocr_n2000.json",
    )

    add_row(
        rows,
        method="Constrained prompt + OCR",
        train_size=0,
        ocr="Yes",
        prompt="constrained",
        path="outputs/metrics/prompt_constrained_with_ocr_n2000.json",
    )

    # LoRA results
    add_row(
        rows,
        method="LoRA",
        train_size=1000,
        ocr="Yes",
        prompt="constrained",
        path="outputs/metrics/lora_1000_constrained_with_ocr_n2000.json",
    )

    add_row(
        rows,
        method="LoRA",
        train_size=5000,
        ocr="Yes",
        prompt="constrained",
        path="outputs/metrics/lora_5000_constrained_with_ocr_n2000.json",
    )

    add_row(
        rows,
        method="LoRA",
        train_size=10000,
        ocr="Yes",
        prompt="constrained",
        path="outputs/metrics/lora_10000_constrained_with_ocr_n2000.json",
    )

    df = pd.DataFrame(rows)

    metric_cols = ["Accuracy", "Token F1", "ROUGE-L", "BLEU"]
    for col in metric_cols:
        df[col] = df[col].round(4)

    output_path = "outputs/tables/final_results_table.csv"
    df.to_csv(output_path, index=False)

    print(df)
    print(f"\nSaved final results table to {output_path}")


if __name__ == "__main__":
    main()