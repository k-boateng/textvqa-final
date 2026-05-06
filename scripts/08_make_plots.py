import os
import sys
sys.path.append(".")

import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir


def save_bar_plot(df, metric, output_path, title):
    labels = df["Method Label"]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, df[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_lora_size_plot(df, output_path):
    lora_df = df[df["Method"] == "LoRA"].copy()
    lora_df = lora_df.sort_values("Train Size")

    plt.figure(figsize=(7, 5))
    plt.plot(lora_df["Train Size"], lora_df["Accuracy"], marker="o")
    plt.title("LoRA Accuracy vs Training Set Size")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_ocr_ablation_plot(df, output_path):
    subset = df[
        df["Method"].isin(
            [
                "Zero-shot naive",
                "Zero-shot naive + OCR",
                "Constrained prompt",
                "Constrained prompt + OCR",
            ]
        )
    ].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(subset["Method Label"], subset["Accuracy"])
    plt.title("OCR Ablation: Accuracy With vs Without OCR Tokens")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    ensure_dir("outputs/plots")

    df = pd.read_csv("outputs/tables/final_results_table.csv")

    df["Method Label"] = df.apply(
        lambda row: (
            f"LoRA {int(row['Train Size'])}"
            if row["Method"] == "LoRA"
            else row["Method"]
        ),
        axis=1,
    )

    save_bar_plot(
        df,
        metric="Accuracy",
        output_path="outputs/plots/final_accuracy_bar.png",
        title="TextVQA Accuracy by Method",
    )

    save_bar_plot(
        df,
        metric="Token F1",
        output_path="outputs/plots/final_token_f1_bar.png",
        title="TextVQA Token F1 by Method",
    )

    save_bar_plot(
        df,
        metric="BLEU",
        output_path="outputs/plots/final_bleu_bar.png",
        title="TextVQA BLEU by Method",
    )

    save_lora_size_plot(
        df,
        output_path="outputs/plots/lora_train_size_accuracy.png",
    )

    save_ocr_ablation_plot(
        df,
        output_path="outputs/plots/ocr_ablation_accuracy.png",
    )

    print("Saved plots:")
    print("outputs/plots/final_accuracy_bar.png")
    print("outputs/plots/final_token_f1_bar.png")
    print("outputs/plots/final_bleu_bar.png")
    print("outputs/plots/lora_train_size_accuracy.png")
    print("outputs/plots/ocr_ablation_accuracy.png")


if __name__ == "__main__":
    main()