import csv
import json
import sys
sys.path.append(".")

from src.utils import ensure_dir, normalize_answer


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def is_correct(record):
    pred = normalize_answer(record["prediction"])
    answers = [normalize_answer(a) for a in record["answers"]]
    return pred in answers


def main():
    ensure_dir("outputs/tables")

    prompt_path = "outputs/predictions/prompt_constrained_with_ocr_n2000.json"
    lora_path = "outputs/predictions/lora_10000_constrained_with_ocr_n2000.json"

    prompt_records = load_json(prompt_path)
    lora_records = load_json(lora_path)

    rows = []

    for prompt_record, lora_record in zip(prompt_records, lora_records):
        prompt_correct = is_correct(prompt_record)
        lora_correct = is_correct(lora_record)

        if prompt_correct and not lora_correct:
            rows.append(
                {
                    "idx": prompt_record["idx"],
                    "image_id": prompt_record.get("image_id", ""),
                    "question": prompt_record.get("question", ""),
                    "answers": " | ".join(prompt_record.get("answers", [])),
                    "ocr_tokens": " | ".join(prompt_record.get("ocr_tokens", [])[:60]),
                    "prompt_prediction": prompt_record.get("prediction", ""),
                    "lora_prediction": lora_record.get("prediction", ""),
                    "regression_type": "",
                    "notes": "",
                }
            )

    output_path = "outputs/tables/lora_regression_analysis_sample.csv"

    fieldnames = [
        "idx",
        "image_id",
        "question",
        "answers",
        "ocr_tokens",
        "prompt_prediction",
        "lora_prediction",
        "regression_type",
        "notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total LoRA regressions: {len(rows)}")
    print(f"Saved to {output_path}")
    print("\nSuggested regression_type labels:")
    print("over_short_answer")
    print("ocr_overreliance")
    print("dataset_prior_bias")
    print("normalization_issue")
    print("ambiguous")
    print("reasoning_error")
    print("other")


if __name__ == "__main__":
    main()