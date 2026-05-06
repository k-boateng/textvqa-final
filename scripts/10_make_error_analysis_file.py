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

    lora_path = "outputs/predictions/lora_10000_constrained_with_ocr_n2000.json"
    records = load_json(lora_path)

    failure_rows = []

    for record in records:
        if is_correct(record):
            continue

        failure_rows.append(
            {
                "idx": record["idx"],
                "image_id": record.get("image_id", ""),
                "question": record.get("question", ""),
                "answers": " | ".join(record.get("answers", [])),
                "prediction": record.get("prediction", ""),
                "ocr_tokens": " | ".join(record.get("ocr_tokens", [])[:50]),
                "error_type": "",
                "notes": "",
            }
        )

    selected = failure_rows[:100]

    output_path = "outputs/tables/error_analysis_lora10000_sample.csv"

    fieldnames = [
        "idx",
        "image_id",
        "question",
        "answers",
        "prediction",
        "ocr_tokens",
        "error_type",
        "notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Total LoRA-10k failures: {len(failure_rows)}")
    print(f"Saved first {len(selected)} failures to {output_path}")
    print("\nUse these labels for error_type:")
    print("ocr_error")
    print("reasoning_error")
    print("hallucination")
    print("ambiguous")
    print("normalization_issue")
    print("other")


if __name__ == "__main__":
    main()