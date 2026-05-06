import json
import os
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


def compact_record(record, method_name):
    return {
        "idx": record["idx"],
        "image_id": record.get("image_id"),
        "method": method_name,
        "question": record.get("question"),
        "answers": record.get("answers"),
        "prediction": record.get("prediction"),
        "correct": is_correct(record),
        "ocr_tokens": record.get("ocr_tokens", [])[:50],
    }


def main():
    ensure_dir("outputs/examples")

    naive_path = "outputs/predictions/prompt_naive_no_ocr_n2000.json"
    prompt_path = "outputs/predictions/prompt_constrained_with_ocr_n2000.json"
    lora_path = "outputs/predictions/lora_10000_constrained_with_ocr_n2000.json"

    naive = load_json(naive_path)
    prompt = load_json(prompt_path)
    lora = load_json(lora_path)

    examples = {
        "all_methods_correct": [],
        "all_methods_wrong": [],
        "prompt_improves_over_naive": [],
        "lora_improves_over_prompt": [],
        "lora_regresses_from_prompt": [],
    }

    for n_rec, p_rec, l_rec in zip(naive, prompt, lora):
        n_correct = is_correct(n_rec)
        p_correct = is_correct(p_rec)
        l_correct = is_correct(l_rec)

        packed = {
            "idx": n_rec["idx"],
            "image_id": n_rec.get("image_id"),
            "question": n_rec.get("question"),
            "answers": n_rec.get("answers"),
            "ocr_tokens": p_rec.get("ocr_tokens", [])[:50],
            "naive_prediction": n_rec.get("prediction"),
            "naive_correct": n_correct,
            "prompt_prediction": p_rec.get("prediction"),
            "prompt_correct": p_correct,
            "lora_prediction": l_rec.get("prediction"),
            "lora_correct": l_correct,
        }

        if n_correct and p_correct and l_correct:
            examples["all_methods_correct"].append(packed)

        if (not n_correct) and (not p_correct) and (not l_correct):
            examples["all_methods_wrong"].append(packed)

        if (not n_correct) and p_correct:
            examples["prompt_improves_over_naive"].append(packed)

        if (not p_correct) and l_correct:
            examples["lora_improves_over_prompt"].append(packed)

        if p_correct and (not l_correct):
            examples["lora_regresses_from_prompt"].append(packed)

    summary = {}

    for category, records in examples.items():
        selected = records[:20]
        output_path = f"outputs/examples/{category}.json"

        with open(output_path, "w") as f:
            json.dump(selected, f, indent=2)

        summary[category] = len(records)
        print(category, "total:", len(records), "saved:", len(selected))

    with open("outputs/examples/qualitative_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved qualitative examples to outputs/examples/")


if __name__ == "__main__":
    main()