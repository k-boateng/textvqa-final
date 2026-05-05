import os
import csv
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset
from src.infer import run_inference
from src.metrics import compute_metrics
from src.model_qwen import Qwen25VLModel
from src.utils import load_config, save_json, load_json, set_seed, ensure_dir


def evaluate_setting(
    setting_name,
    val_subset,
    model,
    config,
    prompt_type,
    use_ocr,
):
    val_size = config["data"]["val_size"]

    prediction_path = f"outputs/predictions/{setting_name}_n{val_size}.json"
    metric_path = f"outputs/metrics/{setting_name}_n{val_size}.json"

    if os.path.exists(prediction_path):
        records = load_json(prediction_path)
        start_idx = len(records)
        print(f"\nFound existing predictions for {setting_name}: {start_idx}/{val_size}")
    else:
        records = []
        start_idx = 0
        print(f"\nStarting new run for {setting_name}")

    if start_idx >= val_size:
        print(f"{setting_name} already complete.")
        valid_records = [r for r in records if not r.get("skipped", False)]
        metrics = compute_metrics(valid_records)
        metrics["setting"] = setting_name
        metrics["prompt_type"] = prompt_type
        metrics["use_ocr"] = use_ocr
        metrics["n_total_including_skipped"] = len(records)
        metrics["n_skipped"] = len(records) - len(valid_records)
        save_json(metrics, metric_path)
        return metrics

    save_every = 1

    for idx in range(start_idx, val_size):
        print(f"\n[{setting_name}] Running index {idx}/{val_size - 1}")

        single_example = val_subset.select([idx])

        try:
            new_record = run_inference(
                dataset_split=single_example,
                model_wrapper=model,
                prompt_type=prompt_type,
                use_ocr=use_ocr,
                max_new_tokens=config["model"]["max_new_tokens"],
                temperature=config["model"]["temperature"],
                do_sample=config["model"]["do_sample"],
            )[0]

            new_record["idx"] = idx
            new_record["setting"] = setting_name
            records.append(new_record)

        except Exception as error:
            print(f"Error on index {idx}: {error}")

            records.append(
                {
                    "idx": idx,
                    "image_id": None,
                    "question": None,
                    "answers": [],
                    "ocr_tokens": [],
                    "prompt_type": prompt_type,
                    "use_ocr": use_ocr,
                    "prediction": "",
                    "setting": setting_name,
                    "skipped": True,
                    "skip_reason": str(error),
                }
            )

        if (idx + 1) % save_every == 0:
            save_json(records, prediction_path)

            valid_records = [r for r in records if not r.get("skipped", False)]
            partial_metrics = compute_metrics(valid_records)
            partial_metrics["setting"] = setting_name
            partial_metrics["prompt_type"] = prompt_type
            partial_metrics["use_ocr"] = use_ocr
            partial_metrics["n_total_including_skipped"] = len(records)
            partial_metrics["n_skipped"] = len(records) - len(valid_records)

            save_json(partial_metrics, metric_path)

            print(f"Saved progress for {setting_name}: {idx + 1}/{val_size}")
            print(partial_metrics)

    valid_records = [r for r in records if not r.get("skipped", False)]

    metrics = compute_metrics(valid_records)
    metrics["setting"] = setting_name
    metrics["prompt_type"] = prompt_type
    metrics["use_ocr"] = use_ocr
    metrics["n_total_including_skipped"] = len(records)
    metrics["n_skipped"] = len(records) - len(valid_records)

    save_json(records, prediction_path)
    save_json(metrics, metric_path)

    return metrics


def save_summary_csv(metrics_list, path):
    ensure_dir(os.path.dirname(path))

    fieldnames = [
        "setting",
        "prompt_type",
        "use_ocr",
        "accuracy",
        "token_f1",
        "rouge_l",
        "bleu",
        "n",
        "n_total_including_skipped",
        "n_skipped",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in metrics_list:
            row = {key: metrics.get(key, None) for key in fieldnames}
            writer.writerow(row)


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    val_size = config["data"]["val_size"]

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])

    val_subset = get_fixed_validation_subset(
        dataset,
        n=val_size,
        seed=config["data"]["seed"],
    )

    model = Qwen25VLModel(
        model_name=config["model"]["name"],
        device=config["experiment"]["device"],
    )

    settings = [
        {
            "setting_name": "prompt_naive_no_ocr",
            "prompt_type": "naive",
            "use_ocr": False,
        },
        {
            "setting_name": "prompt_naive_with_ocr",
            "prompt_type": "naive",
            "use_ocr": True,
        },
        {
            "setting_name": "prompt_ocr_aware_with_ocr",
            "prompt_type": "ocr_aware",
            "use_ocr": True,
        },
        {
            "setting_name": "prompt_constrained_with_ocr",
            "prompt_type": "constrained",
            "use_ocr": True,
        },
        {
            "setting_name": "prompt_constrained_no_ocr",
            "prompt_type": "constrained",
            "use_ocr": False,
        },
    ]

    all_metrics = []

    for setting in settings:
        metrics = evaluate_setting(
            setting_name=setting["setting_name"],
            val_subset=val_subset,
            model=model,
            config=config,
            prompt_type=setting["prompt_type"],
            use_ocr=setting["use_ocr"],
        )

        all_metrics.append(metrics)

    summary_json_path = f"outputs/metrics/prompt_eval_summary_n{val_size}.json"
    summary_csv_path = f"outputs/tables/prompt_eval_summary_n{val_size}.csv"

    save_json(all_metrics, summary_json_path)
    save_summary_csv(all_metrics, summary_csv_path)

    print("\nPrompt evaluation summary:")
    for metrics in all_metrics:
        print(metrics)


if __name__ == "__main__":
    main()