import argparse
import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset
from src.infer import run_inference
from src.metrics import compute_metrics
from src.model_qwen import Qwen25VLModel
from src.utils import load_config, save_json, load_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=[
            "qwen_naive_no_ocr",
            "qwen_constrained_with_ocr",
            "qwen_lora10k_constrained_with_ocr",
        ],
    )

    parser.add_argument(
        "--val_size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
    )

    return parser.parse_args()


def get_setting_config(setting):
    if setting == "qwen_naive_no_ocr":
        return {
            "setting_name": "fullval_qwen_naive_no_ocr",
            "prompt_type": "naive",
            "use_ocr": False,
            "adapter_path": None,
        }

    if setting == "qwen_constrained_with_ocr":
        return {
            "setting_name": "fullval_qwen_constrained_with_ocr",
            "prompt_type": "constrained",
            "use_ocr": True,
            "adapter_path": None,
        }

    if setting == "qwen_lora10k_constrained_with_ocr":
        return {
            "setting_name": "fullval_qwen_lora10000_constrained_with_ocr",
            "prompt_type": "constrained",
            "use_ocr": True,
            "adapter_path": "outputs/lora/qwen_textvqa_10000",
        }

    raise ValueError(f"Unknown setting: {setting}")


def main():
    args = parse_args()

    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    setting = get_setting_config(args.setting)

    val_size = args.val_size
    setting_name = setting["setting_name"]

    prediction_path = f"outputs/predictions/{setting_name}_n{val_size}.json"
    metric_path = f"outputs/metrics/{setting_name}_n{val_size}.json"

    print("Full validation evaluation")
    print("Setting:", setting_name)
    print("Validation size:", val_size)
    print("Prediction path:", prediction_path)
    print("Metric path:", metric_path)

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])

    val_subset = get_fixed_validation_subset(
        dataset,
        n=val_size,
        seed=config["data"]["seed"],
    )

    if os.path.exists(prediction_path):
        records = load_json(prediction_path)
        start_idx = len(records)
        print(f"Found existing predictions: {start_idx}/{val_size}")
    else:
        records = []
        start_idx = 0
        print("No existing predictions found. Starting from 0.")

    if start_idx >= val_size:
        print("All predictions already completed.")
        valid_records = [r for r in records if not r.get("skipped", False)]

        metrics = compute_metrics(valid_records)
        metrics["setting"] = setting_name
        metrics["model"] = "Qwen2.5-VL-3B-Instruct"
        metrics["prompt_type"] = setting["prompt_type"]
        metrics["use_ocr"] = setting["use_ocr"]
        metrics["adapter_path"] = setting["adapter_path"]
        metrics["n_total_including_skipped"] = len(records)
        metrics["n_skipped"] = len(records) - len(valid_records)

        save_json(metrics, metric_path)
        print(metrics)
        return

    model = Qwen25VLModel(
        model_name=config["model"]["name"],
        device=config["experiment"]["device"],
        adapter_path=setting["adapter_path"],
    )

    for idx in range(start_idx, val_size):
        print(f"\n[{setting_name}] Running index {idx}/{val_size - 1}")

        single_example = val_subset.select([idx])

        try:
            new_record = run_inference(
                dataset_split=single_example,
                model_wrapper=model,
                prompt_type=setting["prompt_type"],
                use_ocr=setting["use_ocr"],
                max_new_tokens=config["model"]["max_new_tokens"],
                temperature=config["model"]["temperature"],
                do_sample=config["model"]["do_sample"],
            )[0]

            new_record["idx"] = idx
            new_record["setting"] = setting_name
            new_record["model"] = "Qwen2.5-VL-3B-Instruct"
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
                    "prompt_type": setting["prompt_type"],
                    "use_ocr": setting["use_ocr"],
                    "prediction": "",
                    "setting": setting_name,
                    "model": "Qwen2.5-VL-3B-Instruct",
                    "skipped": True,
                    "skip_reason": str(error),
                }
            )

        if (idx + 1) % args.save_every == 0:
            save_json(records, prediction_path)

            valid_records = [r for r in records if not r.get("skipped", False)]
            partial_metrics = compute_metrics(valid_records)
            partial_metrics["setting"] = setting_name
            partial_metrics["model"] = "Qwen2.5-VL-3B-Instruct"
            partial_metrics["prompt_type"] = setting["prompt_type"]
            partial_metrics["use_ocr"] = setting["use_ocr"]
            partial_metrics["adapter_path"] = setting["adapter_path"]
            partial_metrics["n_total_including_skipped"] = len(records)
            partial_metrics["n_skipped"] = len(records) - len(valid_records)

            save_json(partial_metrics, metric_path)

            print(f"Saved progress: {idx + 1}/{val_size}")
            print(partial_metrics)

    valid_records = [r for r in records if not r.get("skipped", False)]

    metrics = compute_metrics(valid_records)
    metrics["setting"] = setting_name
    metrics["model"] = "Qwen2.5-VL-3B-Instruct"
    metrics["prompt_type"] = setting["prompt_type"]
    metrics["use_ocr"] = setting["use_ocr"]
    metrics["adapter_path"] = setting["adapter_path"]
    metrics["n_total_including_skipped"] = len(records)
    metrics["n_skipped"] = len(records) - len(valid_records)

    save_json(records, prediction_path)
    save_json(metrics, metric_path)

    print("Final full-val metrics:")
    print(metrics)


if __name__ == "__main__":
    main()