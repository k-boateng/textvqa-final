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
        "--train_size",
        type=int,
        required=True,
        choices=[1000, 5000, 10000],
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="constrained",
        choices=["naive", "ocr_aware", "constrained"],
    )

    parser.add_argument(
        "--use_ocr",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    val_size = config["data"]["val_size"]
    model_name = config["model"]["name"]

    adapter_path = f"outputs/lora/qwen_textvqa_{args.train_size}"

    setting_name = (
        f"lora_{args.train_size}_{args.prompt_type}_"
        f"{'with_ocr' if args.use_ocr else 'no_ocr'}"
    )

    prediction_path = f"outputs/predictions/{setting_name}_n{val_size}.json"
    metric_path = f"outputs/metrics/{setting_name}_n{val_size}.json"

    print("Evaluating LoRA adapter")
    print("Adapter:", adapter_path)
    print("Setting:", setting_name)
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
        metrics["train_size"] = args.train_size
        metrics["prompt_type"] = args.prompt_type
        metrics["use_ocr"] = args.use_ocr
        metrics["n_total_including_skipped"] = len(records)
        metrics["n_skipped"] = len(records) - len(valid_records)

        save_json(metrics, metric_path)
        print(metrics)
        return

    model = Qwen25VLModel(
        model_name=model_name,
        device=config["experiment"]["device"],
        adapter_path=adapter_path,
    )

    save_every = 1

    for idx in range(start_idx, val_size):
        print(f"\n[{setting_name}] Running index {idx}/{val_size - 1}")

        single_example = val_subset.select([idx])

        try:
            new_record = run_inference(
                dataset_split=single_example,
                model_wrapper=model,
                prompt_type=args.prompt_type,
                use_ocr=args.use_ocr,
                max_new_tokens=config["model"]["max_new_tokens"],
                temperature=config["model"]["temperature"],
                do_sample=config["model"]["do_sample"],
            )[0]

            new_record["idx"] = idx
            new_record["setting"] = setting_name
            new_record["train_size"] = args.train_size
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
                    "prompt_type": args.prompt_type,
                    "use_ocr": args.use_ocr,
                    "prediction": "",
                    "setting": setting_name,
                    "train_size": args.train_size,
                    "skipped": True,
                    "skip_reason": str(error),
                }
            )

        if (idx + 1) % save_every == 0:
            save_json(records, prediction_path)

            valid_records = [r for r in records if not r.get("skipped", False)]
            partial_metrics = compute_metrics(valid_records)
            partial_metrics["setting"] = setting_name
            partial_metrics["train_size"] = args.train_size
            partial_metrics["prompt_type"] = args.prompt_type
            partial_metrics["use_ocr"] = args.use_ocr
            partial_metrics["n_total_including_skipped"] = len(records)
            partial_metrics["n_skipped"] = len(records) - len(valid_records)

            save_json(partial_metrics, metric_path)

            print(f"Saved progress: {idx + 1}/{val_size}")
            print(partial_metrics)

    valid_records = [r for r in records if not r.get("skipped", False)]

    metrics = compute_metrics(valid_records)
    metrics["setting"] = setting_name
    metrics["train_size"] = args.train_size
    metrics["prompt_type"] = args.prompt_type
    metrics["use_ocr"] = args.use_ocr
    metrics["n_total_including_skipped"] = len(records)
    metrics["n_skipped"] = len(records) - len(valid_records)

    save_json(records, prediction_path)
    save_json(metrics, metric_path)

    print("Final LoRA metrics:")
    print(metrics)


if __name__ == "__main__":
    main()