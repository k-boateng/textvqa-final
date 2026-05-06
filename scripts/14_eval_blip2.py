import argparse
import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset
from src.infer import run_inference
from src.metrics import compute_metrics
from src.model_blip2 import BLIP2Model
from src.utils import load_config, save_json, load_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val_size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["naive", "constrained"],
    )

    parser.add_argument(
        "--use_ocr",
        action="store_true",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    val_size = args.val_size

    setting_name = (
        f"blip2_{args.prompt_type}_"
        f"{'with_ocr' if args.use_ocr else 'no_ocr'}"
    )

    prediction_path = f"outputs/predictions/{setting_name}_n{val_size}.json"
    metric_path = f"outputs/metrics/{setting_name}_n{val_size}.json"

    print("BLIP-2 evaluation")
    print("Model:", args.model_name)
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
        metrics["model"] = args.model_name
        metrics["prompt_type"] = args.prompt_type
        metrics["use_ocr"] = args.use_ocr
        metrics["n_total_including_skipped"] = len(records)
        metrics["n_skipped"] = len(records) - len(valid_records)

        save_json(metrics, metric_path)
        print(metrics)
        return

    model = BLIP2Model(
        model_name=args.model_name,
        device=config["experiment"]["device"],
    )

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
            new_record["model"] = args.model_name
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
                    "model": args.model_name,
                    "skipped": True,
                    "skip_reason": str(error),
                }
            )

        if (idx + 1) % args.save_every == 0:
            save_json(records, prediction_path)

            valid_records = [r for r in records if not r.get("skipped", False)]
            partial_metrics = compute_metrics(valid_records)
            partial_metrics["setting"] = setting_name
            partial_metrics["model"] = args.model_name
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
    metrics["model"] = args.model_name
    metrics["prompt_type"] = args.prompt_type
    metrics["use_ocr"] = args.use_ocr
    metrics["n_total_including_skipped"] = len(records)
    metrics["n_skipped"] = len(records) - len(valid_records)

    save_json(records, prediction_path)
    save_json(metrics, metric_path)

    print("Final BLIP-2 metrics:")
    print(metrics)


if __name__ == "__main__":
    main()