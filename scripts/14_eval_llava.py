import argparse
import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset
from src.infer import run_inference
from src.metrics import compute_metrics
from src.model_llava import LLaVAModel
from src.utils import load_config, save_json, load_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["naive", "constrained"],
    )
    parser.add_argument("--use_ocr", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="xtuner/llava-phi-3-mini-hf",
    )
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only run this many examples (for smoke tests).")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    val_size = args.val_size if args.limit is None else args.limit
    setting_name = (
        f"llava_{args.prompt_type}_"
        f"{'with_ocr' if args.use_ocr else 'no_ocr'}"
    )
    if args.limit is not None:
        setting_name += f"_smoke{args.limit}"

    prediction_path = f"outputs/predictions/{setting_name}_n{val_size}.json"
    metric_path = f"outputs/metrics/{setting_name}_n{val_size}.json"

    print("=" * 60)
    print("LLaVA-Phi-3-mini evaluation")
    print(f"Model:          {args.model_name}")
    print(f"Setting:        {setting_name}")
    print(f"Val size:       {val_size}")
    print(f"Prediction:     {prediction_path}")
    print(f"Metric:         {metric_path}")
    print("=" * 60)

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])
    val_subset = get_fixed_validation_subset(
        dataset,
        n=args.val_size,  # use full val_size for subset selection
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

    model = LLaVAModel(
        model_name=args.model_name,
        device=config["experiment"]["device"],
    )

    for idx in range(start_idx, val_size):
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
            records.append({
                "idx": idx, "image_id": None, "question": None,
                "answers": [], "ocr_tokens": [],
                "prompt_type": args.prompt_type, "use_ocr": args.use_ocr,
                "prediction": "", "setting": setting_name,
                "model": args.model_name,
                "skipped": True, "skip_reason": str(error),
            })

        if (idx + 1) % args.save_every == 0:
            save_json(records, prediction_path)
            valid = [r for r in records if not r.get("skipped", False)]
            partial = compute_metrics(valid)
            partial["setting"] = setting_name
            save_json(partial, metric_path)
            print(f"[{idx+1}/{val_size}] acc={partial.get('accuracy', 0):.4f}")

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
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()