import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset
from src.infer import run_inference
from src.metrics import compute_metrics
from src.model_qwen import Qwen25VLModel
from src.utils import load_config, save_json, load_json, set_seed


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    val_size = config["data"]["val_size"]

    prediction_path = f"outputs/predictions/zero_shot_naive_no_ocr_n{val_size}.json"
    metric_path = f"outputs/metrics/zero_shot_naive_no_ocr_n{val_size}.json"

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
        metrics = compute_metrics(records)
        save_json(metrics, metric_path)
        print(metrics)
        return

    model = Qwen25VLModel(
        model_name=config["model"]["name"],
        device=config["experiment"]["device"],
    )

    save_every = 1

    for idx in range(start_idx, val_size):
        print(f"\nRunning index {idx}/{val_size - 1}")
        single_example = val_subset.select([idx])

        new_record = run_inference(
            dataset_split=single_example,
            model_wrapper=model,
            prompt_type="naive",
            use_ocr=False,
            max_new_tokens=config["model"]["max_new_tokens"],
            temperature=config["model"]["temperature"],
            do_sample=config["model"]["do_sample"],
        )[0]

        new_record["idx"] = idx
        records.append(new_record)

        if (idx + 1) % save_every == 0:
            save_json(records, prediction_path)
            partial_metrics = compute_metrics(records)
            save_json(partial_metrics, metric_path)
            print(f"Saved progress: {idx + 1}/{val_size}")
            print(partial_metrics)

    metrics = compute_metrics(records)

    save_json(records, prediction_path)
    save_json(metrics, metric_path)

    print("Final zero-shot naive no-OCR metrics:")
    print(metrics)


if __name__ == "__main__":
    main()