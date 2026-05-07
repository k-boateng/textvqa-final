import csv
import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset, format_example
from src.utils import load_config, set_seed, ensure_dir


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    csv_path = "outputs/tables/lora_regression_analysis_sample.csv"
    output_dir = "outputs/examples/lora_regression_images"
    ensure_dir(output_dir)

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run scripts/16_make_lora_regression_analysis.py first.")
        return

    print(f"Loading TextVQA dataset...")
    dataset = load_textvqa_dataset(config["data"]["dataset_name"])
    val_subset = get_fixed_validation_subset(
        dataset,
        n=config["data"].get("val_size", 2000),
        seed=config["data"]["seed"],
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} regression cases. Exporting images...")

    saved = 0
    for row in rows:
        idx = int(row["idx"])
        try:
            raw_example = val_subset[idx]
            example = format_example(raw_example, use_ocr=True)
            image = example["image"]
            image_path = os.path.join(output_dir, f"idx_{idx}.jpg")
            image.save(image_path, "JPEG", quality=85)
            saved += 1
        except Exception as e:
            print(f"  Failed idx={idx}: {e}")

    print(f"\nSaved {saved}/{len(rows)} images to {output_dir}")


if __name__ == "__main__":
    main()