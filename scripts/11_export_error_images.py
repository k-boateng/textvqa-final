import csv
import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset, format_example
from src.utils import load_config, set_seed, ensure_dir


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    csv_path = "outputs/tables/error_analysis_lora10000_sample.csv"
    output_dir = "outputs/examples/error_analysis_images"
    ensure_dir(output_dir)

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])
    val_subset = get_fixed_validation_subset(
        dataset,
        n=config["data"]["val_size"],
        seed=config["data"]["seed"],
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        idx = int(row["idx"])
        raw_example = val_subset[idx]
        example = format_example(raw_example, use_ocr=True)

        image = example["image"]
        image_path = os.path.join(output_dir, f"idx_{idx}.jpg")
        image.save(image_path)

    print(f"Saved {len(rows)} images to {output_dir}")


if __name__ == "__main__":
    main()