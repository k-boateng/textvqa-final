import os
import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, format_example
from src.utils import load_config, set_seed, save_json, ensure_dir


def serialize_example(raw_example, idx):
    example = format_example(raw_example, use_ocr=True)

    return {
        "idx": idx,
        "image_id": example["image_id"],
        "question": example["question"],
        "answers": example["answers"],
        "ocr_tokens": example["ocr_tokens"],
    }


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    output_dir = "outputs/lora_subsets"
    ensure_dir(output_dir)

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])
    train = dataset["train"]

    train = train.shuffle(seed=config["data"]["seed"])

    subset_sizes = [1000, 5000, 10000]

    for size in subset_sizes:
        subset = train.select(range(size))

        records = []

        for idx, raw_example in enumerate(subset):
            record = serialize_example(raw_example, idx)
            records.append(record)

        output_path = os.path.join(output_dir, f"train_subset_{size}.json")
        save_json(records, output_path)

        print(f"Saved {size} examples to {output_path}")

    print("Done creating LoRA subsets.")


if __name__ == "__main__":
    main()