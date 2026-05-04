import sys
sys.path.append(".")

from src.data import (
    load_textvqa_dataset,
    get_fixed_validation_subset,
    format_example,
)
from src.utils import load_config, set_seed


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])

    print("Dataset object:")
    print(dataset)

    print("\nAvailable splits:")
    print(dataset.keys())

    if "train" in dataset:
        print("Train size:", len(dataset["train"]))

    if "validation" in dataset:
        print("Validation size:", len(dataset["validation"]))

    if "test" in dataset:
        print("Test size:", len(dataset["test"]))

    val_subset = get_fixed_validation_subset(
        dataset,
        n=5,
        seed=config["data"]["seed"],
    )

    print("\nFirst few formatted validation examples:")

    for i, raw_example in enumerate(val_subset):
        example = format_example(raw_example, use_ocr=True)

        print("=" * 80)
        print("Example:", i)
        print("Image ID:", example["image_id"])
        print("Question:", example["question"])
        print("Answers:", example["answers"])
        print("Number of OCR tokens:", len(example["ocr_tokens"]))
        print("First OCR tokens:", example["ocr_tokens"][:20])
        print("Image type:", type(example["image"]))

        if example["image"] is not None:
            print("Image size:", example["image"].size)


if __name__ == "__main__":
    main()