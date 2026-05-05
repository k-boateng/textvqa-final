import sys
sys.path.append(".")

from src.data import load_textvqa_dataset, get_fixed_validation_subset, format_example
from src.utils import load_config, set_seed


def main():
    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    target_idx = 1411

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])

    val_subset = get_fixed_validation_subset(
        dataset,
        n=config["data"]["val_size"],
        seed=config["data"]["seed"],
    )

    raw_example = val_subset[target_idx]
    example = format_example(raw_example, use_ocr=True)

    print("Index:", target_idx)
    print("Image ID:", example["image_id"])
    print("Question:", example["question"])
    print("Answers:", example["answers"])
    print("Number of OCR tokens:", len(example["ocr_tokens"]))
    print("First 100 OCR tokens:", example["ocr_tokens"][:100])

    image = example["image"]

    print("Image type:", type(image))

    if image is not None:
        print("Image size:", image.size)
        print("Image mode:", image.mode)
        image.save("outputs/examples/problem_example_1411.jpg")
        print("Saved image to outputs/examples/problem_example_1411.jpg")


if __name__ == "__main__":
    main()