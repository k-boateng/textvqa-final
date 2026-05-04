from datasets import load_dataset
from PIL import Image


def load_textvqa_dataset(dataset_name="lmms-lab/textvqa"):
    """
    Load the TextVQA dataset from Hugging Face.
    """
    dataset = load_dataset(dataset_name)
    return dataset


def get_fixed_validation_subset(dataset, n=2000, seed=42):
    """
    Use a fixed validation subset so all experiments compare on the same examples.
    """
    val = dataset["validation"]

    if n is None or n >= len(val):
        return val

    return val.shuffle(seed=seed).select(range(n))


def extract_answers(example):
    """
    Extract answer list from a TextVQA example.

    We keep this flexible because different dataset versions may name
    the answer field slightly differently.
    """
    if "answers" in example and example["answers"] is not None:
        answers = example["answers"]
    elif "answer" in example and example["answer"] is not None:
        answers = example["answer"]
    else:
        answers = []

    if isinstance(answers, str):
        answers = [answers]

    return answers


def extract_ocr_tokens(example):
    """
    Extract OCR tokens if present.
    """
    possible_keys = ["ocr_tokens", "ocr", "ocr_info"]

    for key in possible_keys:
        if key not in example or example[key] is None:
            continue

        value = example[key]

        if isinstance(value, list):
            if len(value) == 0:
                return []

            if isinstance(value[0], str):
                return value

            if isinstance(value[0], dict):
                tokens = []
                for item in value:
                    if "text" in item and item["text"] is not None:
                        tokens.append(str(item["text"]))
                return tokens

    return []


def get_image(example, max_side=1024):
    """
    Extract image from example and resize very large images.

    Large raw TextVQA images can create too many visual tokens for VLM inference.
    Resizing keeps evaluation stable while preserving the visible text reasonably well.
    """
    image = example.get("image", None)

    if isinstance(image, Image.Image):
        image = image.convert("RGB")

        width, height = image.size
        largest_side = max(width, height)

        if largest_side > max_side:
            scale = max_side / largest_side
            new_width = int(width * scale)
            new_height = int(height * scale)

            image = image.resize(
                (new_width, new_height),
                resample=Image.BICUBIC,
            )

        return image

    return image


def format_example(example, use_ocr=True):
    """
    Standardize one raw dataset example into the format our pipeline expects.
    """
    question = example["question"]
    answers = extract_answers(example)
    image = get_image(example)
    ocr_tokens = extract_ocr_tokens(example)

    if not use_ocr:
        ocr_tokens = []

    return {
        "question": question,
        "answers": answers,
        "image": image,
        "ocr_tokens": ocr_tokens,
        "image_id": example.get("image_id", example.get("id", None)),
    }