from tqdm import tqdm

from src.data import format_example
from src.prompts import build_prompt


def run_inference(
    dataset_split,
    model_wrapper,
    prompt_type="naive",
    use_ocr=True,
    max_new_tokens=32,
    temperature=0.0,
    do_sample=False,
    limit=None,
):
    """
    Run inference over a TextVQA split or subset.

    Returns a list of prediction records.
    """
    records = []

    if limit is not None:
        dataset_split = dataset_split.select(range(min(limit, len(dataset_split))))

    for idx, raw_example in enumerate(tqdm(dataset_split)):
        example = format_example(raw_example, use_ocr=use_ocr)

        prompt = build_prompt(
            question=example["question"],
            ocr_tokens=example["ocr_tokens"],
            prompt_type=prompt_type,
        )

        try:
            prediction = model_wrapper.generate_answer(
                image=example["image"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        except Exception as error:
            prediction = ""
            print(f"Error on index {idx}: {error}")

        records.append(
            {
                "idx": idx,
                "image_id": example["image_id"],
                "question": example["question"],
                "answers": example["answers"],
                "ocr_tokens": example["ocr_tokens"],
                "prompt_type": prompt_type,
                "use_ocr": use_ocr,
                "prediction": prediction,
            }
        )

    return records