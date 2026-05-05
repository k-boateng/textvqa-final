from collections import Counter

import torch
from torch.utils.data import Dataset

from qwen_vl_utils import process_vision_info

from src.data import format_example
from src.prompts import build_prompt
from src.utils import normalize_answer


def choose_target_answer(answers):
    """
    TextVQA has multiple human answers.
    We train on the most frequent normalized answer.
    """
    if not answers:
        return ""

    normalized = [normalize_answer(ans) for ans in answers if ans is not None]

    if not normalized:
        return ""

    counts = Counter(normalized)
    return counts.most_common(1)[0][0]


class TextVQALoRADataset(Dataset):
    """
    Dataset wrapper for Qwen2.5-VL LoRA fine-tuning on TextVQA.
    """

    def __init__(
        self,
        hf_split,
        prompt_type="constrained",
        use_ocr=True,
    ):
        self.hf_split = hf_split
        self.prompt_type = prompt_type
        self.use_ocr = use_ocr

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        raw_example = self.hf_split[idx]
        example = format_example(raw_example, use_ocr=self.use_ocr)

        target_answer = choose_target_answer(example["answers"])

        prompt = build_prompt(
            question=example["question"],
            ocr_tokens=example["ocr_tokens"],
            prompt_type=self.prompt_type,
        )

        return {
            "idx": idx,
            "image": example["image"],
            "question": example["question"],
            "answers": example["answers"],
            "ocr_tokens": example["ocr_tokens"],
            "prompt": prompt,
            "target_answer": target_answer,
        }


class QwenVLDataCollator:
    """
    Collator for supervised Qwen2.5-VL training.

    It builds two versions of each sample:

    1. Prompt-only conversation:
       user image + question, with generation prompt.

    2. Full conversation:
       user image + question, assistant answer.

    Labels are masked for the prompt tokens so the loss is only computed
    on the assistant answer.
    """

    def __init__(self, processor):
        self.processor = processor

    def _build_messages(self, image, prompt, target_answer=None):
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

        if target_answer is None:
            return [user_message]

        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": target_answer,
                }
            ],
        }

        return [user_message, assistant_message]

    def __call__(self, examples):
        full_texts = []
        full_images = []

        prompt_lengths = []

        for example in examples:
            image = example["image"]
            prompt = example["prompt"]
            target_answer = example["target_answer"]

            prompt_messages = self._build_messages(
                image=image,
                prompt=prompt,
                target_answer=None,
            )

            full_messages = self._build_messages(
                image=image,
                prompt=prompt,
                target_answer=target_answer,
            )

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_image_inputs, _ = process_vision_info(prompt_messages)
            full_image_inputs, _ = process_vision_info(full_messages)

            # Compute prompt length for label masking.
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=prompt_image_inputs,
                padding=True,
                return_tensors="pt",
            )

            prompt_len = prompt_inputs["input_ids"].shape[1]
            prompt_lengths.append(prompt_len)

            full_texts.append(full_text)

            # Each example has one image.
            if isinstance(full_image_inputs, list):
                full_images.extend(full_image_inputs)
            else:
                full_images.append(full_image_inputs)

        batch = self.processor(
            text=full_texts,
            images=full_images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100

        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels

        return batch