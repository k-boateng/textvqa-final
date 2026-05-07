import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Model:
    """
    Thin wrapper around BLIP-2 OPT-2.7B for TextVQA inference.
    """

    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        self.model_name = model_name
        self.device = device

        print("Requested device:", device)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print("Using dtype:", dtype)

        self.processor = Blip2Processor.from_pretrained(model_name)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

        self.model.eval()

        self.actual_device = next(self.model.parameters()).device
        print("First parameter device:", self.actual_device)

    def _prepare_prompt(self, prompt):
        prompt = prompt.strip()

        if "Answer:" not in prompt:
            prompt = prompt + "\nAnswer:"

        return prompt

    def _clean_output(self, text, prompt):
        text = text.strip()

        # Remove full prompt if the decoder echoed it.
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        # Remove everything before final Answer: if present.
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()

        # Keep only first line.
        text = text.split("\n")[0].strip()

        # Remove common junk prefixes.
        for prefix in ["A:", "answer:", "Answer:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text

    @torch.no_grad()
    def generate_answer(
        self,
        image,
        prompt,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    ):
        prompt = self._prepare_prompt(prompt)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.actual_device)
            for key, value in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_beams=1,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode full output first.
        full_text = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        cleaned = self._clean_output(full_text, prompt)

        # Fallback: if full decode cleans to empty, try slicing new tokens.
        if cleaned == "" and "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]

            if generated_ids.shape[1] > input_len:
                answer_ids = generated_ids[:, input_len:]

                answer_text = self.processor.tokenizer.batch_decode(
                    answer_ids,
                    skip_special_tokens=True,
                )[0]

                cleaned = self._clean_output(answer_text, prompt)

        return cleaned