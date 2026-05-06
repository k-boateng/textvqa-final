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

        print("First parameter device:", next(self.model.parameters()).device)

    @torch.no_grad()
    def generate_answer(
        self,
        image,
        prompt,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    ):
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        # BLIP-2 with OPT can return prompt tokens + answer tokens.
        # We only want the newly generated answer tokens.
        if "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]
            answer_ids = generated_ids[:, input_len:]
        else:
            answer_ids = generated_ids

        output_text = self.processor.tokenizer.batch_decode(
            answer_ids,
            skip_special_tokens=True,
        )[0].strip()

        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        if "Answer:" in output_text:
            output_text = output_text.split("Answer:")[-1].strip()

        # TextVQA answers should be short; keep first generated line.
        output_text = output_text.split("\n")[0].strip()

        return output_text