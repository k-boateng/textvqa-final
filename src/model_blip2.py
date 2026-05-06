import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Model:
    """
    Thin wrapper around BLIP-2 OPT-2.7B for TextVQA inference.
    """

    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        self.model_name = model_name

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

        self.device = next(self.model.parameters()).device
        print("First parameter device:", self.device)

    @torch.no_grad()
    def generate_answer(
        self,
        image,
        prompt,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    ):
        # BLIP-2 behaves better when the prompt explicitly ends with Answer:
        clean_prompt = prompt.strip()

        if "Answer:" not in clean_prompt:
            clean_prompt = clean_prompt + "\nAnswer:"

        inputs = self.processor(
            images=image,
            text=clean_prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )


        decoded = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        output_text = decoded.strip()

        # Remove exact prompt echo if present.
        if output_text.startswith(clean_prompt):
            output_text = output_text[len(clean_prompt):].strip()

        # Remove old prompt form too, just in case.
        old_prompt = prompt.strip()
        if output_text.startswith(old_prompt):
            output_text = output_text[len(old_prompt):].strip()

        # If model includes "Answer:", keep only what follows it.
        if "Answer:" in output_text:
            output_text = output_text.split("Answer:")[-1].strip()

        # Keep only first line for TextVQA.
        output_text = output_text.split("\n")[0].strip()

        return output_text