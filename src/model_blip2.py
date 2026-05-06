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

        # With device_map="auto", put inputs on the first model parameter device.
        model_device = next(self.model.parameters()).device
        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
        }

        # Only pass temperature if sampling is actually enabled.
        if do_sample:
            generate_kwargs["temperature"] = temperature

        generated_ids = self.model.generate(
            **inputs,
            **generate_kwargs,
        )

        input_ids = inputs.get("input_ids", None)

        # BLIP-2 OPT sometimes returns prompt + generated answer.
        # Only slice off the prompt if the generated sequence actually starts with input_ids.
        if input_ids is not None:
            input_len = input_ids.shape[1]

            if (
                generated_ids.shape[1] > input_len
                and torch.equal(generated_ids[:, :input_len], input_ids)
            ):
                answer_ids = generated_ids[:, input_len:]
            else:
                answer_ids = generated_ids
        else:
            answer_ids = generated_ids

        output_text = self.processor.tokenizer.batch_decode(
            answer_ids,
            skip_special_tokens=True,
        )[0].strip()

        # Safety cleanup
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        # Extra cleanup for common prompt leakage cases.
        if "Question:" in output_text:
            output_text = output_text.split("Question:")[-1].strip()

        if "Answer:" in output_text:
            output_text = output_text.split("Answer:")[-1].strip()

        # If the model only returned the prompt/question and no answer, do not save the prompt.
        if output_text.strip() == prompt.strip():
            output_text = ""

        # TextVQA answers should be short.
        output_text = output_text.split("\n")[0].strip()

        return output_text