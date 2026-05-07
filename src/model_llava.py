import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLaVAModel:
    """
    Thin wrapper around LLaVA-Phi-3-mini for TextVQA inference.
    Mirrors the Qwen25VLModel interface so the same run_inference() works.

    Default model: xtuner/llava-phi-3-mini-hf
    """

    def __init__(self, model_name="xtuner/llava-phi-3-mini-hf", device="cuda", adapter_path=None):
        self.model_name = model_name
        self.device = device
        self.adapter_path = adapter_path

        print("Requested device:", device)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print("Using dtype:", dtype)

        offload_env = os.environ.get("OFFLOAD_FOLDER", "outputs/offload")
        use_offload = offload_env != ""
        print(f"Offload enabled: {use_offload} (folder='{offload_env}')")

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        if use_offload:
            load_kwargs["offload_folder"] = offload_env
            load_kwargs["offload_state_dict"] = True

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.model.eval()

        if hasattr(self.model, "hf_device_map"):
            print("Model device map:", self.model.hf_device_map)
        print("First parameter device:", next(self.model.parameters()).device)

    def _build_chat_prompt(self, question_prompt):
        """
        LLaVA-Phi-3-mini uses Phi-3's chat format with explicit special tokens.
        The processor doesn't ship a chat_template attribute, so we build it manually.

        Format:
            <|user|>
            <image>
            {question}<|end|>
            <|assistant|>
        """
        return (
            f"<|user|>\n"
            f"<image>\n"
            f"{question_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )

    @torch.no_grad()
    def generate_answer(
        self,
        image,
        prompt,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    ):
        chat_prompt = self._build_chat_prompt(prompt)

        inputs = self.processor(
            images=image,
            text=chat_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id
                            or self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim the prompt prefix from generated_ids
        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, input_len:]

        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Light cleanup
        output_text = output_text.strip()
        # Strip leftover end tokens that sometimes survive special-token stripping
        for tok in ["<|end|>", "<|endoftext|>", "<|assistant|>"]:
            output_text = output_text.replace(tok, "").strip()
        # Take first line only
        output_text = output_text.split("\n")[0].strip()
        # Strip trailing punctuation/quotes
        output_text = output_text.strip(' ."\'')

        return output_text