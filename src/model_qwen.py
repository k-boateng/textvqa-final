import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen25VLModel:
    """
    Thin wrapper around Qwen2.5-VL for TextVQA inference.
    Supports optional LoRA adapter loading.

    Set OFFLOAD_FOLDER env var to control offloading:
      - unset or non-empty path: offload to that folder (default: outputs/offload)
      - empty string ("")      : disable offloading (use on machines with enough VRAM)
    """

    def __init__(self, model_name, device="cuda", adapter_path=None):
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

        # Resolve offload folder via env var. Empty string => disable offloading.
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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        )

        if adapter_path is not None:
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {adapter_path}")

            peft_kwargs = {}
            if use_offload:
                peft_kwargs["offload_dir"] = offload_env

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                **peft_kwargs,
            )

        self.processor = AutoProcessor.from_pretrained(model_name)

        if hasattr(self.model, "hf_device_map"):
            print("Model device map:", self.model.hf_device_map)
        print("First parameter device:", next(self.model.parameters()).device)

        self.model.eval()

    @torch.no_grad()
    def generate_answer(
        self,
        image,
        prompt,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()