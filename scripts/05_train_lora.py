import argparse
import os
import sys
sys.path.append(".")

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.data import load_textvqa_dataset
from src.lora_dataset import TextVQALoRADataset, QwenVLDataCollator
from src.utils import load_config, set_seed, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        choices=[1000, 5000, 10000],
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for QLoRA-style training.",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="constrained",
        choices=["naive", "ocr_aware", "constrained"],
    )

    parser.add_argument(
        "--use_ocr",
        action="store_true",
        help="Include OCR tokens in the training prompt.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config("configs/qwen25vl_3b.yaml")
    set_seed(config["data"]["seed"])

    model_name = config["model"]["name"]

    output_dir = f"outputs/lora/qwen_textvqa_{args.train_size}"
    ensure_dir(output_dir)

    print("Training LoRA adapter")
    print("Model:", model_name)
    print("Train size:", args.train_size)
    print("Output dir:", output_dir)
    print("Prompt type:", args.prompt_type)
    print("Use OCR:", args.use_ocr)
    print("Use 4-bit:", args.use_4bit)

    dataset = load_textvqa_dataset(config["data"]["dataset_name"])
    train_split = dataset["train"]

    train_split = train_split.shuffle(seed=config["data"]["seed"]).select(
        range(args.train_size)
    )

    processor = AutoProcessor.from_pretrained(model_name)

    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)

    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = TextVQALoRADataset(
        hf_split=train_split,
        prompt_type=args.prompt_type,
        use_ocr=args.use_ocr,
    )

    data_collator = QwenVLDataCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print("Saving final LoRA adapter to:", output_dir)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()