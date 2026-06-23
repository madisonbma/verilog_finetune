"""
Phase 1: QLoRA supervised fine-tuning.

What this does, in one breath: load the base model in 4-bit, attach trainable
LoRA adapters, then use TRL's SFTTrainer to make the model imitate the
(instruction -> good Verilog) pairs in your dataset. The output is a small
adapter saved to results/adapter/.

Run:  python src/train_sft.py --config configs/sft_qlora.yaml
"""

import argparse
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from utils import load_config, load_model_and_tokenizer, build_lora_config


def format_example(example, tokenizer):
    """
    Turn one dataset row into a single training string in the model's chat
    format: the user's instruction followed by the target Verilog answer.
    SFTTrainer trains the model to produce the answer half.
    """
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1. Load the 4-bit base model + tokenizer.
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"], load_in_4bit=cfg["load_in_4bit"]
    )

    # 2. Describe the LoRA adapters we want to train.
    lora_config = build_lora_config(cfg)

    # 3. Load and format the dataset.
    #    Expects a .jsonl where each line has "instruction" and "output".
    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    dataset = dataset.map(
        lambda ex: {"text": format_example(ex, tokenizer)}
    )

    # 4. Configure the training loop (these map 1:1 to the YAML).
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=cfg["bf16"],
        max_seq_length=cfg["max_seq_length"],
        seed=cfg["seed"],
        dataset_text_field="text",
    )

    # 5. Train. SFTTrainer wires the model, adapters, data, and loop together.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,   # passing this makes it a LoRA/QLoRA run
        tokenizer=tokenizer,
    )
    trainer.train()

    # 6. Save the trained adapter (small — just the LoRA weights).
    trainer.save_model(cfg["output_dir"])
    print(f"Adapter saved to {cfg['output_dir']}")
    print("Next: python eval/run_eval.py --adapter", cfg["output_dir"], "--benchmark verilogeval")


if __name__ == "__main__":
    main()
