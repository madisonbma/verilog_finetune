"""
Phase 3 (STRETCH): reinforcement learning with a testbench reward, via GRPO.

Difference from Phase 1: in SFT you showed the model correct answers and it
imitated them. Here you show it only the PROBLEM, let it generate several
attempts, score each with reward.py (does it pass the testbench?), and GRPO
nudges the model toward the higher-scoring attempts. This is the recipe behind
recent state-of-the-art Verilog models.

Only attempt this after Phase 1 + 2 are solid. RL is fiddlier and slower.

Run:  python src/train_grpo.py --config configs/grpo.yaml
"""

import argparse
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from utils import load_config, load_model_and_tokenizer
from reward import evaluate_verilog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1. Warm-start from your Phase-1 fine-tune (not the raw base model).
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"],
        load_in_4bit=cfg["load_in_4bit"],
        adapter_path=cfg.get("sft_adapter"),
    )

    # 2. Dataset here needs the PROBLEM and its TESTBENCH (no answer needed).
    #    Each row: {"prompt": "<instruction>", "testbench": "<verilog tb>"}
    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")

    # 3. The reward function GRPO calls on each generated completion.
    #    TRL passes the generated text(s); we compile+simulate against the
    #    matching testbench and return the numeric reward.
    def reward_fn(completions, testbench, **kwargs):
        from utils import extract_verilog
        rewards = []
        for completion, tb in zip(completions, testbench):
            code = extract_verilog(completion)
            r, _status = evaluate_verilog(
                code, tb,
                reward_compile=cfg["reward_compile"],
                reward_pass=cfg["reward_pass"],
            )
            rewards.append(r)
        return rewards

    # 4. Configure + run GRPO.
    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        bf16=cfg["bf16"],
        seed=cfg["seed"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"GRPO adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
