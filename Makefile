# Shortcut commands. Run `make help` to list them.
.PHONY: help setup data baseline sft eval grpo

help:
	@echo "setup     - install python deps (run scripts/setup_gpu.sh for system tools)"
	@echo "data      - download + format the training dataset"
	@echo "baseline  - score the UNtuned base model (Phase 0)"
	@echo "sft       - QLoRA supervised fine-tune (Phase 1)"
	@echo "eval      - score the fine-tuned adapter"
	@echo "grpo      - RL fine-tune with testbench reward (Phase 3 stretch)"

setup:
	pip install -r requirements.txt

data:
	python data/prepare_data.py --out data/sft_dataset.jsonl

baseline:
	bash scripts/benchmark_base.sh

sft:
	python src/train_sft.py --config configs/sft_qlora.yaml

eval:
	python eval/run_eval.py --adapter results/adapter --benchmark verilogeval

grpo:
	python src/train_grpo.py --config configs/grpo.yaml
