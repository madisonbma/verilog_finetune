#!/usr/bin/env bash
# Phase 0: score the UNtuned base model so you have a baseline to beat.
# This writes results/baseline_verilogeval.json
set -e

MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

echo "==> Scoring base model: $MODEL"
echo "    (no adapter = the untouched, pre-fine-tuning model)"

python eval/run_eval.py \
    --model "$MODEL" \
    --benchmark verilogeval \
    --out results/baseline_verilogeval.json

echo "==> Baseline written to results/baseline_verilogeval.json"
echo "    Keep this number. Every later result is measured against it."
