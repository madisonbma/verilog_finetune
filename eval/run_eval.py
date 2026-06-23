"""
Score a model on a benchmark and report pass@k.

Recommended approach: don't reimplement the benchmark's scoring yourself. Both
VerilogEval and RTLLM ship their own harness that already knows how to compile
and check each problem. This script's job is to (1) generate completions with
your model and (2) hand them to that harness, then (3) save the score.

This file is intentionally a thin wrapper with the integration points marked,
because the exact harness call depends on which benchmark repo you clone.
See eval/README.md for the clone-and-setup steps.

Run:
  python eval/run_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark verilogeval --out results/baseline.json
  python eval/run_eval.py --adapter results/adapter --benchmark verilogeval --out results/sft.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate import generate_vllm_batch   # noqa: E402


BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_benchmark_problems(benchmark):
    """
    Return a list of {"task_id", "prompt"} for the chosen benchmark.

    TODO (integration point): point this at the problems file from the cloned
    benchmark repo. For VerilogEval-v2 that's the spec-to-RTL problem set; for
    RTLLM it's the per-design prompt files. eval/README.md has the paths.
    """
    if benchmark == "verilogeval":
        
    elif benchmark == "rtllm":
        return




def score_with_harness(benchmark, completions_by_task):
    """
    Hand generated completions to the benchmark's own scorer and get pass@k back.

    TODO (integration point): call the benchmark's Makefile / scoring script.
    VerilogEval-v2 uses a Makefile that runs iverilog against hidden testbenches;
    you write each completion to the expected path and invoke it.
    Return a dict like {"pass@1": 0.41, "n_problems": 156}.
    """
    raise NotImplementedError(
        "Wire this to the benchmark's scoring harness. See eval/README.md."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL, help="base model name")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    parser.add_argument("--benchmark", default="verilogeval", choices=["verilogeval", "rtllm"])
    parser.add_argument("--k", type=int, default=1, help="pass@k (use 1 for pass@1)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    problems = load_benchmark_problems(args.benchmark)
    prompts = [p["prompt"] for p in problems]

    # Greedy for pass@1; sample with temperature>0 for pass@k>1.
    temperature = 0.0 if args.k == 1 else 0.8
    completions = generate_vllm_batch(
        args.model, prompts, adapter_path=args.adapter, n=args.k, temperature=temperature
    )

    completions_by_task = {
        p["task_id"]: comps for p, comps in zip(problems, completions)
    }
    scores = score_with_harness(args.benchmark, completions_by_task)

    scores["model"] = args.model
    scores["adapter"] = args.adapter
    scores["benchmark"] = args.benchmark
    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
