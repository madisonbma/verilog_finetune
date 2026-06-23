# Evaluation

You score models with the benchmarks' OWN harnesses rather than reimplementing
scoring — they already know how to compile each problem against hidden
testbenches. `run_eval.py` generates completions with your model and hands them
to that harness.

## VerilogEval (primary)

1. Clone it:
   ```bash
   git clone https://github.com/NVlabs/verilog-eval
   ```
2. Read its README. v2 supports both "code completion" and "spec-to-RTL" tasks
   and uses a Makefile-based flow that calls `iverilog` under the hood (already
   installed by scripts/setup_gpu.sh).
3. Wire the two integration points in `run_eval.py`:
   - `load_benchmark_problems()` -> return the v2 problem prompts.
   - `score_with_harness()` -> write each completion where the Makefile expects
     it and invoke the Makefile, then parse the reported pass@k.

## RTLLM (secondary)

1. Clone it:
   ```bash
   git clone https://github.com/hkust-zhiyao/RTLLM
   ```
2. It's ~50 designs across arithmetic / control / memory / misc, each with a
   reference and a testbench. Same wiring idea as above.

## Why two benchmarks

Reporting movement on *both* VerilogEval and RTLLM is far more convincing than
one — it shows the gain generalizes and isn't an artifact of one test set.

## Per-problem logging

To enable the Phase-2 failure taxonomy, have `score_with_harness()` also dump one
JSON per problem into a directory (task_id, status, design_code, compile_stderr,
sim_stdout). `analyze_failures.py` reads that directory.
