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
import re
import subprocess
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate import generate_vllm_batch   # noqa: E402
from utils import compile_and_simulate     # noqa: E402  shared iverilog+vvp primitive

from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_benchmark_problems(benchmark):
    """
    Return a list of {"task_id", "prompt"} for the chosen benchmark.

    BENCHMARK OPTIONS: verilogEval or RTLLM. Both of these are test benches
    to confirm the validity of a Verilog solution.
    """

    benchmark_problems = []
    if benchmark == "verilogeval":
        #using dataset_spec_to_rtl
        #Formatting of this dir:
        #Prob###_*_prompt.txt, Prob###_*_ref.sv, Prob###_*_test.sv
        root_dir = os.path.dirname(__file__)
        verilog_eval_path = Path(os.path.join(root_dir, '..', 'verilog_eval', 'dataset_spec-to-rtl'))
        for file_path in verilog_eval_path.glob('*_prompt.txt'):
            task_id = file_path.name[:-len("_prompt.txt")]   # -> "Prob001_zero"

            with open(file_path, 'r', encoding='utf-8') as file:
                prompt = file.read()
            benchmark_problems.append({"task_id": task_id, "prompt": prompt})


    elif benchmark == "rtllm":
        #Formatting of this dir (3 levels deep):
        #<Category>/<SubCategory>/<design_name>/design_description.txt
        #task_id is the design's location under RTLLM -> stable + unique.
        root_dir = os.path.dirname(__file__)
        rtllm_path = Path(os.path.join(root_dir, '..', 'RTLLM'))
        for desc_path in rtllm_path.rglob('design_description.txt'):
            design_name = desc_path.parent.name
            with open(desc_path, 'r', encoding='utf-8') as file:
                prompt = file.read()
            benchmark_problems.append({"task_id":design_name, "prompt": prompt})


    #stable, reproducible order regardless of filesystem walk order
    #benchmark_problems.sort(key=lambda p: p["task_id"])
    return benchmark_problems


def _rtllm_design_dirs():
    """
    Map task_id -> design directory, mirroring the rtllm branch of
    load_benchmark_problems() so score_with_harness can find each design's
    testbench.v from the task_id alone.
    """
    root_dir = os.path.dirname(__file__)
    rtllm_path = Path(os.path.join(root_dir, '..', 'RTLLM'))
    mapping = {}
    for desc_path in rtllm_path.rglob('design_description.txt'):
        design_dir = desc_path.parent
        design_name = design_dir.name
        mapping[design_name] = design_dir
    return mapping


def strip_langchain_imports(script_path):
    """
    VerilogEval's sv-iv-analyze has a dead `from langchain.schema import ...` at
    module top (the symbols are never used) that crashes on modern langchain
    (>=0.3 dropped langchain.schema). Comment out any langchain import so the
    analyzer runs without that dependency. Idempotent and self-healing — re-applies
    after a fresh clone, no-ops once already patched. Scoring is unaffected because
    the imports are unused, so the pass@1 number stays canonical.
    """
    if not os.path.exists(script_path):
        return
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    changed = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("from langchain", "import langchain")):
            indent = line[:len(line) - len(stripped)]
            lines[i] = f"{indent}# patched out by run_eval (unused, breaks on langchain>=0.3): {stripped}"
            changed = True
    if changed:
        with open(script_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"==> patched dead langchain import out of {os.path.basename(script_path)}")


def run_make_command(cmd, label, build_path):
    """Run a build command in build_path; print + return False on failure."""
    try:
        result = subprocess.run(
            cmd, cwd=build_path, check=True, text=True, capture_output=True,
        )
        print(f"==> {label}: ok")
        if result.stdout:
            print(result.stdout[-2000:])     # tail; full logs live in build/
        return True
    except subprocess.CalledProcessError as e:
        print(f"!! {label} failed (exit {e.returncode})")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        return False

def score_with_harness(benchmark, completions_by_task):
    """
    Verify the model-generated verilog against on the benchmark's scorer.
    Get pass@k back.
    Hand generated completions to the benchmark's own scorer and get pass@k back.

    TODO (integration point): call the benchmark's Makefile / scoring script.
    VerilogEval-v2 uses a Makefile that runs iverilog against hidden testbenches;
    you write each completion to the expected path and invoke it.
    Return a dict like {"pass@1": 0.41, "n_problems": 156}.
    """

    if benchmark == "verilogeval":
        # Score with VerilogEval's OWN harness (configure + make) so pass@1 is the
        # canonical, published-benchmark number. Flow ("Option B"): pre-place each
        # completion as the sample file the Makefile expects, then run ONLY the
        # test+analyze targets so the harness never invokes its own LLM generation
        # (no langchain / API key needed).
        root_dir = os.path.dirname(os.path.abspath(__file__))
        ve_dir = os.path.join(root_dir, "..", "verilog_eval")
        configure = os.path.abspath(os.path.join(ve_dir, "configure"))
        build_path = os.path.abspath(os.path.join(ve_dir, "build"))
        os.makedirs(build_path, exist_ok=True)
        print(f"Build dir: {build_path}")

        # Neutralize sv-iv-analyze's dead langchain import so the analysis step runs
        # without a langchain dependency (see helper). Safe: the import is unused.
        strip_langchain_imports(os.path.join(ve_dir, "scripts", "sv-iv-analyze"))

        # Samples per problem (= pass@k width). MUST match --with-samples below, or
        # make will try to GENERATE the missing samples via sv-generate (langchain).
        samples = max(len(c) for c in completions_by_task.values())

        # STEP 1: place each completion at build/<prob>/<prob>_sampleNN.sv.
        # The Makefile numbers samples 1-based, zero-padded to 2 digits
        # (seq --format "%02g" 1 N), e.g. Prob001_zero_sample01.sv.
        for task, comps in completions_by_task.items():
            task_dir = os.path.join(build_path, task)
            os.makedirs(task_dir, exist_ok=True)
            for i, code in enumerate(comps, start=1):
                base = os.path.join(task_dir, f"{task}_sample{str(i).zfill(2)}")
                with open(base + ".sv", "w", encoding="utf-8") as f:
                    f.write(code)
                # sv-iv-analyze opens a per-sample -sv-generate.log (for token/cost
                # stats from its own generator). We bypassed generation, so write an
                # empty stub — without it analyze raises FileNotFoundError, which
                # `| tee summary.txt` silently masks as a "clean" build with no output.
                open(base + "-sv-generate.log", "w").close()

        # Restrict the harness to EXACTLY the problems we placed, so a mismatch with
        # the repo's full problems.txt can't trigger generation of a missing one.
        problems_file = os.path.join(build_path, "placed_problems.txt")
        with open(problems_file, "w", encoding="utf-8") as f:
            f.write("\n".join(completions_by_task.keys()) + "\n")

        # STEP 2: configure the build dir (generates Makefile / problems.mk /
        # samples.mk), then run only the compile+sim and analysis targets.
        ok = run_make_command(
            [configure,
             "--with-task=spec-to-rtl",
             "--with-model=manual-rtl-coder",      # never actually called (pre-placed)
             f"--with-problems={problems_file}", #manual problems file to iterate over
             f"--with-samples={samples}", #number of samples, k<=samples
             "--with-examples=0", #unused since using own model
             "--with-temperature=0", #unused since using own model
             "--with-top-p=0.01"], #unused since using own model
            "configure",
            build_path
        )
        if not ok:
            return None

        # sv-iv-test-clean only removes stale binaries/logs, NOT our placed .sv files.
        # (Do NOT run sv-generate-clean — that would delete the samples we wrote.)
        # SHELL=/bin/bash: added for Ubuntu compatibility
        for cmd, label in [
            (["make", "SHELL=/bin/bash", "sv-iv-test-clean"], "make sv-iv-test-clean"),
            (["make", "SHELL=/bin/bash", "-j4", "sv-iv-test"], "make sv-iv-test"),
            (["make", "SHELL=/bin/bash", "sv-iv-analyze"], "make sv-iv-analyze"),
        ]:
            if not run_make_command(cmd, label, build_path):
                return None

        # STEP 3: parse build/summary.csv. Row format (no header):
        #   problem, npass, nsamples, pass_fraction, <per-sample status codes...>
        # pass@1 over the suite = mean of the per-problem pass_fraction (col 3).
        summary = os.path.join(build_path, "summary.csv")
        total = 0.0
        n_problems = 0
        with open(summary, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                total += float(row[3]) #pass %
                n_problems += 1
        pass_at_1 = total / n_problems if n_problems else 0.0
        return {"pass@1": pass_at_1, "n_problems": n_problems}


    elif benchmark == "rtllm":
        # RTLLM has no usable official harness (auto_run.py needs commercial VCS
        # + hardcoded paths), so we score directly: compile each completion with
        # the design's own testbench.v using iverilog, run it, and read stdout.
        # Each RTLLM testbench is self-checking and prints "...Passed" on success.
        design_dirs = _rtllm_design_dirs()
        n_problems = 0
        syntax_passes = 0
        func_passes = 0
        per_problem = {}

        for task_id, comps in completions_by_task.items():
            design_dir = design_dirs.get(task_id)
            if design_dir is None:
                print(f"  ! unknown RTLLM task_id, skipping: {task_id}")
                continue
            testbench = design_dir / "testbench.v"
            if not testbench.exists():
                print(f"  ! no testbench.v for {task_id}, skipping")
                continue

            n_problems += 1
            syntax_ok = False
            func_ok = False
            # pass@k: the design counts as passed if ANY of its k samples passes.
            for completion in comps:
                # top_module=None: let iverilog auto-detect the tb top, since RTLLM
                # testbench module names vary (testbench / main / test_alu / ...).
                status, stdout, _ = compile_and_simulate(
                    [("dut.v", completion), testbench],
                    top_module=None, extra_flags=["-g2012"], timeout=30,
                )
                if status == "fail_compile":
                    continue                      # syntax failure for this sample
                syntax_ok = True                  # compiled (whether it ran or hung)
                if status == "ran" and "pass" in stdout.lower():
                    func_ok = True
                    break
            syntax_passes += int(syntax_ok)
            func_passes += int(func_ok)
            per_problem[task_id] = {"syntax": syntax_ok, "func": func_ok}

        n = n_problems or 1
        # RTLLM reports BOTH syntax and functional pass; functional is the headline.
        return {
            "pass@1": func_passes / n,
            "func_pass@1": func_passes / n,
            "syntax_pass@1": syntax_passes / n,
            "n_problems": n_problems,
            "per_problem": per_problem,
        }

    return None

def load_completions_cache(path, meta):
    """
    Return cached completions_by_task if `path` exists AND was built with the same
    settings (meta), else None. Refusing on a meta mismatch prevents scoring, say,
    base-model completions as if they came from the tuned model.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if blob.get("meta") != meta:
        print(f"!! completions cache {path} was built with different settings; "
              f"ignoring it and regenerating.\n   cached: {blob.get('meta')}\n   now:    {meta}")
        return None
    completions = blob["completions_by_task"]
    print(f"==> Loaded {len(completions)} cached completions from {path} (skipping generation)")
    return completions


def save_completions_cache(path, meta, completions_by_task):
    """Persist completions + the settings they were generated with."""
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "completions_by_task": completions_by_task}, f)
    print(f"==> Saved {len(completions_by_task)} completions to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL, help="base model name")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    parser.add_argument("--benchmark", default="verilogeval", choices=["verilogeval", "rtllm"])
    parser.add_argument("--k", type=int, default=1, help="pass@k (use 1 for pass@1)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--completions-cache", default=None,
                        help="JSON file to cache model completions. If it exists (and was "
                             "built with the same model/adapter/benchmark/k), load it and "
                             "skip generation; otherwise generate and save it here.")
    args = parser.parse_args()

    # Completions depend only on these; reuse a cache only if they all match.
    cache_meta = {
        "model": args.model, "adapter": args.adapter,
        "benchmark": args.benchmark, "k": args.k,
    }
    completions_by_task = load_completions_cache(args.completions_cache, cache_meta)

    if completions_by_task is None:
        # Cache miss (or no cache): generate with the model. This is the slow,
        # GPU-bound step — the cache lets later scorer runs skip it entirely.
        problems = load_benchmark_problems(args.benchmark)
        prompts = [p["prompt"] for p in problems]

        # Greedy for pass@1; sample with temperature>0 for pass@k>1.
        temperature = 0.0 if args.k == 1 else 0.8
        completions = generate_vllm_batch(
            args.model, prompts, adapter_path=args.adapter, n=args.k, temperature=temperature
        )
        #re-index to task_id
        completions_by_task = {
            p["task_id"]: comps for p, comps in zip(problems, completions)
        }
        save_completions_cache(args.completions_cache, cache_meta, completions_by_task)

    scores = score_with_harness(args.benchmark, completions_by_task)

    scores["model"] = args.model
    scores["adapter"] = args.adapter
    scores["benchmark"] = args.benchmark
    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
