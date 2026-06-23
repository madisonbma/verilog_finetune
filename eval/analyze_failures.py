"""
Phase 2a: the failure taxonomy — your differentiator.

After an eval run, this buckets every FAILED generation into categories a
hardware engineer recognizes: syntax, port/interface mismatch, bit-width error,
FSM/state-machine bug, combinational-vs-sequential confusion, etc. An ML-only
candidate can report "43% pass@1." You can report *why the other 57% failed* and
which categories a targeted dataset would fix. That write-up is the part of this
project that proves domain depth.

Run (after an eval run saved per-problem results):
    python eval/analyze_failures.py --results results/eval_run --out results/taxonomy.md
"""

import argparse
import json
import os
import re
from collections import Counter


# Heuristic classifiers. These are deliberately simple and transparent — refine
# them with your own eye as you read real failures. The point is a defensible,
# hardware-aware breakdown, not a perfect classifier.
def classify_failure(design_code, compile_stderr, sim_stdout, status):
    if status == "fail_compile":
        s = compile_stderr.lower()
        if "syntax error" in s or "unexpected" in s:
            return "syntax_error"
        if "port" in s or "connection" in s:
            return "port_interface_mismatch"
        if "width" in s or "truncated" in s:
            return "bit_width_error"
        return "other_compile_error"

    if status == "timeout":
        return "infinite_loop_or_no_clock"   # often a missing/incorrect clock or latch

    if status == "fail_sim":
        # Compiled and ran but produced wrong values. Look at the design for clues.
        code = design_code.lower()
        if re.search(r"\bcase\b|\bstate\b|\bnext_state\b", code):
            return "fsm_logic_error"
        if "posedge" not in code and ("always" in code and ("<=" in code)):
            return "missing_clock_edge"
        if "blocking_vs_nonblocking" in code:  # placeholder; refine by inspection
            return "blocking_nonblocking_misuse"
        return "functional_logic_error"

    return "uncategorized"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="dir of per-problem result json files from an eval run")
    parser.add_argument("--out", default="results/taxonomy.md")
    args = parser.parse_args()

    counts = Counter()
    examples = {}   # keep one example per category for the write-up

    for fname in os.listdir(args.results):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.results, fname)) as f:
            rec = json.load(f)
        if rec.get("status") == "pass":
            continue
        cat = classify_failure(
            rec.get("design_code", ""),
            rec.get("compile_stderr", ""),
            rec.get("sim_stdout", ""),
            rec.get("status", ""),
        )
        counts[cat] += 1
        examples.setdefault(cat, rec.get("task_id", fname))

    # Write a clean markdown report you can drop straight into your repo / writeup.
    total = sum(counts.values())
    with open(args.out, "w") as f:
        f.write("# Failure taxonomy\n\n")
        f.write(f"Total failed problems analyzed: **{total}**\n\n")
        f.write("| Category | Count | % of failures | Example task |\n")
        f.write("|---|---|---|---|\n")
        for cat, n in counts.most_common():
            pct = 100 * n / total if total else 0
            f.write(f"| {cat} | {n} | {pct:.1f}% | {examples[cat]} |\n")
        f.write("\n## Notes\n\n")
        f.write("- TODO: read 3-5 examples per category by hand and add real commentary.\n")
        f.write("- TODO: which category would a targeted dataset most cheaply fix?\n")

    print(f"Wrote taxonomy to {args.out}")
    print(counts.most_common())


if __name__ == "__main__":
    main()
