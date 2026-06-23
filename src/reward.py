"""
The reward function: given a chunk of generated Verilog and a testbench, does it
compile, and does it pass simulation? Returns a numeric reward.

This is the heart of "verification as reward" — and it's the part of the project
where your hardware background is the differentiator. Phase 2b (rejection
sampling) and Phase 3 (GRPO) both call this.

It shells out to `iverilog` (compile) and `vvp` (run), which setup_gpu.sh
installed. No AI here at all — just standard EDA tooling, scripted.
"""

import os
import subprocess
import tempfile


def evaluate_verilog(design_code, testbench_code, reward_compile=0.2, reward_pass=1.0, timeout=20):
    """
    Compile `design_code` together with `testbench_code` and run it.

    Returns (reward, status) where status is one of:
        "pass"           -> simulation ran and the testbench reported success
        "fail_sim"       -> compiled and ran, but testbench reported a mismatch
        "fail_compile"   -> did not compile (syntax / port errors)
        "timeout"        -> simulation hung (often an unintended infinite loop)

    Reward shaping: partial credit for compiling, full credit for passing.
    Tune the weights in configs/grpo.yaml.
    """
    with tempfile.TemporaryDirectory() as d:
        design_path = os.path.join(d, "design.v")
        tb_path = os.path.join(d, "tb.v")
        out_path = os.path.join(d, "sim.out")

        with open(design_path, "w") as f:
            f.write(design_code)
        with open(tb_path, "w") as f:
            f.write(testbench_code)

        # 1. Compile with iverilog.
        compile_proc = subprocess.run(
            ["iverilog", "-o", out_path, design_path, tb_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if compile_proc.returncode != 0:
            return 0.0, "fail_compile"

        # 2. Run the compiled simulation with vvp.
        try:
            run_proc = subprocess.run(
                ["vvp", out_path],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 0.0, "timeout"

        # 3. Decide pass/fail from the testbench output.
        #    Convention here: the benchmark testbenches print a clear marker.
        #    VerilogEval/RTLLM harnesses have their own check — adapt this to match
        #    whichever you're scoring against. A common pattern is to treat the
        #    absence of "mismatch"/"error" and presence of a "passed" marker as success.
        stdout = run_proc.stdout.lower()
        passed = ("passed" in stdout or "all tests passed" in stdout) and "mismatch" not in stdout

        if passed:
            return reward_pass, "pass"
        return reward_compile, "fail_sim"   # partial credit: it at least compiled & ran


# Convenience wrapper used by the GRPO trainer, which passes batches.
def batch_reward(designs, testbenches, **kwargs):
    rewards, statuses = [], []
    for d, tb in zip(designs, testbenches):
        try:
            r, s = evaluate_verilog(d, tb, **kwargs)
        except Exception:
            r, s = 0.0, "fail_compile"
        rewards.append(r)
        statuses.append(s)
    return rewards, statuses
