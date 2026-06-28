"""
The reward function: given a chunk of generated Verilog and a testbench, does it
compile, and does it pass simulation? Returns a numeric reward.

This is the heart of "verification as reward" — and it's the part of the project
where your hardware background is the differentiator. Phase 2b (rejection
sampling) and Phase 3 (GRPO) both call this.

It shells out to `iverilog` (compile) and `vvp` (run), which setup_gpu.sh
installed. No AI here at all — just standard EDA tooling, scripted.
"""

from utils import compile_and_simulate


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
    # Both files are in-memory strings here, so hand them to the shared iverilog
    # primitive as (filename, code) tuples. -g2012 so SystemVerilog testbenches
    # compile; auto-detect the top module.
    status, stdout, _ = compile_and_simulate(
        [("design.v", design_code), ("tb.v", testbench_code)],
        extra_flags=["-g2012"], timeout=timeout,
    )
    if status == "fail_compile":
        return 0.0, "fail_compile"
    if status == "timeout":
        return 0.0, "timeout"

    # status == "ran": decide pass/fail from the testbench output. The benchmark
    # testbenches print a clear marker; treat a "passed" marker with no reported
    # mismatch as success.
    out = stdout.lower()
    passed = ("passed" in out or "all tests passed" in out) and "mismatch" not in out

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
