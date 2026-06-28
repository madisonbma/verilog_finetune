"""
Shared helpers used across training, generation, and evaluation.

Nothing here is exotic — it's the small glue code you'd otherwise rewrite in
every script: loading a 4-bit model, attaching a LoRA adapter, and pulling the
actual Verilog out of a chatty model response.
"""

import re
import yaml


def load_config(path):
    """Read a YAML config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(base_model, load_in_4bit=True, adapter_path=None):
    """
    Load a base model (optionally in 4-bit via bitsandbytes) and its tokenizer.
    If adapter_path is given, attach that trained LoRA adapter on top.

    This is the function every other script calls to get a usable model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = None
    if load_in_4bit:
        # This is the "Q" in QLoRA: store frozen weights in 4-bit to fit on one GPU.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # "normal float 4", the recommended type
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",                      # put it on the GPU automatically
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        # Attach a previously trained LoRA adapter (from Phase 1 or 3).
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def build_lora_config(cfg):
    """Turn the YAML lora_* fields into a peft LoraConfig object."""
    from peft import LoraConfig
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def extract_verilog(text):
    """
    Models love to wrap code in ```verilog ... ``` fences and add chatter.
    Benchmarks need ONLY the module code. Pull out the first code block, or
    fall back to everything between the first 'module' and last 'endmodule'.

    Getting this right is the single most common reason a benchmark run looks
    like "the model failed everything" when it actually didn't.
    """
    # 1. Prefer a fenced code block.
    fenced = re.search(r"```(?:verilog|systemverilog)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # 2. Fall back to module...endmodule span.
    start = text.find("module")
    end = text.rfind("endmodule")
    if start != -1 and end != -1:
        return text[start:end + len("endmodule")].strip()

    # 3. Give up gracefully — return as-is so the simulator error is informative.
    return text.strip()


def to_chat_prompt(tokenizer, instruction):
    """
    Wrap a bare instruction in the model's expected chat template.
    Using the SAME template for base and tuned models is essential for a fair
    before/after comparison.
    """
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def compile_and_simulate(sources, top_module=None, timeout=20, extra_flags=None):
    """
    Compile Verilog with iverilog and run it with vvp. This is the one
    benchmark-agnostic primitive underneath every scorer and the RL reward:
    the caller supplies the source files and reads the returned stdout to decide
    pass/fail however ITS testbench reports success (RTLLM prints "...Passed",
    VerilogEval prints "Mismatches: 0", etc.).

    sources: list of items compiled together in order, each either
        - a (filename, code) tuple  -> written into a temp build dir (use this
          for in-memory model completions and testbench strings), or
        - a path (str / Path) to an existing file -> e.g. a benchmark's
          testbench.v or _ref.sv already on disk.
    top_module: value for iverilog's -s flag, or None to let iverilog
        auto-detect the top (the module nothing else instantiates). Use None for
        RTLLM, whose testbench top names vary (testbench / main / test_alu / ...).
        VerilogEval would pass top_module="tb".
    timeout: seconds allowed for EACH of the compile and run steps.
    extra_flags: extra iverilog flags, e.g. ["-g2012"] for SystemVerilog tests.

    Returns (status, stdout, stderr):
        "fail_compile" -> iverilog returned nonzero, or compilation timed out
                          (no simv produced); stderr holds the diagnostics.
        "timeout"      -> compiled, but the simulation hung past `timeout`.
        "ran"          -> compiled and simulated; inspect stdout for the verdict.
    """
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        compile_paths = []
        for item in sources:
            if isinstance(item, tuple):
                filename, code = item
                path = os.path.join(d, filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                compile_paths.append(path)
            else:
                compile_paths.append(str(item))

        out_path = os.path.join(d, "sim.out")
        cmd = ["iverilog"]
        if extra_flags:
            cmd += extra_flags
        if top_module:
            cmd += ["-s", top_module]
        cmd += ["-o", out_path] + compile_paths

        # 1. Compile. A compile timeout means no simv, so treat it like a
        #    compile failure (matches RTLLM's "syntax_success = simv produced").
        try:
            compile_proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "fail_compile", "", "iverilog compile timed out"
        if compile_proc.returncode != 0:
            return "fail_compile", "", compile_proc.stderr

        # 2. Simulate. A run timeout is usually an unintended infinite loop.
        try:
            run_proc = subprocess.run(
                ["vvp", out_path], capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "", ""

        return "ran", run_proc.stdout, run_proc.stderr
