"""
Run a model on prompts and return its Verilog. Used by the evaluator and the
reward function, and handy for eyeballing outputs by hand.

Two backends:
  - "hf"   : plain transformers generation. Simple, slower. Fine for spot checks.
  - "vllm" : fast batched generation. Use this when scoring a whole benchmark.

Run a quick manual check:
    python src/generate.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --prompt "Write a Verilog module 'adder' that adds two 4-bit inputs."
"""

import argparse
from utils import load_model_and_tokenizer, to_chat_prompt, extract_verilog


def generate_hf(model, tokenizer, prompt, max_new_tokens=1024):
    """Single-prompt generation with plain transformers."""
    import torch
    text = to_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy = deterministic, good for pass@1
            temperature=1.0,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_verilog(decoded)


def generate_vllm_batch(model_name, prompts, adapter_path=None, n=1, temperature=0.0):
    """
    Batched generation with vLLM — much faster for scoring a full benchmark.
    Returns a list (one entry per prompt) of lists (n completions each).
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model_name, enable_lora=adapter_path is not None)
    sampling = SamplingParams(n=n, temperature=temperature, max_tokens=1024)

    lora_req = LoRARequest("adapter", 1, adapter_path) if adapter_path else None
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    results = []
    for o in outputs:
        results.append([extract_verilog(c.text) for c in o.outputs])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, adapter_path=args.adapter)
    print(generate_hf(model, tokenizer, args.prompt))


if __name__ == "__main__":
    main()
