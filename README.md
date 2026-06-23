# verilog-finetune

Fine-tune an open-source code model to write better Verilog, and measure the
improvement on standard benchmarks. The goal of this project is **not** to beat
the state of the art — it's to demonstrate, end to end, that you can run a real
supervised fine-tuning (SFT) pipeline and evaluate it rigorously, with a failure
analysis that only someone with hardware/verification experience could write.

The headline deliverable is one sentence you'll be able to say in an interview:
> "I took `Qwen2.5-Coder-7B`, fine-tuned it with QLoRA on a Verilog dataset, and
> moved pass@1 on VerilogEval from **X%** to **Y%**. Here's the failure taxonomy
> and what I'd do next."

---

## 1. The mental model (read this first)

A "fine-tune" is just this loop:

```
   base model  ──►  show it thousands of (instruction, good answer) pairs  ──►  adjusted model
                         (this is "training")
```

Then you measure whether the adjusted model is actually better:

```
   model  ──►  ask it to solve held-out problems  ──►  compile + simulate each answer  ──►  % that pass
                  ("inference")                            ("evaluation")              ("pass@1")
```

Everything in this repo is one of those two boxes. The "stack" below is just the
specific tools that implement each box. You don't need to know them yet — the
glossary explains each one, and you only touch them in the order the phases lay out.

---

## 2. Glossary — every component, in plain English

**Hugging Face (HF)** — Think "GitHub for AI models and datasets," plus the
Python libraries to use them. You'll download the base model and the training
data from the HF Hub (`huggingface.co`). Free account; you'll generate an access
token and run `huggingface-cli login` once.

**`transformers`** — HF's core library. Gives you `AutoModelForCausalLM` and
`AutoTokenizer` — i.e., "load this model by name and start using it." A
**tokenizer** turns text into the integer IDs the model actually consumes, and
back again.

**Base model (`Qwen2.5-Coder-7B-Instruct`)** — The starting point we adjust.
"7B" = 7 billion parameters. "Coder" = already pretrained heavily on code.
"Instruct" = already tuned to follow instructions in a chat format (so we don't
start from a raw text-completion model). `StarCoder2-7B` is a fine alternative.

**Fine-tuning vs. prompting** — Prompting (what you do today) changes the *input*
to a fixed model. Fine-tuning changes the *model's weights* so it's permanently
better at a task. This project is about the second thing, because that's the
competency you want to prove.

**LoRA (Low-Rank Adaptation)** — Fully fine-tuning a 7B model means updating all
7 billion numbers — needs many expensive GPUs. LoRA instead **freezes** the
original model and inserts tiny new "adapter" matrices (a few million numbers)
that it trains instead. You get ~95% of the benefit for ~1% of the cost. The
output is a small "adapter" file you can attach to the base model.

**Quantization & QLoRA** — A 7B model in full precision is ~28GB and won't fit on
an affordable GPU. **Quantization** stores each weight in 4 bits instead of 16,
shrinking it ~4x with little quality loss. **QLoRA** = load the frozen base model
in 4-bit, then train LoRA adapters on top. This is what lets the whole project
run on a *single* rented GPU. The 4-bit magic comes from a library called
**`bitsandbytes`**.

**`peft`** — HF's "Parameter-Efficient Fine-Tuning" library. This is the package
that actually implements LoRA. You hand it a model and a `LoraConfig` and it
returns a model with the adapters wired in.

**`trl` (Transformer Reinforcement Learning)** — HF's training-loop library. It
gives you ready-made trainers so you don't hand-write the training loop:
- **`SFTTrainer`** — supervised fine-tuning. "Here are (instruction, answer)
  pairs, make the model imitate the answers." This is Phase 1.
- **`GRPOTrainer`** — reinforcement learning. "Don't show the model answers;
  instead let it generate, score each generation with a reward function, and
  push it toward higher-reward behavior." This is the stretch Phase 3, where the
  reward is "did the Verilog pass the testbench."

**`accelerate`** — HF's plumbing for "run this training on whatever hardware is
present" (1 GPU, many GPUs, etc.). You mostly just run `accelerate config` once
and forget it.

**Dataset / instruction format** — The training data is a list of examples, each
roughly `{"instruction": "Write a Verilog module that...", "output": "module ...
endmodule"}`. `data/prepare_data.py` loads a public Verilog dataset (RTLCoder's)
and reshapes it into the chat format the Instruct model expects.

**VerilogEval & RTLLM** — The two standard benchmarks. Each is a set of Verilog
problems *with hidden testbenches*. You generate a solution for each, the
benchmark compiles and simulates it against the testbench, and reports the score.
VerilogEval-v2 ships a Makefile-based harness; RTLLM is a second, complementary
set (50 designs of varying complexity). Using both is more convincing than one.

**`pass@1` / `pass@k`** — The metric. `pass@1` = "ask once, what fraction of
problems pass?" `pass@k` = "allow k attempts per problem, what fraction pass at
least once?" Higher is better. Your before/after `pass@1` numbers are the whole
point of the project.

**Simulators: `iverilog` & `verilator`** — The programs that actually compile and
run Verilog to check if it works. These are not AI — they're standard
open-source EDA tools (you've used commercial equivalents). The benchmarks call
them under the hood. `iverilog` is the simplest to install; `verilator` is faster.

**`cocotb`** — A Python framework for writing Verilog testbenches in Python
instead of in Verilog. Useful in Phase 2/3 if you want to write your own
verification checks for the reward function. Optional but plays to your strength.

**`vLLM`** — A fast inference server. Evaluation means generating hundreds of
completions; plain `transformers` generation is slow. vLLM makes that 10–20x
faster. Optional in Phase 0/1, very nice to have by eval time.

**GPU rental (RunPod / Lambda / Modal)** — You don't own a suitable GPU, so you
rent one by the hour. A single **A100 (40 or 80GB)** is plenty for a 7B QLoRA
run, at roughly **$1–2/hour**. The entire project is **tens of dollars**, not
thousands. RunPod and Lambda give you a Jupyter/SSH box; Modal is
script-driven. Start with RunPod if you want the simplest "rent a box" feel.

---

## 3. Repo layout

```
verilog-finetune/
├── README.md                 ← you are here
├── requirements.txt          ← Python dependencies
├── Makefile                  ← shortcut commands (make sft, make eval, ...)
├── .gitignore
├── configs/
│   ├── sft_qlora.yaml        ← all SFT hyperparameters in one place
│   └── grpo.yaml             ← stretch-phase RL hyperparameters
├── data/
│   ├── README.md             ← which dataset to download and from where
│   └── prepare_data.py       ← load + reformat the dataset for training
├── src/
│   ├── train_sft.py          ← Phase 1: QLoRA supervised fine-tune
│   ├── generate.py           ← run the model on prompts (inference)
│   ├── reward.py             ← compile+simulate a completion → pass/fail (the reward)
│   ├── train_grpo.py         ← Phase 3 stretch: RL with testbench reward
│   └── utils.py              ← shared helpers (load model, extract code, etc.)
├── eval/
│   ├── README.md             ← how to set up VerilogEval + RTLLM
│   ├── run_eval.py           ← run a model through a benchmark → pass@k
│   └── analyze_failures.py   ← Phase 2: categorize what the model got wrong
├── scripts/
│   ├── setup_gpu.sh          ← one-shot install on a fresh rented GPU box
│   └── benchmark_base.sh     ← Phase 0: score the UNtuned base model
└── results/                  ← scores, logs, and your trained adapters land here
```

---

## 4. The plan, mapped to commands

Each phase ends with something you could show someone. Don't start a phase until
the previous one runs clean.

### Phase 0 — Plumbing + baseline (a few days)
Get a GPU, install everything, and **score the base model before touching it.**
That baseline number is what makes every later result meaningful.
```bash
bash scripts/setup_gpu.sh          # install deps + simulators on the rented box
bash scripts/benchmark_base.sh     # → results/baseline_verilogeval.json
```

### Phase 1 — The fine-tune (week 1)
Reformat the dataset, run QLoRA SFT, re-score, compare to baseline.
```bash
python data/prepare_data.py --out data/sft_dataset.jsonl
python src/train_sft.py --config configs/sft_qlora.yaml   # → results/adapter/
python eval/run_eval.py --adapter results/adapter --benchmark verilogeval
```
**Deliverable:** "base = X% pass@1 → my fine-tune = Y% pass@1."

### Phase 2 — Your differentiator (week 2) — pick ONE
- **(a) Failure taxonomy.** Run `analyze_failures.py` to bucket every failed
  generation into syntax / functional / FSM-state / bit-width-mismatch / port
  errors, and write up the patterns. This is the analysis an ML-only candidate
  cannot do credibly — it's your Qualcomm DFT eye on the model's mistakes.
  ```bash
  python eval/analyze_failures.py --results results/eval_run --out results/taxonomy.md
  ```
- **(b) Rejection-sampling SFT.** Generate several candidates per problem, keep
  only the ones that *pass simulation*, and fine-tune on that filtered set. A
  poor-man's "verification as reward." Re-run Phase 1 with the filtered data.

### Phase 3 — Stretch: real RL (only if Phase 2 was smooth)
Replace imitation with reinforcement: the model generates Verilog, `reward.py`
compiles+simulates it, and GRPO pushes the model toward passing code.
```bash
python src/train_grpo.py --config configs/grpo.yaml       # → results/grpo_adapter/
```
Even a small win here signals post-training skill most applicants won't have.

---

## 5. First-time setup

```bash
# 0. On a fresh rented GPU box (RunPod/Lambda), clone this repo, then:
bash scripts/setup_gpu.sh

# 1. Log in to Hugging Face once (free token from huggingface.co/settings/tokens)
huggingface-cli login

# 2. Configure accelerate once (accept defaults for single-GPU)
accelerate config
```

## 6. What to expect (cost & time)

| Phase | Wall-clock | GPU cost |
|-------|-----------|----------|
| 0 — setup + baseline | a few hours of fiddling | a few dollars |
| 1 — QLoRA SFT (7B) | ~1–4 hrs training | ~$5–15 |
| 2 — analysis | mostly your time, little GPU | a few dollars |
| 3 — GRPO (stretch) | several hours | ~$15–40 |

Total: comfortably under $100 if you shut the box down between sessions.
**Always stop/terminate the rented GPU when you're done — that's the #1 way
people accidentally run up a bill.**

## 7. Common gotchas

- **Out-of-memory (OOM):** lower `per_device_train_batch_size` to 1 and raise
  `gradient_accumulation_steps`; make sure 4-bit loading is actually on.
- **The model rambles past `endmodule`:** extract only the code block before
  scoring (handled in `utils.extract_verilog`). Some models need a custom stop.
- **Benchmark "all fail" on first run:** almost always a simulator-not-installed
  or code-extraction problem, not a model problem. Sanity-check by feeding a
  known-good Verilog answer straight into the scorer.
- **Comparing apples to oranges:** score base and tuned models with the *exact
  same* prompt template and decoding settings, or the comparison is meaningless.

---

## 8. Background reading (so you can defend choices)

- VerilogEval (Liu et al.) and RTLLM (Lu et al.) — the benchmarks you're using.
- RTLCoder — the fine-tuned-open-model baseline + dataset you're building on.
- VeriCoder — shows functionally-validated training data beats unvalidated
  (the rationale for Phase 2b).
- VeriReason — RL with testbench feedback (the rationale for Phase 3).
- ChipNeMo — NVIDIA's domain-adapted chip LLM, for the big-picture framing.
