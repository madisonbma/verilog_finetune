# Training data

`train_sft.py` expects a JSONL file where each line is:

```json
{"instruction": "Write a Verilog module 'mux2' that ...", "output": "module mux2 ... endmodule"}
```

`prepare_data.py` builds that file from a public dataset on the Hugging Face Hub.

## Where to get a dataset

1. Go to huggingface.co/datasets and search **"RTLCoder"** or **"Verilog
   instruction"**. RTLCoder published the auto-generated instruction/answer
   dataset described in their paper — it's purpose-built for exactly this.
2. Copy the dataset id (looks like `owner/dataset-name`).
3. Open `prepare_data.py` and set `DEFAULT_DATASET` to that id, and check the
   `INSTRUCTION_COL` / `OUTPUT_COL` names match the dataset's actual columns
   (open the dataset's "Viewer" tab on the Hub to see them).
4. Run:
   ```bash
   python data/prepare_data.py --out data/sft_dataset.jsonl --max_examples 200
   ```
   Start with `--max_examples 200` to smoke-test the whole pipeline fast, then
   re-run without the cap for the real training run.

## Phase 2b note (rejection sampling)

For the "train only on functionally-validated examples" variant, you'll generate
candidate answers with the base model, run each through `src/reward.py`, and keep
only the passing ones as your fine-tuning set. That filtered JSONL goes in the
same place and trains with the same `train_sft.py`.

## Phase 3 note (GRPO)

The RL dataset is different: it needs `{"prompt": ..., "testbench": ...}` per row
(the problem and a testbench to score against), not pre-written answers.
