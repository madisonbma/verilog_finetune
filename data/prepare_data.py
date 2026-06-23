"""
Download a public Verilog instruction dataset and reshape it into the simple
{"instruction", "output"} JSONL that train_sft.py expects.

Default source: the RTLCoder dataset on the Hugging Face Hub — an auto-generated
set of (Verilog design problem, answer) pairs built specifically for fine-tuning.
See data/README.md for the exact dataset name and alternatives.

Run:  python data/prepare_data.py --out data/sft_dataset.jsonl
"""

import argparse
import json
from datasets import load_dataset


# TODO: confirm the current dataset id on huggingface.co before running.
# RTLCoder publishes its training data; search the Hub for "RTLCoder" / "Verilog
# instruction" and pick one. The mapping below assumes columns named
# "Instruction" and "Response" — adjust to whatever the dataset actually uses.

#Using RTLCoder for training data since formatted appropriately already.
DEFAULT_DATASET = "Nellyw888/RTL-Coder2"  
INSTRUCTION_COL = "Instruction"
OUTPUT_COL = "Response"
"""
FORMAT OF RTL-CODER2:
[
  {
    "content": "You are tasked with designing a Verilog module 
    that implements a combinational logic circuit. The module 
    should have one output, Y, and five inputs, A1, A2, B1, C1, 
    and D1. The circuit should implement the following logic:\n\n1. 
    Y should be the output of a buffer gate.\n2. The input to the 
    buffer gate should be the output of a NAND gate.\n3. The NAND 
    gate should have four inputs: C1, B1, D1, and the output of an 
    OR gate.\n4. The OR gate should have two inputs: A1 and A2.\n\n
    Design a Verilog module that implements this circuit.",
    "role": "user"
  },
  {
    "content": "module combinational_logic (\n    Y ,\n    A1,\n    
    A2,\n    B1,\n    C1,\n    D1\n);\n\n    // Module ports\n    
    output Y ;\n    input  A1;\n    input  A2;\n    input  B1;\n    
    input  C1;\n    input  D1;\n\n    // Local signals\n    
    wire or0_out    ;\n    wire nand0_out_Y;\n\n    //   Name   
    Output       Other arguments\n    or   or0   (or0_out    
    , A2, A1             );\n    nand nand0 (nand0_out_Y, C1, 
    B1, D1, or0_out);\n    buf  buf0  (Y          , nand0_out_Y        
    );\n\nendmodule",
    "role": "assistant"
  }
]
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default="data/sft_dataset.jsonl")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="cap dataset size for a fast first run")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split="train")

    n = 0
    with open(args.out, "w") as f:
        for row in ds:
            if len(row) != 2:
                print(f"SKIPPING THIS ROW. Unknown why len>2. {row}")
                continue
            else:
                for dict in row: #len2 guaranteed now
                    if dict.get("role").upper() == "USER":
                        instruction = dict.get('content')
                    elif dict.get('role').upper() == "ASSISTANT":
                        output = dict.get('content')
                    else:
                        print(f"Unknown entry: {dict}")

            if not instruction or not output:
                continue
            f.write(json.dumps({"instruction": instruction, "output": output}) + "\n")
            n += 1
            if args.max_examples and n >= args.max_examples:
                break

    print(f"Wrote {n} examples to {args.out}")
    print("Tip: for a first end-to-end smoke test, use --max_examples 200")


if __name__ == "__main__":
    main()
