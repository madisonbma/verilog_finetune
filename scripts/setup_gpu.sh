#!/usr/bin/env bash
# One-shot setup on a FRESH rented GPU box (RunPod / Lambda / etc.).
# Run this once after cloning the repo onto the machine.
set -e

echo "==> Installing system Verilog simulators (these are EDA tools, not pip packages)"
# iverilog is the simplest; verilator is faster. Install both.
sudo apt-get update
sudo apt-get install -y iverilog verilator git build-essential

echo "==> Creating a Python virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==> Done. Next steps:"
echo "    source .venv/bin/activate"
echo "    huggingface-cli login     # paste a free token from huggingface.co"
echo "    accelerate config         # accept defaults for single-GPU"
echo "    iverilog -V               # sanity-check the simulator is installed"
