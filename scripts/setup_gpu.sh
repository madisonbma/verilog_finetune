#!/usr/bin/env bash
# One-shot setup on a FRESH rented GPU box (Lambda / RunPod / etc.).
# Run from the repo root:  bash scripts/setup_gpu.sh
set -e

REPO_ROOT="$(pwd)"

echo "==> Installing system packages (verilator + build deps for iverilog)"
sudo apt-get update
# verilator from apt is fine. iverilog we build from source (see next step),
# so autoconf/gperf/flex/bison are REQUIRED — build-essential alone isn't enough.
# software-properties-common gives us add-apt-repository for the deadsnakes PPA.
sudo apt-get install -y \
    verilator git build-essential \
    autoconf gperf flex bison \
    software-properties-common

echo "==> Building Icarus Verilog v12 from source"
# Ubuntu 22.04's apt ships iverilog v11, but VerilogEval needs EXACTLY v12
# (v13 dev is also unsupported). Remove any apt iverilog, then build v12.
sudo apt-get remove -y iverilog || true   # || true: fine if it wasn't installed
rm -rf /tmp/iverilog                       # clean slate so re-runs don't fail
git clone https://github.com/steveicarus/iverilog.git /tmp/iverilog
cd /tmp/iverilog
git checkout v12-branch
sh ./autoconf.sh && ./configure && make -j"$(nproc)"
sudo make install                          # installs to /usr/local/bin/iverilog
cd "$REPO_ROOT"                            # back to repo root for the pip step

echo "==> Installing Python 3.11 (22.04's default python3 is 3.10)"
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
# -venv: needed to create venvs.  -dev: provides Python.h, which Triton/vLLM
# need to JIT-compile CUDA helper kernels at engine startup (without it vLLM
# dies with "fatal error: Python.h: No such file or directory").
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

echo "==> Creating a Python 3.11 virtual environment"
python3.11 -m venv .venv
source .venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt
# Note: VerilogEval's sv-iv-analyze has a dead `import langchain` that crashes on
# modern langchain; run_eval.py patches it out at runtime, so no langchain needed.

echo ""
echo "==> Sanity checks"
iverilog -V          # expect: Icarus Verilog version 12.x  (NOT 11, NOT 13)
python --version     # expect: Python 3.11.x

echo ""
echo "==> Done. Next steps (with the venv active):"
echo "    source .venv/bin/activate     # re-activate in any new shell"
echo "    huggingface-cli login         # paste a free token from huggingface.co"
echo "    accelerate config             # accept defaults for single-GPU"
