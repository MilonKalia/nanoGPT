#!/bin/bash
# Run once on the Engaging login node from the repository root (needs internet).
set -e
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy pytest matplotlib
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
.venv/bin/python data/enwik8/prepare.py
mkdir -p logs out
