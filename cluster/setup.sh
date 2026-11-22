#!/bin/bash
# Run once on the Engaging login node from the repository root (needs internet).
# uv-managed Python: the system Python lacks Python.h, which Triton (torch.compile) needs.
set -e
uv venv --python 3.12 --python-preference only-managed .venv
uv pip install --python .venv/bin/python torch numpy pytest matplotlib
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
.venv/bin/python data/enwik8/prepare.py
mkdir -p logs out
