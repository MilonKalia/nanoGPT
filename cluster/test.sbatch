#!/bin/bash
#SBATCH -J enwik8-test
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

export TRITON_DEFAULT_FP_FUSION=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
git rev-parse HEAD
.venv/bin/python -m pytest -q tests
