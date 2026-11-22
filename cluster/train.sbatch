#!/bin/bash
#SBATCH -J enwik8-train
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out

# usage: sbatch cluster/train.sbatch config/enwik8_match.py --learning_rate=1e-3 --out_dir=out/match
# stops cleanly after 5.5 h; submitting the same command again resumes from out_dir/ckpt.pt
export TRITON_DEFAULT_FP_FUSION=1
nvidia-smi --query-gpu=name --format=csv,noheader
git rev-parse HEAD
.venv/bin/python train.py "$@" --max_runtime_minutes=330.0
