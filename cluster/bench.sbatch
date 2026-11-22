#!/bin/bash
#SBATCH -J enwik8-bench
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

# 300 training steps per setting, to measure speed and memory (results are not used)
export TRITON_DEFAULT_FP_FUSION=1
nvidia-smi --query-gpu=name --format=csv,noheader
for model in baseline match; do
  for micro in "64 1" "32 2"; do
    set -- $micro
    echo "=== $model batch_size=$1 grad_accum=$2 ==="
    .venv/bin/python train.py config/enwik8_$model.py --max_bytes=0 --max_iters=300 \
      --eval_interval=1000 --eval_iters=10 --log_interval=50 \
      --batch_size=$1 --gradient_accumulation_steps=$2 --init_from=scratch \
      --out_dir=out/bench-$model-b$1 || echo "FAILED: $model batch_size=$1"
  done
done
