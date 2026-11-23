#!/bin/bash
#SBATCH -J enwik8-attnbench
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

# 1) component timings for match-biased attention, 2) 300 training steps of the match model
# with table copies and with the reference backend (speed only; results are not used)
export TRITON_DEFAULT_FP_FUSION=1
nvidia-smi --query-gpu=name --format=csv,noheader
git rev-parse HEAD
.venv/bin/python tools/bench_attention.py --batch-size 64
for setting in "--match_bias_replicas=64" "--match_bias_replicas=1024" "--attn_backend=reference --batch_size=32 --gradient_accumulation_steps=2"; do
  echo "=== match $setting ==="
  .venv/bin/python train.py config/enwik8_match.py --max_bytes=0 --max_iters=300 \
    --eval_interval=1000 --eval_iters=10 --log_interval=50 --batch_size=64 --gradient_accumulation_steps=1 \
    --init_from=scratch --out_dir=out/bench-match-variant $setting || echo "FAILED: $setting"
done
