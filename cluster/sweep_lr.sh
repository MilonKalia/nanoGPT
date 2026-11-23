#!/bin/bash
# M1: learning-rate sweep, 3 rates x 2 models x 250M bytes (each run is its own GPU job)
# usage (repository root, login node): bash cluster/sweep_lr.sh
for model in baseline match; do
  for lr in 6e-4 1e-3 1.5e-3; do
    sbatch -J m1-$model-$lr cluster/train.sbatch config/enwik8_$model.py \
      --learning_rate=$lr --max_bytes=250000000 --out_dir=out/m1/$model-lr$lr
  done
done
