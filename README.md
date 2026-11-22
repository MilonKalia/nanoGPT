# nanoGPT, byte-level: match-length attention bias on enwik8

A fork of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) for character-level
(byte-level) language modelling on enwik8, with one architectural change under test.

## The idea

Transformers copy from their context with induction heads. Trained induction heads weight an
earlier position by how long the context before it matches the current context, which works like a
soft n-gram back-off model inside attention. Character-level models need long matches (5–30 bytes),
and enwik8 (Wikipedia XML) is full of long repeats.

This fork builds that signal into attention. For a query at position `t` and a key at `j <= t`, let
`m(t, j)` be the number of bytes just before `j` that equal the bytes ending at `t` (capped at 32).
`m` is computed exactly from the raw bytes, mapped to 11 log-spaced buckets, and each head adds a
learned scalar per bucket to its attention logits:

    score(t, j) = q_t · k_j / sqrt(d) + beta[layer, head, bucket(m(t, j))]

The bias tables start at zero, so at initialisation the model is exactly the baseline. They add
12 × 8 × 11 = 1,056 parameters. The bias runs inside a fused FlexAttention kernel on GPU.

## What changed from nanoGPT

- `model.py`: rotary positions, RMSNorm, SwiGLU, no bias terms (the baseline); optional match-length
  bias (`match_bias = none | graded | single | shuffled`, `match_bias_layers`).
- `match_bias.py`: exact match lengths and buckets.
- `train.py`: raw bytes (uint8, vocabulary 256), losses in bits per byte, a training budget in bytes,
  `metrics.jsonl` learning curves, `best.pt` + resumable `ckpt.pt`, automatic resume, wall-clock limit.
- `evaluate.py`: exact dev/test bits per byte with a sliding window (every byte scored once).
- `metrics.py`: the single function used for every bits-per-byte number.
- `data/enwik8/prepare.py`: standard 90M / 5M / 5M byte split.
- `tests/`: match lengths against brute force, causality, zero-bias equivalence, FlexAttention vs
  reference attention (values and gradients), bits per byte, data split, sliding-window evaluation,
  end-to-end training and resume.
- Removed: GPT-2 loading, OpenWebText/Shakespeare data, sampling and benchmarking scripts that no
  longer apply.

## Usage

```bash
pip install -r requirements.txt
python data/enwik8/prepare.py
python -m pytest -q tests
python train.py config/enwik8_baseline.py
python train.py config/enwik8_match.py
python evaluate.py --ckpt out/enwik8-match/best.pt --split test
```

Variants: `--match_bias=single` (one "match of at least 4 bytes" indicator),
`--match_bias=shuffled` (control: match lengths from unrelated text), `--match_bias_layers=2`
(bias in the first two layers only). Slurm scripts for a GPU cluster are in `cluster/`.

## Results

To be added.

## References

- Karpathy, A. (2022). nanoGPT. https://github.com/karpathy/nanoGPT
- Olsson, C. et al. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.
- D'Angelo, F., Yüksel, O. K., Narashiman, S. S. & Flammarion, N. (2026). Induction Heads Interpolate N-Grams. ICML.
- Akyürek, E., Wang, B., Kim, Y. & Andreas, J. (2024). In-Context Language Learning: Architectures and Algorithms. ICML.
- Al-Rfou, R. et al. (2019). Character-Level Language Modeling with Deeper Self-Attention. AAAI.
- Dong, J. et al. (2024). Flex Attention: A Programming Model for Generating Optimized Attention Kernels.

nanoGPT is MIT-licensed (see LICENSE). Code in this fork was written with AI assistance (Claude).
