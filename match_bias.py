"""
Match-length attention bias.

For a query at position t and a key at position j <= t, the match length m(t, j)
is the number of bytes just before j that equal the bytes ending at t:

    m(t, j) = max k >= 0 such that x[t - i] == x[j - 1 - i] for i = 0 .. k-1

so the key's own byte x[j] is "what came next" after an earlier occurrence of the
current context. m is computed exactly from the raw bytes, capped at MAX_MATCH,
and mapped to a small number of buckets. Each attention head then adds a learned
scalar per bucket to its attention logits (see model.CausalSelfAttention).
"""

import torch

MAX_MATCH = 32

# log-spaced buckets, as in T5's relative-position buckets:
# 0, 1, 2, 3, 4, 5-6, 7-8, 9-12, 13-16, 17-24, 25-32
GRADED_UPPER_EDGES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SINGLE_ORDER_THRESHOLD = 4

BIAS_MODES = ("none", "graded", "single", "shuffled")


def num_buckets(mode):
    if mode in ("graded", "shuffled"):
        return len(GRADED_UPPER_EDGES)
    if mode == "single":
        return 2
    raise ValueError(f"no buckets for match_bias={mode!r}")


def bucket_lookup(mode, max_match=MAX_MATCH):
    """Tensor of length max_match + 1 mapping a match length to its bucket id."""
    lengths = torch.arange(max_match + 1)
    if mode == "single":
        return (lengths >= SINGLE_ORDER_THRESHOLD).to(torch.uint8)
    edges = torch.tensor(GRADED_UPPER_EDGES)
    return torch.bucketize(lengths, edges, right=False).to(torch.uint8)


@torch.no_grad()
def match_lengths(idx, max_match=MAX_MATCH):
    """
    idx: (B, T) integer tensor of bytes.
    Returns m: (B, T, T) uint8 with m[b, t, j] = match length for query t and key j.
    Entries with j > t are not meaningful (they are masked by causal attention).
    """
    B, T = idx.shape
    # eq[b, t, j] = (x[t] == x[j-1]); column j = 0 has no previous byte
    eq = torch.zeros(B, T, T, dtype=torch.bool, device=idx.device)
    eq[:, :, 1:] = idx[:, :, None] == idx[:, None, :-1]
    m = torch.zeros(B, T, T, dtype=torch.uint8, device=idx.device)
    run = eq
    for k in range(1, max_match + 1):
        m += run
        if k == max_match:
            break
        # extend each run one step back along its diagonal: needs eq[t-k, j-k]
        shifted = torch.zeros_like(eq)
        shifted[:, k:, k:] = eq[:, :-k, :-k]
        run = run & shifted
    return m


@torch.no_grad()
def match_buckets(idx, mode, max_match=MAX_MATCH):
    """(B, T, T) uint8 bucket ids for the given bias mode."""
    lut = bucket_lookup(mode, max_match).to(idx.device)
    return lut[match_lengths(idx, max_match).int()]
