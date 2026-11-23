"""
Time the pieces of match-biased attention on one GPU (forward + backward, bf16, one layer's shapes)
and check that every variant gives the same output and table gradient as the reference path.

$ python tools/bench_attention.py --batch-size 64
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import match_bias as mb  # noqa: E402
from model import biased_attention_flex, biased_attention_reference, causal_block_mask  # noqa: E402


def cuda_time_ms(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--data', default='data/enwik8/train.bin')
    args = parser.parse_args()
    B, T, H, D = args.batch_size, args.block_size, args.n_head, args.head_dim
    device = 'cuda'
    torch.manual_seed(0)

    data = np.memmap(args.data, dtype=np.uint8, mode='r')
    starts = np.random.default_rng(0).integers(0, len(data) - T, B)
    idx = torch.from_numpy(np.stack([data[s:s + T] for s in starts]).astype(np.int64)).to(device)
    lut = mb.bucket_lookup('graded').to(device)
    buckets = lut[mb.match_lengths(idx).int()]
    counts = torch.bincount(buckets[:, torch.ones(T, T, dtype=torch.bool, device=device).tril()].flatten().long(),
                            minlength=11)
    print("share of (query, key) pairs per bucket:", [f"{c:.4f}" for c in (counts / counts.sum()).tolist()])

    q, k, v = (torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    table = (0.5 * torch.randn(H, 11, device=device)).requires_grad_()
    grad_out = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    block_mask = causal_block_mask(T, device)
    flex = torch.compile(flex_attention)
    biased = torch.compile(biased_attention_flex)

    def step(forward):
        def run():
            out = forward()
            out.backward(grad_out)
            for t in (q, k, v, table):
                t.grad = None
        return run

    match_lengths_compiled = torch.compile(lambda x: lut[mb.match_lengths(x).int()])
    rows = [
        ("match buckets, eager", lambda: lut[mb.match_lengths(idx).int()], 5),
        ("match buckets, compiled", lambda: match_lengths_compiled(idx), 5),
        ("SDPA causal (baseline layer)", step(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)), 10),
        ("flex, causal mask only", step(lambda: flex(q, k, v, block_mask=block_mask)), 10),
        ("flex + bias, frozen table", step(lambda: biased(q, k, v, table.detach(), buckets, block_mask)), 10),
        ("reference (materialised bias + SDPA)", step(lambda: biased_attention_reference(q, k, v, table, buckets)), 5),
    ]
    for r in (1, 16, 64, 256, 1024):
        rows.append((f"flex + bias, trainable, copies={r}",
                     step(lambda r=r: biased(q, k, v, table, buckets, block_mask, r)), 3 if r == 1 else 10))

    print(f"\nB={B} T={T} H={H} D={D}; milliseconds per call (forward + backward for attention rows)")
    for name, fn, iters in rows:
        try:
            print(f"{name:42s} {cuda_time_ms(fn, iters):9.2f}")
        except Exception as e:  # report and keep going
            print(f"{name:42s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")

    out_ref = biased_attention_reference(q, k, v, table, buckets)
    g_ref = torch.autograd.grad(out_ref, table, grad_out)[0]
    for r in (1, 64, 1024):
        out = biased(q, k, v, table, buckets, block_mask, r)
        g = torch.autograd.grad(out, table, grad_out)[0]
        print(f"copies={r:5d}: max |out - ref| = {(out - out_ref).abs().max().item():.2e}, "
              f"table grad relative error = {((g - g_ref).norm() / g_ref.norm()).item():.2e}")


if __name__ == '__main__':
    main()
