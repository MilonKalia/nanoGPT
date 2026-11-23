"""
Look inside trained models.

bias:      plot the learned match-length bias tables (beta[bucket] - beta[0] for every layer and head;
           only differences matter because softmax ignores a constant added to every key)
attention: attention mass each head puts on earlier positions with a match of at least `min_match`
           bytes ("copy keys"), against the mass a uniform head would give them

$ python tools/inspect_model.py bias --ckpt out/m2/match-s1337/best.pt --out out/m4
$ python tools/inspect_model.py attention --ckpt baseline=out/m2/baseline-s1337/best.pt \
      --ckpt match=out/m2/match-s1337/best.pt --out out/m4
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import match_bias as mb  # noqa: E402
from evaluate import load_model  # noqa: E402
from model import apply_rotary  # noqa: E402

BUCKET_LABELS = ("0", "1", "2", "3", "4", "5-6", "7-8", "9-12", "13-16", "17-24", "25-32")


def bias_tables(model):
    """(layers with a table, heads, buckets) array of beta - beta[bucket 0]."""
    tables = [b.attn.match_table.detach().float().cpu() for b in model.transformer.h if b.attn.match_table is not None]
    t = torch.stack(tables)
    return (t - t[..., :1]).numpy()


@torch.no_grad()
def attention_probs(model, idx):
    """Yield (layer, probs) with probs of shape (B, H, T, T), following the model's own forward pass."""
    T = idx.size(1)
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    buckets = model.match_buckets(idx) if model.use_match_bias else None
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    x = model.transformer.wte(idx)
    for layer, block in enumerate(model.transformer.h):
        attn = block.attn
        h = block.ln_1(x)
        B, _, C = h.shape
        q, k, _ = attn.c_attn(h).split(C, dim=2)
        q, k = (t.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2) for t in (q, k))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(attn.head_dim)
        if attn.match_table is not None:
            scores = scores + attn.match_table[:, buckets.int()].transpose(0, 1).float()
        yield layer, scores.masked_fill(~causal, float('-inf')).softmax(-1)
        x = block(x, cos, sin, buckets, None)


@torch.no_grad()
def copy_attention(model, windows, min_match=4, batch_size=8):
    """Mean attention mass on copy keys, and the uniform-attention baseline, per (layer, head).
    Averaged over queries that have at least one copy key."""
    L, H = model.config.n_layer, model.config.n_head
    mass, chance, n_queries = torch.zeros(L, H), torch.zeros(L, H), 0
    for i in range(0, len(windows), batch_size):
        idx = windows[i:i + batch_size]
        T = idx.size(1)
        copy = mb.match_lengths(idx) >= min_match
        copy &= torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        has_copy = copy.any(-1)  # (B, T)
        n_copy = copy.sum(-1).float()
        positions = torch.arange(1, T + 1, device=idx.device).float()
        n_queries += int(has_copy.sum())
        for layer, probs in attention_probs(model, idx):
            m = (probs * copy[:, None]).sum(-1)  # (B, H, T)
            mass[layer] += (m * has_copy[:, None]).sum((0, 2)).cpu()
            chance[layer] += ((n_copy / positions * has_copy)[:, None].expand_as(m)).sum((0, 2)).cpu()
    return (mass / n_queries).numpy(), (chance / n_queries).numpy(), n_queries


def plot_bias(tables, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_layers, n_heads, n_buckets = tables.shape
    cols = 4
    rows = math.ceil(n_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), sharex=True, sharey=True, squeeze=False)
    x = np.arange(n_buckets)
    for layer, ax in enumerate(axes.flat):
        if layer >= n_layers:
            ax.axis('off')
            continue
        for head in range(n_heads):
            ax.plot(x, tables[layer, head], color='#b8b7b1', linewidth=1)
        ax.plot(x, tables[layer].mean(0), color='#eb6834', linewidth=2, marker='o', markersize=3)
        ax.axhline(0, color='#52514e', linewidth=0.8)
        ax.set_title(f"layer {layer + 1}", fontsize=9, loc='left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e5e4e0', linewidth=0.6)
    labels = BUCKET_LABELS if n_buckets == len(BUCKET_LABELS) else [str(i) for i in range(n_buckets)]
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_xlabel('match length (bytes)', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('bias vs. no match', fontsize=8)
    fig.suptitle("Learned match-length bias per layer (grey: each head; orange: mean over heads)",
                 fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_attention(results, min_match, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results) + 1, 5.2), squeeze=False)
    vmax = max(r['mass'].max() for r in results.values())
    for ax, (label, r) in zip(axes[0], results.items()):
        im = ax.imshow(r['mass'], cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        L, H = r['mass'].shape
        for i in range(L):
            for j in range(H):
                v = r['mass'][i, j]
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6,
                        color='white' if v > 0.6 * vmax else '#0b0b0b')
        ax.set_title(f"{label} (uniform head: {r['chance'].mean():.3f})", fontsize=9, loc='left')
        ax.set_xticks(range(H))
        ax.set_xticklabels([str(h + 1) for h in range(H)], fontsize=7)
        ax.set_yticks(range(L))
        ax.set_yticklabels([str(l + 1) for l in range(L)], fontsize=7)
        ax.set_xlabel('head', fontsize=8)
        ax.set_ylabel('layer', fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label='attention mass on copy keys')
    fig.suptitle(f"Attention on earlier positions with a match of at least {min_match} bytes",
                 fontsize=10, x=0.01, ha='left')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def sample_windows(data_dir, split, n_windows, block_size, seed=0):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint8, mode='r')
    starts = np.random.default_rng(seed).integers(0, len(data) - block_size, n_windows)
    return torch.from_numpy(np.stack([data[s:s + block_size] for s in starts]).astype(np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['bias', 'attention'])
    parser.add_argument('--ckpt', action='append', required=True, help='path, or label=path for attention')
    parser.add_argument('--out', required=True)
    parser.add_argument('--data-dir', default='data/enwik8')
    parser.add_argument('--split', default='val')
    parser.add_argument('--n-windows', type=int, default=64)
    parser.add_argument('--min-match', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.command == 'bias':
        model, _ = load_model(args.ckpt[0].split('=')[-1], 'cpu')
        tables = bias_tables(model)
        np.save(os.path.join(args.out, 'bias_tables.npy'), tables)
        print("mean over heads, per layer (bucket order", ", ".join(BUCKET_LABELS[:tables.shape[-1]]), ")")
        for layer, t in enumerate(tables.mean(1)):
            print(f"layer {layer + 1:2d}: " + " ".join(f"{v:+.2f}" for v in t))
        plot_bias(tables, os.path.join(args.out, 'fig_bias_tables.png'))
        return

    results = {}
    for spec in args.ckpt:
        label, path = spec.split('=', 1) if '=' in spec else (os.path.basename(os.path.dirname(spec)), spec)
        model, _ = load_model(path, args.device)
        for block in model.transformer.h:
            block.attn.backend = 'reference'
        windows = sample_windows(args.data_dir, args.split, args.n_windows, model.config.block_size).to(args.device)
        mass, chance, n_queries = copy_attention(model, windows, args.min_match)
        results[label] = dict(mass=mass, chance=chance)
        top = np.dstack(np.unravel_index(np.argsort(-mass, axis=None)[:5], mass.shape))[0]
        print(f"{label}: {n_queries:,} queries with a copy key; uniform head {chance.mean():.3f}; "
              f"top heads (layer, head, mass): " + ", ".join(f"({l + 1},{h + 1},{mass[l, h]:.2f})" for l, h in top))
    with open(os.path.join(args.out, 'copy_attention.json'), 'w') as f:
        json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in results.items()}, f)
    plot_attention(results, args.min_match, os.path.join(args.out, 'fig_copy_attention.png'))


if __name__ == '__main__':
    main()
