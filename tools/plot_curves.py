"""
Learning curves from metrics.jsonl: dev bits per byte against bytes seen, and the paired gap
(match − baseline) per seed. Monitoring estimates from random dev batches; both runs of a pair
use the same seed, so they see the same batches.

$ python tools/plot_curves.py --pair s1337=out/m2/baseline-s1337,out/m2/match-s1337 \
      --pair s2=out/m3/baseline-s2,out/m3/match-s2 --out out/m4
"""

import argparse
import json
import os

import numpy as np

COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')


def load(run_dir):
    """(iterations, bytes in billions, dev bpc), one row per iteration: a resumed run logs the
    iteration it resumed from twice, and the later entry is kept."""
    by_iter = {}
    with open(os.path.join(run_dir, 'metrics.jsonl')) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row['iter'] > 0:
                    by_iter[row['iter']] = row
    its = sorted(by_iter)
    return (np.array(its), np.array([by_iter[i]['bytes'] for i in its]) / 1e9,
            np.array([by_iter[i]['val_bpc'] for i in its]))


def smooth(y, k=5):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def spread_labels(ax, items, min_gap_frac=0.06):
    """Direct labels at line ends, nudged apart vertically so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for x, y, text in items:
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(text, (x, y), xytext=(x, y_text), textcoords='data', color='#0b0b0b', fontsize=8,
                    va='center', ha='left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', action='append', required=True, help='label=baseline_dir,match_dir')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = []
    for spec in args.pair:
        label, dirs = spec.split('=', 1)
        base_dir, match_dir = dirs.split(',')
        (ib, xb, yb), (im, _, ym) = load(base_dir), load(match_dir)
        common = np.intersect1d(ib, im)
        keep_b, keep_m = np.isin(ib, common), np.isin(im, common)
        pairs.append((label, xb[keep_b], yb[keep_b], ym[keep_m]))

    label, x, yb, ym = pairs[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for y, color in ((yb, COLORS[0]), (ym, COLORS[1])):
        ax.plot(x, y, color=color, linewidth=2)
    ax.set_ylim(min(yb.min(), ym.min()) - 0.02, min(yb.min(), ym.min()) + 0.25)
    pad = x[-1] * 0.01
    spread_labels(ax, [(x[-1] + pad, smooth(yb)[-1], 'baseline'), (x[-1] + pad, smooth(ym)[-1], 'match-length bias')])
    ax.set_xlabel('training bytes (billions)')
    ax.set_ylabel('dev bpc (monitor)')
    ax.set_title(f"Dev bits per byte during training (seed {label})", fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#e5e4e0', linewidth=0.8)
    ax.set_xlim(0, x[-1] * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_curves.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0, color='#52514e', linewidth=1)
    ends = []
    for i, (label, x, yb, ym) in enumerate(pairs):
        gap = ym - yb
        color = COLORS[2 + i % 2] if len(pairs) > 1 else COLORS[1]
        ax.plot(x, gap, color=color, linewidth=0.8, alpha=0.35)
        ax.plot(x, smooth(gap), color=color, linewidth=2)
        ends.append((x[-1] * 1.01, smooth(gap)[-1], f"seed {label}"))
    ax.set_xlabel('training bytes (billions)')
    ax.set_ylabel('dev bpc: match − baseline')
    ax.set_title("Gap during training (thin: each evaluation; thick: 5-point moving average)",
                 fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#e5e4e0', linewidth=0.8)
    ax.set_xlim(0, max(p[1][-1] for p in pairs) * 1.15)
    spread_labels(ax, ends)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_gap.png'), dpi=200)
    plt.close(fig)
    for label, x, yb, ym in pairs:
        print(f"seed {label}: mean gap first 10% {np.mean((ym - yb)[:len(x) // 10]):+.4f}, "
              f"last 25% {np.mean((ym - yb)[-len(x) // 4:]):+.4f}")


if __name__ == '__main__':
    main()
