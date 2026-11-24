"""
Where does the gain come from? Compare two models' per-byte losses (saved by
evaluate.py --save-nll) on the same split.

- overall bits per byte of each model and the paired difference, with a 95% block-bootstrap
  interval (blocks of 10k bytes, because neighbouring bytes are correlated; Kuensch 1989)
- the difference stratified by the longest match available in the preceding `window` bytes
  (the information the match-length bias exposes) and by the class of the predicted byte

$ python tools/analyze_bytes.py --split test \
    --baseline out/m2/baseline-s1337/test_nll.npy --match out/m2/match-s1337/test_nll.npy --out out/m4/test
"""

import argparse
import json
import os

import numpy as np

NATS_PER_BIT = np.log(2.0)
MATCH_GROUPS = ((0, 0, "0"), (1, 3, "1-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, 31, "16-31"), (32, 32, "32+"))
CLASS_NAMES = ("lowercase", "uppercase", "digit", "whitespace", "markup/punct.", "non-ASCII")


def run_lengths(eq, cap):
    """Length of the run of True values ending at each position, capped."""
    idx = np.arange(len(eq))
    last_false = np.maximum.accumulate(np.where(eq, -1, idx))
    return np.minimum(idx - last_false, cap)


def longest_available_match(data, window=512, cap=32):
    """
    For each byte i, the longest match length m(i-1, j) over keys j in the preceding `window`
    positions: how many bytes before some earlier position equal the bytes just before i.
    Entry 0 is 0 (no context).
    """
    x = np.asarray(data)
    best = np.zeros(len(x), dtype=np.uint8)
    for d in range(1, window + 1):
        eq = np.zeros(len(x), dtype=bool)
        eq[d:] = x[d:] == x[:-d]
        np.maximum(best, run_lengths(eq, cap).astype(np.uint8), out=best)
    out = np.zeros(len(x), dtype=np.uint8)
    out[1:] = best[:-1]
    return out


def byte_classes(data):
    x = np.asarray(data)
    cls = np.full(len(x), 4, dtype=np.int8)  # markup / punctuation by default
    cls[(x >= ord('a')) & (x <= ord('z'))] = 0
    cls[(x >= ord('A')) & (x <= ord('Z'))] = 1
    cls[(x >= ord('0')) & (x <= ord('9'))] = 2
    cls[np.isin(x, [9, 10, 13, 32])] = 3
    cls[x >= 128] = 5
    return cls


def block_bootstrap_ci(diff_bits, block=10_000, n_boot=2000, seed=0):
    """95% percentile interval of the mean per-byte difference, resampling contiguous blocks."""
    n_blocks = len(diff_bits) // block
    block_means = diff_bits[:n_blocks * block].reshape(n_blocks, block).mean(1)
    rng = np.random.default_rng(seed)
    boots = block_means[rng.integers(0, n_blocks, (n_boot, n_blocks))].mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def stratify(nll_base, nll_match, groups, labels):
    """Rows of (label, share of bytes, bpc baseline, bpc match, difference, share of total gain)."""
    total_gain = (nll_base - nll_match).sum()
    rows = []
    for g, label in enumerate(labels):
        sel = groups == g
        if not sel.any():
            continue
        b, m = nll_base[sel].mean() / NATS_PER_BIT, nll_match[sel].mean() / NATS_PER_BIT
        rows.append(dict(group=label, share_of_bytes=float(sel.mean()), bpc_baseline=float(b), bpc_match=float(m),
                         difference=float(m - b),
                         share_of_gain=float((nll_base[sel] - nll_match[sel]).sum() / total_gain)))
    return rows


def match_group_ids(lengths):
    ids = np.zeros(len(lengths), dtype=np.int8)
    for g, (lo, hi, _) in enumerate(MATCH_GROUPS):
        ids[(lengths >= lo) & (lengths <= hi)] = g
    return ids


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'group':15s} {'bytes':>7s} {'baseline':>9s} {'match':>8s} {'diff':>8s} {'gain share':>11s}")
    for r in rows:
        print(f"{r['group']:15s} {r['share_of_bytes']:7.1%} {r['bpc_baseline']:9.4f} {r['bpc_match']:8.4f} "
              f"{r['difference']:+8.4f} {r['share_of_gain']:11.1%}")


def tick_labels(rows):
    """Group name with its share of bytes underneath; long class names wrap at the slash."""
    return [r['group'].replace('/', '/\n') + f"\n({r['share_of_bytes']:.1%})" for r in rows]


def plot_groups(rows, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    diffs = [r['difference'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = range(len(rows))
    bars = ax.bar(x, diffs, color='#eb6834', width=0.6)
    ax.axhline(0, color='#52514e', linewidth=1)
    for bar, d in zip(bars, diffs):
        y = bar.get_height()
        ax.annotate(f"{d:+.3f}", (bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, -3 if y < 0 else 3), textcoords='offset points',
                    ha='center', va='top' if y < 0 else 'bottom', fontsize=8, color='#0b0b0b')
    ax.set_xticks(list(x))
    ax.set_xticklabels(tick_labels(rows), fontsize=8)
    ax.set_xlabel('group (share of bytes)', fontsize=8)
    ax.set_ylabel('bpc difference (match − baseline)', fontsize=8)
    ax.set_title(title, fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
    ax.set_axisbelow(True)
    lo, hi = min(diffs + [0]), max(diffs + [0])
    pad = (hi - lo) * 0.15 + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--data-dir', default='data/enwik8')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--match', required=True)
    parser.add_argument('--window', type=int, default=512)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = np.fromfile(os.path.join(args.data_dir, f'{args.split}.bin'), dtype=np.uint8)
    nll_b = np.load(args.baseline).astype(np.float64)[1:]
    nll_m = np.load(args.match).astype(np.float64)[1:]
    assert len(nll_b) == len(nll_m) == len(data) - 1

    diff_bits = (nll_m - nll_b) / NATS_PER_BIT
    lo, hi = block_bootstrap_ci(diff_bits)
    overall = dict(split=args.split, n_bytes=len(nll_b), bpc_baseline=float(nll_b.mean() / NATS_PER_BIT),
                   bpc_match=float(nll_m.mean() / NATS_PER_BIT), difference=float(diff_bits.mean()),
                   ci95=[lo, hi], share_of_bytes_better=float((diff_bits < 0).mean()))
    print(json.dumps(overall, indent=2))

    lengths = longest_available_match(data, args.window)[1:]
    by_match = stratify(nll_b, nll_m, match_group_ids(lengths), [g[2] for g in MATCH_GROUPS])
    by_class = stratify(nll_b, nll_m, byte_classes(data)[1:], CLASS_NAMES)
    print_table(f"by longest match in the previous {args.window} bytes", by_match)
    print_table("by class of the predicted byte", by_class)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(dict(overall=overall, by_match=by_match, by_class=by_class), f, indent=2)
    plot_groups(by_match, f"Gain by longest earlier match available ({args.split})",
                os.path.join(args.out, 'fig_gain_by_match.png'))
    plot_groups(by_class, f"Gain by class of the predicted byte ({args.split})",
                os.path.join(args.out, 'fig_gain_by_class.png'))


if __name__ == '__main__':
    main()
