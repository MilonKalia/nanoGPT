"""
Summarise training runs from their metrics.jsonl files.

$ python tools/summarize_runs.py out/m1
"""

import json
import os
import sys


def summarize(run_dir):
    with open(os.path.join(run_dir, 'metrics.jsonl')) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    best = min(rows, key=lambda r: r['val_bpc'])
    last = rows[-1]
    return dict(run=os.path.basename(run_dir), last_iter=last['iter'], bytes_m=last['bytes'] / 1e6,
                last_val_bpc=last['val_bpc'], best_val_bpc=best['val_bpc'], best_iter=best['iter'],
                train_bpc=last['train_bpc'], hours=last['elapsed_s'] / 3600)


def main(root):
    runs = sorted(d for d in (os.path.join(root, n) for n in os.listdir(root))
                  if os.path.exists(os.path.join(d, 'metrics.jsonl')))
    header = f"{'run':28s} {'iter':>6s} {'bytes(M)':>9s} {'train':>7s} {'val':>7s} {'best val':>9s} {'@iter':>6s} {'hours':>6s}"
    print(header)
    print('-' * len(header))
    for s in map(summarize, runs):
        print(f"{s['run']:28s} {s['last_iter']:6d} {s['bytes_m']:9.0f} {s['train_bpc']:7.4f} {s['last_val_bpc']:7.4f} "
              f"{s['best_val_bpc']:9.4f} {s['best_iter']:6d} {s['hours']:6.2f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'out')
