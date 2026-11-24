"""
Compare the gain breakdowns of several runs (summary.json files written by analyze_bytes.py) in
one grouped bar chart per breakdown: by longest earlier match and by class of the predicted byte.

$ python tools/plot_breakdowns.py --run "dev, seed 1337=out/m4/val-s1337/summary.json" \
      --run "test, seed 1337=out/m4/test-s1337/summary.json" --run "test, seed 2=out/m4/test-s2/summary.json" \
      --out out/m4
"""

import argparse
import json
import os

import numpy as np

COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
TITLES = dict(by_match="Gain by longest earlier match available (previous 512 bytes)",
              by_class="Gain by class of the predicted byte")


def load_runs(specs):
    runs = []
    for spec in specs:
        label, path = spec.split('=', 1)
        with open(path) as f:
            runs.append((label, json.load(f)))
    return runs


def grouped_values(runs, key):
    """(group names, array runs x groups of bpc differences); groups in the first run's order."""
    groups = [r['group'] for r in runs[0][1][key]]
    values = np.full((len(runs), len(groups)), np.nan)
    for i, (_, summary) in enumerate(runs):
        by_group = {r['group']: r['difference'] for r in summary[key]}
        values[i] = [by_group.get(g, np.nan) for g in groups]
    return groups, values


def plot_breakdown(runs, key, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups, values = grouped_values(runs, key)
    n_runs = len(runs)
    width = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(groups))
    for i, (label, _) in enumerate(runs):
        # white edges leave a thin gap between neighbouring bars
        ax.bar(x + (i - (n_runs - 1) / 2) * width, values[i], width, color=COLORS[i % len(COLORS)],
               edgecolor='white', linewidth=1, label=label)
    ax.axhline(0, color='#52514e', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('/', '/\n') for g in groups], fontsize=8)
    ax.set_ylabel('bpc difference (match − baseline)', fontsize=8)
    ax.set_title(TITLES[key], fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='label=path/to/summary.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.run)
    for key in ('by_match', 'by_class'):
        plot_breakdown(runs, key, os.path.join(args.out, f'fig_breakdown_{key[3:]}.png'))
    for label, summary in runs:
        o = summary['overall']
        print(f"{label}: {o['difference']:+.4f} bpc, 95% interval [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]")


if __name__ == '__main__':
    main()
