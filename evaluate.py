"""
Exact bits-per-byte evaluation of a checkpoint on the enwik8 dev ("val") or test split.

Sliding window: windows of block_size bytes advance by `stride`; the first window
scores all of its targets and every later window scores only its last `stride`
targets, so every byte after the first is scored exactly once, with at least
block_size - stride bytes of context (Hugging Face, "Perplexity of fixed-length models").
No dynamic evaluation: the model is never updated at test time.

$ python evaluate.py --ckpt out/enwik8-match/best.pt --split test --save-nll out/enwik8-match/test_nll.npy
"""

import argparse
import json
import os
from contextlib import nullcontext

import numpy as np
import torch

from metrics import bits_per_byte
from model import GPT, GPTConfig


def window_plan(n_bytes, block_size, stride):
    """Return (window length, [(start, first scored position in window), ...])."""
    if not 0 < stride <= block_size:
        raise ValueError("need 0 < stride <= block_size")
    length = min(block_size, n_bytes - 1)
    plan = [(0, 0)]
    last_scored = length  # targets are byte indices 1 .. n_bytes-1
    while last_scored < n_bytes - 1:
        end = min(last_scored + stride, n_bytes - 1)
        plan.append((end - length, length - (end - last_scored)))
        last_scored = end
    return length, plan


def sliding_window_nll(data, nll_fn, block_size, stride, batch_size, shuffled_source=False):
    """
    data: 1-D uint8 array. nll_fn(x, y, match_x) -> (B, L) per-byte NLL in nats.
    Returns float64 array of length len(data); entry i is the NLL of byte i (NaN for i = 0).
    shuffled_source: give the model match lengths from an unrelated window (A1 control).
    """
    n = len(data)
    length, plan = window_plan(n, block_size, stride)
    out = np.full(n, np.nan)
    for i in range(0, len(plan), batch_size):
        chunk = plan[i:i + batch_size]
        x = np.stack([data[s:s + length] for s, _ in chunk])
        y = np.stack([data[s + 1:s + 1 + length] for s, _ in chunk])
        match_x = None
        if shuffled_source:
            match_x = np.stack([data[(s + n // 2) % (n - length):][:length] for s, _ in chunk])
        nll = nll_fn(x, y, match_x)
        for (s, first), row in zip(chunk, nll):
            out[s + 1 + first:s + 1 + length] = row[first:]
    return out


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--data-dir', default='data/enwik8')
    parser.add_argument('--stride', type=int, default=None, help='default: block_size // 2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save-nll', default=None, help='optional .npy path for per-byte NLL (float32)')
    args = parser.parse_args()

    model, checkpoint = load_model(args.ckpt, args.device)
    cfg = model.config
    stride = args.stride or cfg.block_size // 2
    forward = torch.compile(model) if args.compile else model
    ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
           if 'cuda' in args.device else nullcontext())

    @torch.no_grad()
    def nll_fn(x, y, match_x):
        to = lambda a: torch.from_numpy(a.astype(np.int64)).to(args.device)
        with ctx:
            _, loss = forward(to(x), to(y), match_idx=None if match_x is None else to(match_x),
                              reduction='none')
        return loss.float().cpu().numpy()

    data = np.memmap(os.path.join(args.data_dir, f'{args.split}.bin'), dtype=np.uint8, mode='r')
    nll = sliding_window_nll(np.asarray(data), nll_fn, cfg.block_size, stride, args.batch_size,
                             shuffled_source=cfg.match_bias == 'shuffled')
    scored = nll[1:]
    assert not np.isnan(scored).any(), "some bytes were not scored"
    result = dict(split=args.split, bpc=bits_per_byte(scored.sum(), len(scored)), n_bytes_scored=len(scored),
                  stride=stride, block_size=cfg.block_size, ckpt=args.ckpt,
                  iter_num=checkpoint.get('iter_num'), match_bias=cfg.match_bias)
    print(json.dumps(result, indent=2))
    out_json = os.path.join(os.path.dirname(args.ckpt), f'eval_{args.split}.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    if args.save_nll:
        np.save(args.save_nll, nll.astype(np.float32))


if __name__ == '__main__':
    main()
