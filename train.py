"""
Train a byte-level GPT on enwik8. Adapted from nanoGPT's train.py (single GPU).

$ python train.py config/enwik8_baseline.py
$ python train.py config/enwik8_match.py --learning_rate=1e-3 --out_dir=out/match-lr1e-3

Changes from nanoGPT: raw-byte data (uint8, vocab 256), losses logged in bits per byte,
a training budget in bytes, metrics.jsonl for learning curves, best.pt (best dev bpc)
next to the resumable ckpt.pt, automatic resume, a wall-clock limit for batch jobs,
and the match-length bias options of model.py.
"""

import json
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from metrics import nats_to_bits
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config: enwik8 baseline (see config/enwik8_*.py)
# I/O
out_dir = 'out/enwik8-baseline'
eval_interval = 250
log_interval = 50
eval_iters = 40
eval_only = False
init_from = 'auto'  # 'scratch', 'resume', or 'auto' (resume if out_dir/ckpt.pt exists)
max_runtime_minutes = 0.0  # >0: save and stop cleanly before a batch-job time limit
# data
data_dir = 'data/enwik8'
gradient_accumulation_steps = 1
batch_size = 64  # micro-batch; bytes per step = grad_accum * batch_size * block_size
block_size = 1024
max_bytes = 2_000_000_000  # training budget in bytes; sets max_iters when > 0
# model
n_layer = 12
n_head = 8
n_embd = 512
mlp_hidden = 1408
dropout = 0.1
rope_base = 10000.0
match_bias = 'none'  # none | graded | single | shuffled
match_bias_layers = -1
attn_backend = 'flex'
match_bias_replicas = 1024  # 54x faster bias backward than 1 copy (tools/bench_attention.py)
# adamw optimizer
learning_rate = 1e-3
max_iters = 30000  # overridden by max_bytes
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate schedule: linear warmup, cosine decay to min_lr_ratio * learning_rate
decay_lr = True
warmup_frac = 0.02
min_lr_ratio = 0.1
# system
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

bytes_per_iter = gradient_accumulation_steps * batch_size * block_size
if max_bytes > 0:
    max_iters = max_bytes // bytes_per_iter
warmup_iters = int(warmup_frac * max_iters)
min_lr = min_lr_ratio * learning_rate
print(f"bytes per iteration: {bytes_per_iter:,}; iterations: {max_iters:,}; "
      f"total bytes: {bytes_per_iter * max_iters:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' or dtype == 'float32' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_batch(split):
    # recreate the memmap every batch to avoid a memory leak (as in nanoGPT)
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint8, mode='r')
    ix =torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    batch = [x, y]
    if match_bias == 'shuffled':
        # A1 control: match lengths from independent random text, so the bias carries no information
        jx = torch.randint(len(data) - block_size, (batch_size,))
        batch.append(torch.stack([torch.from_numpy(data[j:j + block_size].astype(np.int64)) for j in jx]))
    if device_type == 'cuda':
        return [t.pin_memory().to(device, non_blocking=True) for t in batch]
    return [t.to(device) for t in batch]


def model_inputs(batch):
    x, y = batch[0], batch[1]
    return x, y, (batch[2] if len(batch) > 2 else None)


with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    vocab_size = pickle.load(f)['vocab_size']

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, vocab_size=vocab_size,
                  mlp_hidden=mlp_hidden, dropout=dropout, rope_base=rope_base, match_bias=match_bias,
                  match_bias_layers=match_bias_layers, attn_backend=attn_backend,
                  match_bias_replicas=match_bias_replicas)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if init_from == 'auto':
    init_from = 'resume' if os.path.exists(ckpt_path) else 'scratch'

iter_num, best_val_bpc, checkpoint = 0, float('inf'), None
if init_from == 'scratch':
    print("initializing a new model from scratch")
    model = GPT(GPTConfig(**model_args))
elif init_from == 'resume':
    print(f"resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()})
    iter_num, best_val_bpc = checkpoint['iter_num'], checkpoint['best_val_bpc']
else:
    raise ValueError(f"unknown init_from={init_from!r}")
print(f"parameters: {model.get_num_params(non_embedding=False):,} total, "
      f"{model.get_num_params():,} without the byte embedding")
model.to(device)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

raw_model = model
if compile:
    print("compiling the model...")
    model = torch.compile(model)


@torch.no_grad()
def estimate_bpc():
    """Mean bits per byte over eval_iters random batches of each split (monitoring only;
    the exact dev/test numbers come from evaluate.py)."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, match_x = model_inputs(get_batch(split))
            with ctx:
                _, loss = model(x, y, match_idx=match_x)
            losses[k] = loss.item()
        out[split] = nats_to_bits(losses.mean().item())
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (learning_rate - min_lr)


def save(path, with_optimizer):
    state = dict(model=raw_model.state_dict(), model_args=model_args, iter_num=iter_num,
                 best_val_bpc=best_val_bpc, config=config)
    if with_optimizer:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, path)


def log_metrics(**row):
    with open(os.path.join(out_dir, 'metrics.jsonl'), 'a') as f:
        f.write(json.dumps(row) + '\n')


batch = get_batch('train')
t_start = t0 = time.time()
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    last_iter = iter_num >= max_iters
    if iter_num % eval_interval == 0 or last_iter:
        bpc = estimate_bpc()
        print(f"step {iter_num}: train bpc {bpc['train']:.4f}, val bpc {bpc['val']:.4f}")
        log_metrics(iter=iter_num, bytes=iter_num * bytes_per_iter, train_bpc=bpc['train'], val_bpc=bpc['val'],
                    lr=lr, elapsed_s=time.time() - t_start)
        if iter_num > 0:
            if bpc['val'] < best_val_bpc:
                best_val_bpc = bpc['val']
                save(os.path.join(out_dir, 'best.pt'), with_optimizer=False)
            save(ckpt_path, with_optimizer=True)
    if last_iter or eval_only:
        break
    if max_runtime_minutes > 0 and (time.time() - t_start) / 60 > max_runtime_minutes:
        save(ckpt_path, with_optimizer=True)
        print(f"time limit reached at step {iter_num}; run the same command again to resume")
        break

    for micro_step in range(gradient_accumulation_steps):
        x, y, match_x = model_inputs(batch)
        with ctx:
            _, loss = model(x, y, match_idx=match_x)
            loss = loss / gradient_accumulation_steps
        batch = get_batch('train')  # prefetch while the GPU works
        loss.backward()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt, t0 = t1 - t0, t1
    if iter_num % log_interval == 0:
        train_bpc = nats_to_bits(loss.item() * gradient_accumulation_steps)
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        mem = f", peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device_type == 'cuda' else ""
        print(f"iter {iter_num}: train bpc {train_bpc:.4f}, lr {lr:.2e}, {dt * 1000:.0f} ms, "
              f"{bytes_per_iter / dt / 1e3:.0f}k bytes/s, mfu {mfu * 100:.1f}%{mem}")
    iter_num += 1
