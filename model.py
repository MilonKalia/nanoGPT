"""
GPT language model for byte-level modelling. Fork of nanoGPT's model.py.

Changes from nanoGPT:
1) rotary position embeddings instead of learned absolute positions (Su et al., 2021)
2) RMSNorm instead of LayerNorm (Zhang & Sennrich, 2019), no bias terms anywhere
3) SwiGLU MLP (Shazeer, 2020)
4) optional match-length attention bias (match_bias.py): each head adds a learned
   scalar to its attention logits, indexed by how long the query's recent bytes
   match the bytes before each key
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import match_bias as mb


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 256
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    mlp_hidden: int = 1408
    dropout: float = 0.1
    rope_base: float = 10000.0
    match_bias: str = "none"      # none | graded | single | shuffled
    match_bias_layers: int = -1   # bias in the first k layers; -1 = all layers
    match_max_len: int = mb.MAX_MATCH
    attn_backend: str = "flex"    # flex (GPU) | reference (materialised bias, any device)
    match_bias_replicas: int = 1  # flex only: copies of the table that share its gradient work


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def rotary_tables(seq_len, head_dim, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary(x, cos, sin):
    """x: (B, H, T, D); cos, sin: (T, D/2). Rotates pairs (x_i, x_{i+D/2})."""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


_BLOCK_MASKS = {}


@torch.compiler.disable
def causal_block_mask(seq_len, device):
    key = (seq_len, str(device))
    if key not in _BLOCK_MASKS:
        _BLOCK_MASKS[key] = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, None, None, seq_len, seq_len, device=device)
    return _BLOCK_MASKS[key]


def biased_attention_flex(q, k, v, table, buckets, block_mask, replicas=1):
    if replicas == 1:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + table[h, buckets[b, q_idx, kv_idx].to(torch.int32)]
    else:
        # FlexAttention accumulates the bias gradient with atomic adds; with a tiny table every
        # score adds into the same few addresses. Identical copies indexed by key position spread
        # those adds out; autograd sums the copies' gradients back into the one parameter.
        table_copies = table.unsqueeze(-1).expand(*table.shape, replicas).contiguous()

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + table_copies[h, buckets[b, q_idx, kv_idx].to(torch.int32), kv_idx % replicas]
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)


def biased_attention_reference(q, k, v, table, buckets):
    T = q.size(-2)
    bias = table[:, buckets.int()].transpose(0, 1)  # (B, H, T, T)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    bias = bias.masked_fill(~causal, float('-inf')).to(q.dtype)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, use_match_bias):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.backend = config.attn_backend
        self.replicas = config.match_bias_replicas
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.match_table = None
        if use_match_bias:
            # zero init: at step 0 the layer is exactly the baseline layer
            self.match_table = nn.Parameter(torch.zeros(config.n_head, mb.num_buckets(config.match_bias)))

    def forward(self, x, cos, sin, buckets, block_mask):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if self.match_table is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.backend == "flex":
            y = biased_attention_flex(q, k, v, self.match_table, buckets, block_mask, self.replicas)
        else:
            y = biased_attention_reference(q, k, v, self.match_table, buckets)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 2 * config.mlp_hidden, bias=False)
        self.c_proj = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, up = self.c_fc(x).chunk(2, dim=-1)
        return self.dropout(self.c_proj(F.silu(gate) * up))


class Block(nn.Module):

    def __init__(self, config, use_match_bias):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, use_match_bias)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, buckets, block_mask):
        x = x + self.attn(self.ln_1(x), cos, sin, buckets, block_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.match_bias in mb.BIAS_MODES, config.match_bias
        assert config.attn_backend in ("flex", "reference"), config.attn_backend
        self.config = config
        self.use_match_bias = config.match_bias != "none"
        n_bias_layers = config.n_layer if config.match_bias_layers < 0 else config.match_bias_layers

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, self.use_match_bias and i < n_bias_layers)
                             for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in nanoGPT

        cos, sin = rotary_tables(config.block_size, config.n_embd // config.n_head, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if self.use_match_bias:
            self.register_buffer("bucket_lut", mb.bucket_lookup(config.match_bias, config.match_max_len),
                                 persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """Parameter count; the (tied) byte embedding is excluded when non_embedding=True."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def match_buckets(self, source):
        return self.bucket_lut[mb.match_lengths(source, self.config.match_max_len).int()]

    def forward(self, idx, targets=None, match_idx=None, reduction='mean'):
        """
        idx, targets: (B, T) byte tensors.
        match_idx: optional (B, T) bytes to compute match lengths from instead of idx;
            required for the 'shuffled' control, where it must be unrelated text.
        reduction: 'mean' for a scalar loss in nats, 'none' for (B, T) per-byte losses.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block size {self.config.block_size}"

        buckets, block_mask = None, None
        if self.use_match_bias:
            if self.config.match_bias == "shuffled" and match_idx is None:
                raise ValueError("match_bias='shuffled' needs match_idx from unrelated text")
            buckets = self.match_buckets(idx if match_idx is None else match_idx)
            if self.config.attn_backend == "flex":
                block_mask = causal_block_mask(T, idx.device)

        x = self.transformer.drop(self.transformer.wte(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.transformer.h:
            x = block(x, cos, sin, buckets, block_mask)
        x = self.transformer.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)
        if reduction == 'none':
            loss = loss.view(B, T)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # weight decay only on matrices; norms and the match-bias tables are not decayed
        decay = [p for n, p in params.items() if p.dim() >= 2 and 'match_table' not in n]
        no_decay = [p for n, p in params.items() if p.dim() < 2 or 'match_table' in n]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        print(f"decayed parameters: {sum(p.numel() for p in decay):,}; "
              f"not decayed: {sum(p.numel() for p in no_decay):,}")
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                 **(dict(fused=True) if use_fused else {}))

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_flops=362e12):
        """Model FLOPs utilisation; default peak = NVIDIA L40S dense BF16 (362 TFLOPS)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        return flops_per_token * T * fwdbwd_per_iter / dt / peak_flops

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
