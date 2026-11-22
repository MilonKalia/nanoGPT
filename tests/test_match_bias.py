import math

import pytest
import torch
import torch.nn.functional as F

import match_bias as mb
from model import biased_attention_flex, biased_attention_reference, causal_block_mask


def brute_force_match_lengths(seq, max_match):
    T = len(seq)
    m = [[0] * T for _ in range(T)]
    for t in range(T):
        for j in range(1, t + 1):
            k = 0
            while k < max_match and j - 1 - k >= 0 and seq[t - k] == seq[j - 1 - k]:
                k += 1
            m[t][j] = k
    return m


@pytest.mark.parametrize("alphabet,T,max_match", [(2, 40, 32), (3, 50, 5), (256, 30, 32)])
def test_match_lengths_equal_brute_force(alphabet, T, max_match):
    torch.manual_seed(0)
    idx = torch.randint(0, alphabet, (3, T))
    m = mb.match_lengths(idx, max_match)
    for b in range(3):
        expected = torch.tensor(brute_force_match_lengths(idx[b].tolist(), max_match), dtype=torch.uint8)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        assert torch.equal(m[b][causal], expected[causal])


def test_match_lengths_on_repeated_text_and_cap():
    text = b"the cat sat. the cat sat. the cat sat."
    idx = torch.tensor(list(text))[None]
    m = mb.match_lengths(idx, 32)
    t = len(text) - 1  # query at the final '.'
    j = t - 13 + 1     # key right after the previous occurrence of the same 13 bytes
    assert m[0, t, j] == 25  # all 25 bytes before j repeat the 25 bytes ending at t (period 13)
    assert m[0, t].max() == 25
    assert mb.match_lengths(idx, 8)[0, t, j] == 8
    assert m[0, :, 0].eq(0).all()  # the first key has no preceding byte


def test_bucket_lookup():
    lut = mb.bucket_lookup("graded").tolist()
    assert lut[:6] == [0, 1, 2, 3, 4, 5]
    assert lut[6] == 5 and lut[7] == 6 and lut[8] == 6 and lut[9] == 7
    assert lut[16] == 8 and lut[17] == 9 and lut[24] == 9 and lut[25] == 10 and lut[32] == 10
    assert max(lut) + 1 == mb.num_buckets("graded") == 11
    single = mb.bucket_lookup("single").tolist()
    assert single == [0, 0, 0, 0] + [1] * 29 and mb.num_buckets("single") == 2


def naive_biased_attention(q, k, v, table, buckets):
    T = q.size(-2)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    scores = scores + table[torch.arange(table.size(0))[None, :, None, None], buckets[:, None].long()]
    scores = scores.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float('-inf'))
    return scores.softmax(-1) @ v


def random_attention_inputs(B=2, H=3, T=24, D=8, NB=11, requires_grad=False):
    torch.manual_seed(1)
    q, k, v = (torch.randn(B, H, T, D, requires_grad=requires_grad) for _ in range(3))
    table = torch.randn(H, NB, requires_grad=requires_grad)
    buckets = torch.randint(0, NB, (B, T, T), dtype=torch.uint8)
    return q, k, v, table, buckets


def test_reference_attention_matches_naive_values_and_gradients():
    q, k, v, table, buckets = random_attention_inputs(requires_grad=True)
    out_ref = biased_attention_reference(q, k, v, table, buckets)
    grads_ref = torch.autograd.grad(out_ref.square().sum(), (q, k, v, table))
    out_naive = naive_biased_attention(q, k, v, table, buckets)
    grads_naive = torch.autograd.grad(out_naive.square().sum(), (q, k, v, table))
    assert torch.allclose(out_ref, out_naive, atol=1e-5)
    for a, b in zip(grads_ref, grads_naive):
        assert torch.allclose(a, b, atol=1e-4)


def test_flex_attention_forward_matches_reference_on_cpu():
    q, k, v, table, buckets = random_attention_inputs()
    out_flex = biased_attention_flex(q, k, v, table, buckets, causal_block_mask(q.size(-2), 'cpu'))
    assert torch.allclose(out_flex, biased_attention_reference(q, k, v, table, buckets), atol=1e-5)


def test_zero_table_equals_plain_causal_attention():
    q, k, v, table, buckets = random_attention_inputs()
    zero = torch.zeros_like(table)
    plain = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(biased_attention_reference(q, k, v, zero, buckets), plain, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (FlexAttention backward)")
def test_compiled_flex_matches_reference_values_and_gradients_on_gpu():
    q, k, v, table, buckets = random_attention_inputs(B=2, H=4, T=256, D=64)
    tensors = [t.cuda().requires_grad_() for t in (q, k, v, table)]
    buckets = buckets.cuda()
    flex = torch.compile(biased_attention_flex)
    out_flex = flex(*tensors, buckets, causal_block_mask(256, 'cuda'))
    grads_flex = torch.autograd.grad(out_flex.square().sum(), tensors)
    out_ref = biased_attention_reference(*tensors, buckets)
    grads_ref = torch.autograd.grad(out_ref.square().sum(), tensors)
    assert torch.allclose(out_flex, out_ref, atol=2e-3)
    for a, b in zip(grads_flex, grads_ref):
        assert torch.allclose(a, b, atol=2e-2, rtol=1e-2)
