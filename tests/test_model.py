import pytest
import torch

from model import GPT, GPTConfig, apply_rotary, rotary_tables


def tiny_config(**overrides):
    cfg = dict(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=32, mlp_hidden=64,
               dropout=0.0, attn_backend='reference')
    cfg.update(overrides)
    return GPTConfig(**cfg)


def test_default_size_matches_the_plan():
    baseline = GPT(GPTConfig())
    match = GPT(GPTConfig(match_bias='graded'))
    n = baseline.get_num_params()
    assert 38e6 < n < 39e6
    assert match.get_num_params() - n == 12 * 8 * 11


@pytest.mark.parametrize("mode", ["none", "graded", "single"])
def test_causality(mode):
    torch.manual_seed(0)
    model = GPT(tiny_config(match_bias=mode)).eval()
    for block in model.transformer.h:  # non-zero bias so the match path matters
        if block.attn.match_table is not None:
            torch.nn.init.normal_(block.attn.match_table, std=2.0)
    idx = torch.randint(0, 4, (2, 32))
    changed = idx.clone()
    changed[:, 20:] = (changed[:, 20:] + 1) % 4
    targets = torch.zeros_like(idx)
    with torch.no_grad():
        a, _ = model(idx, targets)
        b, _ = model(changed, targets)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-6)
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_zero_initialised_bias_model_equals_baseline():
    torch.manual_seed(0)
    baseline = GPT(tiny_config()).eval()
    match = GPT(tiny_config(match_bias='graded')).eval()
    missing, unexpected = match.load_state_dict(baseline.state_dict(), strict=False)
    assert all('match_table' in k for k in missing) and not unexpected
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        a, la = baseline(idx, idx)
        b, lb = match(idx, idx)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(la, lb, atol=1e-6)


def test_bias_in_first_layers_only():
    model = GPT(tiny_config(n_layer=4, match_bias='graded', match_bias_layers=2))
    has_table = [block.attn.match_table is not None for block in model.transformer.h]
    assert has_table == [True, True, False, False]


def test_shuffled_control_needs_and_uses_separate_match_source():
    torch.manual_seed(0)
    model = GPT(tiny_config(match_bias='shuffled')).eval()
    for block in model.transformer.h:
        torch.nn.init.normal_(block.attn.match_table, std=2.0)
    idx = torch.randint(0, 4, (2, 32))
    with pytest.raises(ValueError):
        model(idx, idx)
    with torch.no_grad():
        _, own = model(idx, idx, match_idx=idx)
        _, other = model(idx, idx, match_idx=torch.randint(0, 4, (2, 32)))
    assert not torch.allclose(own, other)


def test_per_byte_losses_average_to_the_mean_loss():
    torch.manual_seed(0)
    model = GPT(tiny_config(match_bias='graded')).eval()
    idx = torch.randint(0, 256, (3, 32))
    with torch.no_grad():
        _, mean_loss = model(idx, idx)
        _, per_byte = model(idx, idx, reduction='none')
    assert per_byte.shape == (3, 32)
    assert torch.allclose(per_byte.mean(), mean_loss, atol=1e-6)


def test_rotary_scores_depend_only_on_relative_position():
    cos, sin = rotary_tables(64, 16, 10000.0)
    torch.manual_seed(0)
    q, k = torch.randn(16), torch.randn(16)
    def score(i, j):
        qi = apply_rotary(q.expand(1, 1, 64, 16), cos, sin)[0, 0, i]
        kj = apply_rotary(k.expand(1, 1, 64, 16), cos, sin)[0, 0, j]
        return (qi * kj).sum()
    assert torch.allclose(score(10, 3), score(40, 33), atol=1e-4)


def test_match_tables_are_not_weight_decayed():
    model = GPT(tiny_config(match_bias='graded'))
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
    tables = {id(b.attn.match_table) for b in model.transformer.h}
    decayed = {id(p) for p in optimizer.param_groups[0]['params']}
    assert optimizer.param_groups[0]['weight_decay'] == 0.1 and not tables & decayed


@pytest.mark.parametrize("mode", ["none", "graded"])
def test_tiny_model_overfits_one_batch(mode):
    torch.manual_seed(0)
    model = GPT(tiny_config(match_bias=mode))
    optimizer = model.configure_optimizers(0.0, 3e-3, (0.9, 0.95), 'cpu')
    idx = torch.randint(0, 256, (4, 33))
    for _ in range(150):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3


def test_match_bias_learns_to_copy_repeated_random_text():
    """On random text repeated twice, the bias should make copying easy: the second half becomes
    predictable, and the learned table should prefer longer matches."""
    torch.manual_seed(0)
    model = GPT(tiny_config(n_layer=1, match_bias='graded', block_size=64))
    optimizer = model.configure_optimizers(0.0, 1e-2, (0.9, 0.95), 'cpu')
    for _ in range(200):
        half = torch.randint(0, 256, (16, 32))
        seq = torch.cat([half, half, half[:, :1]], dim=1)
        _, loss = model(seq[:, :-1], seq[:, 1:], reduction='none')
        optimizer.zero_grad()
        loss[:, 40:].mean().backward()
        optimizer.step()
    assert loss[:, 40:].mean().item() < 1.0  # vs ln(256) = 5.5 nats without copying
    table = model.transformer.h[0].attn.match_table.detach()
    assert (table[:, 7:].mean(1) > table[:, :2].mean(1)).all()
