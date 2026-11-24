import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import torch

from model import GPT, GPTConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "tools", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = load_tool("analyze_bytes")
im = load_tool("inspect_model")


def brute_longest_match(x, window, cap):
    out = [0] * len(x)
    for i in range(1, len(x)):
        t, best = i - 1, 0
        for j in range(max(1, t - window + 1), t + 1):
            k = 0
            while k < cap and j - 1 - k >= 0 and x[t - k] == x[j - 1 - k]:
                k += 1
            best = max(best, k)
        out[i] = best
    return out


def test_run_lengths():
    eq = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert ab.run_lengths(eq, 32).tolist() == [1, 2, 0, 1, 2, 3, 0, 1]
    assert ab.run_lengths(eq, 2).tolist() == [1, 2, 0, 1, 2, 2, 0, 1]


def test_longest_available_match_equals_brute_force():
    rng = np.random.default_rng(0)
    for alphabet, window, cap in [(2, 16, 32), (3, 8, 5), (4, 40, 32)]:
        x = rng.integers(0, alphabet, 120).astype(np.uint8)
        assert ab.longest_available_match(x, window, cap).tolist() == brute_longest_match(x.tolist(), window, cap)


def test_byte_classes():
    cls = ab.byte_classes(np.frombuffer("aZ5 <\xe9".encode("latin-1"), dtype=np.uint8))
    assert cls.tolist() == [0, 1, 2, 3, 4, 5]


def test_block_bootstrap_interval_covers_the_mean():
    diff = np.random.default_rng(0).normal(-0.01, 0.5, 200_000)
    lo, hi = ab.block_bootstrap_ci(diff, block=1000, n_boot=500)
    assert lo < diff.mean() < hi and hi - lo < 0.02


def test_stratify_shares_sum_to_one():
    rng = np.random.default_rng(0)
    base = rng.random(1000) + 1
    match = base - rng.random(1000) * 0.1
    rows = ab.stratify(base, match, rng.integers(0, 3, 1000), ["a", "b", "c"])
    assert abs(sum(r["share_of_bytes"] for r in rows) - 1) < 1e-9
    assert abs(sum(r["share_of_gain"] for r in rows) - 1) < 1e-9
    assert all(r["difference"] < 0 for r in rows)


def tiny_checkpoint(path, mode):
    torch.manual_seed(0)
    args = dict(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=32, mlp_hidden=64, dropout=0.0,
                match_bias=mode, attn_backend='reference')
    model = GPT(GPTConfig(**args))
    for block in model.transformer.h:
        if block.attn.match_table is not None:
            torch.nn.init.normal_(block.attn.match_table)
    torch.save(dict(model=model.state_dict(), model_args=args, iter_num=0), path)
    return model


def test_copy_attention_is_a_valid_probability_mass(tmp_path):
    model = tiny_checkpoint(tmp_path / "m.pt", "graded").eval()
    half = torch.randint(0, 4, (3, 32))
    windows = torch.cat([half, half], dim=1)
    mass, chance, n = im.copy_attention(model, windows, min_match=4, batch_size=2)
    assert n > 0 and mass.shape == (2, 2)
    assert ((mass >= 0) & (mass <= 1 + 1e-5)).all() and ((chance > 0) & (chance <= 1)).all()


def test_attention_probs_follow_the_model(tmp_path):
    """The probabilities rebuilt by inspect_model must reproduce the model's own attention output."""
    model = tiny_checkpoint(tmp_path / "m.pt", "graded").eval()
    idx = torch.randint(0, 4, (2, 64))
    probs = dict(im.attention_probs(model, idx))
    block = model.transformer.h[0]
    h = block.ln_1(model.transformer.wte(idx))
    _, _, v = block.attn.c_attn(h).split(32, dim=2)
    v = v.view(2, 64, 2, 16).transpose(1, 2)
    y = (probs[0] @ v).transpose(1, 2).reshape(2, 64, 32)
    buckets = model.match_buckets(idx)
    expected = block.attn(h, model.rope_cos[:64], model.rope_sin[:64], buckets, None)
    assert torch.allclose(block.attn.c_proj(y), expected, atol=1e-5)


def test_curves_deduplicate_resumed_iterations(tmp_path):
    pc = load_tool("plot_curves")
    with open(tmp_path / "metrics.jsonl", "w") as f:
        for it, v in [(0, 8.0), (250, 2.0), (500, 1.9), (500, 1.95), (750, 1.8)]:
            f.write(json.dumps(dict(iter=it, bytes=it * 1000, val_bpc=v, train_bpc=v, lr=0, elapsed_s=0)) + "\n")
    its, _, val = pc.load(tmp_path)
    assert its.tolist() == [250, 500, 750] and val.tolist() == [2.0, 1.95, 1.8]


def test_grouped_breakdown_values_follow_the_first_runs_groups():
    pb = load_tool("plot_breakdowns")
    a = dict(by_match=[dict(group="0", difference=-0.1), dict(group="1-3", difference=-0.2)])
    b = dict(by_match=[dict(group="1-3", difference=-0.4)])
    groups, values = pb.grouped_values([("a", a), ("b", b)], "by_match")
    assert groups == ["0", "1-3"]
    assert values[0].tolist() == [-0.1, -0.2] and np.isnan(values[1, 0]) and values[1, 1] == -0.4


def test_tools_run_end_to_end(tmp_path):
    tiny_checkpoint(tmp_path / "match.pt", "graded")
    tiny_checkpoint(tmp_path / "base.pt", "none")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    text = np.frombuffer(b"<page><title>Test</title> hello world, hello match.</page>\n" * 100, dtype=np.uint8)
    text.tofile(data_dir / "val.bin")
    out = tmp_path / "out"
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=REPO, check=True, capture_output=True, text=True)
    run("tools/inspect_model.py", "bias", "--ckpt", str(tmp_path / "match.pt"), "--out", str(out), "--device", "cpu")
    run("tools/inspect_model.py", "attention", "--ckpt", f"baseline={tmp_path / 'base.pt'}",
        "--ckpt", f"match={tmp_path / 'match.pt'}", "--out", str(out), "--data-dir", str(data_dir),
        "--n-windows", "4", "--device", "cpu")
    nll = np.random.default_rng(0).random(len(text)).astype(np.float32)
    np.save(tmp_path / "b.npy", nll)
    np.save(tmp_path / "m.npy", nll * 0.95)
    result = run("tools/analyze_bytes.py", "--split", "val", "--data-dir", str(data_dir), "--baseline",
                 str(tmp_path / "b.npy"), "--match", str(tmp_path / "m.npy"), "--out", str(out), "--window", "64")
    assert "by longest match" in result.stdout
    for name in ("b", "m"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        with open(run_dir / "metrics.jsonl", "w") as f:
            for it in range(0, 2000, 100):
                f.write(json.dumps(dict(iter=it, bytes=it * 65536, val_bpc=2 - it / 2000 - (0.01 if name == "m" else 0),
                                        train_bpc=1.0, lr=1e-3, elapsed_s=it)) + "\n")
    run("tools/plot_curves.py", "--pair", f"x={tmp_path / 'b'},{tmp_path / 'm'}", "--out", str(out))
    summary = str(out / "summary.json")
    run("tools/plot_breakdowns.py", "--run", f"one={summary}", "--run", f"two={summary}", "--out", str(out))
    for fig in ("fig_bias_tables.png", "fig_copy_attention.png", "fig_gain_by_match.png", "fig_gain_by_class.png",
                "fig_curves.png", "fig_gap.png", "fig_breakdown_match.png", "fig_breakdown_class.png"):
        assert (out / fig).exists(), fig
