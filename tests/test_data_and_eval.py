import importlib.util
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from evaluate import sliding_window_nll, window_plan
from metrics import bits_per_byte, nats_to_bits

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("prepare", os.path.join(REPO, "data", "enwik8", "prepare.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_bits_per_byte():
    assert bits_per_byte(1000 * math.log(256), 1000) == pytest.approx(8.0)
    assert nats_to_bits(math.log(2)) == pytest.approx(1.0)
    probs = np.array([0.5, 0.25, 0.25])
    entropy_nats = -(probs * np.log(probs)).sum()
    assert bits_per_byte(entropy_nats * 10, 10) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        bits_per_byte(1.0, 0)


def test_standard_split_sizes():
    train, val, test = prepare.split_bytes(np.zeros(100_000_000, dtype=np.uint8))
    assert (len(train), len(val), len(test)) == (90_000_000, 5_000_000, 5_000_000)


def test_write_splits_round_trip(tmp_path):
    data = np.arange(1000, dtype=np.uint64).astype(np.uint8)
    prepare.write_splits(data, tmp_path, n_val=100, n_test=50)
    parts = [np.fromfile(tmp_path / f"{n}.bin", dtype=np.uint8) for n in ("train", "val", "test")]
    assert [len(p) for p in parts] == [850, 100, 50]
    assert np.array_equal(np.concatenate(parts), data)
    with open(tmp_path / "meta.pkl", "rb") as f:
        assert pickle.load(f) == {"vocab_size": 256}


@pytest.mark.parametrize("n,block,stride", [(5000, 64, 32), (1001, 100, 100), (300, 64, 7), (50, 64, 32)])
def test_every_byte_scored_exactly_once(n, block, stride):
    length, plan = window_plan(n, block, stride)
    counts = np.zeros(n, dtype=int)
    for start, first in plan:
        counts[start + 1 + first:start + 1 + length] += 1
    assert counts[0] == 0 and (counts[1:] == 1).all()
    assert all(first == 0 or length - first <= stride for _, first in plan[1:])


def test_sliding_window_nll_with_a_bigram_model():
    """A model that only looks at the previous byte gives the same per-byte NLL however
    the windows are cut, so the sliding-window result must equal a direct computation."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 5, 2000).astype(np.uint8)
    table = rng.random((256, 256)) + 0.1
    logp = np.log(table / table.sum(1, keepdims=True))

    def nll_fn(x, y, match_x):
        return -logp[x, y]

    nll = sliding_window_nll(data, nll_fn, block_size=64, stride=16, batch_size=7)
    assert np.isnan(nll[0]) and not np.isnan(nll[1:]).any()
    assert np.allclose(nll[1:], -logp[data[:-1], data[1:]])


def test_uniform_model_gives_eight_bits():
    data = np.random.default_rng(0).integers(0, 256, 3000).astype(np.uint8)
    nll = sliding_window_nll(data, lambda x, y, m: np.full(x.shape, math.log(256)), 128, 64, 4)
    assert bits_per_byte(nll[1:].sum(), len(data) - 1) == pytest.approx(8.0)


def test_train_and_evaluate_end_to_end_on_tiny_data(tmp_path):
    """train.py runs, writes metrics and checkpoints, resumes, and evaluate.py scores every byte."""
    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    data_dir.mkdir()
    text = (b"<page><title>Test</title><text>Hello world, hello match bias.</text></page>\n" * 200)
    prepare.write_splits(np.frombuffer(text, dtype=np.uint8), data_dir, n_val=2000, n_test=2000)
    common = [sys.executable, "train.py", "config/enwik8_smoke.py", f"--data_dir={data_dir}",
              f"--out_dir={out_dir}", "--match_bias=graded"]
    subprocess.run(common + ["--max_iters=10"], cwd=REPO, check=True, capture_output=True)
    resumed = subprocess.run(common + ["--max_iters=20"], cwd=REPO, check=True, capture_output=True, text=True)
    assert "resuming training" in resumed.stdout
    lines = (out_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 4 and (out_dir / "best.pt").exists()
    result = subprocess.run([sys.executable, "evaluate.py", f"--ckpt={out_dir / 'best.pt'}", "--split=test",
                             f"--data-dir={data_dir}", "--device=cpu", "--batch-size=8"],
                            cwd=REPO, check=True, capture_output=True, text=True)
    assert '"n_bytes_scored": 1999' in result.stdout
