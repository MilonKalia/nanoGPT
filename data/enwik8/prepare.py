"""
Prepare enwik8 for byte-level language modelling.

Downloads enwik8 (the first 10^8 bytes of an English Wikipedia XML dump, Mahoney 2011)
and writes the standard split as raw bytes (uint8):
    train.bin = bytes [0, 90M), val.bin = [90M, 95M), test.bin = [95M, 100M)
plus meta.pkl with vocab_size = 256.
"""

import os
import pickle
import urllib.request
import zipfile

import numpy as np

URL = "http://mattmahoney.net/dc/enwik8.zip"
TOTAL_BYTES = 100_000_000
N_VAL = 5_000_000
N_TEST = 5_000_000
VOCAB_SIZE = 256


def split_bytes(data, n_val=N_VAL, n_test=N_TEST):
    n_train = len(data) - n_val - n_test
    if n_train <= 0:
        raise ValueError("not enough data for the requested split")
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def load_enwik8(data_dir):
    raw_path = os.path.join(data_dir, "enwik8")
    if not os.path.exists(raw_path):
        zip_path = os.path.join(data_dir, "enwik8.zip")
        if not os.path.exists(zip_path):
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("enwik8", data_dir)
    with open(raw_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if len(data) != TOTAL_BYTES:
        raise ValueError(f"expected {TOTAL_BYTES:,} bytes, got {len(data):,}")
    return data


def write_splits(data, out_dir, n_val=N_VAL, n_test=N_TEST):
    names = ("train", "val", "test")
    for name, part in zip(names, split_bytes(data, n_val, n_test)):
        part.tofile(os.path.join(out_dir, f"{name}.bin"))
        print(f"{name}: {len(part):,} bytes")
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": VOCAB_SIZE}, f)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = load_enwik8(here)
    print(f"distinct byte values: {len(np.unique(data))}")
    write_splits(data, here)
