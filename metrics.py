"""Bits per byte. Every bpc number reported anywhere is computed by this function."""

import math

NATS_PER_BIT = math.log(2.0)


def bits_per_byte(total_nats, n_bytes):
    """Convert a summed negative log-likelihood in nats into bits per byte (base 2)."""
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return total_nats / (n_bytes * NATS_PER_BIT)


def nats_to_bits(mean_nats):
    """Convert a mean loss in nats per byte into bits per byte."""
    return mean_nats / NATS_PER_BIT
