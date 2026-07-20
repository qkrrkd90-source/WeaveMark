"""Keyed PRF that maps a token context + secret key to a 32-bit RNG seed.

Shared by embedding and extraction so both sides derive the same seed.
"""
import hashlib

import torch


def _seed_int(values):
    # comma-delimited so [12,34] and [123,4] don't map to the same seed
    seed_str = ",".join(map(str, values))
    hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
    return int(hash_digest, 16) % 2 ** 32


def prf(seed: torch.LongTensor, secret_key: int):
    # seed: 1-D (one context) -> int, or 2-D (batch) -> list of int.
    if seed.dim() == 1:
        return _seed_int(seed.tolist() + [secret_key])

    return [_seed_int(row.tolist() + [secret_key]) for row in seed]
