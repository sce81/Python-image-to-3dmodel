"""SDPA-backed stand-in for the two xformers ops TRELLIS.2 sparse attention calls.

The official TRELLIS.2 checkout supports only 'xformers' / 'flash_attn' sparse
attention backends, and neither ships Windows wheels for Torch 2.10+cu130.
This shim registers a fake `xformers` module implementing exactly the surface
`trellis2.modules.sparse.attention` uses:

    xformers.ops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen[, kv_seqlen])
    xformers.ops.memory_efficient_attention(q, k, v[, attn_bias])

via torch.nn.functional.scaled_dot_product_attention. Variable-length segments
are bucketed by (q_len, kv_len) so each bucket runs as one batched SDPA call
(no O(T^2) dense mask, no per-window python loop).

Mirrors the install_sparse_sdpa_fallback() pattern generate_props.py already
uses for TRELLIS 1. Only installs when the real xformers is unavailable.
"""

from __future__ import annotations

import sys
import types
from itertools import accumulate

import torch
import torch.nn.functional as F


class BlockDiagonalMask:
    """Stores the segment lengths of a block-diagonal attention pattern."""

    def __init__(self, q_seqlen: list[int], kv_seqlen: list[int]):
        self.q_seqlen = q_seqlen
        self.kv_seqlen = kv_seqlen

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None, **_):
        def to_list(value):
            if hasattr(value, "tolist"):
                value = value.tolist()
            return [int(item) for item in value]

        q = to_list(q_seqlen)
        kv = to_list(kv_seqlen) if kv_seqlen is not None else list(q)
        if len(q) != len(kv):
            raise ValueError(f"segment count mismatch: {len(q)} vs {len(kv)}")
        return cls(q, kv)


def memory_efficient_attention(q, k, v, attn_bias=None, **_):
    """q, k, v: [1, T, H, C] packed sequences; returns [1, Tq, H, Cv]."""
    if attn_bias is None:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2)

    if not isinstance(attn_bias, BlockDiagonalMask):
        raise TypeError(f"unsupported attn_bias for SDPA shim: {type(attn_bias)}")

    q_lens, kv_lens = attn_bias.q_seqlen, attn_bias.kv_seqlen
    q_flat, k_flat, v_flat = q[0], k[0], v[0]
    if q_flat.shape[0] != sum(q_lens) or k_flat.shape[0] != sum(kv_lens):
        raise ValueError("attn_bias segment lengths do not match packed tensors")

    q_offsets = [0, *accumulate(q_lens)]
    kv_offsets = [0, *accumulate(kv_lens)]
    out = q_flat.new_empty(q_flat.shape[0], q_flat.shape[1], v_flat.shape[-1])

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(q_lens, kv_lens)):
        buckets.setdefault(key, []).append(index)

    for (q_len, kv_len), indices in buckets.items():
        if q_len == 0:
            continue
        q_batch = torch.stack([q_flat[q_offsets[i]:q_offsets[i] + q_len] for i in indices])
        k_batch = torch.stack([k_flat[kv_offsets[i]:kv_offsets[i] + kv_len] for i in indices])
        v_batch = torch.stack([v_flat[kv_offsets[i]:kv_offsets[i] + kv_len] for i in indices])
        result = F.scaled_dot_product_attention(
            q_batch.transpose(1, 2), k_batch.transpose(1, 2), v_batch.transpose(1, 2)
        ).transpose(1, 2)
        for position, i in enumerate(indices):
            out[q_offsets[i]:q_offsets[i] + q_len] = result[position]

    return out.unsqueeze(0)


def install() -> bool:
    """Register the shim as `xformers` unless the real package imports."""
    try:
        import xformers.ops  # noqa: F401
        return False
    except Exception:
        pass

    xformers = types.ModuleType("xformers")
    xformers.__version__ = "0.0.0+nexus-sdpa-shim"
    ops = types.ModuleType("xformers.ops")
    fmha = types.ModuleType("xformers.ops.fmha")
    fmha.BlockDiagonalMask = BlockDiagonalMask
    ops.fmha = fmha
    ops.memory_efficient_attention = memory_efficient_attention
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops
    sys.modules["xformers.ops.fmha"] = fmha
    return True
