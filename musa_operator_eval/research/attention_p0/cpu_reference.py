"""Small, NumPy-only semantic references for three pre-task Attention candidates.

These functions deliberately optimise for auditability rather than speed. They do
not import torch, select a MUSA backend, or imply that any planned hardware cell is
supported. All dot products, softmax reductions and outputs use float64.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _float_tensor(name: str, value: np.ndarray, rank: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got {array.ndim}")
    return array


def _integer_vector(name: str, value: Iterable[int], size: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector")
    return raw.astype(np.int64, copy=False)


def _scale_value(scale: float | None, head_dim: int) -> float:
    value = float(scale if scale is not None else 1.0 / math.sqrt(head_dim))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("scale must be finite and positive")
    return value


def _attention_segment(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool,
    scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one SHD segment with bottom-right causal alignment."""
    q64 = _float_tensor("q segment", q, 3)
    k64 = _float_tensor("k segment", k, 3)
    v64 = _float_tensor("v segment", v, 3)
    s_q, h_q, head_dim = q64.shape
    if k64.shape != v64.shape:
        raise ValueError("k and v segment shapes must match")
    s_kv, h_kv, kv_head_dim = k64.shape
    if head_dim != kv_head_dim:
        raise ValueError("q, k and v must share head_dim")
    if head_dim == 0:
        raise ValueError("head_dim must be positive")
    if h_kv == 0 or h_q % h_kv:
        raise ValueError("H_q must be divisible by a non-zero H_kv")

    scale_value = _scale_value(scale, head_dim)
    group_size = h_q // h_kv
    output = np.zeros((s_q, h_q, head_dim), dtype=np.float64)
    lse = np.full((s_q, h_q), -np.inf, dtype=np.float64)

    for query_head in range(h_q):
        kv_head = query_head // group_size
        scores = (q64[:, query_head] @ k64[:, kv_head].T) * scale_value
        for query_index in range(s_q):
            visible_count = s_kv
            if causal:
                # Bottom-right alignment: query i aligns with key
                # i + (S_kv - S_q), which is the common varlen convention.
                visible_count = min(s_kv, query_index + (s_kv - s_q) + 1)
            if visible_count <= 0:
                continue
            visible_scores = scores[query_index, :visible_count]
            row_max = float(np.max(visible_scores))
            weights = np.exp(visible_scores - row_max)
            denominator = float(np.sum(weights))
            probabilities = weights / denominator
            output[query_index, query_head] = probabilities @ v64[:visible_count, kv_head]
            lse[query_index, query_head] = row_max + math.log(denominator)
    return output, lse


def per_sequence_causal_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    q_lens: Iterable[int],
    kv_lens: Iterable[int],
    causal_flags: Iterable[bool],
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense BSHD reference with an independent causal flag per sequence.

    Rows beyond ``q_lens`` are zero with ``-inf`` LSE. A sequence with zero KV
    length follows the same fully-masked-row rule. ``causal_flags`` must be a
    boolean vector; coercing it to one scalar would change the candidate's core
    semantics.
    """
    q64 = _float_tensor("q", q, 4)
    k64 = _float_tensor("k", k, 4)
    v64 = _float_tensor("v", v, 4)
    batch, max_q, h_q, head_dim = q64.shape
    if k64.shape != v64.shape or k64.shape[0] != batch or k64.shape[3] != head_dim:
        raise ValueError("k/v shapes must match and agree with q batch/head_dim")
    max_kv = k64.shape[1]
    q_lengths = _integer_vector("q_lens", q_lens, batch)
    kv_lengths = _integer_vector("kv_lens", kv_lens, batch)
    flags = np.asarray(causal_flags)
    if flags.ndim != 1 or flags.size != batch or flags.dtype != np.bool_:
        raise ValueError("causal_flags must be a boolean vector with one value per sequence")
    if np.any(q_lengths < 0) or np.any(q_lengths > max_q):
        raise ValueError("q_lens values must be inside the padded q extent")
    if np.any(kv_lengths < 0) or np.any(kv_lengths > max_kv):
        raise ValueError("kv_lens values must be inside the padded k/v extent")

    output = np.zeros_like(q64)
    lse = np.full((batch, max_q, h_q), -np.inf, dtype=np.float64)
    for batch_index in range(batch):
        q_length = int(q_lengths[batch_index])
        kv_length = int(kv_lengths[batch_index])
        segment_output, segment_lse = _attention_segment(
            q64[batch_index, :q_length],
            k64[batch_index, :kv_length],
            v64[batch_index, :kv_length],
            causal=bool(flags[batch_index]),
            scale=scale,
        )
        output[batch_index, :q_length] = segment_output
        lse[batch_index, :q_length] = segment_lse
    return output, lse


def _cu_seqlens(name: str, value: Iterable[int], total: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError(f"{name} must contain B+1 integer offsets")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector")
    offsets = raw.astype(np.int64, copy=False)
    if offsets[0] != 0 or offsets[-1] != total or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be non-decreasing from 0 through {total}")
    return offsets


def varlen_packed_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    cu_seqlens_q: Iterable[int],
    cu_seqlens_kv: Iterable[int],
    causal: bool,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Packed THD MHA/GQA with bottom-right causal alignment.

    Empty q or KV segments are legal. Query rows paired with an empty KV segment
    produce zero output and ``-inf`` LSE. There is no dropout, bias or window.
    """
    q64 = _float_tensor("q", q, 3)
    k64 = _float_tensor("k", k, 3)
    v64 = _float_tensor("v", v, 3)
    if k64.shape != v64.shape or q64.shape[2] != k64.shape[2]:
        raise ValueError("k/v shapes must match and q/k/v must share head_dim")
    q_offsets = _cu_seqlens("cu_seqlens_q", cu_seqlens_q, q64.shape[0])
    kv_offsets = _cu_seqlens("cu_seqlens_kv", cu_seqlens_kv, k64.shape[0])
    if q_offsets.size != kv_offsets.size:
        raise ValueError("q and kv offset vectors must describe the same batch")

    output = np.zeros_like(q64)
    lse = np.full((q64.shape[0], q64.shape[1]), -np.inf, dtype=np.float64)
    for batch_index in range(q_offsets.size - 1):
        q_start, q_stop = map(int, q_offsets[batch_index:batch_index + 2])
        kv_start, kv_stop = map(int, kv_offsets[batch_index:batch_index + 2])
        segment_output, segment_lse = _attention_segment(
            q64[q_start:q_stop],
            k64[kv_start:kv_stop],
            v64[kv_start:kv_stop],
            causal=bool(causal),
            scale=scale,
        )
        output[q_start:q_stop] = segment_output
        lse[q_start:q_stop] = segment_lse
    return output, lse


def paged_kv_decode_reference(
    q: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    *,
    block_table: np.ndarray,
    cache_seqlens: Iterable[int],
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paged BSHD decode reference with read-only caches and causal attention.

    Cache layout is ``[physical_page, page_offset, H_kv, D]``. Required page
    identifiers must be in range; entries beyond ``ceil(cache_len/page_size)``
    are ignored. Repeating a valid page identifier is legal aliasing and repeats
    that physical page in the logical KV sequence.
    """
    q64 = _float_tensor("q", q, 4)
    key64 = _float_tensor("key_cache", key_cache, 4)
    value64 = _float_tensor("value_cache", value_cache, 4)
    if key64.shape != value64.shape:
        raise ValueError("key_cache and value_cache shapes must match")
    batch, query_tokens, _, head_dim = q64.shape
    physical_pages, page_size, _, cache_head_dim = key64.shape
    if page_size <= 0 or head_dim != cache_head_dim:
        raise ValueError("page_size must be positive and q/cache head_dim must match")
    table = np.asarray(block_table)
    if table.ndim != 2 or table.shape[0] != batch or not np.issubdtype(table.dtype, np.integer):
        raise ValueError("block_table must be an integer matrix with one row per sequence")
    lengths = _integer_vector("cache_seqlens", cache_seqlens, batch)
    if np.any(lengths < 0):
        raise ValueError("cache_seqlens values must be non-negative")

    output = np.zeros_like(q64)
    lse = np.full((batch, query_tokens, q64.shape[2]), -np.inf, dtype=np.float64)
    for batch_index, cache_length_value in enumerate(lengths):
        cache_length = int(cache_length_value)
        required_pages = (cache_length + page_size - 1) // page_size
        if required_pages > table.shape[1]:
            raise ValueError("block_table row is too short for cache_seqlens")
        page_ids = table[batch_index, :required_pages].astype(np.int64, copy=False)
        if np.any(page_ids < 0) or np.any(page_ids >= physical_pages):
            raise ValueError("a required block_table page id is out of range")
        if required_pages:
            logical_k = key64[page_ids].reshape(-1, key64.shape[2], head_dim)[:cache_length]
            logical_v = value64[page_ids].reshape(-1, value64.shape[2], head_dim)[:cache_length]
        else:
            logical_k = np.empty((0, key64.shape[2], head_dim), dtype=np.float64)
            logical_v = np.empty_like(logical_k)
        segment_output, segment_lse = _attention_segment(
            q64[batch_index], logical_k, logical_v, causal=True, scale=scale
        )
        output[batch_index] = segment_output
        lse[batch_index] = segment_lse
    return output, lse
