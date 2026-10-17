"""Independent CPU float64 oracle for causal ReLU attention."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def relu_causal_attention(
    x: np.ndarray,
    *,
    init_inputs: Sequence[int],
    model_state: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return BSC output using float64 projection and attention arithmetic."""
    n_embd, n_head, max_seqlen = (int(value) for value in init_inputs)
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != n_embd:
        raise ValueError("x must be a BSC tensor whose C axis equals n_embd")
    if values.shape[1] > max_seqlen:
        raise ValueError("the sequence exceeds the declared maximum")
    if n_embd % n_head:
        raise ValueError("n_embd must be divisible by n_head")

    state = {name: np.asarray(value) for name, value in model_state.items()}
    required = {"causal_mask", "c_attn.weight", "c_attn.bias"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"model_state is missing {sorted(missing)}")
    weight = np.asarray(state["c_attn.weight"], dtype=np.float64)
    bias = np.asarray(state["c_attn.bias"], dtype=np.float64)
    if weight.shape != (3 * n_embd, n_embd) or bias.shape != (3 * n_embd,):
        raise ValueError("c_attn parameters do not match the declared embedding width")
    projected = np.matmul(values, weight.T) + bias
    query, key, value = np.split(projected, 3, axis=-1)

    batch, sequence, channels = values.shape
    width = channels // n_head

    def heads(tensor: np.ndarray) -> np.ndarray:
        return tensor.reshape(batch, sequence, n_head, width).transpose(0, 2, 1, 3)

    query, key, value = heads(query), heads(key), heads(value)
    scores = np.matmul(query, key.swapaxes(-1, -2)) / math.sqrt(width)
    causal = np.asarray(state["causal_mask"], dtype=bool)
    if causal.ndim != 4 or causal.shape[-2:] != (max_seqlen, max_seqlen):
        raise ValueError("causal_mask does not match the declared maximum sequence length")
    scores = np.where(causal[:, :, :sequence, :sequence], scores, -np.inf)
    weights = np.maximum(scores, 0.0)
    output = np.matmul(weights, value)
    return output.transpose(0, 2, 1, 3).reshape(batch, sequence, channels)
