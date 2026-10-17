"""Independent NumPy float64 oracle for the windowed cosine classifier."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


DEPTHS = (2, 2, 6, 2)
HEADS = (3, 6, 12, 24)
BASE_DIM = 96
BASE_RESOLUTION = (56, 56)
WINDOW_SIZE = 7


def _linear(values, weight, bias=None):
    result = np.matmul(values, np.asarray(weight, dtype=np.float64).T)
    if bias is not None:
        result = result + np.asarray(bias, dtype=np.float64)
    return result


def _layer_norm(values, weight, bias, epsilon=1e-5):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + epsilon)
    return normalized * np.asarray(weight, dtype=np.float64) + np.asarray(bias, dtype=np.float64)


def _erf(values):
    """Vectorized Numerical Recipes approximation, accurate to about 1e-7."""
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    t = 1.0 / (1.0 + 0.5 * absolute)
    tau = t * np.exp(
        -absolute * absolute
        - 1.26551223
        + t
        * (
            1.00002368
            + t
            * (
                0.37409196
                + t
                * (
                    0.09678418
                    + t
                    * (
                        -0.18628806
                        + t
                        * (
                            0.27886807
                            + t
                            * (
                                -1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                            )
                        )
                    )
                )
            )
        )
    )
    return np.where(values >= 0.0, 1.0 - tau, tau - 1.0)


def _gelu(values):
    values = np.asarray(values, dtype=np.float64)
    return 0.5 * values * (1.0 + _erf(values / math.sqrt(2.0)))


def _softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _sigmoid(values):
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _partition(values, size):
    batch, height, width, channels = values.shape
    values = values.reshape(batch, height // size, size, width // size, size, channels)
    return values.transpose(0, 1, 3, 2, 4, 5).reshape(-1, size, size, channels)


def _reverse(windows, size, height, width):
    batch = windows.shape[0] // ((height // size) * (width // size))
    values = windows.reshape(batch, height // size, width // size, size, size, -1)
    return values.transpose(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


def _patch_convolution(values, weight, bias):
    weight = np.asarray(weight, dtype=np.float64)
    kernel_height, kernel_width = weight.shape[-2:]
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(values, dtype=np.float64),
        (kernel_height, kernel_width),
        axis=(2, 3),
    )
    windows = windows[:, :, ::kernel_height, ::kernel_width, :, :]
    result = np.einsum("bcyxij,ocij->boyx", windows, weight, optimize=True)
    return result + np.asarray(bias, dtype=np.float64)[None, :, None, None]


def _attention(values, state, prefix, heads, mask):
    batch_windows, tokens, channels = values.shape
    q_bias = np.asarray(state[f"{prefix}.q_bias"], dtype=np.float64)
    v_bias = np.asarray(state[f"{prefix}.v_bias"], dtype=np.float64)
    qkv_bias = np.concatenate((q_bias, np.zeros_like(q_bias), v_bias))
    qkv = _linear(values, state[f"{prefix}.qkv.weight"], qkv_bias)
    qkv = qkv.reshape(batch_windows, tokens, 3, heads, -1).transpose(2, 0, 3, 1, 4)
    query, key, value = qkv
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
    key = key / np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-12)
    scores = np.matmul(query, key.swapaxes(-1, -2))
    logit_scale = np.minimum(
        np.asarray(state[f"{prefix}.logit_scale"], dtype=np.float64), math.log(100.0)
    )
    scores = scores * np.exp(logit_scale)

    coordinates = np.asarray(state[f"{prefix}.relative_coords_table"], dtype=np.float64)
    hidden = _linear(
        coordinates,
        state[f"{prefix}.cpb_mlp.0.weight"],
        state[f"{prefix}.cpb_mlp.0.bias"],
    )
    hidden = np.maximum(hidden, 0.0)
    bias_table = _linear(hidden, state[f"{prefix}.cpb_mlp.2.weight"]).reshape(-1, heads)
    indices = np.asarray(state[f"{prefix}.relative_position_index"], dtype=np.int64)
    bias = bias_table[indices.reshape(-1)].reshape(tokens, tokens, heads).transpose(2, 0, 1)
    scores = scores + (16.0 * _sigmoid(bias))[None, ...]

    if mask is not None:
        window_count = mask.shape[0]
        scores = scores.reshape(
            batch_windows // window_count, window_count, heads, tokens, tokens
        )
        scores = scores + mask[None, :, None, :, :]
        scores = scores.reshape(-1, heads, tokens, tokens)
    probabilities = _softmax(scores)
    merged = np.matmul(probabilities, value).transpose(0, 2, 1, 3)
    merged = merged.reshape(batch_windows, tokens, channels)
    return _linear(merged, state[f"{prefix}.proj.weight"], state[f"{prefix}.proj.bias"])


def _block(values, state, prefix, resolution, heads, shift):
    height, width = resolution
    batch, tokens, channels = values.shape
    shortcut = values
    image = values.reshape(batch, height, width, channels)
    if shift:
        image = np.roll(image, shift=(-shift, -shift), axis=(1, 2))
    windows = _partition(image, WINDOW_SIZE).reshape(-1, WINDOW_SIZE**2, channels)
    mask_key = f"{prefix}.attn_mask"
    mask = np.asarray(state[mask_key], dtype=np.float64) if mask_key in state else None
    attended = _attention(windows, state, f"{prefix}.attn", heads, mask)
    image = _reverse(
        attended.reshape(-1, WINDOW_SIZE, WINDOW_SIZE, channels),
        WINDOW_SIZE,
        height,
        width,
    )
    if shift:
        image = np.roll(image, shift=(shift, shift), axis=(1, 2))
    attended = _layer_norm(
        image.reshape(batch, tokens, channels),
        state[f"{prefix}.norm1.weight"],
        state[f"{prefix}.norm1.bias"],
    )
    values = shortcut + attended
    hidden = _linear(values, state[f"{prefix}.mlp.fc1.weight"], state[f"{prefix}.mlp.fc1.bias"])
    hidden = _gelu(hidden)
    hidden = _linear(hidden, state[f"{prefix}.mlp.fc2.weight"], state[f"{prefix}.mlp.fc2.bias"])
    hidden = _layer_norm(hidden, state[f"{prefix}.norm2.weight"], state[f"{prefix}.norm2.bias"])
    return values + hidden


def _patch_merge(values, state, prefix, resolution):
    height, width = resolution
    batch, _, channels = values.shape
    image = values.reshape(batch, height, width, channels)
    merged = np.concatenate(
        (
            image[:, 0::2, 0::2],
            image[:, 1::2, 0::2],
            image[:, 0::2, 1::2],
            image[:, 1::2, 1::2],
        ),
        axis=-1,
    ).reshape(batch, -1, 4 * channels)
    merged = _linear(merged, state[f"{prefix}.reduction.weight"])
    return _layer_norm(merged, state[f"{prefix}.norm.weight"], state[f"{prefix}.norm.bias"])


def swin_v2_classifier(x: np.ndarray, *, model_state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate the frozen four-stage classifier entirely with NumPy operations."""
    state = {name: np.asarray(value) for name, value in model_state.items()}
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (3, 224, 224):
        raise ValueError("x must have BCHW shape (B, 3, 224, 224)")

    values = _patch_convolution(
        values, state["patch_embed.proj.weight"], state["patch_embed.proj.bias"]
    )
    values = values.reshape(values.shape[0], values.shape[1], -1).transpose(0, 2, 1)
    values = _layer_norm(
        values, state["patch_embed.norm.weight"], state["patch_embed.norm.bias"]
    )

    for stage_index, depth in enumerate(DEPTHS):
        resolution = (
            BASE_RESOLUTION[0] // (2**stage_index),
            BASE_RESOLUTION[1] // (2**stage_index),
        )
        for block_index in range(depth):
            prefix = f"layers.{stage_index}.blocks.{block_index}"
            shift = 0 if block_index % 2 == 0 or min(resolution) <= WINDOW_SIZE else WINDOW_SIZE // 2
            values = _block(values, state, prefix, resolution, HEADS[stage_index], shift)
        if stage_index < len(DEPTHS) - 1:
            values = _patch_merge(
                values, state, f"layers.{stage_index}.downsample", resolution
            )

    values = _layer_norm(values, state["norm.weight"], state["norm.bias"]).mean(axis=1)
    return _linear(values, state["head.weight"], state["head.bias"])
