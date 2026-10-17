"""Public execution model for causal ReLU attention.

Correctness is decided by ``reference/relu_causal_reference.py`` and the
generated golden, not by this device-side model. This file defines the public
constructor, input generator and an independently written torch realization used
for evaluator integration and diagnostic timing.
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Project x to Q/K/V, apply top-left causal ReLU attention, return BSC."""

    def __init__(self, n_embd: int, n_head: int, max_seqlen: int):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seqlen, max_seqlen, dtype=torch.bool)).view(
                1, 1, max_seqlen, max_seqlen
            ),
        )
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        head_width = channels // self.n_head

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence, self.n_head, head_width).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_width)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :sequence, :sequence], float("-inf")
        )
        weights = torch.relu(scores)
        output = torch.matmul(weights, v)
        return output.transpose(1, 2).contiguous().view(batch, sequence, channels)


batch_size = 1
max_seqlen = 64
n_embd = 128
n_head = 4


def get_inputs():
    x = torch.rand(batch_size, max_seqlen, n_embd)
    return [x]


def get_init_inputs():
    return [n_embd, n_head, max_seqlen]


TIER = "A_kernel"
