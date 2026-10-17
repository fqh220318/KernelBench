"""Public execution model for the windowed cosine-attention classifier.

Correctness is decided by ``reference/swin_v2_reference.py`` and serialized
goldens. This independently structured torch model supplies the public
constructor, deterministic input generator, device integration, and diagnostic
reference timing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _window_partition(x: torch.Tensor, size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height // size, size, width // size, size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, size, size, channels)


def _window_reverse(windows: torch.Tensor, size: int, height: int, width: int) -> torch.Tensor:
    batch = windows.shape[0] // ((height // size) * (width // size))
    x = windows.reshape(batch, height // size, width // size, size, size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, height, width, -1)


class FeedForward(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: tuple[int, int],
        heads: int,
        pretrained_window_size: tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.heads = heads
        self.logit_scale = nn.Parameter(torch.log(torch.full((heads, 1, 1), 10.0)))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, heads, bias=False),
        )

        height_offsets = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        width_offsets = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(height_offsets, width_offsets, indexing="ij")
        relative_table = torch.stack((grid_h, grid_w), dim=-1).unsqueeze(0)
        denominators = (
            pretrained_window_size[0] - 1 if pretrained_window_size[0] > 0 else window_size[0] - 1,
            pretrained_window_size[1] - 1 if pretrained_window_size[1] > 0 else window_size[1] - 1,
        )
        relative_table[..., 0] /= denominators[0]
        relative_table[..., 1] /= denominators[1]
        relative_table *= 8.0
        relative_table = torch.sign(relative_table) * torch.log2(relative_table.abs() + 1.0) / math.log2(8.0)
        self.register_buffer("relative_coords_table", relative_table)

        token_h, token_w = torch.meshgrid(
            torch.arange(window_size[0]), torch.arange(window_size[1]), indexing="ij"
        )
        coordinates = torch.stack((token_h, token_w)).flatten(1)
        offsets = coordinates[:, :, None] - coordinates[:, None, :]
        offsets = offsets.permute(1, 2, 0).contiguous()
        offsets[..., 0] += window_size[0] - 1
        offsets[..., 1] += window_size[1] - 1
        offsets[..., 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", offsets.sum(-1))

        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(channels))
        self.v_bias = nn.Parameter(torch.zeros(channels))
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch_windows, tokens, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        scores = scores * scale

        bias_table = self.cpb_mlp(self.relative_coords_table).reshape(-1, self.heads)
        bias = bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1).contiguous()
        scores = scores + 16.0 * torch.sigmoid(bias).unsqueeze(0)
        if mask is not None:
            window_count = mask.shape[0]
            scores = scores.reshape(
                batch_windows // window_count, window_count, self.heads, tokens, tokens
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(2)
            scores = scores.reshape(-1, self.heads, tokens, tokens)
        probabilities = torch.softmax(scores, dim=-1)
        merged = (probabilities @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(merged)


class WindowBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        resolution: tuple[int, int],
        heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        pretrained_window_size: int,
    ):
        super().__init__()
        if min(resolution) <= window_size:
            window_size = min(resolution)
            shift_size = 0
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be inside the window")
        self.resolution = resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, _pair(window_size), heads, _pair(pretrained_window_size))
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = FeedForward(channels, int(channels * mlp_ratio))

        if shift_size:
            height, width = resolution
            labels = torch.zeros((1, height, width, 1))
            slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            label = 0
            for height_slice in slices:
                for width_slice in slices:
                    labels[:, height_slice, width_slice, :] = label
                    label += 1
            label_windows = _window_partition(labels, window_size).reshape(-1, window_size**2)
            differences = label_windows.unsqueeze(1) - label_windows.unsqueeze(2)
            attention_mask = differences.masked_fill(differences != 0, -100.0).masked_fill(
                differences == 0, 0.0
            )
        else:
            attention_mask = None
        self.register_buffer("attn_mask", attention_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError("input token count does not match the stage resolution")
        shortcut = x
        image = x.reshape(batch, height, width, channels)
        if self.shift_size:
            image = torch.roll(image, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = _window_partition(image, self.window_size)
        windows = windows.reshape(-1, self.window_size**2, channels)
        attended = self.attn(windows, self.attn_mask)
        attended = attended.reshape(-1, self.window_size, self.window_size, channels)
        image = _window_reverse(attended, self.window_size, height, width)
        if self.shift_size:
            image = torch.roll(image, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.norm1(image.reshape(batch, tokens, channels))
        return x + self.norm2(self.mlp(x))


class PatchMerge(nn.Module):
    def __init__(self, resolution: tuple[int, int], channels: int):
        super().__init__()
        self.resolution = resolution
        self.reduction = nn.Linear(4 * channels, 2 * channels, bias=False)
        self.norm = nn.LayerNorm(2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.resolution
        batch, tokens, channels = x.shape
        if tokens != height * width or height % 2 or width % 2:
            raise ValueError("patch merging requires an even declared resolution")
        image = x.reshape(batch, height, width, channels)
        merged = torch.cat(
            (
                image[:, 0::2, 0::2],
                image[:, 1::2, 0::2],
                image[:, 0::2, 1::2],
                image[:, 1::2, 1::2],
            ),
            dim=-1,
        ).reshape(batch, -1, 4 * channels)
        return self.norm(self.reduction(merged))


class Stage(nn.Module):
    def __init__(
        self,
        channels: int,
        resolution: tuple[int, int],
        depth: int,
        heads: int,
        window_size: int,
        mlp_ratio: float,
        pretrained_window_size: int,
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                WindowBlock(
                    channels,
                    resolution,
                    heads,
                    window_size,
                    0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio,
                    pretrained_window_size,
                )
                for index in range(depth)
            ]
        )
        self.downsample = PatchMerge(resolution, channels) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.downsample(x) if self.downsample is not None else x


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, input_channels: int, channels: int):
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        self.resolution = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.proj = nn.Conv2d(
            input_channels, channels, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError("input image does not match the declared size")
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class Model(nn.Module):
    """Four-stage classifier whose in-scope operator is window attention."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        input_channels: int = 3,
        classes: int = 1000,
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        pretrained_window_sizes=(0, 0, 0, 0),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, input_channels, embed_dim)
        base_resolution = self.patch_embed.resolution
        self.layers = nn.ModuleList()
        for stage_index, depth in enumerate(depths):
            channels = embed_dim * (2**stage_index)
            resolution = (
                base_resolution[0] // (2**stage_index),
                base_resolution[1] // (2**stage_index),
            )
            self.layers.append(
                Stage(
                    channels,
                    resolution,
                    depth,
                    heads[stage_index],
                    window_size,
                    mlp_ratio,
                    pretrained_window_sizes[stage_index],
                    stage_index < len(depths) - 1,
                )
            )
        final_channels = embed_dim * (2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(final_channels)
        self.head = nn.Linear(final_channels, classes) if classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


batch_size = 1
image_size = 224


def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size)]


def get_init_inputs():
    return []


TIER = "A_kernel"
