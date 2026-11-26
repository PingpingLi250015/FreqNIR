"""Dynamic cross-modality attention and transformer blocks."""

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PointDepthProjection(nn.Module):
    """1x1 point-wise followed by 3x3 depth-wise projection (Eq. 9)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.point = nn.Conv2d(channels, channels, 1)
        self.depth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.depth(self.point(x))


class DCAttention(nn.Module):
    """Dynamic Cross-modality Attention from Eqs. (9)-(13).

    Attention rows are evaluated in chunks. This is algebraically equivalent to
    constructing the full HW x HW maps, while keeping peak memory bounded.
    """

    def __init__(self, channels: int, bias_window: int = 16, chunk_size: int = 512) -> None:
        super().__init__()
        self.channels = channels
        self.bias_window = bias_window
        self.chunk_size = chunk_size

        self.rgb_value = PointDepthProjection(channels)
        self.rgb_query = PointDepthProjection(channels)
        self.nir_key_cross = PointDepthProjection(channels)
        self.nir_query = PointDepthProjection(channels)
        self.nir_key_self = PointDepthProjection(channels)
        self.output = nn.Conv2d(channels, channels, 1)

        self.distance_bias = nn.Parameter(torch.zeros(bias_window + 2))
        self.eta = nn.Parameter(torch.tensor(1.0))
        self.log_delta = nn.Parameter(torch.tensor(math.log(math.sqrt(channels))))

    @staticmethod
    def _tokens(x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2)

    def _position_bias(self, start: int, stop: int, length: int, device: torch.device) -> Tensor:
        query = torch.arange(start, stop, device=device).view(-1, 1)
        key = torch.arange(length, device=device).view(1, -1)
        distance = (query - key).abs()
        indices = distance.clamp_max(self.bias_window + 1)
        return self.distance_bias[indices]

    def forward(self, rgb: Tensor, nir: Tensor) -> Tensor:
        if rgb.shape != nir.shape:
            raise ValueError(f"RGB and NIR features must match, got {rgb.shape} and {nir.shape}")
        batch, channels, height, width = rgb.shape
        length = height * width

        value = self._tokens(self.rgb_value(rgb))
        query_rgb = self._tokens(self.rgb_query(rgb))
        key_cross = self._tokens(self.nir_key_cross(nir))
        query_nir = self._tokens(self.nir_query(nir))
        key_self = self._tokens(self.nir_key_self(nir))
        key_cross_t = key_cross.transpose(1, 2)
        key_self_t = key_self.transpose(1, 2)

        scale = math.sqrt(channels)
        delta = self.log_delta.exp().clamp_min(1e-6)
        rows: List[Tensor] = []
        step = length if self.chunk_size <= 0 else self.chunk_size
        for start in range(0, length, step):
            stop = min(start + step, length)
            cross_logits = torch.bmm(query_rgb[:, start:stop], key_cross_t) / scale
            cross_attention = F.softmax(cross_logits, dim=-1)

            self_logits = torch.bmm(query_nir[:, start:stop], key_self_t) / delta
            self_logits = self_logits + self._position_bias(start, stop, length, rgb.device)
            self_attention = F.softmax(self_logits, dim=-1)

            fused_attention = cross_attention + self.eta * self_attention
            rows.append(torch.bmm(fused_attention, value))

        fused = torch.cat(rows, dim=1).transpose(1, 2).reshape(batch, channels, height, width)
        return self.output(fused)


class DCGFFN(nn.Module):
    """Dynamic Cross-modality Gate-FFN from Eq. (14)."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.nir_projection = nn.Conv2d(channels, channels, 1)
        self.input_projection = nn.Conv2d(channels, hidden * 2, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.output_projection = nn.Conv2d(hidden, channels, 1)

    def forward(self, rgb: Tensor, nir: Tensor) -> Tensor:
        fused = rgb + self.nir_projection(nir)
        local, gate = self.input_projection(fused).chunk(2, dim=1)
        gated = F.gelu(self.depthwise(local)) * gate
        return self.output_projection(gated)


class DCFBlock(nn.Module):
    """LayerNorm -> DCAttention -> LayerNorm -> DCG-FFN."""

    def __init__(
        self,
        channels: int,
        bias_window: int = 16,
        attention_chunk_size: int = 512,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm_attention = LayerNorm2d(channels)
        self.attention = DCAttention(channels, bias_window, attention_chunk_size)
        self.norm_ffn = LayerNorm2d(channels)
        self.ffn = DCGFFN(channels, ffn_expansion)

    def forward(self, rgb: Tensor, nir: Tensor) -> Tensor:
        rgb = rgb + self.attention(self.norm_attention(rgb), nir)
        rgb = rgb + self.ffn(self.norm_ffn(rgb), nir)
        return rgb


class DCFGroup(nn.Module):
    def __init__(self, channels: int, depth: int, **block_kwargs: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DCFBlock(channels, **block_kwargs) for _ in range(depth)])

    def forward(self, rgb: Tensor, nir: Tensor) -> Tensor:
        for block in self.blocks:
            rgb = block(rgb, nir)
        return rgb
