"""Complete FreqNIR encoder-decoder."""

from typing import Dict, List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import DCFGroup
from .nfdp import NFDP


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class FreqNIR(nn.Module):
    """Frequency-separated NIR-prior cross-modality denoising network."""

    def __init__(
        self,
        base_channels: int = 30,
        block_depths: Sequence[int] = (2, 2, 2, 2, 2, 2),
        bias_window: int = 16,
        attention_chunk_size: int = 512,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if len(block_depths) != 6 or any(depth < 1 for depth in block_depths):
            raise ValueError("block_depths must contain six positive integers")

        c1, c2, c3 = base_channels * 2, base_channels * 4, base_channels * 8
        block_kwargs = {
            "bias_window": bias_window,
            "attention_chunk_size": attention_chunk_size,
            "ffn_expansion": ffn_expansion,
        }

        self.nfdp = NFDP(base_channels)
        self.rgb_stem = nn.Conv2d(3, c1, 3, stride=2, padding=1)

        self.encoder1 = DCFGroup(c1, block_depths[0], **block_kwargs)
        self.down1 = Downsample(c1, c2)
        self.encoder2 = DCFGroup(c2, block_depths[1], **block_kwargs)
        self.down2 = Downsample(c2, c3)
        self.encoder3 = DCFGroup(c3, block_depths[2], **block_kwargs)

        self.decoder3 = DCFGroup(c3, block_depths[3], **block_kwargs)
        self.up2 = Upsample(c3, c2)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, 1)
        self.decoder2 = DCFGroup(c2, block_depths[4], **block_kwargs)
        self.up1 = Upsample(c2, c1)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, 1)
        self.decoder1 = DCFGroup(c1, block_depths[5], **block_kwargs)

        self.output = nn.Sequential(
            Upsample(c1, base_channels),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def extract_priors(self, nir: Tensor) -> Dict[str, List[Tensor]]:
        return self.nfdp(nir)

    def forward(
        self, rgb: Tensor, nir: Tensor, return_aux: bool = False, stage: str = "full"
    ) -> Union[Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        if nir.ndim != 4 or nir.shape[1] != 1:
            raise ValueError("nir must have shape [B, 1, H, W]")
        if rgb.shape[0] != nir.shape[0] or rgb.shape[-2:] != nir.shape[-2:]:
            raise ValueError("rgb and nir must share batch and spatial dimensions")

        original_height, original_width = rgb.shape[-2:]
        pad_h = (-original_height) % 8
        pad_w = (-original_width) % 8
        if pad_h or pad_w:
            mode = "reflect" if min(original_height, original_width) > 1 else "replicate"
            rgb_padded = F.pad(rgb, (0, pad_w, 0, pad_h), mode=mode)
            nir = F.pad(nir, (0, pad_w, 0, pad_h), mode=mode)
        else:
            rgb_padded = rgb

        nfdp_output = self.nfdp(nir)
        if stage == "nfdp":
            return nfdp_output
        if stage != "full":
            raise ValueError("stage must be 'nfdp' or 'full'")
        z1, z2, z3 = nfdp_output["priors"]

        x1 = self.encoder1(self.rgb_stem(rgb_padded), z1)
        x2 = self.encoder2(self.down1(x1), z2)
        x3 = self.encoder3(self.down2(x2), z3)

        x = self.decoder3(x3, z3)
        x = self.fuse2(torch.cat((self.up2(x), x2), dim=1))
        x = self.decoder2(x, z2)
        x = self.fuse1(torch.cat((self.up1(x), x1), dim=1))
        x = self.decoder1(x, z1)
        restored = self.output(x) + rgb_padded
        restored = restored[..., :original_height, :original_width]

        if not return_aux:
            return restored
        return {
            "output": restored,
            "priors": nfdp_output["priors"],
            "low_predictions": nfdp_output["low_predictions"],
            "prior_predictions": nfdp_output["prior_predictions"],
        }
