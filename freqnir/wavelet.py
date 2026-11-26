"""Differentiable orthonormal Haar wavelet transforms."""

from typing import Tuple

import torch.nn.functional as F
from torch import Tensor

WaveletBands = Tuple[Tensor, Tensor, Tensor, Tensor]


def pad_to_even(x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
    """Reflect-pad the bottom/right edge so both spatial dimensions are even."""
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        mode = "reflect" if min(x.shape[-2:]) > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (pad_h, pad_w)


def haar_dwt2(x: Tensor) -> WaveletBands:
    """Apply a one-level 2-D Haar DWT to a BCHW tensor.

    Returns ``(LL, LH, HL, HH)``. The factor of 1/2 is the product of
    the two 1/sqrt(2) analysis-filter normalizers.
    """
    x, _ = pad_to_even(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_idwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Invert :func:`haar_dwt2` for four equally shaped BCHW tensors."""
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("All inverse-DWT bands must have identical shapes")

    x00 = (ll - lh - hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll + lh + hl + hh) * 0.5

    out = ll.new_empty(ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2)
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out
