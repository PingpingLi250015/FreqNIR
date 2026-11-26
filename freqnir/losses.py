"""Losses described in Sec. 3.4 of the FreqNIR paper."""

from typing import Dict, List, Mapping, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .wavelet import haar_dwt2


def _resize_target(target: Tensor, prediction: Tensor) -> Tensor:
    if target.shape[-2:] == prediction.shape[-2:]:
        return target
    return F.interpolate(target, size=prediction.shape[-2:], mode="bilinear", align_corners=False)


class FourierDiceLoss(nn.Module):
    """Amplitude Dice objective in Eq. (16)."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = _resize_target(target, prediction)
        pred_amplitude = torch.abs(torch.fft.fft2(prediction, norm="ortho"))
        target_amplitude = torch.abs(torch.fft.fft2(target, norm="ortho"))
        numerator = pred_amplitude.square().sum() + target_amplitude.square().sum()
        denominator = 2.0 * (pred_amplitude * target_amplitude).sum()
        return numerator / denominator.clamp_min(self.epsilon)


class WaveletHighFrequencyLoss(nn.Module):
    """L1 distance between the three high-frequency DWT bands (Eq. 15)."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = _resize_target(target, prediction)
        pred_bands = haar_dwt2(prediction)[1:]
        target_bands = haar_dwt2(target)[1:]
        return sum(F.l1_loss(pred, truth) for pred, truth in zip(pred_bands, target_bands)) / 3.0


class LaplacianCharbonnierLoss(nn.Module):
    """Charbonnier penalty on Laplacian responses from Eq. (17)."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def _laplacian(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        weight = self.kernel.expand(channels, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, weight, groups=channels)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        difference = self._laplacian(prediction) - self._laplacian(target)
        return torch.sqrt(difference.square() + self.epsilon**2).mean()


class FreqNIRLoss(nn.Module):
    """Weighted frequency objectives and reconstruction objective (Eq. 18)."""

    def __init__(self, lambda_fourier: float = 1e-3, lambda_wavelet: float = 5e-3) -> None:
        super().__init__()
        self.lambda_fourier = lambda_fourier
        self.lambda_wavelet = lambda_wavelet
        self.fourier = FourierDiceLoss()
        self.wavelet = WaveletHighFrequencyLoss()
        self.reconstruction = LaplacianCharbonnierLoss(epsilon=1e-3)

    def frequency_terms(
        self, low_predictions: List[Tensor], prior_predictions: List[Tensor], target: Tensor
    ) -> Dict[str, Tensor]:
        fourier = sum(self.fourier(prediction, target) for prediction in low_predictions)
        wavelet = sum(self.wavelet(prediction, target) for prediction in prior_predictions)
        fourier = fourier / len(low_predictions)
        wavelet = wavelet / len(prior_predictions)
        return {
            "fourier": fourier,
            "wavelet": wavelet,
            "frequency": self.lambda_fourier * fourier + self.lambda_wavelet * wavelet,
        }

    def forward(
        self,
        outputs: Mapping[str, Union[Tensor, List[Tensor]]],
        target: Tensor,
        stage: str = "full",
    ) -> Dict[str, Tensor]:
        low_predictions = outputs["low_predictions"]
        prior_predictions = outputs["prior_predictions"]
        if not isinstance(low_predictions, list) or not isinstance(prior_predictions, list):
            raise TypeError("NFDP predictions must be lists")
        terms = self.frequency_terms(low_predictions, prior_predictions, target)
        if stage == "nfdp":
            terms["total"] = terms["frequency"]
            return terms
        if stage != "full":
            raise ValueError("stage must be 'nfdp' or 'full'")
        prediction = outputs["output"]
        if not isinstance(prediction, Tensor):
            raise TypeError("Full-model output must be a Tensor")
        reconstruction = self.reconstruction(prediction, target)
        terms["reconstruction"] = reconstruction
        terms["total"] = terms["frequency"] + reconstruction
        return terms
