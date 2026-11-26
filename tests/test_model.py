import torch

from freqnir import FreqNIR
from freqnir.losses import FreqNIRLoss


def tiny_model() -> FreqNIR:
    return FreqNIR(
        base_channels=2,
        block_depths=(1, 1, 1, 1, 1, 1),
        bias_window=2,
        attention_chunk_size=32,
        ffn_expansion=1.0,
    )


def test_model_forward_and_loss_backward() -> None:
    model = tiny_model()
    rgb = torch.rand(1, 3, 16, 16)
    nir = torch.rand(1, 1, 16, 16)
    target = torch.rand_like(rgb)
    outputs = model(rgb, nir, return_aux=True)
    assert outputs["output"].shape == rgb.shape
    losses = FreqNIRLoss()(outputs, target, stage="full")
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_nfdp_has_three_scales() -> None:
    model = tiny_model()
    rgb = torch.rand(1, 3, 16, 24)
    nir = torch.rand(1, 1, 16, 24)
    output = model(rgb, nir, stage="nfdp")
    assert [feature.shape[-2:] for feature in output["priors"]] == [
        (8, 12),
        (4, 6),
        (2, 3),
    ]
