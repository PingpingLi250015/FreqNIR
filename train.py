#!/usr/bin/env python3
"""Two-stage FreqNIR training entry point."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from freqnir import FreqNIR
from freqnir.data import PairedRGBNIRDataset
from freqnir.losses import FreqNIRLoss
from freqnir.metrics import image_metrics
from freqnir.utils import (
    load_checkpoint,
    load_config,
    model_kwargs,
    seed_everything,
    tiled_forward,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=("nfdp", "full"), required=True)
    parser.add_argument("--output", default=None, help="Override experiment output directory")
    parser.add_argument("--resume", default=None, help="Resume a same-stage checkpoint")
    parser.add_argument(
        "--nfdp-checkpoint", default=None, help="Five-epoch NFDP checkpoint for full training"
    )
    return parser.parse_args()


def distributed_setup() -> Tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return True, torch.distributed.get_rank(), world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def make_loader(
    root: str,
    batch_size: int,
    crop_size: Optional[int],
    augment: bool,
    workers: int,
    distributed: bool,
    world_size: int,
    rank: int,
    shuffle: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = PairedRGBNIRDataset(root, crop_size=crop_size, augment=augment)
    sampler = (
        DistributedSampler(dataset, world_size, rank, shuffle=shuffle) if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def reduce_mean(value: Tensor, world_size: int) -> Tensor:
    if world_size == 1:
        return value
    reduced = value.detach().clone()
    torch.distributed.all_reduce(reduced)
    return reduced / world_size


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, tile_size: int, overlap: int
) -> Dict[str, float]:
    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0}
    count = 0
    bare_model = unwrap_model(model)
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        nir = batch["nir"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        for index in range(rgb.shape[0]):
            prediction = tiled_forward(
                bare_model,
                rgb[index : index + 1],
                nir[index : index + 1],
                tile_size,
                overlap,
            )
            metrics = image_metrics(prediction, target[index : index + 1])
            totals["psnr"] += metrics["psnr"].item()
            totals["ssim"] += metrics["ssim"].item()
            count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    distributed, rank, world_size, device = distributed_setup()
    seed_everything(int(config.get("seed", 1234)) + rank)

    data_config = config["data"]
    train_config = config["training"]
    stage_config = train_config[args.stage]
    global_batch = int(train_config.get("batch_size", 8))
    if global_batch % world_size:
        raise ValueError("training.batch_size must be divisible by the distributed world size")
    per_device_batch = global_batch // world_size

    train_loader, train_sampler = make_loader(
        data_config["train_dir"],
        per_device_batch,
        int(data_config.get("crop_size", 128)),
        True,
        int(data_config.get("workers", 4)),
        distributed,
        world_size,
        rank,
        True,
    )
    val_loader = None
    if data_config.get("val_dir") and rank == 0:
        val_loader, _ = make_loader(
            data_config["val_dir"],
            1,
            None,
            False,
            min(2, int(data_config.get("workers", 4))),
            False,
            1,
            0,
            False,
        )

    model = FreqNIR(**model_kwargs(config)).to(device)
    if args.stage == "full" and not args.resume:
        nfdp_checkpoint = args.nfdp_checkpoint or stage_config.get("nfdp_checkpoint")
        if not nfdp_checkpoint or not Path(nfdp_checkpoint).is_file():
            raise FileNotFoundError(
                "Full training requires the five-epoch NFDP checkpoint; pass --nfdp-checkpoint"
            )
        load_checkpoint(model, nfdp_checkpoint, strict=False)

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=args.stage == "nfdp",
        )

    parameters = (
        unwrap_model(model).nfdp.parameters() if args.stage == "nfdp" else model.parameters()
    )
    optimizer = Adam(
        parameters,
        lr=float(train_config.get("learning_rate", 2e-4)),
        betas=(0.9, 0.999),
    )
    epochs = int(stage_config["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_config.get("minimum_learning_rate", 1e-6))
    )
    criterion = FreqNIRLoss(
        lambda_fourier=float(config["loss"].get("lambda_fourier", 1e-3)),
        lambda_wavelet=float(config["loss"].get("lambda_wavelet", 5e-3)),
    ).to(device)
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    best_score = -float("inf")
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    output = Path(
        args.output
        or stage_config.get("output_dir")
        or ("experiments/nfdp" if args.stage == "nfdp" else "experiments/freqnir")
    )
    checkpoint_dir = output / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(output / "resolved_config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        progress = tqdm(train_loader, disable=rank != 0, desc=f"{args.stage} {epoch + 1}/{epochs}")
        for batch in progress:
            rgb = batch["rgb"].to(device, non_blocking=True)
            nir = batch["nir"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast() if use_amp else nullcontext():
                if args.stage == "nfdp":
                    outputs = model(rgb, nir, stage="nfdp")
                else:
                    outputs = model(rgb, nir, return_aux=True)
                losses = criterion(outputs, target, stage=args.stage)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            reduced_loss = reduce_mean(losses["total"], world_size)
            running += reduced_loss.item()
            if rank == 0:
                progress.set_postfix(loss=f"{reduced_loss.item():.4f}")
        scheduler.step()

        if distributed:
            torch.distributed.barrier()
        if rank != 0:
            if distributed:
                torch.distributed.barrier()
            continue

        epoch_loss = running / max(len(train_loader), 1)
        validation = None
        score = -epoch_loss
        if args.stage == "full" and val_loader is not None:
            validation = validate(
                model,
                val_loader,
                device,
                int(data_config.get("evaluation_tile_size", 256)),
                int(data_config.get("tile_overlap", 32)),
            )
            score = validation["psnr"]
        if args.stage == "nfdp":
            score = -epoch_loss

        state = {
            "epoch": epoch,
            "stage": args.stage,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": max(best_score, score),
            "config": config,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if score > best_score:
            best_score = score
            state["best_score"] = best_score
            torch.save(state, checkpoint_dir / "best.pt")

        message = f"epoch={epoch + 1} loss={epoch_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e}"
        if validation:
            message += f" psnr={validation['psnr']:.3f} ssim={validation['ssim']:.4f}"
        print(message, flush=True)
        if distributed:
            torch.distributed.barrier()

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
