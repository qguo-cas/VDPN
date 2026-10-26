"""Train FusionNet and select checkpoints using a separate validation set."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cli import add_data_arguments, make_criterion, make_dataset, validate_loader_arguments
from engine import load_checkpoint, resolve_device, run_epoch, set_seed, validate
from model import FusionNet


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion"))
    parser.add_argument("--num-channels", type=int, default=4, help="Use all channels; must match the TIFFs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint created by this training entry point")
    parser.add_argument("--spectral-weight", type=float, default=0.8)
    parser.add_argument("--spatial-weight", type=float, default=0.2)
    add_data_arguments(parser)
    return parser


def check_disjoint(trainset, valset):
    train_paths = {path.resolve() for sample in trainset.samples for path in sample}
    val_paths = {path.resolve() for sample in valset.samples for path in sample}
    if train_paths & val_paths:
        raise ValueError("Training and validation must use separate input files.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_loader_arguments(parser, args)
    if args.epochs < 1 or args.save_every < 1 or args.num_channels < 1:
        parser.error("--epochs, --save-every, and --num-channels must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    weights = (args.spectral_weight, args.spatial_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        parser.error("Loss weights must be finite, nonnegative, and not all zero")
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.resume and (args.output_dir / "last.pth").exists():
        parser.error("Output already contains a run. Use --resume or choose a new --output-dir.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    trainset = make_dataset(args.train_dir, args, args.num_channels)
    valset = make_dataset(args.val_dir, args, args.num_channels)
    check_disjoint(trainset, valset)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda")
    train_loader = DataLoader(trainset, shuffle=True, **loader_options)
    val_loader = DataLoader(valset, shuffle=False, **loader_options)
    model = FusionNet(num_channels=args.num_channels).to(device)
    criterion = make_criterion(vars(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_loss, history = 0, float("inf"), []
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        saved_config = checkpoint["config"]
        if Path(saved_config["output_dir"]).resolve() != args.output_dir:
            raise ValueError("Resume must use the original --output-dir so best.pth stays with its history.")
        for name in ("num_channels", "spectral_weight", "spatial_weight"):
            if saved_config[name] != getattr(args, name):
                raise ValueError(f"Resume requires matching --{name.replace('_', '-')}: {saved_config[name]}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        if start_epoch >= args.epochs:
            raise ValueError("--epochs must exceed the checkpoint's completed epoch count.")
        # Do not compare historical scores with a different validation split.
        for name in ("train_dir", "val_dir"):
            if Path(saved_config[name]).resolve() != getattr(args, name).expanduser().resolve():
                raise ValueError("Resume must use the original data split paths.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    # The optimizer's saved learning rate takes precedence when resuming.
    config["lr"] = optimizer.param_groups[0]["lr"]
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device}; training: {len(trainset)} samples; validation: {len(valset)} samples")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = validate(model, val_loader, criterion, device)
        row = {"epoch": epoch,
               **{f"train_{key}": value for key, value in train_metrics.items()},
               **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        improved = val_metrics["loss"] < best_loss
        best_loss = min(best_loss, val_metrics["loss"])
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "config": config,
                      "best_val_loss": best_loss, "train_metrics": train_metrics,
                      "val_metrics": val_metrics, "history": history}
        torch.save(checkpoint, args.output_dir / "last.pth")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pth")
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch:04d}.pth")
        with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_metrics['loss']:.6f}, "
              f"val_loss={val_metrics['loss']:.6f}, best_val_loss={best_loss:.6f}")
    print(f"Checkpoints and metrics saved to {args.output_dir}")
    return history


if __name__ == "__main__":
    main()
