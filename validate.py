"""Evaluate a saved FusionNet checkpoint on a separate validation split."""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from cli import add_data_arguments, make_criterion, make_dataset, validate_loader_arguments
from engine import load_checkpoint, resolve_device, validate
from model import FusionNet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--output", type=Path, help="Optional JSON metrics file")
    add_data_arguments(parser)
    args = parser.parse_args(argv)
    validate_loader_arguments(parser, args)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    model = FusionNet(num_channels=config["num_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = make_dataset(args.val_dir, args, config["num_channels"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    metrics = validate(model, loader, make_criterion(config).to(device), device)
    result = {"num_samples": len(dataset), "epoch": checkpoint["epoch"], **metrics}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
