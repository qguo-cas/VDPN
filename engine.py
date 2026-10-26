"""Device-independent train/validation loops and checkpoint helpers."""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu or auto.")
    return device


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Return sample-weighted means; validation never builds a gradient graph."""
    training = optimizer is not None
    previous_mode = model.training
    model.train(training)
    totals = {name: 0.0 for name in ("loss", "spectral", "spatial")}
    count = 0
    try:
        with torch.set_grad_enabled(training):
            for _, pan, upms in loader:
                pan = pan.to(device, non_blocking=True)
                upms = upms.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                fused = model(pan, upms)
                losses = criterion(pan, upms, fused)
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise FloatingPointError("Non-finite loss encountered; inspect input data and learning rate.")
                if training:
                    losses["loss"].backward()
                    optimizer.step()
                batch_size = pan.shape[0]
                count += batch_size
                for name, value in losses.items():
                    totals[name] += value.detach().item() * batch_size
    finally:
        model.train(previous_mode)
    if count == 0:
        raise ValueError("Cannot run an epoch on an empty dataset.")
    return {name: value / count for name, value in totals.items()}


def validate(model, loader, criterion, device):
    return run_epoch(model, loader, criterion, device)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)
