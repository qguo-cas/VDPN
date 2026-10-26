"""Paired TIFF inputs for pansharpening, using every spectral channel."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_image(path):
    """Read a TIFF as float32 CHW and normalize each channel to [0, 1]."""
    import tifffile

    with tifffile.TiffFile(path) as image:
        series = image.series[0]
        array = series.asarray().astype(np.float32)
        axes = series.axes
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        if axes in ("YXS", "YXC"):
            array = np.moveaxis(array, -1, 0)
        elif axes not in ("CYX", "SYX", "QYX", "IYX"):
            raise ValueError(f"Unsupported TIFF axes {axes!r} in {path}; expected CHW or HWC.")
    else:
        raise ValueError(f"Expected a 2D or 3D TIFF, got {array.shape} in {path}.")
    if not np.isfinite(array).all():
        raise ValueError(f"TIFF contains NaN or infinity: {path}")
    minimum = array.min(axis=(1, 2), keepdims=True)
    span = array.max(axis=(1, 2), keepdims=True) - minimum
    array = (array - minimum) / np.where(span > 0, span, 1.0)
    return torch.from_numpy(np.ascontiguousarray(array))


class FusionDataset(Dataset):
    """Match <id>_ms, <id>_pan and <id>_mul TIFFs by the full sample ID."""

    def __init__(self, root, ms_dir="MS", pan_dir="PAN", upms_dir="UPMS", num_channels=4):
        self.root = Path(root).expanduser().resolve()
        self.num_channels = num_channels
        directories = [self.root / name for name in (ms_dir, pan_dir, upms_dir)]
        suffixes = ("_ms", "_pan", "_mul")
        indexed = []
        for directory, suffix in zip(directories, suffixes):
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing input directory: {directory}")
            files = {}
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in (".tif", ".tiff") or not path.stem.endswith(suffix):
                    continue
                sample_id = path.stem[:-len(suffix)]
                if sample_id in files:
                    raise ValueError(f"Duplicate sample ID {sample_id!r} in {directory}")
                files[sample_id] = path
            indexed.append(files)
        ids = set(indexed[0])
        if not ids:
            raise ValueError(f"No *_ms.tif or *_ms.tiff files found in {directories[0]}")
        if any(set(files) != ids for files in indexed[1:]):
            raise ValueError(f"MS, PAN and UPMS sample IDs must match exactly under {self.root}.")
        self.samples = [tuple(files[key] for files in indexed) for key in sorted(ids)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ms, pan, upms = (load_image(path) for path in self.samples[index])
        if ms.shape[0] != self.num_channels or upms.shape[0] != self.num_channels:
            raise ValueError(f"Sample {self.samples[index][0].stem}: expected {self.num_channels} MS channels, "
                             f"got MS={ms.shape[0]}, UPMS={upms.shape[0]}.")
        if pan.shape[0] != 1 or pan.shape[1:] != upms.shape[1:]:
            raise ValueError("PAN must have one channel and the same spatial shape as UPMS.")
        if any(low > high for low, high in zip(ms.shape[1:], upms.shape[1:])):
            raise ValueError("MS spatial dimensions must not exceed UPMS dimensions.")
        return ms, pan, upms
