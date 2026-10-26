"""Shared command-line options for data and loss configuration."""

from dataload import FusionDataset
from loss import FusionLoss


def add_data_arguments(parser):
    parser.add_argument("--ms-dir", default="MS", help="MS subdirectory within each split")
    parser.add_argument("--pan-dir", default="PAN", help="PAN subdirectory within each split")
    parser.add_argument("--upms-dir", default="UPMS", help="Upsampled MS subdirectory within each split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0, etc.")


def make_dataset(root, args, num_channels):
    return FusionDataset(root, args.ms_dir, args.pan_dir, args.upms_dir, num_channels)


def make_criterion(config):
    return FusionLoss(**{name: config[name] for name in ("spectral_weight", "spatial_weight")})


def validate_loader_arguments(parser, args):
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers must be nonnegative")
