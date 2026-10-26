"""Shared, all-channel objective for training and validation."""

from torch import nn

from .edge_loss import EdgeCorrelationLoss
from .spectral_angle_loss import SpectralAngleLoss


class FusionLoss(nn.Module):
    def __init__(self, spectral_weight=0.8, spatial_weight=0.2):
        super().__init__()
        self.spectral = SpectralAngleLoss()
        self.spatial = EdgeCorrelationLoss()
        self.spectral_weight = spectral_weight
        self.spatial_weight = spatial_weight

    def forward(self, pan, upms, fused):
        spectral = self.spectral(upms, fused)
        spatial = self.spatial(pan, fused)
        total = self.spectral_weight * spectral + self.spatial_weight * spatial
        return {"loss": total, "spectral": spectral, "spatial": spatial}
