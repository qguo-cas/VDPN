# VDPN: Variational-Theoretic Dual-Stream Pansharpening Network

This repository contains the pansharpening research code associated with the following paper, with PyTorch workflows for training, per-epoch validation, and standalone validation:

> Q. Zhou and Q. Guo, “Variational-Theoretic Dual-Stream Pansharpening Network With Spectral Hypersphere Difference and Multidirectional Consistency,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 64, pp. 5408417–5408417, 2026, Art. no. 5408417, doi: 10.1109/TGRS.2026.3732241.

[Paper DOI](https://doi.org/10.1109/TGRS.2026.3732241)

**Keywords:** Pansharpening, remote sensing, self-supervised learning, variational theory, dual-stream network, spectral hypersphere difference index (SHDI), multidirectional consistency, interpretability.

## Method Overview

Pansharpening combines complementary information from high-resolution panchromatic (PAN) images and low-resolution multispectral (MS) images to produce high-resolution multispectral images. VDPN uses separate spatial and spectral streams to extract PAN and MS features, and constrains the fused image through spectral hypersphere difference and multidirectional spatial consistency. See the [paper](https://doi.org/10.1109/TGRS.2026.3732241) for the theoretical formulation and experimental settings. The implementation details below refer to the current repository.

| Method Component | Current Implementation |
| --- | --- |
| Spatial feature extraction | `pan_encoder` and `DilationResidualModule` extract spatial features from the PAN image. |
| Spectral feature extraction | `ms_encoder` encodes upsampled MS images (UPMS). `SwinTransformerModule` and `FiveLayerCNNResidual` process these features in parallel. |
| Feature fusion and reconstruction | `ChannelAttention` and `spec_conv` combine the spectral features. The spatial and spectral streams are concatenated and passed through `decoder` with a sigmoid activation to reconstruct the fused image. |
| Spectral supervision | `SpectralAngleLoss` in `loss/spectral_angle_loss.py` preserves the original `SHDILoss` calculation: it computes the angle between corresponding UPMS and fused-image spectral vectors and averages over pixels and samples. |
| Multidirectional spatial consistency | `EdgeCorrelationLoss` compares PAN and fused-image edges in horizontal, vertical, 45°, and 135° directions using Pearson correlation. |
| Training objective | `FusionLoss` combines the spectral and spatial terms as `0.8 × spectral + 0.2 × spatial` by default. Both terms use all spectral channels, and their weights are configurable. |
| Per-epoch validation | Every epoch in `train.py` first trains on the training set, then evaluates the current model on the validation set with gradients disabled. The checkpoint with the lowest mean validation loss is saved as `best.pth`. |

## Implementation Scope and Reproducibility

This version includes configurable paths and devices, training with all spectral bands, per-epoch validation, and optional standalone validation as part of the preparation for public release. To reproduce the paper's experiments, prepare the data and training settings according to the paper:

- **Input data:** Each split contains matching TIFF files in `MS/`, `PAN/`, and `UPMS/`, named `<id>_ms.tif`, `<id>_pan.tif`, and `<id>_mul.tif`. Subdirectory names are configurable. UPMS must be prepared in advance and registered to PAN; the model takes PAN and UPMS as inputs. The loader also reads MS for dataset consistency checks.
- **Preprocessing and channels:** TIFF channels are independently min-max normalized to `[0, 1]`, with constant channels mapped to zero. Set `--num-channels` to the full MS channel count. PAN and UPMS must have matching spatial dimensions, and samples within a batch must have matching shapes. Registration, upsampling, band alignment, radiometric correction, and nodata masking are not performed automatically.
- **Training and validation:** Prepare separate training and validation scenes before extracting patches. Running `train.py` performs both phases in every epoch and records their mean losses in `metrics.csv`. The optional `validate.py` evaluates a saved checkpoint; it is not required for per-epoch validation.
- **Self-supervised evaluation:** Training and validation share the spectral and spatial objective without requiring high-resolution MS ground truth. Validation losses support model selection; they do not replace reference-based image-quality metrics or the paper's evaluation protocol.
- **Checkpoints:** New checkpoints include model and optimizer states, configuration, and metric history. Older checkpoints must match the current architecture; checkpoints containing removed auxiliary branches cannot be resumed directly. Resume does not restore random-generator states, so it is not guaranteed to reproduce an uninterrupted run's sample order exactly.
- **Results:** Synthetic-data checks verify data loading, all-channel training, validation behavior, checkpoint selection, and resume. The paper's real-data metrics and ablation results have not been re-evaluated with this revised version; refer to the paper for reported results.

`dataload/` contains only `dataset.py` and the package entry point `__init__.py`, used for training and validation. Import the dataset with `from dataload import FusionDataset`. `model/` contains `fusion_net.py` and `__init__.py`; import the model with `from model import FusionNet`. `loss/` contains the shared objective and its two component losses, together with the package entry point.

## Citation

If this project is useful for your research, please cite the following paper:

```bibtex
@article{zhou2026vdpn,
  author  = {Zhou, Q. and Guo, Q.},
  title   = {Variational-Theoretic Dual-Stream Pansharpening Network With Spectral Hypersphere Difference and Multidirectional Consistency},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  volume  = {64},
  pages   = {5408417--5408417},
  note    = {Art. no. 5408417},
  doi     = {10.1109/TGRS.2026.3732241},
  url     = {https://doi.org/10.1109/TGRS.2026.3732241}
}
```

## Release Notes

`.gitignore` excludes data, model weights, output directories, and Python caches. Unused model variants and loss modules have been removed. Runtime dependencies are listed in `requirements.txt`, and automated CPU checks are available in `tests/test_pipeline.py`. A code license file has not yet been added; include the chosen project license before publication.
