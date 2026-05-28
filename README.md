# VDPN
Variational Dual-Stream Pansharpening Network with Spectral Hypersphere Difference and Multi-Directional Consistency (VDPN)

Paper Status
Under review.

Overview

This repository hosts VDPN, a variationally grounded, self-supervised pansharpening framework that targets two common challenges in deep methods: limited interpretability and insufficient pixel-wise multi-band spectral consistency and orientation consistency. VDPN follows a dual-stream design—one spatial, one spectral—and reconstructs high-spatial-resolution multispectral (HRMS) images while preserving spectral fidelity.

Method

Variational design. HRMS is modeled as the synthesis of PAN-driven spatial and MS-driven spectral features; UPMS and PAN are modeled as observations of HRMS under spatial and spectral degradations, respectively.

Dual streams.

Spatial stream: dilated convolutions + residual blocks to refine multi-scale edges and textures from PAN.

Spectral stream: Swin Transformer blocks + convolutional layers with a spectral-attention head to extract and refine MS spectral representations.

Decoder. Fuses spatial and spectral features to reconstruct HRMS.

Losses.

MDC (Multi-Directional Consistency): aligns PAN–HRMS edge responses at 0°, 45°, 90°, and 135° to preserve orientation-consistent structures.

SHDI (Spectral Hypersphere Difference Index): enforces pixel-wise multi-band consistency by penalizing angular discrepancies between normalized spectral vectors.

Extensive experiments show that VDPN achieves strong spectral fidelity and spatial detail, outperforming compared methods on standard metrics.

Code Availability

Code and scripts are available here: https://github.com/zqz001/VDPN

Contact

Lead author: Zhou Qingze — zhouqingze22@mails.ucas.ac.cn

Thank you for your interest!
