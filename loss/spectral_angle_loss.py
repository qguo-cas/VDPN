import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAngleLoss(nn.Module):
    """
    利用像素点光谱向量之间的余弦相似度计算角度差，
    并返回整个批次的平均角度差作为损失值。
    """

    def __init__(self):
        super(SpectralAngleLoss, self).__init__()

    def forward(self, ms, hrms):
        """
        参数:
            ms: 多光谱图像，shape [B, C, H, W]
            hrms: 融合结果图像，shape [B, C, H, W]
        返回:
            标量损失值: 整个批次像素点角度差的平均值
        """
        # 将图像从 [B, C, H, W] 转换成 [B, H, W, C]
        ms_flat = ms.permute(0, 2, 3, 1)  # [B, H, W, C]
        hrms_flat = hrms.permute(0, 2, 3, 1)  # [B, H, W, C]

        # 计算每个像素点的余弦相似度
        dot_product = torch.sum(ms_flat * hrms_flat, dim=-1)  # [B, H, W]
        ms_norm = torch.norm(ms_flat, dim=-1)  # [B, H, W]
        hrms_norm = torch.norm(hrms_flat, dim=-1)  # [B, H, W]
        cos_similarity = dot_product / (ms_norm * hrms_norm + 1e-8)

        # 将余弦相似度转换为角度差（弧度），取值范围 [0, π]
        angle = torch.acos(torch.clamp(cos_similarity, -1 + 1e-7, 1 - 1e-7))

        # 计算整个批次的平均角度差，作为损失值
        mean_angle_difference = torch.mean(angle)
        return mean_angle_difference
