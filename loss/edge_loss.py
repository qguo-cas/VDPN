import torch
import torch.nn as nn
import torch.nn.functional as F

def replicate_pan(pan, num_channels):
    # 将 pan 从 1 通道复制到指定的 num_channels
    return pan.repeat(1, num_channels, 1, 1)

def edge_detection(img, direction):
    if direction == 'horizontal':
        kernel = torch.tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=torch.float32)
    elif direction == 'vertical':
        kernel = torch.tensor([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=torch.float32)
    elif direction == '45':
        kernel = torch.tensor([[0, -1, -2],
                               [1, 0, -1],
                               [2, 1, 0]], dtype=torch.float32)
    elif direction == '135':
        kernel = torch.tensor([[2, 1, 0],
                               [1, 0, -1],
                               [0, -1, -2]], dtype=torch.float32)
    else:
        raise ValueError("Invalid direction")
    # 转换为卷积核形状：[out_channels, in_channels, kH, kW]
    kernel = kernel.view(1, 1, 3, 3).to(img.device)
    kernels = kernel.repeat(img.size(1), 1, 1, 1)
    edge = F.conv2d(img, kernels, padding=1, groups=img.size(1))
    return edge

def pearson_corr(x, y, epsilon=1e-8):
    # 计算 x 和 y 在每个通道的 Pearson 相关系数
    x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
    y_mean = torch.mean(y, dim=[2, 3], keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = torch.sum(x_centered * y_centered, dim=[2, 3])
    denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=[2, 3]) * torch.sum(y_centered ** 2, dim=[2, 3]) + epsilon)
    r = numerator / denominator
    # 对通道求均值，得到每个样本的平均 Pearson 相关系数
    r_mean = torch.mean(r, dim=1)
    return r_mean

class EdgeCorrelationLoss(nn.Module):

    def __init__(self, directions=None, epsilon=1e-8):
        super(EdgeCorrelationLoss, self).__init__()
        if directions is None:
            directions = ['horizontal', 'vertical', '45', '135']
        self.directions = directions
        self.epsilon = epsilon

    def forward(self, pan, hrms):

        # 将 pan 扩展到与 hrms 通道数相同
        pan_rep = replicate_pan(pan, hrms.size(1))
        total_corr = 0
        for direction in self.directions:
            edge_pan = edge_detection(pan_rep, direction)
            edge_hrms = edge_detection(hrms, direction)
            r = pearson_corr(edge_pan, edge_hrms, self.epsilon)
            total_corr += r
        avg_corr = total_corr / len(self.directions)  # [B]
        loss = torch.mean((1 - avg_corr) ** 2)
        return loss
