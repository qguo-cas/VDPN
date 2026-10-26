import torch
import torch.nn as nn
import torch.nn.functional as F


class FiveLayerCNNResidual(nn.Module):
    def __init__(self):
        super(FiveLayerCNNResidual, self).__init__()
        # 第一层：输入32通道 -> 输出32通道
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()

        # 第二层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()

        # 第三层
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()

        # 第四层
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.prelu4 = nn.PReLU()

        # 第五层：卷积后加入残差连接，再进行PReLU激活
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.prelu5 = nn.PReLU()

    def forward(self, x):
        # x 的尺寸：[B, 32, 256, 256]
        out = self.prelu1(self.conv1(x))
        out = self.prelu2(self.conv2(out))
        out = self.prelu3(self.conv3(out))
        out = self.prelu4(self.conv4(out))
        out = self.conv5(out)

        out = self.prelu5(out)
        return out

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口还原为完整特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始高度
        W (int): 原始宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """窗口多头注意力机制"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # 生成相对位置索引
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))  # 2, Ws, Ws
        coords_flatten = torch.flatten(coords, 1)  # 2, Ws^2
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ws^2, Ws^2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ws^2, Ws^2, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Ws^2, Ws^2
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size, -1)  # Ws^2, Ws^2, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Ws^2, Ws^2
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer 基本块"""

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        shortcut = x

        # 移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Ws, Ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Ws^2, C]

        # 窗口注意力
        attn_windows = self.attn(x_windows)  # [nW*B, Ws^2, C]

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

        # 逆向移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 残差连接
        x = self.norm1(x + shortcut)

        # MLP
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x


class SwinTransformerModule(nn.Module):
    """完整的Swin Transformer模块"""

    def __init__(self, in_channels=4, embed_dim=64, num_heads=4,
                 window_size=8, num_layers=2):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            # 交替使用移位窗口
            shift_size = window_size // 2 if i % 2 == 0 else 0
            self.blocks.append(
                SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=(256, 256),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size
                )
            )
        self.unembed = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        # 输入维度: [B, C, H, W]
        x = self.embed(x)  # 通道嵌入

        # 通过多个Swin块
        for blk in self.blocks:
            x = blk(x)

        # 恢复原始通道数
        x = self.unembed(x)
        return x

class DilationResidualModule(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(DilationResidualModule, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=2, dilation=2)
        self.conv_3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=3, dilation=3)
        self.conv_4 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=4, dilation=4)
        self.conv_5 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=5, dilation=5)
        # 针对每个扩张率构造一个卷积分支：保持通道数1

        # 使用ReLU激活函数
        self.activation = nn.PReLU()
        self.shp_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        # 输入 x 的尺寸: [B, 1, 256, 256]
        #1th
        conv_out1 = self.activation(self.conv_1(x))
        conv_in2 = torch.cat([x, conv_out1], dim=1)
        #2th
        conv_out2 = self.activation(self.conv_2(conv_in2))
        conv_in3 = torch.cat([x, conv_out2], dim=1)
        #3th
        conv_out3 = self.activation(self.conv_3(conv_in3))
        conv_in4 = torch.cat([x, conv_out3], dim=1)
        #4th
        conv_out4 = self.activation(self.conv_4(conv_in4))
        conv_in5 = torch.cat([x, conv_out4], dim=1)
        #5th
        conv_out5 = self.activation(self.conv_4(conv_in5))
        conv_shp = torch.cat([x, conv_out5], dim=1)
        out=self.shp_conv(conv_shp)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        channel_weights = self.fc(avg_pool).view(x.size(0), -1, 1, 1)
        x = x * channel_weights
        return x


# 2. 新增辅助模块
####################################

# 两层CNN编码器（可用于pan_encoder和ms_encoder）
class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.PReLU()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x

####################################
# 3. 主网络构建
####################################
class FusionNet(nn.Module):
    def __init__(self, num_channels=4):
        super(FusionNet, self).__init__()
        if num_channels < 1:
            raise ValueError("num_channels must be positive")
        self.num_channels = num_channels
        # Encode the single-channel panchromatic input.
        self.pan_encoder = CNNEncoder(in_channels=1, out_channels=32)
        # 空间特征提取模块
        self.spatial_module = DilationResidualModule(in_channels=32, out_channels=32)

        # Encode all multispectral input channels.
        self.ms_encoder = CNNEncoder(in_channels=num_channels, out_channels=32)
        # Swin Transformer分支
        self.ms_swin = SwinTransformerModule(
            in_channels=32,
            embed_dim=64,
            num_heads=4,
            window_size=8,
            num_layers=5
        )
        # 深度CNN分支
        self.ms_dcnn = FiveLayerCNNResidual()  # 假设其已适配32通道输入

        # 融合MS分支：拼接ms_swin与ms_dcnn后，通道数=32+32=64，先进行通道注意力再降维到32通道
        self.channel_att = ChannelAttention(in_channels=64, reduction=16)
        self.spec_conv = nn.Conv2d(64, 32, kernel_size=1)

        # Decode fused features into all multispectral channels.
        self.decoder = nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self, pan, ms):
        if pan.ndim != 4 or ms.ndim != 4:
            raise ValueError("PAN and MS must have shape [B, C, H, W].")
        if pan.shape[1] != 1 or ms.shape[1] != self.num_channels:
            raise ValueError(f"Expected one PAN channel and {self.num_channels} MS channels.")
        if pan.shape[0] != ms.shape[0] or pan.shape[2:] != ms.shape[2:]:
            raise ValueError("PAN and upsampled MS must have matching batch and spatial dimensions.")
        height, width = ms.shape[-2:]
        padding = (0, (-width) % 8, 0, (-height) % 8)
        pan = F.pad(pan, padding, mode="replicate")
        ms = F.pad(ms, padding, mode="replicate")
        # PAN分支
        pan_enc = self.pan_encoder(pan)  # [B, 32, 256, 256]
        F_spa = self.spatial_module(pan_enc)  # [B, 32, 256, 256]

        # MS分支
        ms_enc = self.ms_encoder(ms)  # [B, 32, 256, 256]
        # 分支a：Swin Transformer特征提取
        ms_swtran = self.ms_swin(ms_enc)  # [B, 32, 256, 256]
        # 分支b：深度CNN特征提取
        ms_dcnn = self.ms_dcnn(ms_enc)  # [B, 32, 256, 256]
        # 拼接两支(ms_swtran与ms_dcnn) -> [B, 64, 256, 256]
        ms_cat = torch.cat([ms_swtran, ms_dcnn], dim=1)
        # 通道注意力机制进行光谱特征精炼
        ms_ca = self.channel_att(ms_cat)  # [B, 64, 256, 256]
        F_spe = self.spec_conv(ms_ca)  # 降维到32通道 -> [B, 32, 256, 256]

        # 融合空间与光谱特征
        fused = torch.cat([F_spa, F_spe], dim=1)  # [B, 64, 256, 256]
        # Decode the fused multispectral image.
        hrms = self.sigmoid(self.decoder(fused))  # [B, C, H, W]

        return hrms[..., :height, :width]
