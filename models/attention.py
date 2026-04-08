"""
注意力机制模块
包含通道注意力（SE）、空间注意力和CBAM
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块（Squeeze-and-Excitation）
    通过全局平均池化和两层全连接网络学习通道权重
    """
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入特征通道数
            reduction: 降维比例，控制中间层通道数
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x).view(b, c)
        # Excitation: 学习通道权重 [B, C] -> [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 通道加权
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    通过通道维度的平均和最大池化学习空间权重
    """
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小，用于空间特征提取
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的平均和最大池化 [B, C, H, W] -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接 [B, 2, H, W]
        y = torch.cat([avg_out, max_out], dim=1)
        # 学习空间权重 [B, 2, H, W] -> [B, 1, H, W]
        y = self.sigmoid(self.conv(y))
        # Scale: 空间加权
        return x * y


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Args:
            channels: 输入特征通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先通道注意力，再空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class LightChannelAttention(nn.Module):
    """
    轻量级通道注意力
    使用1D卷积替代全连接层，减少参数量
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y
