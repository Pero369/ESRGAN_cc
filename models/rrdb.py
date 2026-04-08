import torch
import torch.nn as nn
from .attention import ChannelAttention, SpatialAttention, CBAM

class DenseBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels * 3, channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels * 4, channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels * 5, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.beta = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * self.beta + x

class RRDB(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.dense1 = DenseBlock(channels)
        self.dense2 = DenseBlock(channels)
        self.dense3 = DenseBlock(channels)
        self.beta = 0.2

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * self.beta + x


# ---- 轻量化版本 ----

class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class LightDenseBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = DSConv(channels, channels)
        self.conv2 = DSConv(channels * 2, channels)
        self.conv3 = DSConv(channels * 3, channels)
        self.conv4 = DSConv(channels * 4, channels)
        self.conv5 = DSConv(channels * 5, channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.beta = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * self.beta + x


class LightRRDB(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.dense1 = LightDenseBlock(channels)
        self.dense2 = LightDenseBlock(channels)
        self.dense3 = LightDenseBlock(channels)
        self.beta = 0.2

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * self.beta + x


# ---- 带注意力机制的轻量化版本 ----

class LightDenseBlockWithCA(nn.Module):
    """带通道注意力的轻量DenseBlock"""
    def __init__(self, channels=32, reduction=16):
        super().__init__()
        self.conv1 = DSConv(channels, channels)
        self.conv2 = DSConv(channels * 2, channels)
        self.conv3 = DSConv(channels * 3, channels)
        self.conv4 = DSConv(channels * 4, channels)
        self.conv5 = DSConv(channels * 5, channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.ca = ChannelAttention(channels, reduction)
        self.beta = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        x5 = self.ca(x5)  # 在DenseBlock输出后应用通道注意力
        return x5 * self.beta + x


class LightRRDBWithCA(nn.Module):
    """带通道注意力的轻量RRDB"""
    def __init__(self, channels=32, reduction=16, attention_position='rrdb'):
        """
        Args:
            channels: 通道数
            reduction: 注意力降维比例
            attention_position: 'dense' - 在每个DenseBlock后加注意力
                              'rrdb' - 在整个RRDB后加注意力（推荐，参数更少）
        """
        super().__init__()
        if attention_position == 'dense':
            # 在DenseBlock级别添加注意力
            self.dense1 = LightDenseBlockWithCA(channels, reduction)
            self.dense2 = LightDenseBlockWithCA(channels, reduction)
            self.dense3 = LightDenseBlockWithCA(channels, reduction)
            self.ca = None
        else:
            # 在RRDB级别添加注意力（推荐）
            self.dense1 = LightDenseBlock(channels)
            self.dense2 = LightDenseBlock(channels)
            self.dense3 = LightDenseBlock(channels)
            self.ca = ChannelAttention(channels, reduction)

        self.beta = 0.2

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        if self.ca is not None:
            out = self.ca(out)  # 在RRDB输出后应用通道注意力
        return out * self.beta + x


class LightRRDBWithCBAM(nn.Module):
    """带CBAM注意力的轻量RRDB"""
    def __init__(self, channels=32, reduction=16):
        super().__init__()
        self.dense1 = LightDenseBlock(channels)
        self.dense2 = LightDenseBlock(channels)
        self.dense3 = LightDenseBlock(channels)
        self.cbam = CBAM(channels, reduction)
        self.beta = 0.2

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.cbam(out)
        return out * self.beta + x
