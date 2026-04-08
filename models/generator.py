import torch
import torch.nn as nn
from .rrdb import RRDB, LightRRDB, LightRRDBWithCA, LightRRDBWithCBAM

class Generator(nn.Module):
    def __init__(self, num_rrdb=23, channels=64):
        super().__init__()
        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(channels) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

        self.upconv1 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.rrdb_blocks(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle(self.upconv2(feat)))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class LightGenerator(nn.Module):
    def __init__(self, num_rrdb=8, channels=32, enable_attention=False,
                 attention_type='CA', attention_reduction=16, attention_position='rrdb'):
        """
        轻量化生成器
        Args:
            num_rrdb: RRDB块数量
            channels: 特征通道数
            enable_attention: 是否启用注意力机制
            attention_type: 注意力类型 'CA' 或 'CBAM'
            attention_reduction: 注意力降维比例
            attention_position: 注意力位置 'dense' 或 'rrdb'
        """
        super().__init__()
        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)

        # 根据配置选择RRDB类型
        if enable_attention:
            if attention_type == 'CBAM':
                rrdb_block = lambda: LightRRDBWithCBAM(channels, attention_reduction)
            else:  # 'CA'
                rrdb_block = lambda: LightRRDBWithCA(channels, attention_reduction, attention_position)
        else:
            rrdb_block = lambda: LightRRDB(channels)

        self.rrdb_blocks = nn.Sequential(*[rrdb_block() for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

        self.upconv1 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.rrdb_blocks(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle(self.upconv2(feat)))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
