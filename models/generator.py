import torch
import torch.nn as nn
from .rrdb import RRDB

class Generator(nn.Module):
    def __init__(self, num_rrdb=23, channels=64, enable_cbam=False):
        super().__init__()
        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(channels, enable_cbam=enable_cbam) for _ in range(num_rrdb)])
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
