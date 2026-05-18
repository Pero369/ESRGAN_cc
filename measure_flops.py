"""
FLOPs / 参数量 / 推理时间 统一测量脚本
"""
import torch
import torch.nn as nn
import time
import csv
import os
from thop import profile
from models import LightGenerator, Generator
from config import Config


def measure_inference(model, device, size=256, runs=30, warmup=5):
    x = torch.randn(1, 3, size, size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000


# ---- DSC消融：标准卷积版 DenseBlock ----
class StdConvDenseBlock(nn.Module):
    """使用标准卷积的DenseBlock（与LightDenseBlock结构相同，仅替换DSC为标准Conv）"""
    def __init__(self, channels=32):
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


class StdConvRRDB(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.dense1 = StdConvDenseBlock(channels)
        self.dense2 = StdConvDenseBlock(channels)
        self.dense3 = StdConvDenseBlock(channels)
        self.beta = 0.2

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * self.beta + x


class StdConvGenerator(nn.Module):
    """使用标准卷积的轻量生成器（用于DSC消融对比）"""
    def __init__(self, num_rrdb=8, channels=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[StdConvRRDB(channels) for _ in range(num_rrdb)])
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


def measure(name, model, device):
    model.eval()
    x = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(x,), verbose=False)
    infer = measure_inference(model, device)
    return {
        "模型": name,
        "参数量(M)": f"{params/1e6:.2f}",
        "FLOPs(G)": f"{flops/1e9:.2f}",
        "推理时间(ms)": f"{infer:.1f}",
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    rows = []

    # 原版
    print("测量: 原版ESRGAN ...")
    m = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)
    rows.append(measure("原版ESRGAN", m, device))

    # DSC消融：标准卷积 vs DSC
    print("测量: 标准卷积版 (blocks=8,ch=32,无DSC) ...")
    m_std = StdConvGenerator(8, 32).to(device)
    rows.append(measure("标准卷积版 (blocks=8,ch=32)", m_std, device))

    # 轻量版各变体
    variants = [
        ("轻量版 基线 (blocks=8,ch=32)", 8, 32, False, "CA", "dense"),
        ("轻量版 + CA Dense级",          8, 32, True,  "CA", "dense"),
        ("轻量版 + CA RRDB级",          8, 32, True,  "CA", "rrdb"),
        ("轻量版 + CBAM RRDB级",        8, 32, True,  "CBAM", "rrdb"),
        ("轻量版 blocks=4",             4, 32, False, "CA", "dense"),
        ("轻量版 blocks=6",             6, 32, False, "CA", "dense"),
        ("轻量版 blocks=10",            10, 32, False, "CA", "dense"),
        ("轻量版 channels=24",          8, 24, False, "CA", "dense"),
        ("轻量版 channels=40",          8, 40, False, "CA", "dense"),
    ]

    for name, blocks, ch, att, att_type, att_pos in variants:
        print(f"测量: {name} ...")
        m = LightGenerator(blocks, ch, enable_attention=att,
                          attention_type=att_type, attention_position=att_pos).to(device)
        rows.append(measure(name, m, device))

    # 打印表格
    print(f"\n{'模型':<32} {'参数量(M)':>10} {'FLOPs(G)':>10} {'推理(ms)':>10}")
    print("-" * 67)
    for r in rows:
        print(f"{r['模型']:<32} {r['参数量(M)']:>10} {r['FLOPs(G)']:>10} {r['推理时间(ms)']:>10}")

    # 保存
    output_csv = "./results/.csv/model_flops.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["模型", "参数量(M)", "FLOPs(G)", "推理时间(ms)"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已保存: {output_csv}")
