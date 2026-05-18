"""
FLOPs / 参数量 / 推理时间 统一测量脚本
"""
import torch
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
