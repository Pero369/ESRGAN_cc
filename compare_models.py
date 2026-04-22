"""
多模型横向对比脚本
自动发现 checkpoints/ 下的模型，批量评估 PSNR/SSIM/参数量/推理速度
"""
import torch
import time
import os
import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from models import LightGenerator, Generator
from config import Config


# ── 模型配置注册表 ──────────────────────────────────────────────
# 每条记录描述一个实验目录对应的模型构建参数
MODEL_REGISTRY = {
    # COMPREHENSIVE_EXPERIMENTS
    "COMPREHENSIVE_EXPERIMENTS/original":                  dict(light=False, attention=False, gradient=False),
    "COMPREHENSIVE_EXPERIMENTS/light":                     dict(light=True,  attention=False, gradient=False),
    "COMPREHENSIVE_EXPERIMENTS/light_att_CA_rrdb":         dict(light=True,  attention=True,  att_type="CA",   att_pos="rrdb",  gradient=False),
    "COMPREHENSIVE_EXPERIMENTS/light_grad0.1":             dict(light=True,  attention=False, gradient=True,   lambda_grad=0.1),
    "COMPREHENSIVE_EXPERIMENTS/light_att_CA_rrdb_grad0.1": dict(light=True,  attention=True,  att_type="CA",   att_pos="rrdb",  gradient=True, lambda_grad=0.1),
    "COMPREHENSIVE_EXPERIMENTS/enable_attentionFalse":     dict(light=True,  attention=False, gradient=False),
    # ABLATION_EXPERIMENTS
    "ABLATION_EXPERIMENTS/enable_attentionFalse_enable_gradient_lossFalse": dict(light=True, attention=False, gradient=False),
    "ABLATION_EXPERIMENTS/att_CA_rrdb":                    dict(light=True,  attention=True,  att_type="CA",   att_pos="rrdb",  gradient=False),
    "ABLATION_EXPERIMENTS/grad0.1":                        dict(light=True,  attention=False, gradient=True,   lambda_grad=0.1),
    "ABLATION_EXPERIMENTS/att_CA_rrdb_grad0.1":            dict(light=True,  attention=True,  att_type="CA",   att_pos="rrdb",  gradient=True, lambda_grad=0.1),
    # ATTENTION_EXPERIMENTS
    "ATTENTION_EXPERIMENTS/enable_attentionFalse":         dict(light=True,  attention=False, gradient=False),
    "ATTENTION_EXPERIMENTS/att_CA_dense":                  dict(light=True,  attention=True,  att_type="CA",   att_pos="dense", gradient=False),
    "ATTENTION_EXPERIMENTS/att_CA_rrdb":                   dict(light=True,  attention=True,  att_type="CA",   att_pos="rrdb",  gradient=False),
    "ATTENTION_EXPERIMENTS/att_CBAM_rrdb":                 dict(light=True,  attention=True,  att_type="CBAM", att_pos="rrdb",  gradient=False),
    # GRADIENT_LOSS_EXPERIMENTS
    "GRADIENT_LOSS_EXPERIMENTS/enable_gradient_lossFalse": dict(light=True,  attention=False, gradient=False),
    "GRADIENT_LOSS_EXPERIMENTS/grad0.05":                  dict(light=True,  attention=False, gradient=True,   lambda_grad=0.05),
    "GRADIENT_LOSS_EXPERIMENTS/grad0.1":                   dict(light=True,  attention=False, gradient=True,   lambda_grad=0.1),
    "GRADIENT_LOSS_EXPERIMENTS/grad0.2":                   dict(light=True,  attention=False, gradient=True,   lambda_grad=0.2),
}


def build_model(cfg: dict, device):
    """根据配置字典构建模型"""
    if cfg["light"]:
        model = LightGenerator(
            Config.light_num_rrdb_blocks,
            Config.light_num_channels,
            enable_attention=cfg.get("attention", False),
            attention_type=cfg.get("att_type", "CA"),
            attention_reduction=Config.attention_reduction,
            attention_position=cfg.get("att_pos", "rrdb"),
        )
    else:
        model = Generator(Config.num_rrdb_blocks, Config.num_channels)
    return model.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_time(model, device, size=256, runs=30, warmup=5):
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
    return (time.time() - t0) / runs * 1000  # ms


# ── 图像质量评估 ────────────────────────────────────────────────
def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(a, b):
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(a, b, channel_axis=2, data_range=255)
    except ImportError:
        return None


def to_np(t):
    return np.clip(t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)


def evaluate_quality(model, test_dir, device):
    """返回 (avg_psnr, avg_ssim)，test_dir 内放 HR 图像"""
    import torchvision.transforms.functional as TF

    files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        return None, None

    psnr_list, ssim_list = [], []
    model.eval()
    for fname in tqdm(files, desc="  评估", leave=False):
        hr = TF.to_tensor(Image.open(os.path.join(test_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
        h, w = hr.shape[2], hr.shape[3]
        lr = TF.resize(hr, (h // 4, w // 4), interpolation=Image.BICUBIC)
        with torch.no_grad():
            sr = model(lr)
        hr_np, sr_np = to_np(hr), to_np(sr)
        mh, mw = min(hr_np.shape[0], sr_np.shape[0]), min(hr_np.shape[1], sr_np.shape[1])
        hr_np, sr_np = hr_np[:mh, :mw], sr_np[:mh, :mw]
        psnr_list.append(psnr(hr_np, sr_np))
        s = ssim(hr_np, sr_np)
        if s is not None:
            ssim_list.append(s)

    return np.mean(psnr_list), (np.mean(ssim_list) if ssim_list else None)


# ── 主流程 ──────────────────────────────────────────────────────
def find_checkpoint(exp_dir, checkpoint_root, prefer_gan=True):
    """在 exp_dir 下找最新的 generator_*.pth"""
    full = os.path.join(checkpoint_root, exp_dir)
    if not os.path.isdir(full):
        return None
    pths = sorted(f for f in os.listdir(full) if f.startswith("generator_") and f.endswith(".pth"))
    if not pths:
        return None
    if prefer_gan:
        gan_pths = [p for p in pths if "gan" in p]
        pths = gan_pths if gan_pths else pths
    return os.path.join(full, pths[-1])


def run_comparison(exp_keys, checkpoint_root, test_dir, device, output_csv, bench_size=256):
    rows = []
    print(f"\n{'模型':<45} {'参数量':>8} {'推理(ms)':>10} {'PSNR':>8} {'SSIM':>8}")
    print("-" * 85)

    for key in exp_keys:
        cfg = MODEL_REGISTRY.get(key)
        if cfg is None:
            print(f"[跳过] 未在注册表中找到: {key}")
            continue

        ckpt = find_checkpoint(key, checkpoint_root)
        if ckpt is None:
            print(f"[跳过] 未找到权重: {key}")
            continue

        model = build_model(cfg, device)
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        except Exception as e:
            print(f"[跳过] 加载失败 {key}: {e}")
            continue

        params = count_params(model)
        infer_ms = measure_time(model, device, size=bench_size)

        avg_psnr, avg_ssim = (None, None)
        if test_dir and os.path.isdir(test_dir):
            avg_psnr, avg_ssim = evaluate_quality(model, test_dir, device)

        name = key.split("/")[-1]
        psnr_str = f"{avg_psnr:.2f}" if avg_psnr else "N/A"
        ssim_str = f"{avg_ssim:.4f}" if avg_ssim else "N/A"
        print(f"{name:<45} {params/1e6:>6.2f}M {infer_ms:>10.1f} {psnr_str:>8} {ssim_str:>8}")

        rows.append({
            "实验": key,
            "参数量(M)": f"{params/1e6:.2f}",
            "推理时间(ms)": f"{infer_ms:.1f}",
            "PSNR": psnr_str,
            "SSIM": ssim_str,
            "权重文件": os.path.basename(ckpt),
        })

    print("-" * 85)

    if output_csv and rows:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n结果已保存: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模型横向对比")
    parser.add_argument("--checkpoint_root", default="./checkpoints")
    parser.add_argument("--test_dir", default="", help="HR测试图像目录（留空则跳过质量评估）")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--bench_size", type=int, default=256, help="推理速度测试的输入分辨率")
    parser.add_argument("--output_csv", default="./results/comparison.csv")
    parser.add_argument(
        "--group",
        choices=["all", "comprehensive", "ablation", "attention", "gradient"],
        default="comprehensive",
        help="要对比的实验组",
    )
    parser.add_argument("--keys", nargs="+", help="手动指定实验 key（覆盖 --group）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    GROUP_MAP = {
        "comprehensive": [k for k in MODEL_REGISTRY if k.startswith("COMPREHENSIVE")],
        "ablation":      [k for k in MODEL_REGISTRY if k.startswith("ABLATION")],
        "attention":     [k for k in MODEL_REGISTRY if k.startswith("ATTENTION")],
        "gradient":      [k for k in MODEL_REGISTRY if k.startswith("GRADIENT")],
        "all":           list(MODEL_REGISTRY.keys()),
    }

    keys = args.keys if args.keys else GROUP_MAP[args.group]
    run_comparison(keys, args.checkpoint_root, args.test_dir or None, device, args.output_csv, args.bench_size)
