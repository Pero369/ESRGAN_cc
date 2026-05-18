"""
通道数敏感性评估脚本
评估 channels=24/32/40 三组模型（均使用 blocks=8, grad=0.15）在 Set5/Set14 上的表现
"""
import torch
import time
import os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips

from models import LightGenerator
from config import Config


EXPERIMENTS = [
    ("channels=24", "CHANNELS_EXPERIMENTS/light_grad0.15_rrdb8_ch24"),
    ("channels=32", "CHANNELS_EXPERIMENTS/light_grad0.15_rrdb8_ch32"),
    ("channels=40", "CHANNELS_EXPERIMENTS/light_grad0.15_rrdb8_ch40"),
]

DATASETS = ["Set5", "Set14"]


def build_generator(channels, device):
    return LightGenerator(8, channels=channels).to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def load_tensor(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_np(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def rgb_to_y(img_np):
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    y = 16 / 255 + (65.481 * r + 128.553 * g + 24.966 * b)
    return y


def calc_psnr_ssim(sr_np, hr_np, y_channel=False):
    if y_channel:
        sr_y = rgb_to_y(sr_np)
        hr_y = rgb_to_y(hr_np)
        p = psnr(hr_y, sr_y, data_range=255.0)
        s = ssim(hr_y, sr_y, data_range=255.0)
    else:
        p = psnr(hr_np, sr_np, data_range=1.0)
        s = ssim(hr_np, sr_np, data_range=1.0, channel_axis=2)
    return p, s


def np_to_lpips(img_np, device):
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2 - 1


def evaluate_one(model, dataset_dir, lpips_fn, device, y_channel=False):
    srf4_dir = os.path.join(dataset_dir, 'image_SRF_4')
    lr_files = sorted(f for f in os.listdir(srf4_dir) if '_LR' in f)

    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for lr_name in lr_files:
            hr_name = lr_name.replace('_LR', '_HR')
            lr_t = load_tensor(os.path.join(srf4_dir, lr_name))
            hr_np = np.array(Image.open(os.path.join(srf4_dir, hr_name)).convert('RGB')).astype(np.float32) / 255.0

            sr_t = model(lr_t.to(device))
            sr_np = tensor_to_np(sr_t)

            h, w = hr_np.shape[:2]
            sr_np = sr_np[:h, :w]

            p, s = calc_psnr_ssim(sr_np, hr_np, y_channel)
            lp = lpips_fn(np_to_lpips(sr_np, device), np_to_lpips(hr_np, device)).item()

            psnr_list.append(p)
            ssim_list.append(s)
            lpips_list.append(lp)

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    all_rows = []

    for label, exp_dir in EXPERIMENTS:
        ckpt_path = os.path.join(Config.checkpoint_dir, exp_dir, 'generator_gan_150.pth')
        if not os.path.exists(ckpt_path):
            print(f"\n[跳过] 未找到权重: {ckpt_path}")
            continue

        channels = int(label.split('=')[1])

        print(f"\n{'='*60}")
        print(f"评估: {label} (channels={channels}, blocks=8, grad=0.15)")
        print(f"权重: {ckpt_path}")
        print(f"{'='*60}")

        model = build_generator(channels, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        params_m = count_params(model) / 1e6
        infer_ms = measure_inference(model, device)

        print(f"参数量: {params_m:.2f}M  推理时间: {infer_ms:.1f}ms")

        for ds_name in DATASETS:
            ds_dir = os.path.join('data', ds_name)
            if not os.path.isdir(ds_dir):
                print(f"  [跳过] 数据集不存在: {ds_dir}")
                continue

            psnr_rgb, ssim_rgb, lpips_rgb = evaluate_one(model, ds_dir, lpips_fn, device, y_channel=False)
            psnr_y, ssim_y, lpips_y = evaluate_one(model, ds_dir, lpips_fn, device, y_channel=True)

            print(f"  {ds_name}  RGB: PSNR={psnr_rgb:.2f}  SSIM={ssim_rgb:.4f}  LPIPS={lpips_rgb:.4f}")
            print(f"  {ds_name}  Y:   PSNR={psnr_y:.2f}  SSIM={ssim_y:.4f}  LPIPS={lpips_y:.4f}")

            all_rows.append({
                'channels': channels,
                'dataset': ds_name,
                'channel': 'RGB',
                'PSNR': f'{psnr_rgb:.2f}',
                'SSIM': f'{ssim_rgb:.4f}',
                'LPIPS': f'{lpips_rgb:.4f}',
                'params_M': f'{params_m:.2f}',
                'infer_ms': f'{infer_ms:.1f}',
            })
            all_rows.append({
                'channels': channels,
                'dataset': ds_name,
                'channel': 'Y',
                'PSNR': f'{psnr_y:.2f}',
                'SSIM': f'{ssim_y:.4f}',
                'LPIPS': f'{lpips_y:.4f}',
                'params_M': f'{params_m:.2f}',
                'infer_ms': f'{infer_ms:.1f}',
            })

    output_csv = './results/.csv/channels_results.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['channels', 'dataset', 'channel', 'PSNR', 'SSIM', 'LPIPS', 'params_M', 'infer_ms'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n结果已保存: {output_csv}")


if __name__ == '__main__':
    main()
