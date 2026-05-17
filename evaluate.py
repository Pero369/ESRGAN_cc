import torch
import numpy as np
import os
import argparse
import csv
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
from models import LightGenerator
from config import Config


def load_tensor(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0), img


def tensor_to_np(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def rgb_to_y(img_np):
    # ITU-R BT.601，与超分领域标准一致
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    y = 16/255 + (65.481*r + 128.553*g + 24.966*b)
    return y


def calc_metrics(sr_np, hr_np, y_channel=False):
    if y_channel:
        sr_np = rgb_to_y(sr_np)
        hr_np = rgb_to_y(hr_np)
        p = psnr(hr_np, sr_np, data_range=255.0)
        s = ssim(hr_np, sr_np, data_range=255.0)
    else:
        p = psnr(hr_np, sr_np, data_range=1.0)
        s = ssim(hr_np, sr_np, data_range=1.0, channel_axis=2)
    return p, s


def np_to_lpips_tensor(img_np, device):
    # LPIPS 期望 [-1, 1] 范围的 (1, C, H, W) tensor
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2 - 1


def evaluate(model_path, dataset_dir, output_csv=None, y_channel=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = LightGenerator(num_rrdb=Config.light_num_rrdb_blocks, channels=Config.light_num_channels).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()

    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    srf4_dir = os.path.join(dataset_dir, 'image_SRF_4')
    lr_files = sorted(f for f in os.listdir(srf4_dir) if '_LR' in f)

    rows = []
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for lr_name in lr_files:
            hr_name = lr_name.replace('_LR', '_HR')
            lr_t, _ = load_tensor(os.path.join(srf4_dir, lr_name))
            hr_np = np.array(Image.open(os.path.join(srf4_dir, hr_name)).convert('RGB')).astype(np.float32) / 255.0

            sr_t = gen(lr_t.to(device))
            sr_np = tensor_to_np(sr_t)

            h, w = hr_np.shape[:2]
            sr_np = sr_np[:h, :w]

            p, s = calc_metrics(sr_np, hr_np, y_channel)
            lp = lpips_fn(np_to_lpips_tensor(sr_np, device), np_to_lpips_tensor(hr_np, device)).item()

            psnr_list.append(p)
            ssim_list.append(s)
            lpips_list.append(lp)
            rows.append({'image': lr_name, 'PSNR': f'{p:.4f}', 'SSIM': f'{s:.4f}', 'LPIPS': f'{lp:.4f}'})
            print(f'{lr_name}: PSNR={p:.2f} dB, SSIM={s:.4f}, LPIPS={lp:.4f}')

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    print(f'\n平均 PSNR:  {avg_psnr:.4f} dB')
    print(f'平均 SSIM:  {avg_ssim:.4f}')
    print(f'平均 LPIPS: {avg_lpips:.4f}')

    if output_csv:
        rows.append({'image': 'AVERAGE', 'PSNR': f'{avg_psnr:.4f}', 'SSIM': f'{avg_ssim:.4f}', 'LPIPS': f'{avg_lpips:.4f}'})
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'PSNR', 'SSIM', 'LPIPS'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'结果已保存到 {output_csv}')

    return avg_psnr, avg_ssim, avg_lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True, help='Set5 或 Set14 目录')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--y_channel', action='store_true', help='在Y通道计算指标（与公开论文数据对齐）')
    args = parser.parse_args()
    evaluate(args.model_path, args.dataset_dir, args.output_csv, args.y_channel)
