import torch
import argparse
import os
import time
import numpy as np
from models import Generator, LightGenerator
from utils import save_image
from config import Config
from PIL import Image


def sr_tile(generator, lr_img, tile=256, overlap=16):
    if lr_img.device.type == 'cpu':
        return generator(lr_img)
    b, c, h, w = lr_img.shape
    scale = 4
    out = torch.zeros(b, c, h * scale, w * scale, device=lr_img.device)
    step = tile - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, x1 = min(y, h - tile), min(x, w - tile)
            y2, x2 = y1 + tile, x1 + tile
            sr_patch = generator(lr_img[:, :, y1:y2, x1:x2])
            py = overlap * scale // 2 if y1 > 0 else 0
            px = overlap * scale // 2 if x1 > 0 else 0
            out[:, :, y1*scale+py:y2*scale, x1*scale+px:x2*scale] = sr_patch[:, :, py:, px:]
    return out


def prepare_lr(img_path, scale=4, max_lr_side=512):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    if min(w, h) > max_lr_side:
        new_w, new_h = w // scale, h // scale
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f'输入图片较大（{w}×{h}），已自动下采样至 {new_w}×{new_h}')
    return img


def load_pil_to_tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.light_model:
        generator = LightGenerator(
            num_rrdb=Config.light_num_rrdb_blocks,
            channels=Config.light_num_channels
        ).to(device)
        print(f'使用轻量化模型: {Config.light_num_rrdb_blocks} RRDB块, {Config.light_num_channels} 通道')
    else:
        generator = Generator(
            num_rrdb=Config.num_rrdb_blocks,
            channels=Config.num_channels
        ).to(device)
        print(f'使用原版模型: {Config.num_rrdb_blocks} RRDB块, {Config.num_channels} 通道')

    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    exp_name = os.path.basename(os.path.dirname(args.model_path))
    model_tag = f'{exp_name}_{model_name}' if exp_name else model_name

    with torch.no_grad():
        if os.path.isfile(args.input_path):
            pil_lr = prepare_lr(args.input_path)
            lr_img = load_pil_to_tensor(pil_lr).to(device)
            t0 = time.time()
            sr_img = sr_tile(generator, lr_img)
            print(f'推理耗时: {(time.time() - t0)*1000:.1f} ms')
            base = os.path.splitext(os.path.basename(args.input_path))[0]
            save_image(sr_img[0], os.path.join(args.output_dir, f'{base}_{model_tag}_sr.png'))
            # 保存 LR nearest 放大图（与 SR 同尺寸，方便对比）
            sr_w = sr_img.shape[3]
            sr_h = sr_img.shape[2]
            pil_lr.resize((sr_w, sr_h), Image.NEAREST).save(
                os.path.join(args.output_dir, f'{base}_{model_tag}_lr_nearest.png'))
            # 若提供 HR 原图，裁剪/缩放到相同尺寸保存
            if args.hr_path and os.path.isfile(args.hr_path):
                Image.open(args.hr_path).convert('RGB').resize((sr_w, sr_h), Image.BICUBIC).save(
                    os.path.join(args.output_dir, f'{base}_{model_tag}_hr.png'))
        else:
            for img_name in os.listdir(args.input_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    lr_img = load_pil_to_tensor(prepare_lr(os.path.join(args.input_path, img_name))).to(device)
                    t0 = time.time()
                    sr_img = sr_tile(generator, lr_img)
                    print(f'{img_name} 推理耗时: {(time.time() - t0)*1000:.1f} ms')
                    stem = os.path.splitext(img_name)[0]
                    save_image(sr_img[0], os.path.join(args.output_dir, f'{stem}_{model_tag}_sr.png'))
                    del lr_img, sr_img
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    print(f'超分辨率结果已保存到 {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--light_model', action='store_true', help='使用轻量化模型（默认使用原版模型）')
    parser.add_argument('--hr_path', type=str, default=None, help='HR原图路径（可选，用于对比保存）')
    args = parser.parse_args()
    test(args)





""" 
  # 只保存 SR + LR对比图
  python test.py --input_path ./data/val_hr/0802.png --output_dir ./results --model_path
  ./checkpoints/GRADIENT_LOSS_EXPERIMENTS/grad0.1/generator_gan_150.pth --light_model

  # 同时保存 HR 原图用于三图对比
  python test.py --input_path ./data/val_hr/0802.png --output_dir ./results --model_path
  ./checkpoints/GRADIENT_LOSS_EXPERIMENTS/grad0.1/generator_gan_150.pth --light_model --hr_path ./data/val_hr/0802.png
"""