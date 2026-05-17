import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.quantization as quant
import numpy as np
import time
import csv
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from models import LightGenerator
from config import Config


def rgb_to_y(img_np):
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    return 16/255 + (65.481*r + 128.553*g + 24.966*b)


def calc_psnr_ssim(sr_np, hr_np, y_channel=True):
    if y_channel:
        sr_np = rgb_to_y(sr_np)
        hr_np = rgb_to_y(hr_np)
        return psnr(hr_np, sr_np, data_range=255.0), ssim(hr_np, sr_np, data_range=255.0)
    return psnr(hr_np, sr_np, data_range=1.0), ssim(hr_np, sr_np, data_range=1.0, channel_axis=2)


def load_quantized_model(fp32_weights_path):
    model = LightGenerator(
        Config.light_num_rrdb_blocks, Config.light_num_channels,
        enable_attention=Config.enable_attention,
        attention_type=Config.attention_type,
        attention_reduction=Config.attention_reduction,
        attention_position=Config.attention_position
    ).cpu()
    model.load_state_dict(torch.load(fp32_weights_path, map_location='cpu'))
    model.eval()
    return quant.quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)


def load_fp32_model(weights_path):
    model = LightGenerator(
        Config.light_num_rrdb_blocks, Config.light_num_channels,
        enable_attention=Config.enable_attention,
        attention_type=Config.attention_type,
        attention_reduction=Config.attention_reduction,
        attention_position=Config.attention_position
    ).cpu()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def measure_inference_time(model, input_size=(1, 3, 64, 64), n_warmup=5, n_runs=20):
    x = torch.randn(input_size)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    return (time.time() - t0) / n_runs * 1000


def evaluate_on_dataset(model, dataset_dir, y_channel=True):
    srf4_dir = os.path.join(dataset_dir, 'image_SRF_4')
    lr_files = sorted(f for f in os.listdir(srf4_dir) if '_LR' in f)
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for lr_name in lr_files:
            hr_name = lr_name.replace('_LR', '_HR')
            lr_np = np.array(Image.open(os.path.join(srf4_dir, lr_name)).convert('RGB')).astype(np.float32) / 255.0
            hr_np = np.array(Image.open(os.path.join(srf4_dir, hr_name)).convert('RGB')).astype(np.float32) / 255.0
            lr_t = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0)
            sr_t = model(lr_t)
            sr_np = sr_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
            h, w = hr_np.shape[:2]
            sr_np = sr_np[:h, :w]
            p, s = calc_psnr_ssim(sr_np, hr_np, y_channel)
            psnr_list.append(p)
            ssim_list.append(s)
    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == '__main__':
    fp32_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'GRADIENT_LOSS_EXPERIMENTS', 'grad0.1', 'generator_gan_150.pth')
    set5_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Set5')
    set14_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Set14')

    print('加载模型...')
    fp32_model = load_fp32_model(fp32_path)
    quant_model = load_quantized_model(fp32_path)

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    # 动态量化模型存在内存中，大小通过torch.save临时文件测量
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(quant_model.state_dict(), tmp_path)
    quant_size = os.path.getsize(tmp_path) / (1024 * 1024)
    os.unlink(tmp_path)
    print(f'FP32 模型大小: {fp32_size:.2f} MB')
    print(f'INT8 模型大小: {quant_size:.2f} MB  (压缩 {(1-quant_size/fp32_size)*100:.0f}%)')

    print('\n测量推理时间 (CPU, 64×64 LR输入)...')
    fp32_time = measure_inference_time(fp32_model)
    quant_time = measure_inference_time(quant_model)
    print(f'FP32 推理时间: {fp32_time:.1f} ms')
    print(f'INT8 推理时间: {quant_time:.1f} ms  (加速 {fp32_time/quant_time:.2f}x)')

    print('\n评估 Set5 (Y通道)...')
    fp32_set5_psnr, fp32_set5_ssim = evaluate_on_dataset(fp32_model, set5_dir)
    quant_set5_psnr, quant_set5_ssim = evaluate_on_dataset(quant_model, set5_dir)
    print(f'FP32  Set5: PSNR={fp32_set5_psnr:.4f} dB, SSIM={fp32_set5_ssim:.4f}')
    print(f'INT8  Set5: PSNR={quant_set5_psnr:.4f} dB, SSIM={quant_set5_ssim:.4f}')

    print('\n评估 Set14 (Y通道)...')
    fp32_set14_psnr, fp32_set14_ssim = evaluate_on_dataset(fp32_model, set14_dir)
    quant_set14_psnr, quant_set14_ssim = evaluate_on_dataset(quant_model, set14_dir)
    print(f'FP32 Set14: PSNR={fp32_set14_psnr:.4f} dB, SSIM={fp32_set14_ssim:.4f}')
    print(f'INT8 Set14: PSNR={quant_set14_psnr:.4f} dB, SSIM={quant_set14_ssim:.4f}')

    rows = [
        {'指标': '模型大小(MB)', 'FP32轻量版': f'{fp32_size:.2f}', 'INT8量化版': f'{quant_size:.2f}', '变化': f'-{(1-quant_size/fp32_size)*100:.0f}%'},
        {'指标': 'CPU推理时间(ms)', 'FP32轻量版': f'{fp32_time:.1f}', 'INT8量化版': f'{quant_time:.1f}', '变化': f'-{(1-quant_time/fp32_time)*100:.0f}%'},
        {'指标': 'Set5 PSNR(dB)', 'FP32轻量版': f'{fp32_set5_psnr:.4f}', 'INT8量化版': f'{quant_set5_psnr:.4f}', '变化': f'{quant_set5_psnr-fp32_set5_psnr:+.4f}'},
        {'指标': 'Set5 SSIM', 'FP32轻量版': f'{fp32_set5_ssim:.4f}', 'INT8量化版': f'{quant_set5_ssim:.4f}', '变化': f'{quant_set5_ssim-fp32_set5_ssim:+.4f}'},
        {'指标': 'Set14 PSNR(dB)', 'FP32轻量版': f'{fp32_set14_psnr:.4f}', 'INT8量化版': f'{quant_set14_psnr:.4f}', '变化': f'{quant_set14_psnr-fp32_set14_psnr:+.4f}'},
        {'指标': 'Set14 SSIM', 'FP32轻量版': f'{fp32_set14_ssim:.4f}', 'INT8量化版': f'{quant_set14_ssim:.4f}', '变化': f'{quant_set14_ssim-fp32_set14_ssim:+.4f}'},
    ]
    out_csv = os.path.join(os.path.dirname(__file__), '..', 'results', '.csv', 'eval_quantization.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['指标', 'FP32轻量版', 'INT8量化版', '变化'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n结果已保存到 {out_csv}')
