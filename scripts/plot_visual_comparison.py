"""
图4-5 视觉质量对比图 + 图4-6 细节放大对比图

用法（服务器上运行）:
  python scripts/plot_visual_comparison.py --gpu 0

默认会对比以下模型，可通过 --models 自定义:
  - Bicubic（无需checkpoint）
  - 原版ESRGAN
  - 轻量版基线
  - 最优方案(grad=0.15)

输出: figures/fig4-5_visual_comparison.png
      figures/fig4-6_detail_crop.png
"""
import torch
import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 把项目根目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import Generator, LightGenerator
from config import Config

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
TEMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', '.temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ─── 模型定义 ───────────────────────────────────────────

# 每项: (显示名称, 模型类型, checkpoint路径, 额外kwargs)
# 模型类型: 'bicubic' | 'original' | 'light'
DEFAULT_MODELS = [
    ('Bicubic', 'bicubic', None, {}),
    ('原版ESRGAN', 'original',
     'checkpoints/COMPREHENSIVE_EXPERIMENTS/original/generator_gan_150.pth',
     {'num_rrdb': 23, 'channels': 64}),
    ('轻量版基线', 'light',
     'checkpoints/GRADIENT_LOSS_EXPERIMENTS/enable_gradient_lossFalse/generator_gan_150.pth',
     {'num_rrdb': 8, 'channels': 32}),
    ('最优方案\n(GradLoss=0.15)', 'light',
     'checkpoints/CHANNELS_EXPERIMENTS/light_grad0.15_rrdb8_ch32/generator_gan_150.pth',
     {'num_rrdb': 8, 'channels': 32}),
]


# ─── 测试图像选择 ───────────────────────────────────────

# Set5 + Set14 代表性图像: (数据集, 图片编号, 简称)
TEST_IMAGES = [
    # Set5
    ('Set5', 'img_003', 'butterfly'),   # 蝴蝶 — 丰富纹理色彩
    ('Set5', 'img_002', 'baby'),         # 婴儿 — 人脸感知
    # Set14
    ('Set14', 'img_008', 'baboon'),      # 狒狒 — 毛发纹理
    ('Set14', 'img_012', 'monarch'),     # 蝴蝶 — 细纹图案
]

# 细节放大区域: (数据集, 图片编号, x, y, w, h) — 在HR图像上的坐标
CROP_REGIONS = {
    ('Set5', 'img_003'): (220, 80, 120, 120),    # butterfly 翅膀纹理
    ('Set5', 'img_002'): (160, 100, 100, 100),    # baby 眼睛区域
    ('Set14', 'img_008'): (80, 280, 100, 100),    # baboon 胡须纹理
    ('Set14', 'img_012'): (100, 120, 100, 100),   # monarch 翅膀细纹
}


# ─── 工具函数 ──────────────────────────────────────────

def load_image(path):
    """加载图像为 float32 tensor [1, C, H, W]"""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t):
    """tensor [1, C, H, W] → PIL Image"""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def bicubic_upscale(lr_path, scale=4):
    """Bicubic上采样"""
    img = Image.open(lr_path).convert('RGB')
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)


def build_model(model_type, ckpt_path, device, kwargs):
    """构建模型并加载权重"""
    if model_type == 'original':
        model = Generator(
            num_rrdb=kwargs.get('num_rrdb', 23),
            channels=kwargs.get('channels', 64)
        )
    elif model_type == 'light':
        model = LightGenerator(
            num_rrdb=kwargs.get('num_rrdb', 8),
            channels=kwargs.get('channels', 32)
        )
    else:
        return None

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def sr_inference(model, lr_tensor, device):
    """推理（大图自动切块）"""
    if lr_tensor.device != device:
        lr_tensor = lr_tensor.to(device)

    _, _, h, w = lr_tensor.shape
    if h <= 512 and w <= 512:
        with torch.no_grad():
            return model(lr_tensor)

    # 大图分块处理
    tile, overlap = 256, 16
    scale = 4
    _, c, _, _ = lr_tensor.shape
    out = torch.zeros(1, c, h * scale, w * scale, device=device)
    step = tile - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y, h - tile)
            x1 = min(x, w - tile)
            y2, x2 = y1 + tile, x1 + tile
            patch = lr_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                sr_patch = model(patch)
            py = overlap * scale // 2 if y1 > 0 else 0
            px = overlap * scale // 2 if x1 > 0 else 0
            out[:, :, y1*scale+py:y2*scale, x1*scale+px:x2*scale] = sr_patch[:, :, py:, px:]

    return out


# ─── 图4-5: 视觉质量对比图 ──────────────────────────────

def fig_4_5(all_sr, all_hr, args):
    """图4-5：整图对比，水平排列"""
    n_models = len(all_sr[0])  # Bicubic + 各模型
    n_rows = len(TEST_IMAGES)

    fig, axes = plt.subplots(n_rows, n_models + 1, figsize=(3.5 * (n_models + 1), 3.2 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_labels = ['HR (Ground Truth)'] + [m[0] for m in args.models]

    for row_idx, (ds, img_id, name) in enumerate(TEST_IMAGES):
        # HR
        ax = axes[row_idx, 0]
        ax.imshow(all_hr[(ds, img_id)])
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title('HR', fontsize=12, fontweight='bold')
        # 行标签
        ax.set_ylabel(f'{name}\n{img_id}', fontsize=10, fontweight='bold', rotation=0,
                      labelpad=40, ha='right', va='center')

        for col_idx in range(n_models):
            ax = axes[row_idx, col_idx + 1]
            ax.imshow(all_sr[(ds, img_id)][col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_labels[col_idx + 1], fontsize=11, fontweight='bold')

    # 指标标注（PSNR/SSIM）
    if args.annotate_metrics:
        # 这里可以读取已有的评估结果
        pass

    plt.suptitle('图4-5 视觉质量对比（Set5 / Set14 代表性图像）', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4-5_visual_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图4-5 已保存: {path}')


# ─── 图4-6: 细节放大对比图 ──────────────────────────────

def fig_4_6(all_sr, all_hr, args):
    """图4-6：局部裁剪放大对比"""
    n_models = len(all_sr[list(all_sr.keys())[0]])
    n_rows = len(TEST_IMAGES)

    fig, axes = plt.subplots(n_rows, n_models + 1, figsize=(3.5 * (n_models + 1), 3.2 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (ds, img_id, name) in enumerate(TEST_IMAGES):
        crop = CROP_REGIONS.get((ds, img_id))
        if crop is None:
            continue
        x, y, w, h = crop
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')

        # HR crop
        ax = axes[row_idx, 0]
        hr_img = all_hr[(ds, img_id)]
        hr_crop = hr_img.crop((x, y, x + w, y + h))
        ax.imshow(hr_crop)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title('HR (原图)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{name}', fontsize=10, fontweight='bold', rotation=0,
                      labelpad=30, ha='right', va='center')

        for col_idx in range(n_models):
            ax = axes[row_idx, col_idx + 1]
            sr_pil = all_sr[(ds, img_id)][col_idx]
            sr_crop = sr_pil.crop((x, y, x + w, y + h))
            ax.imshow(sr_crop)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(args.models[col_idx][0], fontsize=11, fontweight='bold')

            # 标注PSNR（如果提供）
            psnr_val = args.crop_psnr.get((ds, img_id, col_idx), None)
            if psnr_val:
                ax.set_xlabel(f'PSNR={psnr_val:.2f}dB', fontsize=9)

    plt.suptitle('图4-6 细节放大对比（边缘/纹理局部放大）', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4-6_detail_crop.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图4-6 已保存: {path}')


# ─── 主流程 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='生成图4-5和图4-6：视觉质量对比')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--annotate_metrics', action='store_true', help='是否标注PSNR/SSIM指标')
    parser.add_argument('--skip_fig5', action='store_true')
    parser.add_argument('--skip_fig6', action='store_true')
    args = parser.parse_args()

    # 模型配置可通过命令行覆盖（用逗号分隔）
    # 格式: --models "名称,类型,ckpt_path,num_rrdb,channels|..."
    # 暂用默认配置

    models = DEFAULT_MODELS
    args.models = models  # 供绘图函数使用
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'对比模型: {[m[0] for m in models]}')

    # ── 加载模型 ──
    loaded_models = {}
    for name, mtype, ckpt_path, kwargs in models:
        if mtype == 'bicubic':
            loaded_models[name] = None
            continue
        if not os.path.exists(ckpt_path):
            print(f'[警告] checkpoint不存在: {ckpt_path}')
            print(f'  跳过模型: {name}')
            loaded_models[name] = None
            continue
        print(f'加载 {name}: {ckpt_path}')
        loaded_models[name] = build_model(mtype, ckpt_path, device, kwargs)
        if loaded_models[name] is not None:
            n_params = sum(p.numel() for p in loaded_models[name].parameters()) / 1e6
            print(f'  参数量: {n_params:.2f}M')

    # ── 生成所有SR结果 ──
    all_sr = {}  # (dataset, img_id) → [PIL Image, ...]
    all_hr = {}  # (dataset, img_id) → PIL Image

    for ds, img_id, _ in TEST_IMAGES:
        lr_path = os.path.join('data', ds, 'image_SRF_4', f'{img_id}_SRF_4_LR.png')
        hr_path = os.path.join('data', ds, 'image_SRF_4', f'{img_id}_SRF_4_HR.png')

        if not os.path.exists(lr_path):
            print(f'[警告] 文件不存在: {lr_path}')
            continue

        print(f'\n处理: {ds}/{img_id}')
        hr_img = Image.open(hr_path).convert('RGB')
        all_hr[(ds, img_id)] = hr_img

        sr_list = []
        for name, mtype, ckpt_path, kwargs in models:
            model = loaded_models.get(name)

            if mtype == 'bicubic':
                sr_pil = bicubic_upscale(lr_path, scale=4)
                print(f'  Bicubic: {sr_pil.size}')
            elif model is None:
                # 模型加载失败，用空白图占位
                sr_pil = Image.new('RGB', hr_img.size, (128, 128, 128))
                print(f'  {name}: 跳过（模型未加载）')
            else:
                lr_tensor = load_image(lr_path).to(device)
                sr_tensor = sr_inference(model, lr_tensor, device)
                sr_pil = tensor_to_pil(sr_tensor)
                # 裁剪到HR尺寸（对齐评估）
                sr_pil = sr_pil.resize(hr_img.size, Image.BICUBIC)
                print(f'  {name}: {sr_pil.size}')

            sr_list.append(sr_pil)

        all_sr[(ds, img_id)] = sr_list

    # ── 生成图表 ──
    if not args.skip_fig5:
        fig_4_5(all_sr, all_hr, args)

    if not args.skip_fig6:
        fig_4_6(all_sr, all_hr, args)

    print(f'\n全部图表已保存到: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
