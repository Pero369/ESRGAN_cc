"""
图像质量评估脚本
在标准测试集上评估PSNR、SSIM等指标
"""
import torch
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

from models import LightGenerator, Generator
from config import Config


def calculate_psnr(img1, img2):
    """
    计算PSNR

    Args:
        img1, img2: numpy arrays, 范围[0, 255]

    Returns:
        PSNR值
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    计算SSIM（需要skimage库）

    Args:
        img1, img2: numpy arrays, 范围[0, 255]

    Returns:
        SSIM值
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        # 转换为灰度图或使用多通道SSIM
        if len(img1.shape) == 3:
            return ssim(img1, img2, channel_axis=2, data_range=255)
        else:
            return ssim(img1, img2, data_range=255)
    except ImportError:
        print("警告: 未安装scikit-image，无法计算SSIM")
        print("安装命令: pip install scikit-image")
        return None


def load_image(image_path):
    """加载图像为tensor"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def tensor_to_numpy(tensor):
    """将tensor转换为numpy array [0, 255]"""
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def bicubic_upscale(lr_img, scale=4):
    """使用Bicubic插值上采样"""
    from PIL import Image
    import torchvision.transforms.functional as TF

    # tensor to PIL
    lr_pil = TF.to_pil_image(lr_img.squeeze(0))

    # 上采样
    h, w = lr_pil.size
    hr_pil = lr_pil.resize((w * scale, h * scale), Image.BICUBIC)

    # PIL to tensor
    hr_tensor = TF.to_tensor(hr_pil).unsqueeze(0)
    return hr_tensor


def evaluate_model(model_path, test_dir, output_dir, use_light_model=True, save_images=True):
    """
    评估模型在测试集上的性能

    Args:
        model_path: 模型路径
        test_dir: 测试集目录（包含HR图像）
        output_dir: 输出目录
        use_light_model: 是否使用轻量化模型
        save_images: 是否保存SR图像

    Returns:
        dict: 评估结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    if use_light_model:
        model = LightGenerator(
            Config.light_num_rrdb_blocks,
            Config.light_num_channels,
            enable_attention=Config.enable_attention,
            attention_type=Config.attention_type,
            attention_reduction=Config.attention_reduction,
            attention_position=Config.attention_position
        ).to(device)
    else:
        model = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建输出目录
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    # 获取测试图像列表
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    results = []
    psnr_list = []
    ssim_list = []
    psnr_bicubic_list = []
    ssim_bicubic_list = []

    print(f"评估模型: {model_path}")
    print(f"测试集: {test_dir} ({len(image_files)} 张图像)")

    for img_file in tqdm(image_files, desc="评估中"):
        # 加载HR图像
        hr_path = os.path.join(test_dir, img_file)
        hr_img = load_image(hr_path).to(device)

        # 生成LR图像（下采样4倍）
        import torchvision.transforms.functional as TF
        h, w = hr_img.shape[2], hr_img.shape[3]
        lr_img = TF.resize(hr_img, (h // 4, w // 4), interpolation=Image.BICUBIC)

        # 模型推理
        with torch.no_grad():
            sr_img = model(lr_img)

        # Bicubic上采样（基线）
        bicubic_img = bicubic_upscale(lr_img, scale=4)

        # 转换为numpy
        hr_np = tensor_to_numpy(hr_img)
        sr_np = tensor_to_numpy(sr_img)
        bicubic_np = tensor_to_numpy(bicubic_img)

        # 裁剪到相同尺寸（处理尺寸不匹配）
        min_h = min(hr_np.shape[0], sr_np.shape[0], bicubic_np.shape[0])
        min_w = min(hr_np.shape[1], sr_np.shape[1], bicubic_np.shape[1])
        hr_np = hr_np[:min_h, :min_w]
        sr_np = sr_np[:min_h, :min_w]
        bicubic_np = bicubic_np[:min_h, :min_w]

        # 计算PSNR和SSIM
        psnr_sr = calculate_psnr(hr_np, sr_np)
        ssim_sr = calculate_ssim(hr_np, sr_np)
        psnr_bicubic = calculate_psnr(hr_np, bicubic_np)
        ssim_bicubic = calculate_ssim(hr_np, bicubic_np)

        psnr_list.append(psnr_sr)
        if ssim_sr is not None:
            ssim_list.append(ssim_sr)
        psnr_bicubic_list.append(psnr_bicubic)
        if ssim_bicubic is not None:
            ssim_bicubic_list.append(ssim_bicubic)

        results.append({
            'image': img_file,
            'psnr_sr': psnr_sr,
            'ssim_sr': ssim_sr,
            'psnr_bicubic': psnr_bicubic,
            'ssim_bicubic': ssim_bicubic
        })

        # 保存SR图像
        if save_images:
            sr_pil = Image.fromarray(sr_np)
            sr_pil.save(os.path.join(output_dir, f'{os.path.splitext(img_file)[0]}_sr.png'))

    # 计算平均值
    avg_psnr_sr = np.mean(psnr_list)
    avg_ssim_sr = np.mean(ssim_list) if ssim_list else None
    avg_psnr_bicubic = np.mean(psnr_bicubic_list)
    avg_ssim_bicubic = np.mean(ssim_bicubic_list) if ssim_bicubic_list else None

    summary = {
        'avg_psnr_sr': avg_psnr_sr,
        'avg_ssim_sr': avg_ssim_sr,
        'avg_psnr_bicubic': avg_psnr_bicubic,
        'avg_ssim_bicubic': avg_ssim_bicubic,
        'psnr_improvement': avg_psnr_sr - avg_psnr_bicubic,
        'ssim_improvement': (avg_ssim_sr - avg_ssim_bicubic) if avg_ssim_sr else None,
        'details': results
    }

    return summary


def print_results(summary, model_name="Model"):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"评估结果: {model_name}")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'Bicubic':<15} {'模型':<15} {'提升':<15}")
    print(f"{'-'*60}")

    # PSNR
    psnr_bicubic = summary['avg_psnr_bicubic']
    psnr_sr = summary['avg_psnr_sr']
    psnr_imp = summary['psnr_improvement']
    print(f"{'PSNR (dB)':<20} {psnr_bicubic:>8.2f}      {psnr_sr:>8.2f}      {psnr_imp:>+8.2f}")

    # SSIM
    if summary['avg_ssim_sr'] is not None:
        ssim_bicubic = summary['avg_ssim_bicubic']
        ssim_sr = summary['avg_ssim_sr']
        ssim_imp = summary['ssim_improvement']
        print(f"{'SSIM':<20} {ssim_bicubic:>8.4f}      {ssim_sr:>8.4f}      {ssim_imp:>+8.4f}")

    print(f"{'='*60}\n")


def save_results_to_csv(summary, output_path, model_name="Model"):
    """保存结果到CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入汇总信息
        writer.writerow(['模型', model_name])
        writer.writerow(['平均PSNR (SR)', f"{summary['avg_psnr_sr']:.2f}"])
        writer.writerow(['平均SSIM (SR)', f"{summary['avg_ssim_sr']:.4f}" if summary['avg_ssim_sr'] else 'N/A'])
        writer.writerow(['平均PSNR (Bicubic)', f"{summary['avg_psnr_bicubic']:.2f}"])
        writer.writerow(['平均SSIM (Bicubic)', f"{summary['avg_ssim_bicubic']:.4f}" if summary['avg_ssim_bicubic'] else 'N/A'])
        writer.writerow([])

        # 写入详细结果
        writer.writerow(['图像', 'PSNR (SR)', 'SSIM (SR)', 'PSNR (Bicubic)', 'SSIM (Bicubic)'])
        for result in summary['details']:
            writer.writerow([
                result['image'],
                f"{result['psnr_sr']:.2f}",
                f"{result['ssim_sr']:.4f}" if result['ssim_sr'] else 'N/A',
                f"{result['psnr_bicubic']:.2f}",
                f"{result['ssim_bicubic']:.4f}" if result['ssim_bicubic'] else 'N/A'
            ])

    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像质量评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试集目录（HR图像）')
    parser.add_argument('--output_dir', type=str, default='./results/quality_eval', help='输出目录')
    parser.add_argument('--light_model', action='store_true', help='使用轻量化模型')
    parser.add_argument('--model_name', type=str, default='Model', help='模型名称')
    parser.add_argument('--save_images', action='store_true', help='保存SR图像')
    parser.add_argument('--csv_output', type=str, default='./results/quality_results.csv',
                        help='CSV输出路径')

    args = parser.parse_args()

    # 评估模型
    summary = evaluate_model(
        args.model_path,
        args.test_dir,
        args.output_dir,
        args.light_model,
        args.save_images
    )

    # 打印结果
    print_results(summary, args.model_name)

    # 保存到CSV
    save_results_to_csv(summary, args.csv_output, args.model_name)
