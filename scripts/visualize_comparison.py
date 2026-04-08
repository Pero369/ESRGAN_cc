"""
可视化对比脚本
生成模型对比的图表和可视化结果
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import argparse
import csv
from PIL import Image

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_parameter_comparison(results_list, output_path):
    """
    绘制参数量对比柱状图

    Args:
        results_list: 模型分析结果列表
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r['name'] for r in results_list]
    params = [r['total_params'] / 1e6 for r in results_list]

    bars = ax.bar(names, params, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(names)])

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M',
                ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('参数量 (百万)', fontsize=12)
    ax.set_title('模型参数量对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"参数量对比图已保存: {output_path}")
    plt.close()


def plot_inference_speed(benchmark_results, output_path):
    """
    绘制推理速度对比折线图

    Args:
        benchmark_results: 基准测试结果
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 示例数据结构（需要根据实际benchmark结果调整）
    # benchmark_results = {
    #     'model1': {'256': 10, '512': 40, '1024': 160},
    #     'model2': {'256': 5, '512': 20, '1024': 80}
    # }

    input_sizes = ['256×256', '512×512', '1024×1024']
    markers = ['o', 's', '^', 'D']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, (model_name, results) in enumerate(benchmark_results.items()):
        times = list(results.values())
        ax.plot(input_sizes, times, marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)], linewidth=2, markersize=8,
                label=model_name)

    ax.set_xlabel('输入尺寸', fontsize=12)
    ax.set_ylabel('推理时间 (ms)', fontsize=12)
    ax.set_title('推理速度对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"推理速度对比图已保存: {output_path}")
    plt.close()


def plot_accuracy_efficiency_tradeoff(models_data, output_path):
    """
    绘制精度-效率权衡散点图

    Args:
        models_data: 模型数据列表，每项包含 {'name', 'params', 'psnr'}
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [d['name'] for d in models_data]
    params = [d['params'] / 1e6 for d in models_data]
    psnrs = [d['psnr'] for d in models_data]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (name, param, psnr) in enumerate(zip(names, params, psnrs)):
        ax.scatter(param, psnr, s=200, color=colors[i % len(colors)],
                   alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(name, (param, psnr), xytext=(10, 10),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    ax.set_xlabel('参数量 (百万)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('精度-效率权衡', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"精度-效率权衡图已保存: {output_path}")
    plt.close()


def create_visual_comparison(image_paths, labels, output_path, crop_region=None):
    """
    创建视觉质量对比图

    Args:
        image_paths: 图像路径列表 [HR, Bicubic, Model1, Model2, ...]
        labels: 标签列表
        output_path: 输出路径
        crop_region: 裁剪区域 (x, y, w, h) 用于细节对比
    """
    n_images = len(image_paths)
    fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))

    if n_images == 1:
        axes = [axes]

    for ax, img_path, label in zip(axes, image_paths, labels):
        img = Image.open(img_path).convert('RGB')

        # 如果指定了裁剪区域
        if crop_region:
            x, y, w, h = crop_region
            img = img.crop((x, y, x+w, y+h))

        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"视觉对比图已保存: {output_path}")
    plt.close()


def plot_quality_comparison_table(quality_results, output_path):
    """
    绘制质量对比表格图

    Args:
        quality_results: 质量评估结果列表
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    headers = ['模型', 'PSNR (dB)', 'SSIM', 'vs Bicubic (PSNR)', 'vs Bicubic (SSIM)']
    table_data = []

    for result in quality_results:
        row = [
            result['name'],
            f"{result['psnr']:.2f}",
            f"{result['ssim']:.4f}" if result.get('ssim') else 'N/A',
            f"+{result['psnr_improvement']:.2f}" if result.get('psnr_improvement') else 'N/A',
            f"+{result['ssim_improvement']:.4f}" if result.get('ssim_improvement') else 'N/A'
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('图像质量对比', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"质量对比表格已保存: {output_path}")
    plt.close()


def generate_all_visualizations(complexity_csv, quality_csv, output_dir):
    """
    从CSV文件生成所有可视化图表

    Args:
        complexity_csv: 模型复杂度CSV路径
        quality_csv: 质量评估CSV路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取复杂度数据
    complexity_data = []
    if os.path.exists(complexity_csv):
        with open(complexity_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                complexity_data.append({
                    'name': row['模型'],
                    'total_params': float(row['参数量'].replace('M', '')) * 1e6,
                    'rrdb_blocks': int(row['RRDB块数']),
                    'channels': int(row['通道数'])
                })

        # 生成参数量对比图
        plot_parameter_comparison(complexity_data, os.path.join(output_dir, 'parameter_comparison.png'))

    # 读取质量数据
    quality_data = []
    if os.path.exists(quality_csv):
        # 这里需要根据实际CSV格式解析
        # 示例代码，需要调整
        pass

    print(f"\n所有可视化图表已生成到: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成可视化对比图表')
    parser.add_argument('--complexity_csv', type=str, help='模型复杂度CSV路径')
    parser.add_argument('--quality_csv', type=str, help='质量评估CSV路径')
    parser.add_argument('--output_dir', type=str, default='./results/visualizations',
                        help='输出目录')
    parser.add_argument('--mode', type=str, choices=['all', 'params', 'speed', 'quality', 'tradeoff'],
                        default='all', help='生成模式')

    args = parser.parse_args()

    if args.mode == 'all' and args.complexity_csv:
        generate_all_visualizations(args.complexity_csv, args.quality_csv, args.output_dir)
    else:
        print("请提供必要的CSV文件路径")
        print("示例: python visualize_comparison.py --complexity_csv results/model_complexity.csv --output_dir results/visualizations")
