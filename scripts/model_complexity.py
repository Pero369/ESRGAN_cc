"""
模型复杂度分析脚本
统计参数量、FLOPs、模型大小等指标
"""
import torch
import os
import argparse
from models import LightGenerator, Generator
from config import Config


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_flops(model, input_size=(1, 3, 256, 256)):
    """
    计算FLOPs（使用thop库）

    Args:
        model: 模型
        input_size: 输入尺寸

    Returns:
        FLOPs, MACs
    """
    try:
        from thop import profile, clever_format

        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)

        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")

        return flops, params
    except ImportError:
        print("警告: 未安装thop库，无法计算FLOPs")
        print("安装命令: pip install thop")
        return "N/A", "N/A"


def get_model_size(model_path):
    """获取模型文件大小（MB）"""
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 ** 2)
        return size_mb
    return None


def analyze_model(model_path, use_light_model=True, model_name="Model"):
    """
    分析单个模型的复杂度

    Args:
        model_path: 模型路径
        use_light_model: 是否使用轻量化模型
        model_name: 模型名称

    Returns:
        dict: 包含各项指标的字典
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
        rrdb_blocks = Config.light_num_rrdb_blocks
        channels = Config.light_num_channels
        conv_type = "DSConv"
    else:
        model = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)
        rrdb_blocks = Config.num_rrdb_blocks
        channels = Config.num_channels
        conv_type = "Standard"

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # 统计参数量
    total_params, trainable_params = count_parameters(model)

    # 计算FLOPs
    flops, _ = calculate_flops(model, input_size=(1, 3, 256, 256))

    # 获取模型大小
    model_size = get_model_size(model_path) if model_path else None

    results = {
        'name': model_name,
        'rrdb_blocks': rrdb_blocks,
        'channels': channels,
        'conv_type': conv_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'model_size_mb': model_size
    }

    return results


def print_comparison_table(results_list):
    """
    打印对比表格

    Args:
        results_list: 模型分析结果列表
    """
    print("\n" + "="*100)
    print("模型复杂度对比")
    print("="*100)

    # 表头
    header = f"{'模型':<20} {'RRDB块':<10} {'通道数':<10} {'卷积类型':<12} {'参数量':<15} {'FLOPs':<15} {'模型大小':<12}"
    print(header)
    print("-"*100)

    # 数据行
    for result in results_list:
        params_m = result['total_params'] / 1e6
        size_str = f"{result['model_size_mb']:.2f} MB" if result['model_size_mb'] else "N/A"

        row = (f"{result['name']:<20} "
               f"{result['rrdb_blocks']:<10} "
               f"{result['channels']:<10} "
               f"{result['conv_type']:<12} "
               f"{params_m:>8.2f}M      "
               f"{str(result['flops']):<15} "
               f"{size_str:<12}")
        print(row)

    print("="*100)

    # 计算减少比例（如果有多个模型）
    if len(results_list) >= 2:
        baseline = results_list[0]
        for i, result in enumerate(results_list[1:], 1):
            print(f"\n{result['name']} vs {baseline['name']}:")

            # 参数量减少
            param_reduction = (1 - result['total_params'] / baseline['total_params']) * 100
            print(f"  参数量减少: {param_reduction:.1f}%")

            # RRDB块减少
            rrdb_reduction = (1 - result['rrdb_blocks'] / baseline['rrdb_blocks']) * 100
            print(f"  RRDB块减少: {rrdb_reduction:.1f}%")

            # 通道数减少
            channel_reduction = (1 - result['channels'] / baseline['channels']) * 100
            print(f"  通道数减少: {channel_reduction:.1f}%")

            # 模型大小减少
            if result['model_size_mb'] and baseline['model_size_mb']:
                size_reduction = (1 - result['model_size_mb'] / baseline['model_size_mb']) * 100
                print(f"  模型大小减少: {size_reduction:.1f}%")

    print()


def save_results_to_csv(results_list, output_path):
    """
    保存结果到CSV文件

    Args:
        results_list: 模型分析结果列表
        output_path: 输出文件路径
    """
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(['模型', 'RRDB块数', '通道数', '卷积类型', '参数量', 'FLOPs', '模型大小(MB)'])

        # 写入数据
        for result in results_list:
            params_m = f"{result['total_params'] / 1e6:.2f}M"
            size_mb = f"{result['model_size_mb']:.2f}" if result['model_size_mb'] else "N/A"

            writer.writerow([
                result['name'],
                result['rrdb_blocks'],
                result['channels'],
                result['conv_type'],
                params_m,
                result['flops'],
                size_mb
            ])

    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型复杂度分析')
    parser.add_argument('--models', type=str, nargs='+',
                        help='模型路径列表（格式: path:type:name，type为light或original）')
    parser.add_argument('--output', type=str, default='./results/model_complexity.csv',
                        help='输出CSV文件路径')

    args = parser.parse_args()

    results_list = []

    if args.models:
        # 分析多个模型
        for model_spec in args.models:
            parts = model_spec.split(':')
            model_path = parts[0]
            model_type = parts[1] if len(parts) > 1 else 'light'
            model_name = parts[2] if len(parts) > 2 else os.path.basename(model_path)

            use_light = (model_type == 'light')
            result = analyze_model(model_path, use_light, model_name)
            results_list.append(result)
    else:
        # 默认对比：原版 vs 轻量版
        print("未指定模型，使用默认配置对比原版和轻量版")

        # 原版模型
        result_original = analyze_model(None, False, "原版ESRGAN")
        results_list.append(result_original)

        # 轻量版模型
        result_light = analyze_model(None, True, "轻量化ESRGAN")
        results_list.append(result_light)

    # 打印对比表格
    print_comparison_table(results_list)

    # 保存到CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results_to_csv(results_list, args.output)
