"""
性能基准测试脚本
对比不同模型版本的推理速度、内存占用和精度
"""
import torch
import time
import argparse
import os
from models import LightGenerator, Generator
from config import Config
from utils import load_image
import numpy as np


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(model, input_tensor, num_runs=50, warmup=5):
    """
    测量推理时间

    Args:
        model: 模型
        input_tensor: 输入张量
        num_runs: 测试次数
        warmup: 预热次数

    Returns:
        平均推理时间（毫秒）
    """
    model.eval()
    device = next(model.parameters()).device

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # 同步GPU（如果使用GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    # 同步GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒

    return avg_time


def measure_memory(model, input_tensor):
    """
    测量显存/内存占用

    Args:
        model: 模型
        input_tensor: 输入张量

    Returns:
        峰值内存占用（MB）
    """
    device = next(model.parameters()).device

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        with torch.no_grad():
            _ = model(input_tensor)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        # CPU内存测量（简化版）
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 ** 2)

        with torch.no_grad():
            _ = model(input_tensor)

        mem_after = process.memory_info().rss / (1024 ** 2)
        peak_memory = mem_after - mem_before

    return peak_memory


def benchmark_model(model_path, use_light_model=True, device='cuda', input_sizes=None):
    """
    对模型进行基准测试

    Args:
        model_path: 模型路径
        use_light_model: 是否使用轻量化模型
        device: 设备 'cuda' 或 'cpu'
        input_sizes: 测试的输入尺寸列表
    """
    if input_sizes is None:
        input_sizes = [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 1024, 1024)]

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

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
        model_type = '轻量化模型'
    else:
        model = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)
        model_type = '原版模型'

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 统计参数量
    num_params = count_parameters(model)
    print(f'\n{model_type}')
    print(f'参数量: {num_params / 1e6:.2f}M')

    # 模型大小
    model_size = os.path.getsize(model_path) / (1024 ** 2)
    print(f'模型大小: {model_size:.2f} MB')

    # 测试不同输入尺寸
    print(f'\n{"="*60}')
    print(f'{"输入尺寸":<15} {"推理时间":<15} {"内存占用":<15} {"吞吐量"}')
    print(f'{"="*60}')

    for input_size in input_sizes:
        # 创建测试输入
        test_input = torch.randn(input_size).to(device)

        # 测量推理时间
        try:
            inference_time = measure_inference_time(model, test_input)

            # 测量内存占用
            memory_usage = measure_memory(model, test_input)

            # 计算吞吐量
            throughput = 1000 / inference_time  # images/sec

            size_str = f'{input_size[2]}×{input_size[3]}'
            print(f'{size_str:<15} {inference_time:>8.2f} ms    {memory_usage:>8.2f} MB    {throughput:>6.2f} img/s')

        except RuntimeError as e:
            if 'out of memory' in str(e):
                size_str = f'{input_size[2]}×{input_size[3]}'
                print(f'{size_str:<15} {"OOM":<15} {"OOM":<15} {"N/A"}')
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise e

    print(f'{"="*60}\n')


def compare_models(model_paths, model_types, device='cuda'):
    """
    对比多个模型

    Args:
        model_paths: 模型路径列表
        model_types: 模型类型列表 ['light', 'original', ...]
        device: 设备
    """
    print('\n' + '='*80)
    print('模型对比基准测试')
    print('='*80)

    for model_path, model_type in zip(model_paths, model_types):
        use_light = (model_type == 'light')
        benchmark_model(model_path, use_light, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESRGAN模型性能基准测试')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--light_model', action='store_true', help='使用轻量化模型')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='测试设备')
    parser.add_argument('--input_sizes', type=int, nargs='+',
                        default=[256, 512, 1024],
                        help='测试的输入尺寸（正方形）')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='对比多个模型（格式: path1:type1 path2:type2）')

    args = parser.parse_args()

    if args.compare:
        # 对比模式
        model_paths = []
        model_types = []
        for item in args.compare:
            path, mtype = item.split(':')
            model_paths.append(path)
            model_types.append(mtype)
        compare_models(model_paths, model_types, args.device)
    else:
        # 单模型测试
        input_sizes = [(1, 3, size, size) for size in args.input_sizes]
        benchmark_model(args.model_path, args.light_model, args.device, input_sizes)
