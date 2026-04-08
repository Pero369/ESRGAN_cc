"""
动态量化脚本
将训练好的模型进行INT8动态量化，减小模型大小并提升CPU推理速度
"""
import torch
import torch.quantization as quant
import os
import argparse
from models import LightGenerator, Generator
from config import Config


def dynamic_quantize(model_path, output_path, use_light_model=True):
    """
    对模型进行动态量化

    Args:
        model_path: 原始模型路径
        output_path: 量化后模型保存路径
        use_light_model: 是否使用轻量化模型
    """
    device = torch.device('cpu')  # 量化在CPU上进行

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
        print(f'加载轻量化模型: {Config.light_num_rrdb_blocks} RRDB块, {Config.light_num_channels} 通道')
    else:
        model = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)
        print(f'加载原版模型: {Config.num_rrdb_blocks} RRDB块, {Config.num_channels} 通道')

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 获取原始模型大小
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f'原始模型大小: {original_size:.2f} MB')

    # 动态量化
    print('开始动态量化...')
    quantized_model = quant.quantize_dynamic(
        model,
        {torch.nn.Conv2d, torch.nn.Linear},  # 量化卷积层和全连接层
        dtype=torch.qint8
    )

    # 保存量化模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)

    # 获取量化后模型大小
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'量化后模型大小: {quantized_size:.2f} MB')
    print(f'压缩比例: {(1 - quantized_size/original_size)*100:.1f}%')
    print(f'量化模型已保存到: {output_path}')

    return quantized_model


def test_quantized_model(model, test_input_size=(1, 3, 256, 256)):
    """
    测试量化模型的推理

    Args:
        model: 量化后的模型
        test_input_size: 测试输入尺寸
    """
    import time

    model.eval()
    device = torch.device('cpu')

    # 创建测试输入
    test_input = torch.randn(test_input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)

    # 测试推理时间
    num_runs = 20
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    print(f'\n推理性能测试 (输入尺寸: {test_input_size[2]}x{test_input_size[3]}):')
    print(f'平均推理时间: {avg_time:.2f} ms')
    print(f'吞吐量: {1000/avg_time:.2f} images/sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='动态量化ESRGAN模型')
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--output_path', type=str, default='./quantization/quantized_model.pth',
                        help='量化后模型保存路径')
    parser.add_argument('--light_model', action='store_true', help='使用轻量化模型')
    parser.add_argument('--test', action='store_true', help='测试量化后模型的推理性能')

    args = parser.parse_args()

    # 执行量化
    quantized_model = dynamic_quantize(args.model_path, args.output_path, args.light_model)

    # 可选：测试性能
    if args.test:
        test_quantized_model(quantized_model)
        print('\n提示: 量化模型主要优化CPU推理速度，GPU上可能没有加速效果')
