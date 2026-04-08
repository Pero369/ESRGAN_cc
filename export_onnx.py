"""
ONNX模型导出脚本
将PyTorch模型导出为ONNX格式，支持跨平台部署
"""
import torch
import argparse
import os
from models import LightGenerator, Generator
from config import Config


def export_onnx(model_path, output_path, use_light_model=True, input_size=(1, 3, 256, 256)):
    """
    导出模型为ONNX格式

    Args:
        model_path: PyTorch模型路径
        output_path: ONNX模型保存路径
        use_light_model: 是否使用轻量化模型
        input_size: 输入尺寸 (batch, channels, height, width)
    """
    device = torch.device('cpu')  # ONNX导出在CPU上进行

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

    # 创建示例输入
    dummy_input = torch.randn(input_size).to(device)

    # 导出ONNX
    print(f'导出ONNX模型 (输入尺寸: {input_size})...')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    # 获取模型大小
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'ONNX模型大小: {onnx_size:.2f} MB')
    print(f'ONNX模型已保存到: {output_path}')

    return output_path


def verify_onnx(onnx_path, pytorch_model_path, use_light_model=True):
    """
    验证ONNX模型的正确性

    Args:
        onnx_path: ONNX模型路径
        pytorch_model_path: PyTorch模型路径
        use_light_model: 是否使用轻量化模型
    """
    import onnx
    import onnxruntime as ort
    import numpy as np

    print('\n验证ONNX模型...')

    # 检查ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('✓ ONNX模型格式正确')

    # 加载PyTorch模型
    device = torch.device('cpu')
    if use_light_model:
        pytorch_model = LightGenerator(
            Config.light_num_rrdb_blocks,
            Config.light_num_channels,
            enable_attention=Config.enable_attention,
            attention_type=Config.attention_type,
            attention_reduction=Config.attention_reduction,
            attention_position=Config.attention_position
        ).to(device)
    else:
        pytorch_model = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)

    pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    pytorch_model.eval()

    # 创建测试输入
    test_input = torch.randn(1, 3, 128, 128).to(device)

    # PyTorch推理
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).cpu().numpy()

    # ONNX推理
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(
        None,
        {'input': test_input.cpu().numpy()}
    )[0]

    # 比较输出
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    print(f'✓ 输出对比:')
    print(f'  最大差异: {max_diff:.6f}')
    print(f'  平均差异: {mean_diff:.6f}')

    if max_diff < 1e-4:
        print('✓ ONNX模型验证通过！')
    else:
        print('⚠ 警告: 输出差异较大，请检查模型')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出ESRGAN模型为ONNX格式')
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--output_path', type=str, default='./onnx_models/esrgan.onnx',
                        help='ONNX模型保存路径')
    parser.add_argument('--light_model', action='store_true', help='使用轻量化模型')
    parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 256, 256],
                        help='输入尺寸 (batch channels height width)')
    parser.add_argument('--verify', action='store_true', help='验证ONNX模型正确性')

    args = parser.parse_args()

    # 导出ONNX
    onnx_path = export_onnx(
        args.model_path,
        args.output_path,
        args.light_model,
        tuple(args.input_size)
    )

    # 可选：验证
    if args.verify:
        try:
            verify_onnx(onnx_path, args.model_path, args.light_model)
        except ImportError:
            print('\n提示: 需要安装 onnx 和 onnxruntime 来验证模型')
            print('运行: pip install onnx onnxruntime')
