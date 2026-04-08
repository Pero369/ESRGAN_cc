"""
使用ONNX Runtime进行推理
提供跨平台、高性能的推理能力
"""
import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse
import os
import time


def load_image(image_path):
    """
    加载图像并转换为模型输入格式

    Args:
        image_path: 图像路径

    Returns:
        numpy array: [1, 3, H, W] 范围[0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    # 添加batch维度
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def save_image(img_array, output_path):
    """
    保存图像

    Args:
        img_array: [1, 3, H, W] 范围[0, 1]
        output_path: 输出路径
    """
    # 移除batch维度并转换为HWC
    img_array = np.squeeze(img_array, axis=0)
    img_array = np.transpose(img_array, (1, 2, 0))
    # 转换为uint8
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    # 保存
    img = Image.fromarray(img_array)
    img.save(output_path)


def inference_onnx(onnx_model_path, input_path, output_dir, benchmark=False):
    """
    使用ONNX Runtime进行推理

    Args:
        onnx_model_path: ONNX模型路径
        input_path: 输入图像路径或目录
        output_dir: 输出目录
        benchmark: 是否进行性能测试
    """
    # 创建ONNX Runtime会话
    print(f'加载ONNX模型: {onnx_model_path}')

    # 配置推理选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 创建会话（自动选择最佳执行提供者）
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, sess_options, providers=providers)

    print(f'使用执行提供者: {session.get_providers()}')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理单张图像或目录
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    print(f'找到 {len(image_paths)} 张图像')

    # 推理
    total_time = 0
    for img_path in image_paths:
        print(f'处理: {img_path}')

        # 加载图像
        input_array = load_image(img_path)

        # 推理
        start_time = time.time()
        output_array = session.run(
            None,
            {'input': input_array}
        )[0]
        inference_time = time.time() - start_time
        total_time += inference_time

        # 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_sr.png')
        save_image(output_array, output_path)

        print(f'  推理时间: {inference_time*1000:.2f} ms')
        print(f'  保存到: {output_path}')

    # 性能统计
    if len(image_paths) > 0:
        avg_time = total_time / len(image_paths)
        print(f'\n性能统计:')
        print(f'  平均推理时间: {avg_time*1000:.2f} ms')
        print(f'  吞吐量: {1/avg_time:.2f} images/sec')

    # 可选：性能基准测试
    if benchmark:
        print('\n运行性能基准测试...')
        test_input = load_image(image_paths[0])

        # 预热
        for _ in range(5):
            _ = session.run(None, {'input': test_input})

        # 测试
        num_runs = 50
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {'input': test_input})
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        print(f'  基准测试 ({num_runs} 次运行):')
        print(f'  平均推理时间: {avg_time*1000:.2f} ms')
        print(f'  吞吐量: {1/avg_time:.2f} images/sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用ONNX Runtime进行超分辨率推理')
    parser.add_argument('--model_path', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--input_path', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output_dir', type=str, default='./results_onnx', help='输出目录')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')

    args = parser.parse_args()

    inference_onnx(args.model_path, args.input_path, args.output_dir, args.benchmark)
