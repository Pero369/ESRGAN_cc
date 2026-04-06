"""
测试退化模块功能

此脚本用于验证图像退化模块的各项功能：
1. 测试单独的退化操作（模糊、噪声、JPEG压缩）
2. 测试组合退化效果
3. 生成对比图像
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.degradation import DegradationPipeline
from config import Config
from PIL import Image
import numpy as np

def test_degradation():
    """测试退化模块的各项功能"""

    # 检查测试图像
    test_image_path = './data/train_hr'
    if not os.path.exists(test_image_path):
        print(f"错误：找不到训练数据目录 {test_image_path}")
        return

    # 获取第一张图像作为测试
    image_files = [f for f in os.listdir(test_image_path)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"错误：{test_image_path} 中没有图像文件")
        return

    test_img_path = os.path.join(test_image_path, image_files[0])
    print(f"使用测试图像: {test_img_path}")

    # 读取图像
    img = Image.open(test_img_path).convert('RGB')
    # 裁剪到256x256便于观察
    w, h = img.size
    img = img.crop((0, 0, min(w, 256), min(h, 256)))

    # 创建输出目录
    output_dir = './degradation_test'
    os.makedirs(output_dir, exist_ok=True)

    # 保存原图
    img.save(f'{output_dir}/0_original.png')
    print(f"✓ 保存原图: {output_dir}/0_original.png")

    # 测试1：只启用高斯模糊
    print("\n=== 测试1：高斯模糊 ===")
    config_blur = type('Config', (), {
        'enable_degradation': True,
        'enable_blur': True,
        'enable_noise': False,
        'enable_jpeg': False,
        'blur_kernel_range': (7, 21),
        'blur_sigma_range': (0.1, 3.0),
        'blur_prob': 1.0  # 100%应用
    })()

    pipeline_blur = DegradationPipeline(config_blur)
    img_blur = pipeline_blur.apply(img)
    img_blur.save(f'{output_dir}/1_blur.png')
    print(f"✓ 保存模糊图像: {output_dir}/1_blur.png")

    # 测试2：只启用高斯噪声
    print("\n=== 测试2：高斯噪声 ===")
    config_noise = type('Config', (), {
        'enable_degradation': True,
        'enable_blur': False,
        'enable_noise': True,
        'enable_jpeg': False,
        'noise_sigma_range': (0, 15),
        'noise_prob': 1.0
    })()

    pipeline_noise = DegradationPipeline(config_noise)
    img_noise = pipeline_noise.apply(img)
    img_noise.save(f'{output_dir}/2_noise.png')
    print(f"✓ 保存噪声图像: {output_dir}/2_noise.png")

    # 测试3：只启用JPEG压缩
    print("\n=== 测试3：JPEG压缩 ===")
    config_jpeg = type('Config', (), {
        'enable_degradation': True,
        'enable_blur': False,
        'enable_noise': False,
        'enable_jpeg': True,
        'jpeg_quality_range': (60, 95),
        'jpeg_prob': 1.0
    })()

    pipeline_jpeg = DegradationPipeline(config_jpeg)
    img_jpeg = pipeline_jpeg.apply(img)
    img_jpeg.save(f'{output_dir}/3_jpeg.png')
    print(f"✓ 保存JPEG压缩图像: {output_dir}/3_jpeg.png")

    # 测试4：组合所有退化（使用Config配置）
    print("\n=== 测试4：组合退化（使用Config配置） ===")
    pipeline_combined = DegradationPipeline(Config)
    img_combined = pipeline_combined.apply(img)
    img_combined.save(f'{output_dir}/4_combined.png')
    print(f"✓ 保存组合退化图像: {output_dir}/4_combined.png")

    # 测试5：生成多个随机退化样本
    print("\n=== 测试5：生成随机退化样本 ===")
    for i in range(5):
        img_random = pipeline_combined.apply(img)
        img_random.save(f'{output_dir}/5_random_{i+1}.png')
        print(f"✓ 保存随机样本{i+1}: {output_dir}/5_random_{i+1}.png")

    # 测试6：测试下采样后的效果
    print("\n=== 测试6：退化+下采样 ===")
    img_degraded = pipeline_combined.apply(img)
    img_lr = img_degraded.resize((img.width // 4, img.height // 4), Image.BICUBIC)
    img_lr.save(f'{output_dir}/6_degraded_lr.png')
    print(f"✓ 保存退化+下采样图像: {output_dir}/6_degraded_lr.png")

    # 对比：不退化的下采样
    img_lr_clean = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)
    img_lr_clean.save(f'{output_dir}/6_clean_lr.png')
    print(f"✓ 保存干净下采样图像: {output_dir}/6_clean_lr.png")

    print("\n" + "="*50)
    print("测试完成！")
    print(f"所有测试图像已保存到: {output_dir}/")
    print("\n图像说明：")
    print("  0_original.png      - 原始图像")
    print("  1_blur.png          - 仅高斯模糊")
    print("  2_noise.png         - 仅高斯噪声")
    print("  3_jpeg.png          - 仅JPEG压缩")
    print("  4_combined.png      - 组合退化")
    print("  5_random_*.png      - 随机退化样本（5张）")
    print("  6_degraded_lr.png   - 退化+下采样")
    print("  6_clean_lr.png      - 干净下采样（对比）")
    print("\n请检查图像以验证退化效果是否正常。")
    print("="*50)

def test_color_channels():
    """测试BGR/RGB转换是否正确"""
    print("\n" + "="*50)
    print("测试颜色通道转换...")
    print("="*50)

    # 创建纯色测试图像
    output_dir = './degradation_test'
    os.makedirs(output_dir, exist_ok=True)

    # 创建红色、绿色、蓝色纯色图像
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255)
    }

    config_blur = type('Config', (), {
        'enable_degradation': True,
        'enable_blur': True,
        'enable_noise': False,
        'enable_jpeg': False,
        'blur_kernel_range': (7, 7),
        'blur_sigma_range': (2.0, 2.0),
        'blur_prob': 1.0
    })()

    pipeline = DegradationPipeline(config_blur)

    for color_name, rgb in colors.items():
        # 创建纯色图像
        img = Image.new('RGB', (100, 100), rgb)
        img.save(f'{output_dir}/color_test_{color_name}_original.png')

        # 应用模糊
        img_blurred = pipeline.apply(img)
        img_blurred.save(f'{output_dir}/color_test_{color_name}_blurred.png')

        # 检查颜色是否保持
        arr_original = np.array(img)
        arr_blurred = np.array(img_blurred)

        print(f"\n{color_name.upper()}:")
        print(f"  原始: R={arr_original[50,50,0]}, G={arr_original[50,50,1]}, B={arr_original[50,50,2]}")
        print(f"  模糊: R={arr_blurred[50,50,0]}, G={arr_blurred[50,50,1]}, B={arr_blurred[50,50,2]}")

        # 验证主色调是否保持
        if color_name == 'red' and arr_blurred[50,50,0] < 200:
            print("  ⚠️  警告：红色通道值异常低！")
        elif color_name == 'green' and arr_blurred[50,50,1] < 200:
            print("  ⚠️  警告：绿色通道值异常低！")
        elif color_name == 'blue' and arr_blurred[50,50,2] < 200:
            print("  ⚠️  警告：蓝色通道值异常低！")
        else:
            print("  ✓ 颜色通道正常")

    print("\n颜色测试图像已保存到: {}/color_test_*.png".format(output_dir))
    print("="*50)

if __name__ == '__main__':
    print("="*50)
    print("ESRGAN 退化模块测试")
    print("="*50)

    # 测试退化功能
    test_degradation()

    # 测试颜色通道
    test_color_channels()

    print("\n全部测试完成！")
