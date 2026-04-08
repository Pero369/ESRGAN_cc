# ESRGAN 超分辨率模型

基于PyTorch的ESRGAN（Enhanced Super-Resolution Generative Adversarial Networks）实现，用于图像4倍超分辨率重建。支持原版重型模型和轻量化版本。

## 特性

- RRDB（Residual in Residual Dense Block）网络结构
- 相对判别器（Relativistic GAN）
- 感知损失（基于VGG19特征）
- 两阶段训练：PSNR预训练 + GAN训练
- **轻量化版本**：Depthwise Separable Convolution + 8个RRDB块，参数量 ~2.3M（原版 ~16M）
- **注意力机制**：通道注意力（SE）和CBAM，提升细节恢复能力
- **边缘感知损失**：梯度损失强化边缘清晰度
- **推理加速**：INT8量化和ONNX导出，支持跨平台部署

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

将高分辨率训练图像放入以下目录：
```
data/train_hr/  # 训练集
data/val_hr/    # 验证集
```

## 训练

### 基础训练

```bash
python train.py
```

训练分为两个阶段：
1. PSNR预训练（50 epochs）：使用L1像素损失
2. GAN训练（150 epochs）：使用感知损失和对抗损失

检查点保存在 `checkpoints/` 目录，样本图像保存在 `samples/` 目录。

**模型选择**：
- 训练会自动根据 `config.py` 中的 `use_light_model` 参数选择模型
- `use_light_model = True`：训练轻量化模型（默认）
- `use_light_model = False`：训练原版模型

### 批量参数扫描训练

使用 `train_sweep.py` 进行批量实验，测试不同参数组合：

```bash
python train_sweep.py
```

在脚本中定义实验参数：
```python
EXPERIMENTS = [
    # 轻量化模型实验
    {"use_light_model": True, "lambda_pixel": 0.08},
    {"use_light_model": True, "lambda_pixel": 0.11},
    
    # 自定义轻量化配置
    {
        "use_light_model": True,
        "light_num_rrdb_blocks": 6,
        "light_num_channels": 32,
        "lambda_pixel": 0.10
    },
    
    # 原版模型对比实验
    {"use_light_model": False, "lambda_pixel": 0.08},
]
```

每组实验的结果会保存到独立的子目录中。

## 推理

### 轻量化模型推理

单张图像：
```bash
python test.py --input_path ./test.png --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth --light_model
```

批量处理：
```bash
python test.py --input_path ./test_images --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth --light_model
```

### 原版模型推理

单张图像：
```bash
python test.py --input_path ./test.png --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth
```

批量处理：
```bash
python test.py --input_path ./test_images --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth
```

**参数说明**：
- `--input_path`: 输入图像路径或目录
- `--output_dir`: 输出目录（默认 `./results`）
- `--model_path`: 模型权重文件路径
- `--light_model`: 使用轻量化模型（如果checkpoint是轻量化模型训练的，必须加此参数）

## 配置

在 `config.py` 中修改训练参数：
- `num_rrdb_blocks`: RRDB块数量（默认23）
- `batch_size`: 批次大小（默认16）
- `hr_size`: 高分辨率patch大小（默认128）
- `lr_g`, `lr_d`: 学习率（默认1e-4）

**轻量化开关**：
- `use_light_model`: 是否使用轻量版（默认 `True`）
- `light_num_rrdb_blocks`: 轻量版RRDB块数（默认8）
- `light_num_channels`: 轻量版通道数（默认32）

**注意力机制配置**：
- `enable_attention`: 是否启用注意力（默认 `False`）
- `attention_type`: 注意力类型 `'CA'`（通道）或 `'CBAM'`
- `attention_reduction`: 降维比例（默认16）
- `attention_position`: 注意力位置 `'rrdb'` 或 `'dense'`

**边缘感知损失配置**：
- `enable_gradient_loss`: 是否启用梯度损失（默认 `False`）
- `lambda_gradient`: 梯度损失权重（建议0.05~0.2）
- `gradient_loss_stage`: 应用阶段 `'psnr'`, `'gan'`, `'both'`

## 模型结构

| 版本 | RRDB块数 | 通道数 | 卷积类型 | 参数量 |
|------|---------|--------|---------|--------|
| 原版 | 23 | 64 | 标准卷积 | ~16M |
| 轻量版 | 8 | 32 | Depthwise Separable | ~2.3M |

- **判别器**: VGG风格卷积网络
- **输入**: 任意尺寸RGB图像
- **输出**: 4倍放大的RGB图像

## 预期效果

- PSNR: 28-30dB
- SSIM: 0.85-0.90
- 训练时间: 约2-3天（单GPU）

## 高级功能

### 性能基准测试

测试模型的推理速度和内存占用：

```bash
# 测试单个模型
python benchmark.py --model_path ./checkpoints/generator_gan_150.pth --light_model --device cuda

# 对比多个模型
python benchmark.py --compare ./checkpoints/light_model.pth:light ./checkpoints/original_model.pth:original
```

### 模型量化

将模型量化为INT8，减小模型大小并提升CPU推理速度：

```bash
python quantization/dynamic_quant.py --model_path ./checkpoints/generator_gan_150.pth --light_model --output_path ./quantization/quantized_model.pth --test
```

### ONNX导出

导出为ONNX格式，支持跨平台部署：

```bash
# 导出ONNX模型
python export_onnx.py --model_path ./checkpoints/generator_gan_150.pth --light_model --output_path ./onnx_models/esrgan.onnx --verify

# 使用ONNX Runtime推理
python inference_onnx.py --model_path ./onnx_models/esrgan.onnx --input_path ./test.png --output_dir ./results_onnx --benchmark
```
