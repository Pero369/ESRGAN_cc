# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

请始终使用简体中文与我对话，并在回答时保持专业、简洁。

## 项目概述

这是一个ESRGAN（Enhanced Super-Resolution Generative Adversarial Networks）的PyTorch实现，用于4倍图像超分辨率重建。

## 核心架构

### 两阶段训练流程

训练必须按顺序执行两个阶段，不可跳过：

1. **PSNR预训练阶段**（50 epochs）
   - 仅训练生成器
   - 使用L1像素损失优化PSNR指标
   - 为GAN训练提供良好的初始化

2. **GAN训练阶段**（150 epochs）
   - 交替训练判别器和生成器
   - 使用感知损失（VGG19特征）+ 相对对抗损失
   - 每50 epochs学习率衰减0.5

### 网络结构关键点

**RRDB模块**（models/rrdb.py）：
- DenseBlock：5层卷积密集连接，每层输出concat到后续层
- RRDB：3个DenseBlock串联
- 残差缩放beta=0.2：防止深层网络训练不稳定

**生成器**（models/generator.py）：
- 初始卷积 → 23个RRDB块 → 中间卷积（与初始特征残差连接）→ 2个PixelShuffle上采样块 → 输出卷积
- PixelShuffle用于上采样，避免转置卷积的棋盘效应

**判别器**（models/discriminator.py）：
- VGG风格架构，用于相对判别器（Relativistic Average GAN）
- 判断"SR比HR更真"而非简单的"真/假"

**损失函数**（models/losses.py）：
- PerceptualLoss：使用预训练VGG19的features.34层（conv5_4），不使用ImageNet标准化
- GANLoss：相对判别器损失实现
- PixelLoss：L1损失，用于PSNR预训练和GAN训练（防止颜色偏移）

**退化模块**（data/degradation.py）：
- DegradationPipeline：模拟真实场景的图像退化
- 三种退化操作：高斯模糊、JPEG压缩、高斯噪声
- 在bicubic下采样前应用，生成更真实的低分辨率图像
- 注意：使用OpenCV时已处理BGR/RGB转换

## 常用命令

### 环境设置
```bash
pip install -r requirements.txt
```

### 数据准备
将高分辨率图像放入：
- `data/train_hr/` - 训练集
- `data/val_hr/` - 验证集（可选）

### 训练
```bash
python train.py
```

### 测试退化模块
```bash
python test_degradation.py
```
生成测试图像到 `degradation_test/` 目录，验证退化效果和颜色通道转换。

### 推理
单张图像：
```bash
python test.py --input_path ./test.png --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth
```

批量处理：
```bash
python test.py --input_path ./test_images --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth
```

## 配置修改

所有训练参数在 `config.py` 中集中管理：
- 网络参数：`num_rrdb_blocks`（23）、`num_channels`（64）、`scale_factor`（4）
- 训练参数：`batch_size`（16）、学习率、epoch数
- 损失权重：`lambda_pixel`（0.01）、`lambda_perceptual`（1.0）、`lambda_adversarial`（0.005）
- 数据参数：`hr_size`（128）、`lr_size`（32）
- 退化参数：`enable_degradation`（True/False）、各退化操作的开关和强度范围

## 重要约定

### 数据处理流程
- LR图像自动生成：HR图像 → [可选退化] → bicubic下采样（4倍）
- 退化顺序：高斯模糊 → JPEG压缩 → 高斯噪声（模拟真实图像采集过程）
- 退化模块可通过 `enable_degradation` 开关控制

### 训练流程
- 检查点保存：每10 epochs保存一次到 `checkpoints/`
- 样本图像：每10 epochs保存4张样本到 `samples/`
- 学习率衰减：GAN训练阶段每50 epochs衰减0.5
- 残差缩放beta=0.2：修改此值会影响训练稳定性

### 颜色准确性
- GAN阶段必须包含PixelLoss：防止颜色偏移（偏蓝/偏红）
- PerceptualLoss不使用ImageNet标准化：避免通道不平衡导致的颜色偏移
- 如遇颜色问题，参考 `docs/color_fix_analysis.md` 和 `docs/color_tuning_guide.md`

### OpenCV使用注意
- OpenCV默认BGR格式，PIL默认RGB格式
- 退化模块中已处理颜色空间转换（RGB→BGR→处理→RGB）
- 修改退化模块时务必注意颜色通道顺序

## 文档参考

- `docs/color_fix_analysis.md` - 颜色偏移问题的详细分析和修复记录
- `docs/color_tuning_guide.md` - 颜色微调和损失权重调优指南
