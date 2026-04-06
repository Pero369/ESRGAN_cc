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
- PerceptualLoss：使用预训练VGG19的features.34层（conv5_4）
- GANLoss：相对判别器损失实现
- PixelLoss：L1损失，仅用于PSNR预训练

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
- 损失权重：`lambda_perceptual`（1.0）、`lambda_adversarial`（0.005）
- 数据参数：`hr_size`（128）、`lr_size`（32）

## 重要约定

- 数据集自动生成LR图像：使用bicubic下采样，无需手动准备LR图像
- 检查点保存：每10 epochs保存一次到 `checkpoints/`
- 样本图像：每10 epochs保存4张样本到 `samples/`
- 学习率衰减：GAN训练阶段每50 epochs衰减0.5
- 残差缩放beta=0.2：修改此值会影响训练稳定性
