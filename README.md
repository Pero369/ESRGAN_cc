# ESRGAN 超分辨率模型

基于PyTorch的ESRGAN（Enhanced Super-Resolution Generative Adversarial Networks）基础实现，用于图像4倍超分辨率重建。

## 特性

- RRDB（Residual in Residual Dense Block）网络结构
- 相对判别器（Relativistic GAN）
- 感知损失（基于VGG19特征）
- 两阶段训练：PSNR预训练 + GAN训练

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

```bash
python train.py
```

训练分为两个阶段：
1. PSNR预训练（50 epochs）：使用L1像素损失
2. GAN训练（150 epochs）：使用感知损失和对抗损失

检查点保存在 `checkpoints/` 目录，样本图像保存在 `samples/` 目录。

## 推理

单张图像：
```bash
python test.py --input_path ./test.png --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth
```

批量处理：
```bash
python test.py --input_path ./test_images --output_dir ./results --model_path ./checkpoints/generator_gan_150.pth
```

## 配置

在 `config.py` 中修改训练参数：
- `num_rrdb_blocks`: RRDB块数量（默认23）
- `batch_size`: 批次大小（默认16）
- `hr_size`: 高分辨率patch大小（默认128）
- `lr_g`, `lr_d`: 学习率（默认1e-4）

## 模型结构

- **生成器**: 23个RRDB块 + PixelShuffle上采样
- **判别器**: VGG风格卷积网络
- **输入**: 任意尺寸RGB图像
- **输出**: 4倍放大的RGB图像

## 预期效果

- PSNR: 28-30dB
- SSIM: 0.85-0.90
- 训练时间: 约2-3天（单GPU）
