# train_sweep.py 使用指南

批量参数扫描训练脚本，支持多种实验配置的自动化训练。

## 功能特性

- ✅ 支持所有新增优化参数（注意力、梯度损失等）
- ✅ 自动生成简洁的实验名称
- ✅ 详细的训练日志和损失分解
- ✅ 异常处理和实验继续选项
- ✅ 预定义多种实验模板

## 支持的参数

### 模型配置
- `use_light_model`: 是否使用轻量化模型
- `light_num_rrdb_blocks`: 轻量版RRDB块数
- `light_num_channels`: 轻量版通道数
- `num_rrdb_blocks`: 原版RRDB块数
- `num_channels`: 原版通道数

### 注意力机制
- `enable_attention`: 是否启用注意力
- `attention_type`: 注意力类型 ('CA', 'CBAM')
- `attention_reduction`: 降维比例
- `attention_position`: 注意力位置 ('rrdb', 'dense')

### 边缘感知损失
- `enable_gradient_loss`: 是否启用梯度损失
- `lambda_gradient`: 梯度损失权重
- `gradient_loss_stage`: 应用阶段 ('psnr', 'gan', 'both')

### 损失权重
- `lambda_pixel`: 像素损失权重
- `lambda_perceptual`: 感知损失权重
- `lambda_adversarial`: 对抗损失权重

### 训练参数
- `batch_size`: 批次大小
- `lr_g`: 生成器学习率
- `lr_d`: 判别器学习率
- `num_epochs_psnr`: PSNR预训练轮数
- `num_epochs_gan`: GAN训练轮数

## 预定义实验模板

### 1. 损失权重调优
```python
LOSS_WEIGHT_EXPERIMENTS = [
    {"lambda_pixel": 0.08},
    {"lambda_pixel": 0.11},
    {"lambda_pixel": 0.13},
]
```

### 2. 注意力机制对比
```python
ATTENTION_EXPERIMENTS = [
    {"enable_attention": False},                                    # 基线
    {"enable_attention": True, "attention_type": "CA"},            # 通道注意力
    {"enable_attention": True, "attention_type": "CBAM"},          # CBAM
    {"enable_attention": True, "attention_type": "CA",
     "attention_position": "dense"},                               # DenseBlock级
]
```

### 3. 边缘感知损失对比
```python
GRADIENT_LOSS_EXPERIMENTS = [
    {"enable_gradient_loss": False},                               # 基线
    {"enable_gradient_loss": True, "lambda_gradient": 0.05},      # 低权重
    {"enable_gradient_loss": True, "lambda_gradient": 0.1},       # 中权重
    {"enable_gradient_loss": True, "lambda_gradient": 0.2},       # 高权重
]
```

### 4. 消融实验
```python
ABLATION_EXPERIMENTS = [
    # 基线
    {"enable_attention": False, "enable_gradient_loss": False},

    # 单独添加注意力
    {"enable_attention": True, "attention_type": "CA",
     "enable_gradient_loss": False},

    # 单独添加梯度损失
    {"enable_attention": False,
     "enable_gradient_loss": True, "lambda_gradient": 0.1},

    # 完整版（两者都有）
    {"enable_attention": True, "attention_type": "CA",
     "enable_gradient_loss": True, "lambda_gradient": 0.1},
]
```

### 5. 轻量化程度对比
```python
LIGHTWEIGHT_EXPERIMENTS = [
    {"light_num_rrdb_blocks": 6, "light_num_channels": 32},
    {"light_num_rrdb_blocks": 8, "light_num_channels": 32},
    {"light_num_rrdb_blocks": 10, "light_num_channels": 32},
    {"light_num_rrdb_blocks": 8, "light_num_channels": 24},
    {"light_num_rrdb_blocks": 8, "light_num_channels": 40},
]
```

### 6. 综合优化实验
```python
COMPREHENSIVE_EXPERIMENTS = [
    # 原版模型（对比基线）
    {"use_light_model": False, "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化基线
    {"use_light_model": True, "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化 + 注意力
    {"use_light_model": True, "enable_attention": True, "attention_type": "CA",
     "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化 + 梯度损失
    {"use_light_model": True, "enable_gradient_loss": True, "lambda_gradient": 0.1,
     "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化 + 注意力 + 梯度损失（完整版）
    {"use_light_model": True, "enable_attention": True, "attention_type": "CA",
     "enable_gradient_loss": True, "lambda_gradient": 0.1,
     "num_epochs_psnr": 10, "num_epochs_gan": 10},
]
```

## 使用方法

### 步骤1: 选择实验模板

编辑 `train_sweep.py`，修改最后一行：

```python
# 选择要运行的实验组
EXPERIMENTS = ABLATION_EXPERIMENTS  # 修改这里
```

可选项：
- `LOSS_WEIGHT_EXPERIMENTS`
- `ATTENTION_EXPERIMENTS`
- `GRADIENT_LOSS_EXPERIMENTS`
- `ABLATION_EXPERIMENTS`
- `LIGHTWEIGHT_EXPERIMENTS`
- `COMPREHENSIVE_EXPERIMENTS`

### 步骤2: 运行训练

```bash
python train_sweep.py
```

### 步骤3: 查看结果

```
checkpoints/
├── light_att_CA_rrdb/              # 实验1
│   ├── generator_psnr_10.pth
│   ├── generator_gan_10.pth
│   └── ...
├── light_grad0.1/                  # 实验2
│   └── ...
└── light_att_CA_rrdb_grad0.1/      # 实验3
    └── ...

samples/
├── light_att_CA_rrdb/
│   ├── psnr_epoch10_sample0.png
│   └── ...
└── ...
```

## 实验名称规则

脚本会自动生成简洁的实验名称：

| 配置 | 实验名称 |
|------|---------|
| 轻量化 + 通道注意力(RRDB级) | `light_att_CA_rrdb` |
| 轻量化 + 梯度损失(0.1) | `light_grad0.1` |
| 轻量化 + 注意力 + 梯度损失 | `light_att_CA_rrdb_grad0.1` |
| 8个RRDB块 + 32通道 | `light_rrdb8_ch32` |
| 像素损失权重0.11 | `light_pix0.11` |

## 训练日志示例

```
======================================================================
批量参数扫描训练
======================================================================
总实验数: 4
======================================================================

======================================================================
实验 1/4: light
======================================================================

============================================================
实验配置: light
============================================================
模型: 轻量化生成器
  RRDB块: 8
  通道数: 32
损失权重:
  Pixel: 0.13
  Perceptual: 1.0
  Adversarial: 0.005
训练参数:
  Batch size: 16
  PSNR epochs: 30
  GAN epochs: 30
  Learning rate: G=0.0001, D=0.0001
============================================================

[light] 阶段1: PSNR预训练
PSNR 1/30: 100%|████████████| 50/50 [00:45<00:00,  1.10it/s]
Epoch 1, Total: 0.0523, Pixel: 0.0523

[light] 阶段2: GAN训练
GAN 1/30: 100%|████████████| 50/50 [01:23<00:00,  0.60it/s]
Epoch 1, G: 0.0234, D: 0.1234 [Pix: 0.0012, Perc: 0.0198, Adv: 0.0024]

✓ 实验 light 完成
```

## 自定义实验

### 创建自己的实验组

```python
MY_EXPERIMENTS = [
    {
        "use_light_model": True,
        "light_num_rrdb_blocks": 8,
        "light_num_channels": 32,
        "enable_attention": True,
        "attention_type": "CA",
        "enable_gradient_loss": True,
        "lambda_gradient": 0.1,
        "lambda_pixel": 0.11,
        "num_epochs_psnr": 50,
        "num_epochs_gan": 150,
    },
    # 添加更多配置...
]

EXPERIMENTS = MY_EXPERIMENTS
```

### 快速测试（减少训练轮数）

```python
QUICK_TEST = [
    {"enable_attention": True, "num_epochs_psnr": 5, "num_epochs_gan": 5},
    {"enable_gradient_loss": True, "num_epochs_psnr": 5, "num_epochs_gan": 5},
]

EXPERIMENTS = QUICK_TEST
```

## 异常处理

如果某个实验失败，脚本会：
1. 打印错误信息和堆栈跟踪
2. 询问是否继续下一个实验
3. 输入 `y` 继续，输入 `n` 停止

```
✗ 实验 light_att_CA_rrdb 失败: CUDA out of memory

是否继续下一个实验? (y/n): y
```

## 最佳实践

### 1. 先快速测试
```python
# 使用少量epoch快速验证配置
{"enable_attention": True, "num_epochs_psnr": 5, "num_epochs_gan": 5}
```

### 2. 逐步增加复杂度
```python
# 先测试单个优化
EXPERIMENTS = [
    {"enable_attention": False},
    {"enable_attention": True},
]

# 再测试组合
EXPERIMENTS = [
    {"enable_attention": True, "enable_gradient_loss": True},
]
```

### 3. 记录实验结果
创建 `experiments_log.md` 记录每个实验的结果：

```markdown
## 实验记录

### light_att_CA_rrdb
- 配置: 轻量化 + 通道注意力(RRDB级)
- PSNR: 28.5 dB
- 训练时间: 2小时
- 备注: 边缘细节有改善

### light_grad0.1
- 配置: 轻量化 + 梯度损失(0.1)
- PSNR: 28.3 dB
- 训练时间: 2.5小时
- 备注: 边缘更锐利，但略有过锐化
```

### 4. 使用对比脚本评估
```bash
# 训练完成后，使用对比脚本评估
python scripts/evaluate_quality.py \
  --model_path ./checkpoints/light_att_CA_rrdb/generator_gan_30.pth \
  --test_dir ./data/val_hr \
  --light_model
```

## 论文实验建议

### 消融实验（验证各模块贡献）
```python
EXPERIMENTS = ABLATION_EXPERIMENTS
```
运行后对比：
- 基线 vs +注意力：验证注意力的贡献
- 基线 vs +梯度损失：验证梯度损失的贡献
- 基线 vs 完整版：验证组合效果

### 参数敏感性分析
```python
# 测试梯度损失权重
EXPERIMENTS = [
    {"enable_gradient_loss": True, "lambda_gradient": 0.05},
    {"enable_gradient_loss": True, "lambda_gradient": 0.1},
    {"enable_gradient_loss": True, "lambda_gradient": 0.15},
    {"enable_gradient_loss": True, "lambda_gradient": 0.2},
]
```

### 轻量化程度分析
```python
EXPERIMENTS = LIGHTWEIGHT_EXPERIMENTS
```
分析参数量与性能的权衡。

## 常见问题

### Q1: 如何暂停和恢复训练？
A: 脚本不支持断点续训。建议：
- 减少单次实验的epoch数
- 使用多个小实验组分批运行

### Q2: 显存不足怎么办？
A: 减小batch_size：
```python
{"batch_size": 8, "enable_attention": True}
```

### Q3: 如何只训练GAN阶段？
A: 设置PSNR epoch为0：
```python
{"num_epochs_psnr": 0, "num_epochs_gan": 30}
```

### Q4: 如何使用预训练模型？
A: 需要修改 `run_experiment` 函数，添加模型加载逻辑。

## 总结

train_sweep.py 是进行批量实验的强大工具，特别适合：
- 超参数调优
- 消融实验
- 对比实验
- 参数敏感性分析

合理使用可以大大提高实验效率，为论文提供充分的实验数据。
