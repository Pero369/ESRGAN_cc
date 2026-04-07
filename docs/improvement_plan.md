# ESRGAN 改进方案文档

## 概述

在基础ESRGAN项目上实现4个改进模块，支持消融实验。每个模块通过 `config.py` 开关独立控制。

---

## 改进1：频域损失（FFT Loss）

**目标**：补充高频细节约束，改善纹理过平滑问题

**原理**：对SR和HR图像做快速傅里叶变换，计算幅度谱的L1损失，迫使网络在频域层面对齐高频细节。

**改动文件**：
- `models/losses.py`：新增 `FFTLoss` 类
- `models/__init__.py`：导出 `FFTLoss`
- `config.py`：新增 `enable_fft_loss = True`，`lambda_fft = 0.1`
- `train.py`：GAN阶段 g_loss 加入 FFT 损失项

---

## 改进2：自适应损失权重调度

**目标**：训练前期 `lambda_pixel` 大（稳定颜色），后期线性衰减（释放GAN细节生成能力）

**原理**：GAN训练初期像素损失权重高，保证颜色不偏移；随训练进行逐步降低，让感知损失和对抗损失主导，生成更丰富的纹理细节。

**调度公式**：
```
lambda_pixel_t = lambda_pixel * (1 - adaptive_pixel_decay * t / num_epochs_gan)
```
从初始值线性衰减到 `(1 - adaptive_pixel_decay)` 倍。

**改动文件**：
- `config.py`：新增 `enable_adaptive_pixel_weight = True`，`adaptive_pixel_decay = 0.8`
- `train.py`：GAN训练循环中按 epoch 动态计算当前权重

---

## 改进3：CBAM 注意力机制

**目标**：让网络聚焦重要特征区域，提升细节恢复能力

**原理**：CBAM（Convolutional Block Attention Module）由通道注意力和空间注意力串联组成，插入每个 RRDB 块末尾，自适应地强调重要通道和空间位置。

**结构**：
```
ChannelAttention: 全局平均池化 + 全局最大池化 → 共享MLP → sigmoid → 加权
SpatialAttention: 通道维度avg+max → 7×7卷积 → sigmoid → 加权
CBAM: ChannelAttention → SpatialAttention
```

**改动文件**：
- `models/rrdb.py`：新增 `ChannelAttention`、`SpatialAttention`、`CBAM` 类；`RRDB` 末尾插入 CBAM
- `models/generator.py`：将 `enable_cbam` 参数传入 RRDB
- `config.py`：新增 `enable_cbam = True`

---

## 改进4：二阶退化管道

**目标**：模拟更复杂的真实退化，提升模型对真实低质量图像的泛化能力

**原理**：参考 Real-ESRGAN，将退化操作应用两次。第一阶模拟主要退化，第二阶用更轻微的参数模拟二次退化（如传输、再压缩等），使训练数据更贴近真实场景。

**退化顺序**：
```
原图 → [模糊→JPEG→噪声]（第一阶）→ [模糊→JPEG→噪声]（第二阶，参数更轻微）→ 下采样 → LR
```

**改动文件**：
- `data/degradation.py`：`apply()` 方法中增加第二阶退化逻辑
- `config.py`：新增 `enable_second_order = True` 及第二阶参数

---

## 涉及文件汇总

| 文件 | 改动内容 |
|------|---------|
| `models/losses.py` | 新增 FFTLoss |
| `models/rrdb.py` | 新增 CBAM 相关模块 |
| `models/generator.py` | RRDB 传入 enable_cbam |
| `models/__init__.py` | 导出 FFTLoss |
| `data/degradation.py` | 新增二阶退化逻辑 |
| `config.py` | 新增4个改进的开关和参数 |
| `train.py` | 接入 FFTLoss、自适应权重调度 |

---

## 消融实验配置

在 `train_sweep.py` 的 `EXPERIMENTS` 中：

```python
EXPERIMENTS = [
    {},                                          # baseline
    {"enable_fft_loss": True},                   # +FFT Loss
    {"enable_adaptive_pixel_weight": True},      # +自适应权重
    {"enable_cbam": True},                       # +CBAM注意力
    {"enable_second_order": True},               # +二阶退化
    {                                            # 全部开启
        "enable_fft_loss": True,
        "enable_adaptive_pixel_weight": True,
        "enable_cbam": True,
        "enable_second_order": True,
    },
]
```

---

## 验证方式

1. 运行 `python train_sweep.py` 跑消融实验
2. 用 `python test.py` 对同一张测试图推理，对比各组输出
3. 对比指标：PSNR、SSIM、视觉质量（颜色准确性、纹理细节、伪影）








  改动文件：

  ┌─────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │        文件         │                               改动内容                               │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ config.py           │ 新增8个开关和参数                                                    │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ models/losses.py    │ 新增 FFTLoss                                                         │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ models/rrdb.py      │ 新增 CBAM、ChannelAttention、SpatialAttention；RRDB 支持 enable_cbam │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ models/generator.py │ Generator 传入 enable_cbam                                           │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ models/__init__.py  │ 导出 FFTLoss                                                         │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ data/degradation.py │ 新增二阶退化逻辑和三个方法                                           │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ train.py            │ 接入 FFTLoss、自适应权重、enable_cbam                                │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ train_sweep.py      │ 同步以上所有改动                                                     │
  └─────────────────────┴──────────────────────────────────────────────────────────────────────┘

  默认全部关闭，按需在 config.py 开启：
  enable_fft_loss = True# 改进1
  enable_adaptive_pixel_weight = True  # 改进2
  enable_cbam = True                   # 改进3
  enable_second_order = True           # 改进4