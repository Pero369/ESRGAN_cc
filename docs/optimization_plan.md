# 轻量级ESRGAN优化计划

## 概述

本文档规划了三项高优先级优化，旨在提升轻量级ESRGAN的性能和实用性，为毕业设计论文提供充实的实验内容。

**优化目标**：
1. 注意力机制增强 - 提升细节恢复能力
2. 边缘感知损失 - 改善边缘清晰度
3. 推理加速优化 - 提升部署可行性

**预计完成时间**：7-10天

---

## 优化一：注意力机制增强

### 技术方案

在LightRRDB模块中集成轻量级注意力机制，使网络能够自适应地关注重要特征。

**选择方案：通道注意力（Channel Attention）**
- 基于Squeeze-and-Excitation (SE) 模块
- 参数量极小（~1% 增加）
- 实现简单，效果显著

**SE模块结构**：
```
输入特征 → 全局平均池化 → FC(降维) → ReLU → FC(升维) → Sigmoid → 通道加权
```

### 实现步骤

#### 1. 创建注意力模块（1天）

**文件**：`models/attention.py`

**内容**：
- `ChannelAttention` 类：SE模块实现
- `SpatialAttention` 类（可选）：空间注意力实现
- `CBAM` 类（可选）：通道+空间联合注意力

**关键参数**：
- `reduction_ratio=16`：通道降维比例（平衡性能和参数量）

#### 2. 集成到RRDB模块（0.5天）

**修改文件**：`models/rrdb.py`

**修改点**：
- 在 `LightDenseBlock` 的每个卷积层后添加 `ChannelAttention`
- 或在 `LightRRDB` 的每个 `DenseBlock` 后添加（推荐，参数更少）

**新增类**：
- `LightRRDBWithCA`：带通道注意力的轻量RRDB

#### 3. 配置开关（0.5天）

**修改文件**：`config.py`

**新增参数**：
```python
# 注意力机制配置
enable_attention = True          # 注意力总开关
attention_type = 'CA'            # 'CA', 'SA', 'CBAM'
attention_reduction = 16         # SE模块降维比例
attention_position = 'rrdb'      # 'dense', 'rrdb', 'both'
```

#### 4. 训练和评估（2天）

**实验设计**：
- **基线**：LightGenerator（无注意力）
- **对比组1**：LightGenerator + CA（RRDB级别）
- **对比组2**：LightGenerator + CA（DenseBlock级别）
- **对比组3**（可选）：LightGenerator + CBAM

**评估指标**：
- PSNR / SSIM（定量）
- 参数量 / FLOPs
- 推理速度
- 视觉质量（细节恢复、纹理清晰度）

**数据集**：
- 训练：现有训练集
- 测试：Set5 / Set14

### 预期效果

- **PSNR提升**：+0.2~0.5 dB
- **SSIM提升**：+0.005~0.01
- **参数增加**：<5%（~50K参数）
- **速度影响**：<10%降低
- **视觉改善**：纹理细节更丰富，高频信息恢复更好

---

## 优化二：边缘感知损失

### 技术方案

在现有损失函数基础上，增加边缘感知损失，强化网络对边缘结构的学习。

**选择方案：梯度损失（Gradient Loss）**
- 使用Sobel算子提取图像梯度
- 计算SR和HR图像梯度的L1距离
- 相比边缘检测损失更平滑，更易优化

**损失计算**：
```
Gradient Loss = L1(∇SR, ∇HR)
其中 ∇ = √(Gx² + Gy²)，Gx/Gy为Sobel算子响应
```

### 实现步骤

#### 1. 实现梯度损失（0.5天）

**修改文件**：`models/losses.py`

**新增类**：
```python
class GradientLoss(nn.Module):
    """边缘感知梯度损失"""
    def __init__(self):
        super().__init__()
        # Sobel算子（x方向和y方向）
        self.sobel_x = torch.tensor([...])
        self.sobel_y = torch.tensor([...])
        self.criterion = nn.L1Loss()
    
    def forward(self, sr, hr):
        # 计算梯度幅值
        sr_grad = self.compute_gradient(sr)
        hr_grad = self.compute_gradient(hr)
        return self.criterion(sr_grad, hr_grad)
```

#### 2. 集成到训练流程（0.5天）

**修改文件**：`train.py`

**修改点**：
- 在PSNR阶段：`loss = pixel_loss + λ_grad * gradient_loss`
- 在GAN阶段：`loss_g = ... + λ_grad * gradient_loss`

#### 3. 配置权重（0.5天）

**修改文件**：`config.py`

**新增参数**：
```python
# 边缘感知损失配置
enable_gradient_loss = True      # 梯度损失开关
lambda_gradient = 0.1            # 梯度损失权重（建议0.05~0.2）
gradient_loss_stage = 'both'     # 'psnr', 'gan', 'both'
```

#### 4. 消融实验（1.5天）

**实验设计**：
- **基线**：无梯度损失
- **对比组1**：λ_grad = 0.05
- **对比组2**：λ_grad = 0.1
- **对比组3**：λ_grad = 0.2

**重点评估**：
- 边缘清晰度（主观评价）
- 文字/建筑物边缘的PSNR
- 梯度相似度指标

### 预期效果

- **边缘清晰度**：显著提升（主观）
- **PSNR影响**：±0.1 dB（可能略降，因为优化目标变化）
- **SSIM提升**：+0.005~0.015（结构相似度改善）
- **视觉改善**：文字、线条、建筑物边缘更锐利

---

## 优化三：推理加速优化

### 技术方案

通过模型量化和格式转换，提升推理速度和部署灵活性。

**三个子任务**：
1. **动态量化**：PyTorch原生支持，快速实现
2. **静态量化**：需要校准数据，精度更高
3. **ONNX导出**：跨平台部署，支持多种推理引擎

### 实现步骤

#### 1. 动态量化（1天）

**新增文件**：`quantization/dynamic_quant.py`

**功能**：
- 加载训练好的模型
- 应用动态量化（INT8）
- 保存量化模型
- 对比量化前后的速度和精度

**关键代码**：
```python
import torch.quantization as quant

# 动态量化
model_quantized = quant.quantize_dynamic(
    model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
```

#### 2. 静态量化（1.5天）

**新增文件**：`quantization/static_quant.py`

**功能**：
- 准备校准数据集（100-200张图像）
- 配置量化参数（qconfig）
- 校准和量化
- 评估量化模型

**步骤**：
1. 模型准备：插入量化/反量化节点
2. 校准：在校准数据上运行，收集统计信息
3. 转换：将浮点模型转为INT8模型

#### 3. ONNX导出（1天）

**新增文件**：`export_onnx.py`

**功能**：
- 导出PyTorch模型为ONNX格式
- 验证ONNX模型正确性
- 使用ONNX Runtime推理
- 性能对比

**关键代码**：
```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

#### 4. 性能基准测试（1天）

**新增文件**：`benchmark.py`

**测试内容**：
- **模型大小**：原始 vs 量化 vs ONNX
- **推理速度**：CPU/GPU，不同batch size
- **精度损失**：PSNR/SSIM对比
- **内存占用**：峰值内存使用

**测试环境**：
- CPU：Intel/AMD（记录型号）
- GPU：NVIDIA（记录型号）
- 输入尺寸：256×256, 512×512, 1024×1024

#### 5. 部署示例（0.5天）

**新增文件**：`inference_onnx.py`

**功能**：
- 使用ONNX Runtime进行推理
- 提供简单的API接口
- 支持批量处理

### 预期效果

**动态量化**：
- 模型大小：减少75%（~10MB → ~2.5MB）
- 推理速度：提升1.5~2倍（CPU）
- 精度损失：PSNR -0.1~0.2 dB

**静态量化**：
- 模型大小：减少75%
- 推理速度：提升2~3倍（CPU）
- 精度损失：PSNR -0.05~0.15 dB（优于动态量化）

**ONNX导出**：
- 跨平台部署能力
- 推理速度：与PyTorch相当或略快
- 支持移动端/嵌入式设备部署

---

## 实验设计总览

### 对比实验矩阵

| 模型版本 | 注意力 | 梯度损失 | 量化 | 参数量 | PSNR | SSIM | 速度 |
|---------|--------|---------|------|--------|------|------|------|
| 基线（LightGen） | ✗ | ✗ | ✗ | 2.5M | 基准 | 基准 | 基准 |
| +CA | ✓ | ✗ | ✗ | 2.6M | +0.3 | +0.008 | -5% |
| +GradLoss | ✗ | ✓ | ✗ | 2.5M | +0.1 | +0.01 | 0% |
| +CA+GradLoss | ✓ | ✓ | ✗ | 2.6M | +0.4 | +0.015 | -5% |
| 最终版（量化） | ✓ | ✓ | ✓ | 0.65M | -0.1 | -0.002 | +150% |

### 消融实验

**目的**：验证每个模块的独立贡献

**实验组**：
1. 基线
2. 基线 + CA
3. 基线 + GradLoss
4. 基线 + CA + GradLoss（完整版）

**评估维度**：
- 定量指标：PSNR, SSIM, LPIPS
- 定性指标：边缘清晰度、纹理细节、颜色保真度
- 效率指标：参数量、FLOPs、推理时间

### 标准数据集评估

**测试集**：
- **Set5**：5张经典测试图像
- **Set14**：14张多样化图像
- **DIV2K验证集**：100张高质量图像（可选）

**对比基线**：
- Bicubic插值
- SRCNN
- 原版ESRGAN（重型）
- 本项目LightESRGAN（各优化版本）

---

## 时间安排

### 第1-2天：注意力机制
- Day 1：实现CA模块，集成到RRDB
- Day 2：训练和初步评估

### 第3-4天：边缘感知损失
- Day 3：实现梯度损失，集成到训练流程
- Day 4：消融实验和权重调优

### 第5-7天：推理加速
- Day 5：动态量化和静态量化
- Day 6：ONNX导出和验证
- Day 7：性能基准测试

### 第8-9天：综合实验
- Day 8：完整版模型训练（CA + GradLoss）
- Day 9：标准数据集评估

### 第10天：文档整理
- 实验结果汇总
- 可视化对比图表
- 论文素材准备

---

## 论文贡献点

### 技术创新
1. **轻量化设计**：Depthwise Separable Conv + 减少RRDB块
2. **注意力增强**：轻量级通道注意力机制
3. **边缘优化**：梯度感知损失函数
4. **部署优化**：量化和跨平台导出

### 实验验证
1. **消融实验**：验证各模块贡献
2. **对比实验**：与原版ESRGAN和其他方法对比
3. **效率分析**：参数量、速度、精度的权衡
4. **实用性验证**：真实场景图像测试

### 应用价值
1. **移动端部署**：模型小、速度快
2. **边缘计算**：适合资源受限设备
3. **实时处理**：量化后可达实时性能

---

## 风险和应对

### 风险1：注意力机制效果不明显
**应对**：
- 尝试不同的注意力位置（DenseBlock vs RRDB）
- 调整reduction ratio
- 尝试空间注意力或CBAM

### 风险2：梯度损失导致过锐化
**应对**：
- 降低λ_grad权重
- 仅在PSNR阶段使用
- 结合平滑正则化

### 风险3：量化精度损失过大
**应对**：
- 使用静态量化替代动态量化
- 增加校准数据量
- 尝试混合精度量化（部分层保持FP32）

### 风险4：时间不足
**应对**：
- 优先完成注意力机制和边缘损失（核心创新）
- 量化部分可简化为动态量化+ONNX导出
- 减少消融实验组数

---

## 成功标准

### 最低目标（必须达成）
- ✓ 实现并验证注意力机制（PSNR +0.2 dB）
- ✓ 实现并验证边缘感知损失（视觉改善）
- ✓ 完成模型量化（速度提升1.5倍以上）

### 理想目标
- ✓ 完整的消融实验和对比实验
- ✓ 标准数据集评估（Set5/Set14）
- ✓ ONNX导出和跨平台验证
- ✓ 详细的性能分析报告

### 加分项
- ✓ 与其他轻量级超分方法对比（FSRCNN, CARN等）
- ✓ 真实场景应用演示（老照片修复、监控视频增强等）
- ✓ 开源代码和预训练模型

---

## 参考资料

### 注意力机制
- Squeeze-and-Excitation Networks (CVPR 2018)
- CBAM: Convolutional Block Attention Module (ECCV 2018)
- Image Super-Resolution Using Very Deep Residual Channel Attention Networks (ECCV 2018)

### 边缘感知
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (CVPR 2017)
- Deep Edge Guided Recurrent Residual Learning for Image Super-Resolution (TIP 2017)

### 模型量化
- PyTorch Quantization Documentation
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (CVPR 2018)

---

## 附录：代码结构规划

```
毕业设计/
├── models/
│   ├── attention.py          # 新增：注意力模块
│   ├── rrdb.py              # 修改：集成注意力
│   ├── losses.py            # 修改：添加梯度损失
│   └── ...
├── quantization/            # 新增目录
│   ├── dynamic_quant.py     # 动态量化
│   ├── static_quant.py      # 静态量化
│   └── calibration_data/    # 校准数据
├── export_onnx.py           # 新增：ONNX导出
├── benchmark.py             # 新增：性能测试
├── inference_onnx.py        # 新增：ONNX推理
├── config.py                # 修改：添加新配置项
├── train.py                 # 修改：集成新损失
└── docs/
    ├── optimization_plan.md      # 本文档
    ├── attention_analysis.md     # 待创建：注意力机制分析
    ├── edge_enhancement.md       # 待创建：边缘增强分析
    └── quantization_report.md    # 待创建：量化报告
```
