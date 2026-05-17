import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 从CSV读取数据 ──────────────────────────────────────────────────────────────
df1 = pd.read_csv('results/comparison_comprehensive.csv')
models = ['原版ESRGAN', '轻量版\n(基线)', '轻量版\n+CA', '轻量版\n+GradLoss', '轻量版\n+CA+GradLoss']
params  = df1['参数量(M)'].tolist()
times   = df1['推理时间(ms)'].tolist()
psnr    = df1['PSNR'].tolist()
ssim    = df1['SSIM'].tolist()

df2 = pd.read_csv('results/comparison_gradient.csv')
weights = [0.0, 0.05, 0.1, 0.2]
psnr_w  = df2['PSNR'].tolist()
ssim_w  = df2['SSIM'].tolist()

x = np.arange(len(models))
w = 0.2

# ── 图1：综合对比柱状图 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('图4-1 综合对比实验结果', fontsize=14, fontweight='bold')

datasets = [
    (params, '参数量 (M)',    'steelblue'),
    (times,  '推理时间 (ms)', 'darkorange'),
    (psnr,   'PSNR (dB)',    'seagreen'),
    (ssim,   'SSIM',         'mediumpurple'),
]

sub_titles = ['(a) 参数量对比', '(b) 推理时间对比', '(c) PSNR对比', '(d) SSIM对比']
short_labels = ['原版\nESRGAN', '轻量版\n基线', '+CA', '+Grad\nLoss', '+CA\n+Grad']

for i, (ax, (data, ylabel, color)) in enumerate(zip(axes, datasets)):
    bars = ax.bar(x, data, color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(sub_titles[i], fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 参数量子图使用对数坐标，使轻量版柱子清晰可见
    if i == 0:
        ax.set_yscale('log')
        ax.set_ylim(0.3, 60)
        for bar, val in zip(bars, data):
            y_pos = val * 1.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)
    else:
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig('fig1_comparison.png', dpi=300, bbox_inches='tight')
print('已保存 fig1_comparison.png')

# ── 图2：梯度损失权重 vs 性能（倒U型）────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

l1, = ax1.plot(weights, psnr_w, 'o-', color='seagreen',     linewidth=2, markersize=7, label='PSNR (dB)')
l2, = ax2.plot(weights, ssim_w, 's--', color='mediumpurple', linewidth=2, markersize=7, label='SSIM')

ax1.set_xlabel('梯度损失权重 λ', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12, color='seagreen')
ax2.set_ylabel('SSIM',      fontsize=12, color='mediumpurple')
ax1.tick_params(axis='y', labelcolor='seagreen')
ax2.tick_params(axis='y', labelcolor='mediumpurple')

ax1.axvline(x=0.1, color='gray', linestyle=':', linewidth=1.5, label='最优权重 λ=0.1')

# 在最优点添加数值标注
ax1.annotate('25.84 dB', xy=(0.1, psnr_w[2]), xytext=(0.10, 25.70),
             fontsize=10, color='seagreen', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='seagreen', lw=1))
ax2.annotate('0.7342', xy=(0.1, ssim_w[2]), xytext=(0.14, 0.730),
             fontsize=10, color='mediumpurple', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='mediumpurple', lw=1))

ax1.set_title('图4-2 梯度损失权重与重建质量的关系', fontsize=13, fontweight='bold')
ax1.grid(linestyle='--', alpha=0.4)

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig('fig2_gradloss_curve.png', dpi=300, bbox_inches='tight')
print('已保存 fig2_gradloss_curve.png')
