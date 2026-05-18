"""
绘制第4章图表：图4-1 ~ 图4-4
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']


def fig_4_1():
    """图4-1：综合对比柱状图（PSNR、SSIM、参数量、推理时间四子图）"""
    models = ['原版\nESRGAN', '轻量版\n(基线)', '轻量版\n+CA', '轻量版\n+GradLoss', '轻量版\n+CA+GradLoss']
    params = [38.55, 0.58, 0.58, 0.58, 0.58]
    flops = [40.50, 0.85, 0.85, 0.85, 0.85]
    inf_time = [104.2, 13.7, 14.2, 13.7, 14.2]
    psnr = [24.07, 24.91, 25.55, 25.81, 25.73]
    ssim = [0.5695, 0.6774, 0.7072, 0.7225, 0.7265]
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) PSNR
    ax = axes[0, 0]
    bars = ax.bar(range(len(models)), psnr, color=colors, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, psnr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title('(a) PSNR 对比', fontsize=13, fontweight='bold')
    ax.set_ylim(22, 27.5)
    ax.grid(axis='y', alpha=0.3)
    # highlight best
    bars[-2].set_edgecolor('black')
    bars[-2].set_linewidth(2)

    # (b) SSIM
    ax = axes[0, 1]
    bars = ax.bar(range(len(models)), ssim, color=colors, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, ssim):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('SSIM', fontsize=11)
    ax.set_title('(b) SSIM 对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0.50, 0.78)
    ax.grid(axis='y', alpha=0.3)

    # (c) 参数量 (log scale for visibility since 原版 is huge)
    ax = axes[1, 0]
    bars = ax.bar(range(len(models)), params, color=colors, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f'{v:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('参数量 (M)', fontsize=11)
    ax.set_title('(c) 参数量对比', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # (d) 推理时间
    ax = axes[1, 1]
    bars = ax.bar(range(len(models)), inf_time, color=colors, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, inf_time):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.2,
                f'{v:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('推理时间 (ms)', fontsize=11)
    ax.set_title('(d) 推理时间对比', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 加速比标注
    ax.annotate(f'加速 {104.2/13.7:.1f}×', xy=(1, 13.7), xytext=(0.5, 60),
                fontsize=11, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    plt.suptitle('图4-1 综合对比实验', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4-1_comprehensive_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图4-1 已保存: {path}')


def fig_4_2():
    """图4-2：梯度损失权重与性能关系曲线（倒U型）"""
    grad_weights = [0, 0.05, 0.10, 0.15, 0.20]
    psnr = [24.10, 25.20, 25.37, 25.45, 25.18]
    ssim = [0.6403, 0.7209, 0.7307, 0.7370, 0.7280]
    lpips = [0.3725, 0.3546, 0.3501, 0.3411, 0.3522]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_psnr = '#e74c3c'
    color_ssim = '#3498db'
    color_lpips = '#2ecc71'

    ax1.plot(grad_weights, psnr, 'o-', color=color_psnr, linewidth=2.5, markersize=10,
             markerfacecolor='white', markeredgewidth=2, label='PSNR')
    ax1.set_xlabel(r'梯度损失权重 $\lambda_{grad}$', fontsize=13)
    ax1.set_ylabel('PSNR (dB)', fontsize=13, color=color_psnr)
    ax1.tick_params(axis='y', labelcolor=color_psnr)
    ax1.set_ylim(23.5, 26.0)

    ax2 = ax1.twinx()
    ax2.plot(grad_weights, ssim, 's-', color=color_ssim, linewidth=2.5, markersize=10,
             markerfacecolor='white', markeredgewidth=2, label='SSIM')
    ax2.set_ylabel('SSIM', fontsize=13, color=color_ssim)
    ax2.tick_params(axis='y', labelcolor=color_ssim)
    ax2.set_ylim(0.60, 0.78)

    # LPIPS on a third axis (or just annotate)
    # Annotate each point with values
    for i, (w, p, s, l) in enumerate(zip(grad_weights, psnr, ssim, lpips)):
        ax1.annotate(f'PSNR={p:.2f}\nSSIM={s:.4f}\nLPIPS={l:.4f}',
                     (w, p), textcoords='offset points',
                     xytext=(0, -40) if i == 2 else (0, 18),
                     fontsize=8, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Highlight optimal
    ax1.axvline(x=0.15, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.annotate(r'最优权重 $\lambda=0.15$', xy=(0.15, 25.45), xytext=(0.18, 25.75),
                 fontsize=11, fontweight='bold', color='#e74c3c',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

    ax1.grid(alpha=0.2)
    ax1.set_xticks(grad_weights)
    plt.title('图4-2 梯度损失权重与性能关系（倒U型曲线）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4-2_gradient_loss_weight.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图4-2 已保存: {path}')


def fig_4_3():
    """图4-3：网络规模与性能关系曲线（参数量-PSNR散点图，标注帕累托前沿）"""
    blocks = [4, 6, 8, 10]
    params = [0.34, 0.46, 0.58, 0.70]
    psnr_set5 = [25.10, 25.05, 25.45, 25.18]
    psnr_set14 = [23.30, 23.18, 23.45, 23.27]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(params, psnr_set5, 'o-', color='#e74c3c', linewidth=2.5, markersize=14,
            markerfacecolor='white', markeredgewidth=2.5, label='Set5', zorder=5)
    ax.plot(params, psnr_set14, 's--', color='#3498db', linewidth=2.5, markersize=14,
            markerfacecolor='white', markeredgewidth=2.5, label='Set14', zorder=5)

    # Annotate each point with blocks number
    y_offset_set5 = [0.15, -0.25, 0.15, 0.15]
    y_offset_set14 = [0.15, 0.15, 0.15, -0.25]
    for i, (b, p, s5, s14, yo5, yo14) in enumerate(zip(blocks, params, psnr_set5,
                                                         psnr_set14, y_offset_set5,
                                                         y_offset_set14)):
        ax.annotate(f'blocks={b}', (p, s5 + yo5),
                    fontsize=10, fontweight='bold', color='#e74c3c', ha='center')
        ax.annotate(f'blocks={b}', (p, s14 + yo14),
                    fontsize=10, fontweight='bold', color='#3498db', ha='center')

    # Highlight optimal at blocks=8
    ax.annotate('最优 (blocks=8)\nPSNR=25.45 dB, 0.58M',
                xy=(0.58, 25.45), xytext=(0.48, 24.55),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='orange', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                ha='center')

    # Mark the "performance dip" at blocks=6
    ax.annotate('性能凹陷\n(blocks=6)', xy=(0.46, 25.05), xytext=(0.34, 24.75),
                fontsize=9, color='#e74c3c',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2))

    ax.set_xlabel('参数量 (M)', fontsize=13)
    ax.set_ylabel('PSNR (dB)', fontsize=13)
    ax.set_title('图4-3 网络规模与性能关系 (参数量-PSNR)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.25, 0.80)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4-3_network_scale.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图4-3 已保存: {path}')


def fig_4_4():
    """图4-4：通道数与参数量/FLOPs/推理时间的关系柱状图"""
    channels = [24, 32, 40]
    params = [0.35, 0.58, 0.87]
    flops = [0.50, 0.85, 1.29]
    inf_time = [9.9, 13.7, 26.3]
    psnr = [24.75, 25.45, 24.21]

    x = np.arange(len(channels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 7))

    bars1 = ax1.bar(x - width, params, width, color='#3498db', edgecolor='white', linewidth=0.8,
                    label='参数量 (M)')
    bars2 = ax1.bar(x, flops, width, color='#2ecc71', edgecolor='white', linewidth=0.8,
                    label='FLOPs (G)')
    bars3 = ax1.bar(x + width, inf_time, width, color='#9b59b6', edgecolor='white', linewidth=0.8,
                    label='推理时间 (ms)')

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars3:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('通道数', fontsize=13)
    ax1.set_ylabel('参数量 / FLOPs / 推理时间', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'ch={c}' for c in channels], fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.2)

    # Right axis: PSNR line
    ax2 = ax1.twinx()
    ax2.plot(x, psnr, 'o-', color='#e74c3c', linewidth=3, markersize=12,
             markerfacecolor='white', markeredgewidth=2.5, label='Set5 PSNR (dB)', zorder=5)
    for i, (c, p) in enumerate(zip(channels, psnr)):
        ax2.annotate(f'{p:.2f} dB', (i, p), textcoords='offset points',
                     xytext=(0, 16), fontsize=11, fontweight='bold', color='#e74c3c', ha='center')
    ax2.set_ylabel('Set5 PSNR (dB)', fontsize=13, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(23.5, 26.2)

    # Highlight optimal
    ax2.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.annotate('最优 (ch=32)', xy=(1, 25.45), xytext=(1.3, 25.9),
                 fontsize=12, fontweight='bold', color='#e74c3c',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.title('图4-4 通道数与计算开销/性能关系', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4-4_channel_sensitivity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'图4-4 已保存: {path}')


if __name__ == '__main__':
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_4_4()
    print(f'\n所有图表已保存到: {OUTPUT_DIR}')
