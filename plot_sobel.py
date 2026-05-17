import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import sys
img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/Set14/image_SRF_2/img_001_SRF_2_HR.png'
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
mag = np.sqrt(gx**2 + gy**2)

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
items = [
    (img_rgb,  '(a) 原始HR图像',   'viridis', False),
    (gx,       '(b) 水平梯度 Gx',  'RdBu_r',  True),
    (gy,       '(c) 垂直梯度 Gy',  'RdBu_r',  True),
    (mag,      '(d) 梯度幅值 |∇I|','hot',      True),
]

for ax, (data, title, cmap, do_norm) in zip(axes, items):
    if do_norm:
        ax.imshow(norm(data), cmap=cmap)
    else:
        ax.imshow(data)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('fig3_sobel.png', dpi=150, bbox_inches='tight')
print('已保存 fig3_sobel.png')
