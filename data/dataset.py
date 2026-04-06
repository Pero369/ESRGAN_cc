import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import random
from .degradation import DegradationPipeline

class SRDataset(Dataset):
    def __init__(self, hr_dir, hr_size=128, scale=4, config=None):
        self.hr_dir = hr_dir
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        self.image_files = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.degradation = DegradationPipeline(config) if config else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        hr_img = Image.open(img_path).convert('RGB')

        # 随机裁剪
        w, h = hr_img.size
        x = random.randint(0, max(0, w - self.hr_size))
        y = random.randint(0, max(0, h - self.hr_size))
        hr_img = hr_img.crop((x, y, x + self.hr_size, y + self.hr_size))

        # 应用退化操作
        if self.degradation:
            hr_img_degraded = self.degradation.apply(hr_img)
        else:
            hr_img_degraded = hr_img

        # 生成LR图像（从退化后的HR图像下采样）
        lr_img = hr_img_degraded.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        # 随机翻转
        if random.random() > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转换为tensor
        hr_tensor = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor
