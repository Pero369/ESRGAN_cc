import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features
        self.feature_extractor = nn.Sequential(*list(vgg)[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        # 直接使用[0,1]范围的图像，不进行ImageNet标准化
        # 因为我们关心的是特征的相对差异，而不是绝对值
        sr_feat = self.feature_extractor(sr)
        hr_feat = self.feature_extractor(hr)
        return self.criterion(sr_feat, hr_feat)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, d_out_real, d_out_fake, is_disc=True):
        if is_disc:
            real_loss = self.criterion(d_out_real - d_out_fake.mean(), torch.ones_like(d_out_real))
            fake_loss = self.criterion(d_out_fake - d_out_real.mean(), torch.zeros_like(d_out_fake))
            return (real_loss + fake_loss) / 2
        else:
            return self.criterion(d_out_fake - d_out_real.mean(), torch.ones_like(d_out_fake))


class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        return self.criterion(sr, hr)


class GradientLoss(nn.Module):
    """
    边缘感知梯度损失
    使用Sobel算子计算图像梯度，强化边缘结构学习
    """
    def __init__(self):
        super().__init__()
        # Sobel算子 - X方向
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Sobel算子 - Y方向
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # 为RGB三通道复制Sobel算子
        self.sobel_x = nn.Parameter(sobel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y.repeat(3, 1, 1, 1), requires_grad=False)

        self.criterion = nn.L1Loss()

    def compute_gradient(self, img):
        """
        计算图像梯度幅值
        Args:
            img: [B, 3, H, W] 输入图像
        Returns:
            gradient: [B, 3, H, W] 梯度幅值
        """
        # 分别对每个通道计算梯度
        grad_x = nn.functional.conv2d(img, self.sobel_x, padding=1, groups=3)
        grad_y = nn.functional.conv2d(img, self.sobel_y, padding=1, groups=3)

        # 计算梯度幅值: sqrt(Gx^2 + Gy^2)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return gradient

    def forward(self, sr, hr):
        """
        计算SR和HR图像的梯度损失
        Args:
            sr: [B, 3, H, W] 超分辨率图像
            hr: [B, 3, H, W] 高分辨率图像
        Returns:
            loss: 梯度L1损失
        """
        sr_grad = self.compute_gradient(sr)
        hr_grad = self.compute_gradient(hr)
        return self.criterion(sr_grad, hr_grad)
