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
        # VGG19 预训练时使用的 ImageNet 标准化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, sr, hr):
        sr_feat = self.feature_extractor(self._normalize(sr))
        hr_feat = self.feature_extractor(self._normalize(hr))
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
