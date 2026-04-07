import torch
import torch.nn as nn
import torch.nn.functional as F
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

class FFTLoss(nn.Module):
    def forward(self, sr, hr):
        sr_fft = torch.abs(torch.fft.rfft2(sr))
        hr_fft = torch.abs(torch.fft.rfft2(hr))
        return F.l1_loss(sr_fft, hr_fft)
