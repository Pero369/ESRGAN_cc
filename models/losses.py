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
