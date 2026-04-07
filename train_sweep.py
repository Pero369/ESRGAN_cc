"""
批量参数扫描训练脚本
在 EXPERIMENTS 列表中定义每组参数，脚本会依次训练并将结果保存到独立子目录。
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models import Generator, Discriminator, PerceptualLoss, GANLoss, PixelLoss
from data import SRDataset
from config import Config
from utils import save_image

# ============================================================
# 在这里定义要扫描的参数组，每个 dict 只需写与默认值不同的参数
# ============================================================
EXPERIMENTS = [
    {"lambda_pixel": 0.08},
    {"lambda_pixel": 0.10},
    {"lambda_pixel": 0.12},
    {"lambda_pixel": 0.15},
    {"lambda_pixel": 0.20},
]
# ============================================================


def make_config(overrides: dict):
    """用 overrides 覆盖 Config 默认值，返回一个临时配置对象"""
    cfg = type("Cfg", (), {k: getattr(Config, k) for k in dir(Config) if not k.startswith("__")})()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_experiment(cfg, exp_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(cfg.checkpoint_dir, exp_name)
    sample_dir = os.path.join(cfg.sample_dir, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    generator = Generator(cfg.num_rrdb_blocks, cfg.num_channels).to(device)
    discriminator = Discriminator().to(device)

    pixel_loss = PixelLoss()
    perceptual_loss = PerceptualLoss().to(device)
    gan_loss = GANLoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    train_dataset = SRDataset(cfg.train_hr_path, cfg.hr_size, cfg.scale_factor, config=cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # 阶段1: PSNR预训练
    print(f"\n[{exp_name}] 阶段1: PSNR预训练")
    for epoch in range(cfg.num_epochs_psnr):
        generator.train()
        epoch_loss = 0
        for lr_img, hr_img in tqdm(train_loader, desc=f"PSNR {epoch+1}/{cfg.num_epochs_psnr}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer_g.zero_grad()
            sr_img = generator(lr_img)
            loss = pixel_loss(sr_img, hr_img)
            loss.backward()
            optimizer_g.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"{ckpt_dir}/generator_psnr_{epoch+1}.pth")
            with torch.no_grad():
                sr_sample = generator(lr_img[:4])
                for i in range(4):
                    save_image(sr_sample[i], f"{sample_dir}/psnr_epoch{epoch+1}_sample{i}.png")

    # 阶段2: GAN训练
    print(f"\n[{exp_name}] 阶段2: GAN训练")
    for epoch in range(cfg.num_epochs_gan):
        generator.train()
        discriminator.train()
        epoch_g_loss = epoch_d_loss = 0

        for lr_img, hr_img in tqdm(train_loader, desc=f"GAN {epoch+1}/{cfg.num_epochs_gan}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optimizer_d.zero_grad()
            sr_img = generator(lr_img).detach()
            d_loss = gan_loss(discriminator(hr_img), discriminator(sr_img), is_disc=True)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            sr_img = generator(lr_img)
            d_real = discriminator(hr_img).detach()
            pix_loss = pixel_loss(sr_img, hr_img)
            perc_loss = perceptual_loss(sr_img, hr_img)
            adv_loss = gan_loss(d_real, discriminator(sr_img), is_disc=False)
            g_loss = pix_loss * cfg.lambda_pixel + perc_loss * cfg.lambda_perceptual + adv_loss * cfg.lambda_adversarial
            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        print(f"Epoch {epoch+1}, G: {epoch_g_loss/len(train_loader):.4f}, D: {epoch_d_loss/len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"{ckpt_dir}/generator_gan_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{ckpt_dir}/discriminator_gan_{epoch+1}.pth")
            with torch.no_grad():
                sr_sample = generator(lr_img[:4])
                for i in range(4):
                    save_image(sr_sample[i], f"{sample_dir}/gan_epoch{epoch+1}_sample{i}.png")

        if (epoch + 1) % 50 == 0:
            for pg in optimizer_g.param_groups:
                pg["lr"] *= 0.5
            for pg in optimizer_d.param_groups:
                pg["lr"] *= 0.5


if __name__ == "__main__":
    for overrides in EXPERIMENTS:
        # 用参数值生成实验名，例如 lambda_pixel0.12
        exp_name = "_".join(f"{k}{v}" for k, v in overrides.items())
        print(f"\n{'='*50}\n开始实验: {exp_name}\n{'='*50}")
        cfg = make_config(overrides)
        run_experiment(cfg, exp_name)
    print("\n所有实验完成！")
