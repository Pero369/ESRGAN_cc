import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models import Generator, Discriminator, PerceptualLoss, GANLoss, PixelLoss
from data import SRDataset
from config import Config
from utils import save_image

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.sample_dir, exist_ok=True)

    # 初始化模型
    generator = Generator(Config.num_rrdb_blocks, Config.num_channels).to(device)
    discriminator = Discriminator().to(device)

    # 损失函数
    pixel_loss = PixelLoss()
    perceptual_loss = PerceptualLoss().to(device)
    gan_loss = GANLoss()

    # 优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=Config.lr_g, betas=(Config.beta1, Config.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=Config.lr_d, betas=(Config.beta1, Config.beta2))

    # 数据加载
    train_dataset = SRDataset(Config.train_hr_path, Config.hr_size, Config.scale_factor)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    # 阶段1: PSNR预训练
    print("阶段1: PSNR预训练")
    for epoch in range(Config.num_epochs_psnr):
        generator.train()
        epoch_loss = 0
        for lr_img, hr_img in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.num_epochs_psnr}'):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optimizer_g.zero_grad()
            sr_img = generator(lr_img)
            loss = pixel_loss(sr_img, hr_img)
            loss.backward()
            optimizer_g.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'{Config.checkpoint_dir}/generator_psnr_{epoch+1}.pth')
            with torch.no_grad():
                sr_sample = generator(lr_img[:4])
                for i in range(4):
                    save_image(sr_sample[i], f'{Config.sample_dir}/psnr_epoch{epoch+1}_sample{i}.png')

    # 阶段2: GAN训练
    print("\n阶段2: GAN训练")
    for epoch in range(Config.num_epochs_gan):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0
        epoch_d_loss = 0

        for lr_img, hr_img in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.num_epochs_gan}'):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            # 训练判别器
            optimizer_d.zero_grad()
            sr_img = generator(lr_img).detach()
            d_real = discriminator(hr_img)
            d_fake = discriminator(sr_img)
            d_loss = gan_loss(d_real, d_fake, is_disc=True)
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            sr_img = generator(lr_img)
            d_real = discriminator(hr_img).detach()
            d_fake = discriminator(sr_img)
            perc_loss = perceptual_loss(sr_img, hr_img)
            adv_loss = gan_loss(d_real, d_fake, is_disc=False)
            g_loss = perc_loss * Config.lambda_perceptual + adv_loss * Config.lambda_adversarial
            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        print(f'Epoch {epoch+1}, G_Loss: {epoch_g_loss/len(train_loader):.4f}, D_Loss: {epoch_d_loss/len(train_loader):.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'{Config.checkpoint_dir}/generator_gan_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'{Config.checkpoint_dir}/discriminator_gan_{epoch+1}.pth')
            with torch.no_grad():
                sr_sample = generator(lr_img[:4])
                for i in range(4):
                    save_image(sr_sample[i], f'{Config.sample_dir}/gan_epoch{epoch+1}_sample{i}.png')

        if (epoch + 1) % 50 == 0:
            for param_group in optimizer_g.param_groups:
                param_group['lr'] *= 0.5
            for param_group in optimizer_d.param_groups:
                param_group['lr'] *= 0.5

if __name__ == '__main__':
    train()
