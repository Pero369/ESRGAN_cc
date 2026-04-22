"""
批量参数扫描训练脚本
在 EXPERIMENTS 列表中定义每组参数，脚本会依次训练并将结果保存到独立子目录。

支持的参数：
- 损失权重: lambda_pixel, lambda_perceptual, lambda_adversarial, lambda_gradient
- 模型配置: use_light_model, light_num_rrdb_blocks, light_num_channels
- 注意力机制: enable_attention, attention_type, attention_reduction, attention_position
- 边缘感知: enable_gradient_loss, gradient_loss_stage
- 训练参数: batch_size, lr_g, lr_d, num_epochs_psnr, num_epochs_gan
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models import Generator, LightGenerator, Discriminator, PerceptualLoss, GANLoss, PixelLoss, GradientLoss
from data import SRDataset
from config import Config
from utils import save_image

# ============================================================
# 在这里定义要扫描的参数组，每个 dict 只需写与默认值不同的参数
# ============================================================

# 示例1: 损失权重调优
LOSS_WEIGHT_EXPERIMENTS = [
    {"lambda_pixel": 0.08},
    {"lambda_pixel": 0.11},
    {"lambda_pixel": 0.13},
]

# 示例2: 注意力机制对比
ATTENTION_EXPERIMENTS = [
    {"enable_attention": False},                                    # 基线
    {"enable_attention": True, "attention_type": "CA"},            # 通道注意力
    {"enable_attention": True, "attention_type": "CBAM"},          # CBAM
    {"enable_attention": True, "attention_type": "CA",
     "attention_position": "dense"},                               # DenseBlock级注意力
]

# 示例3: 边缘感知损失对比
GRADIENT_LOSS_EXPERIMENTS = [
    {"enable_gradient_loss": False},                               # 基线
    {"enable_gradient_loss": True, "lambda_gradient": 0.05},      # 低权重
    {"enable_gradient_loss": True, "lambda_gradient": 0.1},       # 中权重
    {"enable_gradient_loss": True, "lambda_gradient": 0.2},       # 高权重
]

# 示例4: 消融实验（验证各模块贡献）
ABLATION_EXPERIMENTS = [
    # 基线
    {"enable_attention": False, "enable_gradient_loss": False},

    # 单独添加注意力
    {"enable_attention": True, "attention_type": "CA",
     "enable_gradient_loss": False},

    # 单独添加梯度损失
    {"enable_attention": False,
     "enable_gradient_loss": True, "lambda_gradient": 0.1},

    # 完整版（两者都有）
    {"enable_attention": True, "attention_type": "CA",
     "enable_gradient_loss": True, "lambda_gradient": 0.1},
]

# 示例5: 轻量化程度对比
LIGHTWEIGHT_EXPERIMENTS = [
    {"light_num_rrdb_blocks": 6, "light_num_channels": 32},
    {"light_num_rrdb_blocks": 8, "light_num_channels": 32},
    {"light_num_rrdb_blocks": 10, "light_num_channels": 32},
    {"light_num_rrdb_blocks": 8, "light_num_channels": 24},
    {"light_num_rrdb_blocks": 8, "light_num_channels": 40},
]

# 示例6: 综合优化实验
COMPREHENSIVE_EXPERIMENTS = [
    # 原版模型（对比基线）
    {"use_light_model": False, "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化基线
    {"use_light_model": True, "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化 + 注意力
    {"use_light_model": True, "enable_attention": True, "attention_type": "CA",
     "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化 + 梯度损失
    {"use_light_model": True, "enable_gradient_loss": True, "lambda_gradient": 0.1,
     "num_epochs_psnr": 10, "num_epochs_gan": 10},

    # 轻量化 + 注意力 + 梯度损失（完整版）
    {"use_light_model": True, "enable_attention": True, "attention_type": "CA",
     "enable_gradient_loss": True, "lambda_gradient": 0.1,
     "num_epochs_psnr": 10, "num_epochs_gan": 10},
]

# 选择要运行的实验组（修改这里来切换实验）
EXPERIMENTS = ABLATION_EXPERIMENTS  # 默认运行消融实验

# ============================================================


def make_config(overrides: dict):
    """用 overrides 覆盖 Config 默认值，返回一个临时配置对象"""
    cfg = type("Cfg", (), {k: getattr(Config, k) for k in dir(Config) if not k.startswith("__")})()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_experiment(cfg, exp_name: str, run_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ckpt = run_dir if run_dir else cfg.checkpoint_dir
    base_sample = os.path.join(os.path.dirname(base_ckpt), "samples", os.path.basename(base_ckpt)) if run_dir else cfg.sample_dir
    ckpt_dir = os.path.join(base_ckpt, exp_name)
    sample_dir = os.path.join(base_sample, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 打印实验配置
    print(f"\n{'='*60}")
    print(f"实验配置: {exp_name}")
    print(f"{'='*60}")

    # 根据配置选择模型
    if cfg.use_light_model:
        generator = LightGenerator(
            cfg.light_num_rrdb_blocks,
            cfg.light_num_channels,
            enable_attention=cfg.enable_attention,
            attention_type=cfg.attention_type,
            attention_reduction=cfg.attention_reduction,
            attention_position=cfg.attention_position
        ).to(device)
        print(f"模型: 轻量化生成器")
        print(f"  RRDB块: {cfg.light_num_rrdb_blocks}")
        print(f"  通道数: {cfg.light_num_channels}")
        if cfg.enable_attention:
            print(f"  注意力: {cfg.attention_type} (位置: {cfg.attention_position})")
    else:
        generator = Generator(cfg.num_rrdb_blocks, cfg.num_channels).to(device)
        print(f"模型: 原版生成器")
        print(f"  RRDB块: {cfg.num_rrdb_blocks}")
        print(f"  通道数: {cfg.num_channels}")

    discriminator = Discriminator().to(device)

    # 损失函数
    pixel_loss = PixelLoss()
    perceptual_loss = PerceptualLoss().to(device)
    gan_loss = GANLoss()

    # 梯度损失（可选）
    gradient_loss = None
    if cfg.enable_gradient_loss:
        gradient_loss = GradientLoss().to(device)
        print(f"梯度损失: 启用 (权重={cfg.lambda_gradient}, 阶段={cfg.gradient_loss_stage})")

    # 打印损失权重
    print(f"损失权重:")
    print(f"  Pixel: {cfg.lambda_pixel}")
    print(f"  Perceptual: {cfg.lambda_perceptual}")
    print(f"  Adversarial: {cfg.lambda_adversarial}")
    if cfg.enable_gradient_loss:
        print(f"  Gradient: {cfg.lambda_gradient}")

    print(f"训练参数:")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  PSNR epochs: {cfg.num_epochs_psnr}")
    print(f"  GAN epochs: {cfg.num_epochs_gan}")
    print(f"  Learning rate: G={cfg.lr_g}, D={cfg.lr_d}")
    print(f"{'='*60}\n")

    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    train_dataset = SRDataset(cfg.train_hr_path, cfg.hr_size, cfg.scale_factor, config=cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # 阶段1: PSNR预训练
    print(f"\n[{exp_name}] 阶段1: PSNR预训练")
    for epoch in range(cfg.num_epochs_psnr):
        generator.train()
        epoch_loss = 0
        epoch_pixel_loss = 0
        epoch_grad_loss = 0

        for lr_img, hr_img in tqdm(train_loader, desc=f"PSNR {epoch+1}/{cfg.num_epochs_psnr}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer_g.zero_grad()
            sr_img = generator(lr_img)

            # 像素损失
            pix_loss = pixel_loss(sr_img, hr_img)
            loss = pix_loss
            epoch_pixel_loss += pix_loss.item()

            # 添加梯度损失（如果启用且在PSNR阶段）
            if cfg.enable_gradient_loss and cfg.gradient_loss_stage in ['psnr', 'both']:
                grad_loss = gradient_loss(sr_img, hr_img)
                loss = loss + grad_loss * cfg.lambda_gradient
                epoch_grad_loss += grad_loss.item()

            loss.backward()
            optimizer_g.step()
            epoch_loss += loss.item()

        # 打印损失
        avg_loss = epoch_loss / len(train_loader)
        avg_pixel = epoch_pixel_loss / len(train_loader)
        log_str = f"Epoch {epoch+1}, Total: {avg_loss:.4f}, Pixel: {avg_pixel:.4f}"
        if cfg.enable_gradient_loss and cfg.gradient_loss_stage in ['psnr', 'both']:
            avg_grad = epoch_grad_loss / len(train_loader)
            log_str += f", Grad: {avg_grad:.4f}"
        print(log_str)

        if (epoch + 1) % 10 == 0:
            # 只保留最新的PSNR检查点
            prev = f"{ckpt_dir}/generator_psnr_{epoch+1-10}.pth"
            if os.path.exists(prev):
                os.remove(prev)
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
        epoch_pix = epoch_perc = epoch_adv = epoch_grad = 0

        for lr_img, hr_img in tqdm(train_loader, desc=f"GAN {epoch+1}/{cfg.num_epochs_gan}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            # 训练判别器
            optimizer_d.zero_grad()
            sr_img = generator(lr_img).detach()
            d_loss = gan_loss(discriminator(hr_img), discriminator(sr_img), is_disc=True)
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            sr_img = generator(lr_img)
            d_real = discriminator(hr_img).detach()

            pix_loss = pixel_loss(sr_img, hr_img)
            perc_loss = perceptual_loss(sr_img, hr_img)
            adv_loss = gan_loss(d_real, discriminator(sr_img), is_disc=False)

            g_loss = (pix_loss * cfg.lambda_pixel +
                     perc_loss * cfg.lambda_perceptual +
                     adv_loss * cfg.lambda_adversarial)

            # 添加梯度损失（如果启用且在GAN阶段）
            if cfg.enable_gradient_loss and cfg.gradient_loss_stage in ['gan', 'both']:
                grad_loss = gradient_loss(sr_img, hr_img)
                g_loss = g_loss + grad_loss * cfg.lambda_gradient
                epoch_grad += grad_loss.item()

            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_pix += pix_loss.item()
            epoch_perc += perc_loss.item()
            epoch_adv += adv_loss.item()

        # 打印详细损失
        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        avg_pix = epoch_pix / len(train_loader)
        avg_perc = epoch_perc / len(train_loader)
        avg_adv = epoch_adv / len(train_loader)

        log_str = (f"Epoch {epoch+1}, G: {avg_g:.4f}, D: {avg_d:.4f} "
                  f"[Pix: {avg_pix:.4f}, Perc: {avg_perc:.4f}, Adv: {avg_adv:.4f}")
        if cfg.enable_gradient_loss and cfg.gradient_loss_stage in ['gan', 'both']:
            avg_grad = epoch_grad / len(train_loader)
            log_str += f", Grad: {avg_grad:.4f}"
        log_str += "]"
        print(log_str)

        if (epoch + 1) % 10 == 0:
            # 只保留最新的GAN检查点
            prev = epoch + 1 - 10
            for f in [f"{ckpt_dir}/generator_gan_{prev}.pth", f"{ckpt_dir}/discriminator_gan_{prev}.pth"]:
                if os.path.exists(f):
                    os.remove(f)
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


def generate_exp_name(overrides):
    """生成简洁的实验名称"""
    parts = []

    # 模型类型
    if 'use_light_model' in overrides:
        parts.append('light' if overrides['use_light_model'] else 'original')

    # 注意力
    if overrides.get('enable_attention'):
        att_type = overrides.get('attention_type', 'CA')
        att_pos = overrides.get('attention_position', 'rrdb')
        parts.append(f"att_{att_type}_{att_pos}")

    # 梯度损失
    if overrides.get('enable_gradient_loss'):
        grad_weight = overrides.get('lambda_gradient', 0.1)
        parts.append(f"grad{grad_weight}")

    # RRDB配置
    if 'light_num_rrdb_blocks' in overrides:
        parts.append(f"rrdb{overrides['light_num_rrdb_blocks']}")
    if 'light_num_channels' in overrides:
        parts.append(f"ch{overrides['light_num_channels']}")

    # 损失权重
    if 'lambda_pixel' in overrides:
        parts.append(f"pix{overrides['lambda_pixel']}")
    if 'lambda_gradient' in overrides and not overrides.get('enable_gradient_loss'):
        parts.append(f"grad{overrides['lambda_gradient']}")

    # 如果没有特殊配置，使用所有参数
    if not parts:
        parts = [f"{k}{v}" for k, v in overrides.items()]

    return "_".join(parts)


if __name__ == "__main__":
    exp_group_name = next((k for k, v in globals().items() if v is EXPERIMENTS and k != "EXPERIMENTS"), "sweep")
    run_dir = os.path.join(Config.checkpoint_dir, exp_group_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"批量参数扫描训练")
    print(f"{'='*70}")
    print(f"总实验数: {len(EXPERIMENTS)}")
    print(f"运行目录: {run_dir}")
    print(f"{'='*70}\n")

    for idx, overrides in enumerate(EXPERIMENTS, 1):
        exp_name = generate_exp_name(overrides)

        print(f"\n{'='*70}")
        print(f"实验 {idx}/{len(EXPERIMENTS)}: {exp_name}")
        print(f"{'='*70}")

        cfg = make_config(overrides)

        try:
            run_experiment(cfg, exp_name, run_dir)
            print(f"\n✓ 实验 {exp_name} 完成")
        except Exception as e:
            print(f"\n✗ 实验 {exp_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 询问是否继续
            response = input("\n是否继续下一个实验? (y/n): ")
            if response.lower() != 'y':
                break

    print(f"\n{'='*70}")
    print("所有实验完成！")
    print(f"结果保存在: {Config.checkpoint_dir}")
    print(f"样本保存在: {Config.sample_dir}")
    print(f"{'='*70}\n")
