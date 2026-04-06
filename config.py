class Config:
    # 网络参数
    num_rrdb_blocks = 23
    num_channels = 64
    scale_factor = 4

    # 训练参数
    batch_size = 16
    num_epochs_psnr = 50
    num_epochs_gan = 150
    lr_g = 1e-4
    lr_d = 1e-4
    beta1 = 0.9
    beta2 = 0.999

    # 损失权重
    lambda_perceptual = 1.0
    lambda_adversarial = 0.005
    lambda_pixel = 0.01

    # 数据参数
    hr_size = 128
    lr_size = 32

    # 路径
    train_hr_path = './data/train_hr'
    val_hr_path = './data/val_hr'
    checkpoint_dir = './checkpoints'
    sample_dir = './samples'
