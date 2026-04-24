class Config:
    # 网络参数
    num_rrdb_blocks = 23
    num_channels = 64
    scale_factor = 4

    # 轻量化配置
    use_light_model = True
    light_num_rrdb_blocks = 8
    light_num_channels = 32

    # 训练参数
    batch_size = 16
    #训练轮数，加快测试，实际训练时可以设置更大
    num_epochs_psnr = 50
    num_epochs_gan = 150
    lr_g = 1e-4
    lr_d = 1e-4
    beta1 = 0.9
    beta2 = 0.999

    # 损失权重
    lambda_perceptual = 1.0
    lambda_adversarial = 0.005
    #合适的参数是0.11和0.13
    lambda_pixel = 0.13

    # 数据参数
    hr_size = 128
    lr_size = 32

    # 路径
    train_hr_path = './data/train_hr'
    val_hr_path = './data/val_hr'
    checkpoint_dir = './checkpoints'
    sample_dir = './samples'

    # 退化参数
    enable_degradation = False  # 总开关（启用退化模块）
    enable_blur = False         # 高斯模糊开关
    enable_noise = False        # 高斯噪声开关
    enable_jpeg = False         # JPEG压缩开关

    # 高斯模糊参数
    blur_kernel_range = (7, 21)      # kernel size范围（奇数）
    blur_sigma_range = (0.1, 3.0)    # sigma范围
    blur_prob = 0.8                  # 应用概率

    # 高斯噪声参数
    noise_sigma_range = (0, 15)      # 噪声标准差范围（0-255尺度）
    noise_prob = 0.7                 # 应用概率

    # JPEG压缩参数
    jpeg_quality_range = (60, 95)    # JPEG质量范围
    jpeg_prob = 0.8                  # 应用概率

    # 注意力机制配置
    enable_attention = False         # 注意力机制总开关
    attention_type = 'CA'            # 注意力类型: 'CA'(通道), 'SA'(空间), 'CBAM'(两者结合)
    attention_reduction = 16         # SE模块降维比例
    attention_position = 'rrdb'      # 注意力位置: 'dense'(DenseBlock级), 'rrdb'(RRDB级), 'both'

    # 边缘感知损失配置
    enable_gradient_loss = False     # 梯度损失开关
    lambda_gradient = 0.1            # 梯度损失权重（建议0.05~0.2）
    gradient_loss_stage = 'both'     # 应用阶段: 'psnr', 'gan', 'both'
