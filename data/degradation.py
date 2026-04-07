import cv2
import numpy as np
from PIL import Image
import io
import random


class DegradationPipeline:
    """图像退化处理管道

    用于模拟真实场景下的低质量图像，包含以下退化操作：
    1. 高斯模糊 - 模拟镜头失焦或运动模糊
    2. JPEG压缩 - 模拟图像压缩失真
    3. 高斯噪声 - 模拟传感器噪声

    退化操作按顺序应用：模糊 → JPEG压缩 → 噪声
    """

    def __init__(self, config):
        """初始化退化管道

        Args:
            config: 配置对象，包含退化相关的参数
        """
        self.config = config

    def apply(self, img):
        """应用退化操作到PIL Image

        Args:
            img: PIL Image对象（RGB格式）

        Returns:
            退化后的PIL Image对象
        """
        if not self.config.enable_degradation:
            return img

        # 转换为numpy数组进行处理
        img_np = np.array(img)

        # 按顺序应用退化操作
        if self.config.enable_blur:
            img_np = self._apply_blur(img_np)

        if self.config.enable_jpeg:
            img_np = self._apply_jpeg_compression(img_np)

        if self.config.enable_noise:
            img_np = self._apply_noise(img_np)

        if getattr(self.config, 'enable_second_order', False):
            if self.config.enable_blur:
                img_np = self._apply_blur_second(img_np)
            if self.config.enable_jpeg:
                img_np = self._apply_jpeg_second(img_np)
            if self.config.enable_noise:
                img_np = self._apply_noise_second(img_np)

        return Image.fromarray(img_np)

    def _apply_blur(self, img_np):
        """应用高斯模糊

        Args:
            img_np: numpy数组，shape为(H, W, 3)，dtype为uint8，RGB格式

        Returns:
            模糊后的numpy数组，RGB格式
        """
        if random.random() > self.config.blur_prob:
            return img_np

        # 随机选择kernel size（必须是奇数）
        kernel_size = random.randrange(
            self.config.blur_kernel_range[0],
            self.config.blur_kernel_range[1] + 1,
            2  # 步长为2确保是奇数
        )

        # 随机选择sigma值
        sigma = random.uniform(*self.config.blur_sigma_range)

        # OpenCV使用BGR格式，需要转换
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blurred_bgr = cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), sigma)
        blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

        return blurred_rgb

    def _apply_noise(self, img_np):
        """应用高斯噪声

        Args:
            img_np: numpy数组，shape为(H, W, 3)，dtype为uint8

        Returns:
            添加噪声后的numpy数组
        """
        if random.random() > self.config.noise_prob:
            return img_np

        # 随机选择噪声强度
        sigma = random.uniform(*self.config.noise_sigma_range)

        # 生成高斯噪声
        noise = np.random.normal(0, sigma, img_np.shape)

        # 添加噪声并clip到有效范围
        noisy = img_np.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy

    def _apply_jpeg_compression(self, img_np):
        """应用JPEG压缩失真

        通过真实的JPEG编码/解码过程产生压缩失真

        Args:
            img_np: numpy数组，shape为(H, W, 3)，dtype为uint8

        Returns:
            JPEG压缩后的numpy数组
        """
        if random.random() > self.config.jpeg_prob:
            return img_np

        # 随机选择JPEG质量
        quality = random.randint(*self.config.jpeg_quality_range)

        # 转换为PIL Image
        img_pil = Image.fromarray(img_np)

        # 使用内存缓冲区进行JPEG编码/解码
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)

        # 转换回numpy数组
        return np.array(compressed_img)

    def _apply_blur_second(self, img_np):
        if random.random() > self.config.second_blur_prob:
            return img_np
        sigma = random.uniform(*self.config.second_blur_sigma_range)
        kernel_size = 7
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), sigma)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    def _apply_jpeg_second(self, img_np):
        if random.random() > self.config.second_jpeg_prob:
            return img_np
        quality = random.randint(*self.config.second_jpeg_quality_range)
        buffer = io.BytesIO()
        Image.fromarray(img_np).save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer))

    def _apply_noise_second(self, img_np):
        if random.random() > self.config.second_noise_prob:
            return img_np
        sigma = random.uniform(*self.config.second_noise_sigma_range)
        noisy = img_np.astype(np.float32) + np.random.normal(0, sigma, img_np.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
