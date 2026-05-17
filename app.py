import sys
import os
import torch
import gradio as gr
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.generator import LightGenerator, Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.abspath(__file__))


def sr_tile(model, x, tile=256, overlap=16):
    b, c, h, w = x.shape
    scale = 4
    out = torch.zeros(b, c, h * scale, w * scale, device=x.device)
    step = tile - overlap
    for y in range(0, h, step):
        for xp in range(0, w, step):
            y1, x1 = min(y, h - tile), min(xp, w - tile)
            sr = model(x[:, :, y1:y1+tile, x1:x1+tile])
            py = overlap * scale // 2 if y1 > 0 else 0
            px = overlap * scale // 2 if x1 > 0 else 0
            out[:, :, y1*scale+py:(y1+tile)*scale, x1*scale+px:(x1+tile)*scale] = sr[:, :, py:, px:]
    return out

MODELS = {
    "轻量版（最优，PSNR 25.84）": {
        "cls": LightGenerator,
        "kwargs": {"num_rrdb": 8, "channels": 32},
        "ckpt": "checkpoints/GRADIENT_LOSS_EXPERIMENTS/grad0.1/generator_gan_150.pth",
    },
    "原版 ESRGAN（对比用）": {
        "cls": Generator,
        "kwargs": {"num_rrdb": 23, "channels": 64},
        "ckpt": "checkpoints/COMPREHENSIVE_EXPERIMENTS/original/generator_gan_150.pth",
    },
}

_cache = {}


def load_model(name):
    if name in _cache:
        return _cache[name]
    cfg = MODELS[name]
    model = cfg["cls"](**cfg["kwargs"])
    ckpt_path = os.path.join(BASE, cfg["ckpt"])
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    _cache[name] = model
    return model


def run_sr(input_image: Image.Image, model_name: str):
    if input_image is None:
        return None, None
    model = load_model(model_name)
    rgb = input_image.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = sr_tile(model, x)
    out_arr = out.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    sr = Image.fromarray((out_arr * 255).astype(np.uint8))
    # 将原图放大到与超分结果相同尺寸（nearest，保留像素感）
    lr_upscaled = rgb.resize(sr.size, Image.NEAREST)
    return lr_upscaled, sr


with gr.Blocks(title="ESRGAN 图像超分辨率") as demo:
    gr.Markdown("## ESRGAN 4x 图像超分辨率演示")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="上传低质量图片")
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="轻量版（最优，PSNR 25.84）",
                label="选择模型",
            )
            btn = gr.Button("开始超分", variant="primary")
    with gr.Row():
        lr_out = gr.Image(type="pil", label="原图（放大4x，nearest）")
        sr_out = gr.Image(type="pil", label="超分结果（4x）")
    btn.click(fn=run_sr, inputs=[inp, model_choice], outputs=[lr_out, sr_out])

if __name__ == "__main__":
    demo.launch()
