import numpy as np
from PIL import Image
import torch

def save_image(tensor, path):
    img = tensor.detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    Image.fromarray(img).save(path)

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img.unsqueeze(0)
