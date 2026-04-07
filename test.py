import torch
import argparse
import os
from models import Generator
from utils import load_image, save_image

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        if os.path.isfile(args.input_path):
            lr_img = load_image(args.input_path).to(device)
            sr_img = generator(lr_img)
            base = os.path.splitext(os.path.basename(args.input_path))[0]
            save_image(sr_img[0], os.path.join(args.output_dir, f'{base}_sr.png'))
        else:
            for img_name in os.listdir(args.input_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    lr_img = load_image(os.path.join(args.input_path, img_name)).to(device)
                    sr_img = generator(lr_img)
                    save_image(sr_img[0], os.path.join(args.output_dir, img_name))
                    del lr_img, sr_img
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    print(f'超分辨率结果已保存到 {args.output_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    args = parser.parse_args()
    test(args)
