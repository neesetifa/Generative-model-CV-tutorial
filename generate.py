import pdb
import argparse

import torch
from utils import *
from model.unet import UNet, UNet_conditional
from ddm import Diffusion

def generate(args):
    device = 'cuda'
    if args.run_name == "DDM_Uncondtional":
        model = UNet()
    elif args.run_name == "DDM_Conditional":
        model = UNet_conditional(num_classes=args.num_classes)
    else:
        raise ValueError(f'Invalid run_name, {args.run_name}')
    
    ckpt = torch.load(args.pretrained_weight)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=image_size, device=device)
    if args.sample_step == 'ddpm':
        x = diffusion.sample(model, number_of_generate_example)
    elif args.sample_step == 'ddim':
        x = diffusion.sample_ddim(model, number_of_generate_example)
    save_images_cv2(x, save_image_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDM_Uncondtional", help='model config py file')
    parser.add_argument('--sample_step', type=str, default='ddim', help='ddpm or ddim')
    parser.add_argument('--image_size', type=int, default=64, help='default image size 64x64')
    parser.add_argument('--number_of_generate_example', type=int, default=9, help='')
    parser.add_argument('--pretrained_weight', type=str, default='saved_model/DDM_Uncondtional/church_lsun/round_1/ckpt_147.pt', help='checkpoint file')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    generate(args)
