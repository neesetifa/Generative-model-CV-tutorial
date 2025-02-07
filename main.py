import os
import pdb
import argparse

from tqdm import tqdm
import torch
from torch import optim

from utils import *
from torch.utils.tensorboard import SummaryWriter
from dataset import CustomDataset
from model.unet import UNet, UNet_conditional, EMA
from ddm import Diffusion

def train(args):
    if not os.path.exists(os.path.join('saved_model', args.run_name, args.dataset)):
        os.mkdir(os.path.join('saved_model', args.run_name, args.dataset))
    if not os.path.exists(os.path.join('epoch_results', args.run_name, args.dataset)):
        os.mkdir(os.path.join('epoch_results', args.run_name, args.dataset))
    args.log_dir = os.path.join('saved_model', args.run_name, args.dataset)
    device = 'cuda'
    
    #setup_logging(args.run_name)
    #logger = SummaryWriter(os.path.join("runs", args.run_name, args.dataset))
    logger = get_logger(args.log_dir)

    # Dataset
    dataset = CustomDataset(args.image_size, args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             drop_last=False,
                                             pin_memory=True)

    # Models and Module
    if args.run_name == "DDM_Uncondtional":
        model = UNet()
    elif args.run_name == "DDM_Conditional":
        model = UNet_conditional(num_classes=args.num_classes)
    else:
        raise ValueError(f'Invalid run_name, {args.run_name}')
        
    if args.pretrained_weight is not None:
        model.load_state_dict(torch.load(args.pretrained_weight,map_location='cpu'))
    model = model.to(device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # Loss and Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_epochs, gamma = 0.1)
    mse = torch.nn.MSELoss()
    
    # Log
    l = len(dataloader)
    logger.info('\n'
                '========= Training info =========\n'
                'model: {} \n'
                'dataset: {} \n'
                'total training examples: {} \n'
                'image_size: {} \n'
                'start_epoch: {} \n'
                'max_epoch: {} \n'
                'batch_size: {} \n'
                'learning_rate: {} \n'
                'lr epochs: {} \n'
                'pretrained_weight: {} \n'
                '=================================='.format(args.run_name,
                                                            args.dataset,
                                                            len(dataset),
                                                            args.image_size,
                                                            args.start_epoch,
                                                            args.max_epoch,
                                                            args.batch_size,
                                                            args.lr,
                                                            args.lr_epochs,
                                                            args.pretrained_weight))

    # Train
    best_loss = 999
    for epoch in range(args.start_epoch, args.max_epoch):
        logger.info(f"Starting epoch {epoch}:")
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Current Learning rate = {lr}')
        pbar = tqdm(dataloader)
        mse_loss = AverageMeter()
        for i, batch_data in enumerate(pbar):
            if len(batch_data) == 2:
                images, labels = batch_data
            else:
                images, labels = batch_data[0], None
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if labels:
                # For conditional DDM
                predicted_noise = model(x_t, t, labels)
            else:
                # For unconditional DDM
                predicted_noise = model(x_t, t)
                
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            mse_loss.update(loss.item())
            #pbar.set_postfix(MSE=mse_loss.avg)
            #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            if i%500 == 0:
                logger.info('Epoch: [{0}][{1}/{2}] '
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(epoch,
                                                                           i,
                                                                           len(dataloader),
                                                                           loss=mse_loss))

        scheduler.step()
        
        sampled_images = diffusion.sample(model, n=args.num_test_images)
        save_images_cv2(sampled_images, os.path.join("epoch_results", args.run_name, args.dataset, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("saved_model", args.run_name, args.dataset, "ckpt.pt"))
        if mse_loss.avg<best_loss:
            best_loss = mse_loss.avg
            torch.save(model.state_dict(), os.path.join("saved_model", args.run_name, args.dataset, "ckpt_best.pt"))
            torch.save(ema_model.state_dict(), os.path.join("saved_model", args.run_name, args.dataset, "ckpt_best_ema.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDM_Uncondtional", help='DDM_Uncondtional or DDM_Condtional')
    parser.add_argument('--num_classes', type=int, default=1, help='For DDM Conditional')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=64, help='default image size 64x64')
    parser.add_argument('--dataset', type=str, default='church_lsun', help='')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--lr_epochs', type=list, default=[120], help='Multi-Step learning rate')
    parser.add_argument('--num_test_images', type=int, default=3, help='number of test images generated during training')
    parser.add_argument('--pretrained_weight', type=str, default=None, help='pretrained weight') #'pretrained_model/church_lsun_ckpt.pt'
    return parser.parse_args()

    
if __name__ == '__main__':
    args = parse_args()
    train(args)

