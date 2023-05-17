import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import logging
import cv2


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images_pil(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # channel must be RGB 
    im = Image.fromarray(ndarr)
    im.save(path)

def save_images_cv2(images, path, **kwargs):
    # images = [batch, 3, H, W]
    nrow = int(images.shape[0]**0.5)
    nrow = 3 if nrow<=3 else nrow
    grid = torchvision.utils.make_grid(images, nrow, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy().astype(int)
    # channel must be BGR
    cv2.imwrite(path, ndarr)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)), # 80*80->64*64
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(log_dir = None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)

    if log_dir is None:
        file_name = 'logs/log.txt'
    else:
        file_name = os.path.join(log_dir, 'log.txt')
    filehandler = logging.FileHandler(filename = file_name) # filemode='w', 每次生成新的
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    # level可以每个handler单独设置, 也可以统一设置
    logger.setLevel(logging.INFO)
    
    return logger
