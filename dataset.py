import pdb
import torch
import numpy as np
import cv2
import sys
import os
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_size, dataset):
        self.dataset_root = os.path.join('data', dataset)
        file_list = os.path.join(self.dataset_root, 'image_list.txt')
        with open(file_list, 'r') as f:
            self.lines = f.readlines()
        self.image_size = image_size
            
    def __getitem__(self, index):
        l = self.lines[index].strip()
        image = cv2.imread(os.path.join(self.dataset_root, l))
        h,w,_ = image.shape
        shorter_side = min(h, w)
        left = np.random.randint(0, max(w-shorter_side, 1))
        up = np.random.randint(0, max(h-shorter_side, 1))
        cropped_image = image[up:up+h, left:left+w,:]
        image = cv2.resize(cropped_image, (self.image_size,self.image_size))
        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1) # [3,H,W]
        image = (image/255.-0.5)/0.5
        return image

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = 'data/cat/image_list.txt'
    dataset = CustomDataset(96, file_list)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=False)
    
    for image in dataloader:
        image = (image*0.5+0.5)*255
        image = image.permute(0,2,3,1).numpy().astype(np.int32) 
        cv2.imwrite('test.jpg',image[0])
        pdb.set_trace()
