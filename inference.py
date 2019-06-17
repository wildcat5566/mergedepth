from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import time
import os, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.loss import *
from src.model import *
from src.dataset import *

"""
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
--output_dir inference/left \
--model_pth train/gpu5_0616/left.pth \
--input_dir data/left_imgs/2011_09_26_drive_0001_sync \
"""

def config():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='BJ4')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with inference inputs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for inferenced outputs')
    parser.add_argument('--model_pth', type=str, required=True, help='Load model state dict from .pth file')

    args = parser.parse_args()
    return args

def create_dataset(im_dir, batch_size):
    kitti_ds = KittiTest(
        im_dir=glob.glob(os.path.join(im_dir, "*")), 
        transform=transforms.Compose([transforms.Resize((192,640)),
                                      transforms.ToTensor()])
    )
    test_loader = DataLoader(dataset=kitti_ds, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=6)
    
    return test_loader

def save_image(ptd, fileno, save_path, crop=True):
    gray = np.transpose(ptd.cpu().detach().numpy(), (1,2,0)).squeeze()
    if crop:
        gray = gray[5:-5, 10:-10]

    gray = (gray-np.amin(gray)) / (np.amax(gray) - np.amin(gray) + 1e-4)
    gray = (gray*255).astype(np.uint8)
    save_name=os.path.join(save_path, fileno)
    cv2.imwrite(save_name, gray)

def main():
    args = config()
    print('------Load models------')
    
    M = Halfscale()
    M = torch.nn.DataParallel(M)
    M.load_state_dict(torch.load(args.model_pth))
    
    print('------Create dataset------')
    test_loader = create_dataset(im_dir=args.input_dir,
                                 batch_size=args.batch_size)
    print("Dataloader length: \t{}".format(len(test_loader)))
    for images, fileno in test_loader:
        images = images.cuda()
        depths = M(images)
        
        for i in range(len(depths)):
            save_image(ptd=depths[i], fileno=fileno[i], save_path=args.output_dir)

if __name__ == "__main__":
    main()
    