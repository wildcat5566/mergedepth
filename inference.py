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

def config():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='set batch size')

    parser.add_argument('--l_input_dir', type=str, required=True, help='Inference inputs for left view')
    parser.add_argument('--r_input_dir', type=str, required=True, help='Inference inputs for right view')
    parser.add_argument('--l_output_dir', type=str, required=True, help='Inferenced outputs for left view')
    parser.add_argument('--r_output_dir', type=str, required=True, help='Inferenced outputs for right view')
    parser.add_argument('--l_model_pth', type=str, required=True, help='Load left view model from path')
    parser.add_argument('--r_model_pth', type=str, required=True, help='Load right view model from path')
    
    #data/left_imgs/2011_09_26_drive_0005_sync/
    
    args = parser.parse_args()
    return args

def create_dataset(im_left_dir, im_right_dir, batch_size):
    kitti_ds = KittiTest(
        im_left_dir=glob.glob(os.path.join(im_left_dir, "*")), 
        im_right_dir=glob.glob(os.path.join(im_right_dir, "*")),
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
    L = torch.load(args.l_model_pth).cuda()
    R = torch.load(args.r_model_pth).cuda()

    # Turn off dropout
    L.eval()
    R.eval()
    
    print('------Create dataset------')
    test_loader = create_dataset(im_left_dir=args.l_input_dir,
                                 im_right_dir=args.r_input_dir,
                                 batch_size=args.batch_size)
    
    for images_l, images_r, fileno in test_loader:
        images_l = images_l.cuda()
        images_r = images_r.cuda()
        depths_l = L(images_l)
        depths_r = R(images_r)
        
        for i in range(len(depths_l)):
            save_image(ptd=depths_l[i], fileno=fileno[i], save_path=args.l_output_dir)
            save_image(ptd=depths_r[i], fileno=fileno[i], save_path=args.r_output_dir)

if __name__ == "__main__":
    main()
    