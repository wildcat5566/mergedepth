import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np

class KittiTrain(Dataset):
    def __init__(self, im_left_dir, im_right_dir, gt_left_dir, gt_right_dir, \
                 transform=None, augmentation=True):
        self.im_left_dir = im_left_dir
        self.im_right_dir = im_right_dir
        self.gt_left_dir = gt_left_dir
        self.gt_right_dir = gt_right_dir
        
        self.im_left_dir.sort()
        self.im_right_dir.sort()
        self.gt_left_dir.sort()
        self.gt_right_dir.sort()
        
        self.transform = transform
        self.augmentation = augmentation
        
    def __getitem__(self, index):
        iml = Image.open(self.im_left_dir[index])
        imr = Image.open(self.im_right_dir[index])
        gtl = self.gt_left_dir[index]
        gtr = self.gt_right_dir[index]

        if self.transform:
            iml = self.transform(iml)
            imr = self.transform(imr)
            
        if self.augmentation:
            iml = self.augmentation_gaussian_noise(iml)
            imr = self.augmentation_gaussian_noise(imr)

        return iml, imr, gtl, gtr

    def augmentation_gaussian_noise(self, img, mag=0.5, p=0.2): #3*h*w
        r = np.random.random()
        if r < p:
            img += (mag * torch.rand(3, img.shape[1], img.shape[2]))
            img = img / (1 + mag)
        return img
    
    def __len__(self):
        return len(self.im_left_dir)
    
class KittiTest(Dataset): #single view respectively
    def __init__(self, im_dir, transform=None):
        self.im_dir = im_dir
        self.im_dir.sort()
        self.transform = transform 
        
    def __getitem__(self, index):
        im = Image.open(self.im_dir[index])
        fileno = self.im_dir[index][-14:]
        if self.transform:
            im = self.transform(im)
        return im, fileno
    
    def __len__(self):
        return len(self.im_dir)
    