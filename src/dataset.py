import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2

class KittiStereoLidar(Dataset):
    def __init__(self, im_left_dir, im_right_dir, gt_left_dir, gt_right_dir, transform=None):
        self.im_left_dir = im_left_dir
        self.im_right_dir = im_right_dir
        self.gt_left_dir = gt_left_dir
        self.gt_right_dir = gt_right_dir
        
        self.im_left_dir.sort()
        self.im_right_dir.sort()
        self.gt_left_dir.sort()
        self.gt_right_dir.sort()
        
        self.transform = transform 
        
    def __getitem__(self, index):
        iml = Image.open(self.im_left_dir[index])
        imr = Image.open(self.im_right_dir[index])
        gtl = self.gt_left_dir[index]
        gtr = self.gt_right_dir[index]

        if self.transform:
            iml = self.transform(iml)
            imr = self.transform(imr)

        return iml, imr, gtl, gtr
    
    def __len__(self):
        return len(self.im_left_dir)
    
class KittiStereoLidarGray(Dataset):
    def __init__(self, im_left_dir, im_right_dir, gt_left_dir, gt_right_dir, transform=None):
        self.im_left_dir = im_left_dir
        self.im_right_dir = im_right_dir
        self.gt_left_dir = gt_left_dir
        self.gt_right_dir = gt_right_dir
        
        self.im_left_dir.sort()
        self.im_right_dir.sort()
        self.gt_left_dir.sort()
        self.gt_right_dir.sort()
        
        self.transform = transform 
        
    def __getitem__(self, index):
        iml = Image.open(self.im_left_dir[index])
        imr = Image.open(self.im_right_dir[index])
        gtl = self.gt_left_dir[index]
        gtr = self.gt_right_dir[index]

        if self.transform:
            iml = self.transform(iml)
            imr = self.transform(imr)
            iml = torch.stack([iml.squeeze(), iml.squeeze(), iml.squeeze()])
            imr = torch.stack([imr.squeeze(), imr.squeeze(), imr.squeeze()])

        return iml, imr, gtl, gtr
    
    def __len__(self):
        return len(self.im_left_dir)
    