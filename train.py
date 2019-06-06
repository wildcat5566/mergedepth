from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import time
import os, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.utils import *
from src.loss import *
from src.model import *
from src.dataset import *

def config():
    parser = ArgumentParser()
    parser.add_argument('--epochs', required=True, help='Epochs for training')
    parser.add_argument('--lr', required=True, help='set learning rate')

    parser.add_argument('--batch_size', default=2, help='set batch size')
    parser.add_argument('--alpha', default=1.0, help='set SUPERVISED loss weight')
    parser.add_argument('--beta', default=.01, help='set L-R RECONSTRUCTION loss weight')
    parser.add_argument('--gamma', default=.01, help='set DEPTH MAP CONSISTENCY loss weight')

    parser.add_argument('--r_mask', type=bool, default=True, help='add weight mask to reconstruction loss')
    parser.add_argument('--c_mask', type=bool, default=False, help='add weight mask to consistency loss')

    parser.add_argument('--image_output_dir', help='Save samples while training. Leave none if not needed')
    parser.add_argument('--model_output_dir', help='Save model after training. Leave none if not needed')
    
    args = parser.parse_args()
    return args

def create_dataset(batch_size, num_workers=6):
    kitti_ds = KittiStereoLidar(
        im_left_dir=glob.glob("data/left_imgs/*/*"), 
        im_right_dir=glob.glob("data/right_imgs/*/*"),
        gt_left_dir=glob.glob("data/left_gt/*/*"), 
        gt_right_dir=glob.glob("data/right_gt/*/*"),
        transform=transforms.Compose([transforms.Resize((192,640)),
                                      transforms.ToTensor()])
    )
    data_loader = DataLoader(dataset=kitti_ds, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=num_workers)

    return data_loader
    
def normalize_prediction(map_input, scale=100):
    M, m=np.amax(map_input), np.amin(map_input)
    return (map_input - m)*(scale / (M-m))

def get_su_loss(depth_maps, scan_files):
    batch_loss = 0
    for[dep, scan_file] in zip(depth_maps, scan_files):
        dots = np.load(scan_file) 

        sample_loss = gt_loss(dep, dots)
        batch_loss += sample_loss

    return batch_loss / args.batch_size

def get_recon_loss(depth_maps, src_imgs, tar_imgs, direction, dates):
    batch_loss = 0
    for[dep, src, tar, dat] in zip(depth_maps, src_imgs, tar_imgs, dates):
        if dat=='2011_09_26':
            recf = reconstruct_functions[0]
        elif dat=='2011_09_28':
            recf = reconstruct_functions[1]
        elif dat=='2011_09_29':
            recf = reconstruct_functions[2]
        elif dat=='2011_09_30':
            recf = reconstruct_functions[3]
        elif dat=='2011_10_03':
            recf = reconstruct_functions[4]

        sample_loss, _ = recf.compute_loss(dep, src, tar, direction, weighting=args.r_mask)
        batch_loss += sample_loss

    return batch_loss / args.batch_size

def get_con_loss(depth_maps, tar_imgs, direction, dates):  
    batch_loss = 0
    for[dep, tar, dat] in zip(depth_maps, tar_imgs, dates):
        if dat=='2011_09_26':
            recf = consistency_functions[0]
        elif dat=='2011_09_28':
            recf = consistency_functions[1]
        elif dat=='2011_09_29':
            recf = consistency_functions[2]
        elif dat=='2011_09_30':
            recf = consistency_functions[3]
        elif dat=='2011_10_03':
            recf = consistency_functions[4]

        sample_loss, _ = recf.compute_loss(dep, tar, direction, weighting=args.c_mask)
        batch_loss += sample_loss

    return batch_loss / args.batch_size

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def save_image(image, tempname, save_path):
    save_name = os.path.join(save_path, tempname)
    image = (image*255).astype(np.uint8)
    cv2.imwrite(image, save_name)

def main():
    args = config()
    print("------Argument Configurations------")
    print("Epochs: \t{}".format(args.epochs))
    print("Learning Rate: \t{}".format(args.lr))
    print("Batch Size: \t{}".format(args.batch_size))
    
    print("Supervised loss weight (alpha): \t\t{}".format(args.alpha))
    print("Reconstruction loss weight (beta): \t\t{} \t(mask: {})".format(args.beta, args.r_mask))
    print("Depth map consistency loss weight (gamma): \t{} \t(mask: {})".format(args.gamma, args.c_mask))

    if args.images_output_dir is not None:
        print("Save imgs (during training progress) to: " + args.images_output_dir)
    else:
        print("Not saving imgs while training.")
        
    if args.model_output_dir is not None:
        print("Save right view model to: " + os.path.join(args.model_output_dir, 'right.pth'))
        print("Save left view model to: " + os.path.join(args.model_output_dir, 'left.pth'))
    else:
        print("Not saving model.")

    train_loader = create_dataset(batch_size=args.batch_size)
    print("------Training set created------")
    print("Training samples : {}".format(len(train_loader)))

    sc = 320/1242
    reconstruct_functions = [Reconstruction(date='2011_09_26',scaling=sc), 
                             Reconstruction(date='2011_09_28',scaling=sc),
                             Reconstruction(date='2011_09_29',scaling=sc), 
                             Reconstruction(date='2011_09_30',scaling=sc),
                             Reconstruction(date='2011_10_03',scaling=sc)]

    consistency_functions = [Consistency(date='2011_09_26',scaling=sc), 
                             Consistency(date='2011_09_28',scaling=sc),
                             Consistency(date='2011_09_29',scaling=sc), 
                             Consistency(date='2011_09_30',scaling=sc),
                             Consistency(date='2011_10_03',scaling=sc)]

    # Depth prediction networks for left & right view sets respectively
    L = Network()
    L = torch.nn.DataParallel(L).cuda()
    L_optimizer = torch.optim.Adam(L.parameters(), lr=args.lr)
    
    R = Network()
    R = torch.nn.DataParallel(R).cuda()
    R_optimizer = torch.optim.Adam(R.parameters(), lr=args.lr)

    L.train()
    R.train()
    print("------Start training------")

    for epoch in range(args.epochs):
        # monitor training loss
        train_loss = 0.0
        s_step, r_step, c_step = 0.0, 0.0, 0.0

        batch_count = 1
        sample_count = 1
        print_every = 5 #20
        time_start = time.time()

        for images_l, images_r, scans_l, scans_r in train_loader:
            L_optimizer.zero_grad()
            R_optimizer.zero_grad()
        
            # Move to cuda
            images_l = images_l.cuda()
            images_r = images_r.cuda()
        
            # Forward pass, make predictions
            depths_l = L(images_l)
            depths_r = R(images_r)

            drive_dates = [s[13:23] for s in scans_l]
        
            # Compute supervised (lidar) losses
            s_loss_L = get_su_loss(depth_maps=depths_l, scan_files=scans_l)
            s_loss_R = get_su_loss(depth_maps=depths_r, scan_files=scans_r)
        
            # Compute reconstruction losses
            r_loss_L = get_recon_loss(depth_maps=depths_l, 
                                      src_imgs=images_l, 
                                      tar_imgs=images_r, 
                                      direction='L2R', dates=drive_dates)
            r_loss_R = get_recon_loss(depth_maps=depths_r, 
                                      src_imgs=images_l, 
                                      tar_imgs=images_l, 
                                      direction='R2L', dates=drive_dates)
        
            # Compute depth map L-R consistency losses
            c_loss_L = get_con_loss(depth_maps=depths_l,  
                                    tar_imgs=depths_r, 
                                    direction='L2R', dates=drive_dates)
            c_loss_R = get_con_loss(depth_maps=depths_r,  
                                    tar_imgs=depths_l, 
                                    direction='R2L', dates=drive_dates)
        
            # Weight & sum losses
            loss = (args.alpha*(s_loss_L + s_loss_R) \
                  + args.beta *(r_loss_L + r_loss_R) \
                  + args.gamma*(c_loss_L + c_loss_R)) / (2*(args.alpha + args.beta + args.gamma))

            # Back propagation & optimize
            loss.backward()
            L_optimizer.step()
            R_optimizer.step()

            train_loss += loss.item()
            s_step += (s_loss_L.item() + s_loss_R.item())
            r_step += (r_loss_L.item() + r_loss_R.item())
            c_step += (c_loss_L.item() + c_loss_R.item())
        
            step_loss = train_loss / (batch_count * args.batch_size)
            if batch_count % print_every == 0:
                print('Epoch: {} ({:.2f}%)\tStep Loss: {:.6f} \n\tSu: {:.6f} \tUnsu: {:.6f} \tCon: {:.6f} \n\tmax/min:{}'.format(
                    epoch+1,
                    100*(batch_count / len(train_loader)), 
                    step_loss,
                    s_step / (batch_count * batch_size),
                    r_step / (batch_count * batch_size),
                    c_step / (batch_count * batch_size),
                    (depths_l[0].max(), depths_l[0].min())
                ))
                npd = np.transpose(depths_l[0].cpu().detach().numpy(), (1,2,0)).squeeze()
                npd = npd[5:-5, 10:-10]
                npd = (npd-np.amin(npd)) / (np.amax(npd) - np.amin(npd) + 1e-4) #normalize display, prevent div by 0
                npi = np.transpose(images_l[0].cpu().detach().numpy(), (1,2,0))
                npi = npi[10:-10, 20:-20]

                sample_count += 1
                if args.image_output_dir is not None:
                    tempname='epoch_'+str(epoch)+'_'+str(sample_count)+'.jpg'
                    save_image(npd, tempname=tempname, save_path=args.image_output_dir)
                    tempname='epoch_'+str(epoch)+'_'+str(sample_count)+'_ref.jpg'
                    save_image(npi, tempname=tempname, save_path=args.image_output_dir)

            batch_count += 1

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler) #image pair count
        time_elapsed = time.time() - time_start
        print('Epoch: {} \tTraining Loss: {:.6f} \tTime: {} s'.format(
            epoch+1, 
            train_loss,
            round(time_elapsed, 4)
        ))

    print("------Finished training------")
    if args.model_output_dir is not None:
        save_model(R, os.path.join(args.model_output_dir, 'right.pth'))
        save_model(L, os.path.join(args.model_output_dir, 'left.pth'))

if __name__ == "__main__":
    main()
