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
    """
    {Function}
    Parse training arguments & configurations.
    
    {Outputs}
    args: Argument parser containing a set of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--epochs',         type=int,   default=5,    help='Epochs for training')
    parser.add_argument('--lr',             type=float, default=1e-3, help='set learning rate')
    parser.add_argument('--lr_decay',       type=int,   default=1,    help='decay learning rate')
    parser.add_argument('--lr_decay_rate',  type=float, default=0.9,  help='lr decay rate')
    parser.add_argument('--lr_decay_every', type=int,   default=2,    help='decay lr n times per epoch')
    
    parser.add_argument('--batch_size',   type=int,   default=8, help='set batch size')
    parser.add_argument('--print_every',  type=int,   default=5, help='print loss per _ batches')
    
    parser.add_argument('--augmentation', type=int,   default=1,    help='Augment data')
    parser.add_argument('--fullscale',    type=int,   default=0,    help='True when training large model')
    parser.add_argument('--train_frac',   type=float, default=0.05, help='training set fraction')
    
    parser.add_argument('--alpha',  type=float, default=1.0, help='set SUPERVISED loss weight')
    parser.add_argument('--gamma',  type=float, default=.01, help='set DEPTH MAP CONSISTENCY loss weight')
    parser.add_argument('--reg',    type=float, default=0,   help='set WEIGHT REGULARIZATION loss factor')
    parser.add_argument('--c_mask', type=int,   default=0,   help='add weight mask to consistency loss')

    parser.add_argument('--image_output_dir', type=str, default=None, \
                        help='Save samples while training. Leave none if not needed')
    parser.add_argument('--model_output_dir', type=str, default=None, \
                        help='Save model after training. Leave none if not needed')
    
    args = parser.parse_args()
    return args

def msg_format(args):
    """
    {Function}
    Format argument message for display & write to txt file.
    
    {Inputs}
    args: Set of parsed training arguments
    
    {Outputs}
    msg: Formatted message containing arguments.
    """
    msg = "Training Phase: 1  \
    \nEpochs: \t{} \nLearning Rate: \t{} \t(decay rate: {}, decay {} times per epoch) \
    \nBatch Size: \t{} \nAugmentation: \t{} \
    \nTraining set fraction: \t{} \
    \n\nSupervised loss weight (alpha): \t\t{} \
    \nDepth map consistency loss weight (gamma): \t{} \t(mask: {}) \
    \nWeight regularization loss factor: \t\t{}".format(
        args.epochs, args.lr, args.lr_decay*args.lr_decay_rate, args.lr_decay*args.lr_decay_every,
        args.batch_size, args.augmentation, args.train_frac,
        args.alpha, args.gamma, args.c_mask, args.reg)
        
    if args.image_output_dir is not None:
        msg+=("\n\nSave imgs (during training progress) to: " + args.image_output_dir)
    else:
        msg+=("\n\nNot saving imgs while training.")
        
    if args.model_output_dir is not None:
        msg+=("\nSave right view model to: \t" + os.path.join(args.model_output_dir, 'right.pth'))
        msg+=("\nSave left view model to: \t" + os.path.join(args.model_output_dir, 'left.pth'))
    else:
        msg+=("\nNot saving model.")
        
    return msg

def create_datasets(batch_size, augmentation, train_frac, num_workers=6):
    """
    {Function}
    Create subsets & dataloaders from training data.
    h*w = 96*320
    
    {Inputs}
    batch_size  : size of each data batch
    val&type    : default=8, dtype=int
    
    augmentation: augment data
    val&type    : default=1(True), dtype=int
    
    train_frac  : specify fraction of entire dataset used to create training set.
    val&type    : default=0.05 (~1000 samples), dtype=float
    
    {Outputs}
    train_loader: Pytorch DataLoader of training subset. With specified size, batch_size & augmentation.
    val&type:     PyTorch DataLoader
    
    valid_loader: Pytorch DataLoader of validation subset. Subset is 1/2 size of train_loader.
    val&type:     PyTorch DataLoader
    
    test_loader : Pytorch DataLoader of training subset. Subset size is 1/10 size of train_loader.
    val&type    : PyTorch DataLoader
    
    {Datasets}
    images_l, images_r: Left & right view images
    dims              : batch_size*3*h*w
    val&type          : [0, 1], dtype=torch.FloatTensor
    
    scans_l, scans_r  : Left & right view lidar scan projection file paths
    dims              : batch_size
    val&type          : dtype=str
    """
    kitti_ds = KittiTrain(
        im_left_dir=glob.glob( "data/left_imgs/*/*"), 
        im_right_dir=glob.glob("data/right_imgs/*/*"),
        gt_left_dir=glob.glob( "data/left_gt/*/*"), 
        gt_right_dir=glob.glob("data/right_gt/*/*"),
        transform = transforms.Compose([transforms.Resize((192,640)), 
                                        transforms.RandomGrayscale(p=augmentation*0.2),
                                        transforms.ToTensor()]),
        augmentation=augmentation
    )
    
    indices = list(range(len(kitti_ds)))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_frac * len(indices)))
    valid_split = int(np.floor((train_frac * 1.5) * len(indices)))
    test_split = int(np.floor((train_frac * 1.6) * len(indices)))
    
    train_indices = indices[:train_split]
    valid_indices = indices[train_split:valid_split]
    test_indices = indices[valid_split:test_split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset=kitti_ds, 
                              batch_size=batch_size, 
                              sampler=train_sampler,
                              shuffle=False, 
                              num_workers=num_workers)

    valid_loader = DataLoader(dataset=kitti_ds, 
                              batch_size=batch_size, 
                              sampler=valid_sampler,
                              shuffle=False, 
                              num_workers=num_workers)

    test_loader = DataLoader(dataset=kitti_ds, 
                             batch_size=batch_size, 
                             sampler=test_sampler,
                             shuffle=False, 
                             num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def get_su_loss(depth_maps, scan_files, batch_size):
    """
    {Function}
    Get supervised loss from lidar ground truth.
    h*w = 96*320
    
    {Inputs}
    depth_maps: Predicted depth maps.
    dims      : batch_size*1*h*w
    val&type  : [0, 1], dtype=torch.FloatTensor
    
    scan_files: File path of corresponding lidar scan file (ground truth).
    dims      : batch_size
    val&type  : dtype=str
    
    batch_size: size of each batch of training data.
    dims      : 1
    val&type  : default=8, dtype=int
    
    {Outputs}
    avg_batch_loss   : batch_loss / batch_size
    dims             : 1
    val&type         : [0, 1], dtype=torch.FloatTensor
    
    avg_accuracy_loss: 1 - (batch_accuracy / batch_size)
    dims             : 1
    val&type         : [0, 1], dtype=torch.FloatTensor
    """
    batch_loss = 0
    batch_accuracy = 0
    for[dep, scan_file] in zip(depth_maps, scan_files):
        dots = np.load(scan_file) 

        sample_loss, sample_accuracy = gt_loss(dep, dots, gt_augmentation=True)
        batch_loss += sample_loss
        batch_accuracy += sample_accuracy

    return batch_loss / batch_size, 1 - (batch_accuracy / batch_size)

# move to 2nd cuda device if training fullscale model
def get_con_loss(functions, depth_maps, tar_imgs, direction, dates, batch_size, weighting):  
    """
    {Function}
    Get L-R consistency loss from pair of predicted depth maps.
    h*w = 96*320
    
    {Inputs}
    functions : pixel-wise mapping functions defined in loss.py
    
    depth_maps: Predicted depth maps.
    dims      : batch_size*1*h*w
    val&type  : [0, 1], dtype=torch.FloatTensor
    
    tar_imgs  : Predicted depth maps of complementary view.
    dims      : batch_size*1*h*w
    val&type  : [0, 1], dtype=torch.FloatTensor
    
    direction : View transforming direction.
    dims      : 1
    val&type  : 'L2R' or 'R2L', dtype=str
    
    dates     : Dataset dates, used to match camera calibration parameters.
    dims      : 1
    val&type  : dtype=str
    
    batch_size
    dims      : 1
    val&type  : default=8, dtype=int
    
    weighting : Use weighting mask in loss function or not.
    dims      : 1
    val&type  : default=False(args.c_mask=0), dtype=int
    
    {Outputs}
    avg_batch_loss: batch_loss / batch_size
    dims          : 1
    val&type      : [0, 1], dtype=torch.FloatTensor
    """
    batch_loss = 0
    for[dep, tar, dat] in zip(depth_maps, tar_imgs, dates):
        if dat=='2011_09_26':
            f = functions[0]
        elif dat=='2011_09_28':
            f = functions[1]
        elif dat=='2011_09_29':
            f = functions[2]
        elif dat=='2011_09_30':
            f = functions[3]
        elif dat=='2011_10_03':
            f = functions[4]

        sample_loss = f.compute_loss(dep, tar, direction, weighting=weighting)
        batch_loss += sample_loss

    return batch_loss / batch_size

def get_reg_loss(model, factor):
    """
    {Function}
    Sum & weight all model parameters.
    
    {Inputs}
    model   : Specified model.
    
    factor  : weighting factor for parameter regularization.
    dims    : 1
    val&type: [0, 1], dtype=float
    
    {Outputs}
    factor*reg_loss: (weighted) sum of all model parameters.
    dims           : 1
    val&type       : dtype=float
    """
    l1_crit = nn.L1Loss(size_average=False)
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += l1_crit(param, target=torch.zeros_like(param).cuda())

    return(factor * reg_loss)

def adjust_learning_rate(optimizer, lr, lr_decay_rate):
    """
    {Function}
    Adjust learning rate of specified model optimizer.
    
    {Inputs}
    optimizer    : Specified model optimizer.
    
    lr           : learning rate.
    dims         : 1
    val&type     : [0, 1], dtype=float
    
    lr_decay_rate: decay rate (multiplier) of learning rate.
    dims         : 1
    val&type     : [0, 1], dtype=float
    
    {Outputs}
    lr      : adjusted learning rate.
    dims    : 1
    val&type: [0, 1], dtype=float
    """
    lr = lr * lr_decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def save_model(model, optimizer, save_path):
    """
    {Functions}
    Save model & optimizer state to specified path.
    
    {Inputs}
    model: Specified model to save.
    optimizer: Optimizer of model specified above.
    save_path: Specified path to save model (******.pth).
    """
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}

    torch.save(state, save_path)

def save_image(pti, ptd, tempname, save_path, crop=False):
    """
    {Function}
    Save concatenated source & predicted depth map image to file system.
    H*W = 192*640
    h*w = 96*320
    
    {Inputs}
    pti      : Sample image of training set. Might be augmented.
    dims     : 3*H*W
    val&type : [0, 1], dtype=torch.FloatTensor
    
    ptd      : Predicted depth map of image above (pti).
    dims     : 1*h*w
    val&type : [0, 1], dtype=torch.FloatTensor
    
    tempname : specified saving file name.
    dims     : 1
    val&type : dtype=str
    
    save_path: specified path of saving image (******.jpg)
    dims     : 1
    val&type : dtype=str
    
    {Outputs}
    cat     : Vertically concatenated image of sample image & predicted depth map. 
    dims    : 3*(h*2)*w
    val&type: [0, 255], dype=uint8
    """
    
    # to numpy
    gray = np.transpose(ptd.cpu().detach().numpy(), (1,2,0)).squeeze()
    rgb = np.transpose(pti.cpu().detach().numpy(), (1,2,0))
    
    # Crop edges for test. yes: 86*300, no: 96*320
    if crop:
        gray = gray[5:-5, 10:-10]
        rgb = rgb[10:-10, 20:-20]
    
    # match dimensions, normalize, format
    gray = (gray-np.amin(gray)) / (np.amax(gray) - np.amin(gray) + 1e-4)
    gray = np.stack([gray for i in range(3)],axis=2)
    if rgb.shape[1]!= gray.shape[1]:
        rgb = cv2.resize(rgb, (gray.shape[1], gray.shape[0]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
    cat = np.concatenate((rgb, gray), axis=0) #horizontal
    cat = (cat*255).astype(np.uint8)
    save_name = os.path.join(save_path, tempname)
    cv2.imwrite(save_name, cat)

def main():
    args = config()
    
    print("Hello! May the force be with you!")
    print("------Argument configurations------")
    settings = msg_format(args)
    print(settings)

    if args.model_output_dir:
        settings_fname = os.path.join(args.model_output_dir, 'settings.txt')
    elif args.image_output_dir:
        settings_fname = os.path.join(args.image_output_dir, 'settings.txt')
    else:
        settings_fname = 'settings.txt'
        
    print("\n------Write training settings to {}------".format(settings_fname))
    with open(settings_fname, "w") as text_file:
        print(settings, file=text_file)
    text_file.close()

    train_loader, valid_loader, test_loader = create_datasets(batch_size=args.batch_size, 
                                                              augmentation=args.augmentation, 
                                                              train_frac=args.train_frac)
    print("\n------Create subsets------")
    print("# Training samples: \t{}".format(len(train_loader) * args.batch_size))
    print("# Validation samples: \t{}".format(len(valid_loader) * args.batch_size))
    print("# Test samples: \t{}".format(len(test_loader) * args.batch_size))

    sc = 320/1242
    conf = [Consistency(date='2011_09_26',scaling=sc,device=args.fullscale), 
            Consistency(date='2011_09_28',scaling=sc,device=args.fullscale),
            Consistency(date='2011_09_29',scaling=sc,device=args.fullscale), 
            Consistency(date='2011_09_30',scaling=sc,device=args.fullscale),
            Consistency(date='2011_10_03',scaling=sc,device=args.fullscale)]

    print("\n------Create networks & optimizers------")
    if args.fullscale == False: #Half scale model
        L = Halfscale()
        L = torch.nn.DataParallel(L).cuda()
        R = Halfscale()
        R = torch.nn.DataParallel(R).cuda()
        
    else: #Full scale model
        L = Fullscale(split_gpus=True)
        R = Fullscale(split_gpus=True)
        
    L_optimizer = torch.optim.Adam(L.parameters(), lr=args.lr)
    R_optimizer = torch.optim.Adam(R.parameters(), lr=args.lr)

    L.train()
    R.train()
    print("\n------Start training------")

    for epoch in range(args.epochs):
        train_loss = 0.0
        s_step, c_step = 0.0, 0.0
        a_step = 0.0

        batch_count = 1
        n = 1
        time_start = time.time()

        for images_l, images_r, scans_l, scans_r in train_loader:
            # Zero optimizer gradients
            L_optimizer.zero_grad()
            R_optimizer.zero_grad()
        
            # Move to cuda
            images_l = images_l.cuda()
            images_r = images_r.cuda()
        
            # Forward pass, make predictions (sigmoided)
            depths_l = L(images_l)
            depths_r = R(images_r)
            drive_dates = [s[13:23] for s in scans_l]
        
            # Compute supervised (lidar) losses (device=0)
            s_loss_L, s_acc_L = get_su_loss(depth_maps=depths_l, scan_files=scans_l, batch_size=args.batch_size)
            s_loss_R, s_acc_R = get_su_loss(depth_maps=depths_r, scan_files=scans_r, batch_size=args.batch_size)

            # Compute depth map L-R consistency losses (device=1 on halfscale)
            c_loss_L = get_con_loss(functions=conf,
                                    depth_maps=depths_l,  
                                    tar_imgs=depths_r, 
                                    direction='L2R', dates=drive_dates, 
                                    batch_size=args.batch_size, weighting=args.c_mask)
            c_loss_R = get_con_loss(functions=conf,
                                    depth_maps=depths_r,  
                                    tar_imgs=depths_l, 
                                    direction='R2L', dates=drive_dates, 
                                    batch_size=args.batch_size, weighting=args.c_mask)
        
            # Weight & sum losses. Losses are batch-normalized
            loss = (args.alpha*(s_loss_L + s_loss_R) \
                  + args.gamma*(c_loss_L + c_loss_R)) / (2*(args.alpha + args.gamma))

            # Back propagation & optimize
            loss.backward()
            L_optimizer.step()
            R_optimizer.step()

            train_loss += loss.item()
            step_loss = train_loss / batch_count
            
            s_step += ((s_loss_L.item() + s_loss_R.item()) / 2)
            a_step += ((s_acc_L.item() + s_acc_R.item()) / 2)
            c_step += ((c_loss_L.item() + c_loss_R.item()) / 2)
            
            if batch_count % args.print_every == 0:
                print('Epoch: {} ({:.2f}%)\tStep Loss: {:.6f} \
                       \n\tSu: {:.6f} \tAcc_loss: {:.6f} \tCon: {:.6f}'.format(
                    epoch+1,
                    100*(batch_count / len(train_loader)), 
                    step_loss,
                    s_step / batch_count,
                    a_step / batch_count,
                    c_step / batch_count
                ))

                if args.image_output_dir is not None:
                    tempname = "L_epoch_{:02}_{:03}.jpg".format(epoch+1, n)
                    save_image(pti=images_l[0], ptd=depths_l[0], tempname=tempname, save_path=args.image_output_dir)
                    tempname = "R_epoch_{:02}_{:03}.jpg".format(epoch+1, n)
                    save_image(pti=images_r[0], ptd=depths_r[0], tempname=tempname, save_path=args.image_output_dir)
                n += 1    
            
            # decay learning rate
            if batch_count % int(len(train_loader) / args.lr_decay_every) == int(len(train_loader) / args.lr_decay_every) - 1:
                if args.lr_decay:
                    args.lr = adjust_learning_rate(L_optimizer, args.lr, args.lr_decay_rate)
                    args.lr = adjust_learning_rate(R_optimizer, args.lr, args.lr_decay_rate)
                    for param_group in L_optimizer.param_groups:
                        print("Adaptive learning rate:{:.6f}".format(param_group['lr']))
                        
            batch_count += 1

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler) #image pair count
        time_elapsed = time.time() - time_start
        print('Epoch: {} \tTraining Loss: {:.6f} \tTime: {} s'.format(
            epoch+1, 
            train_loss,
            round(time_elapsed, 4)
        ))
        
        if args.model_output_dir is not None:
            fname = 'right_{:02}.pth'.format(epoch+1)
            save_model(R, R_optimizer, os.path.join(args.model_output_dir, fname))
            fname = 'left_{:02}.pth'.format(epoch+1)
            save_model(L, L_optimizer, os.path.join(args.model_output_dir, fname))
            print("\n------Finish saving models------")

    print("\n------Finish training------")
    return

if __name__ == "__main__":
    main()
