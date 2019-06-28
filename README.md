# mergedepth

## Introduction
Deep learning-based binocular (stereo) scene depth estimation. 

## Network structure
ResNet50-based (remove 2 final layers of original ResNet50) encoder & up-projection blocks for decoder.

### Training phase 
inputs:
- 1 stereo image pair (2 images, left view & right view).
- 1 lidar scan projection pair (for left & right view images respectively) -- serving as scene depth ground truth.
augmentation (only on images):
- Random grayscale.
- Gaussian noise.
outputs:
- 2 depth map predictions (corresponding to both views of training image pair).

### Inference phase 
inputs:
- 1 single image (inference left/right views independently).
outputs:
- 1 depth map prediction.

## Loss function definition
### Supervised loss
Compare predicted depth map with available ground truth data provided by Lidar, in pixel-wise manner. <br />
See examples/Demo_lidar_gt_loss.ipynb for more details. <br />

### Reconstruction loss
With intrinsic, extrinsic camera parameters & a fine depth prediction map, we can reconstruct the left view image using the corresponding right view image, or vice versa. <br />
Therefore, the wellness of reconstructed complementary view image could indicate how good the predicted depth map is. <br />
Reconstructed image is compared to the real target image in a pixel-wise manner. <br />
See examples/Demo_reconstruction_loss.ipynb for more details. <br />

### Depth map consistency loss
Repeat reconstruction loss computation progress but taking the left & right depth maps predicted as inputs instead. <br />
The goal is to ensure consistency of predicted right & left depth maps, especially for the upper region of captured views where lidar ground truth is not available. <br />
See examples/Demo_consistency_loss.ipynb for more details. <br />

### Training phases
Phase 1: Supervised loss + L-R consistency loss. <br />
Phase 2: Reconstruction loss + L-R consistency loss. <br />

### Dataset preperation
The training code is based on KITTI dataset. <br />
Please have training datasets organized as following structure: <br />
<br />
left_imgs:  Training images(left view) <br />
right_imgs: Training images(right view) <br />
left_gt:    Ground truth(left view) <br />
right_gt:   Ground truth(right view) <br />
<br />
data/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;|--left_imgs/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--2011_09_26_drive_0001_sync/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--0000000000.jpg <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;|--right_imgs/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--2011_09_26_drive_0001_sync/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--0000000000.jpg <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;|--left_gt/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--2011_09_26_drive_0001_sync/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--0000000000.npy <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;|--right_gt/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--2011_09_26_drive_0001_sync/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--0000000000.npy <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--... <br />
<br />
OR modify training set directories at def create_dataset() in train.py. <br />

### Training
> cd workspace/mergedepth <br />
> mkdir train <br />
> CUDA_VISIBLE_DEVICES=N python3 train.py --lr 1e-3 --epochs 5 --batch_size 8 --alpha 1 --gamma .5 \ <br />
> --image_output_dir ./train/gpu5 --model_output_dir ./train/gpu5 --lr_decay_rate 0.9 <br />

### Train from existing model
> cd workspace/mergedepth <br />
> CUDA_VISIBLE_DEVICES=N python3 keep_training.py --beta .1 --gamma 1 --lr 1e-5 \ <br />
> --l_model_pth train/archive/su0625/left_10.pth --r_model_pth train/archive/su0625/right_10.pth \ <br />
> --model_output_dir train/gpu0 --image_output_dir train/gpu0  <br />

### Inference
> cd workspace/mergedepth <br />
> mkdir inference <br />
> CUDA_VISIBLE_DEVICES=0 python3 inference.py \ <br />
--output_dir inference/left \ <br />
--model_pth train/gpu5_0616/left.pth \ <br />
--input_dir data/left_imgs/2011_09_26_drive_0001_sync \ <br />

## References
### Papers
- "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", Eigen et al., NIPS'14.
- "Semi-Supervised Deep Learning for Monocular Depth Map Prediction", Kuznietsov et al., CVPR'17.
- "Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue", Garg et al., ECCV'16.
- "Deeper depth prediction with fully convolutional residual networks", Laina et al., 3DV'16.

### Repositories
- Kitti foundation: https://github.com/windowsub0406/KITTI_Tutorial
