# mergedepth

## Introduction
Deep learning-based binocular (stereo) scene depth estimation. 

## Network structure
ResNet-50 based.
Redisual blocks for encoder & up-projection blocks for decoder.

Training phase inputs:
- 1 stereo image pair (2 images, left view & right view).
- 1 lidar scan projection pair (for left & right view images respectively) -- serving as scene depth ground truth. 

Inference phase inputs:
- 1 stereo image pair

Network output:
2 depth map predictions (corresponding to both views of training image pair).

## Loss function definition
### Supervised loss
Compare predicted depth map with available ground truth data provided by Lidar, in pixel-wise manner.
See examples/Demo_lidar_gt_loss.ipynb for more details.

### Reconstruction loss
With intrinsic, extrinsic camera parameters & a fine depth prediction map, we can reconstruct the left view image using the corresponding right view image, or vice versa.
Therefore, the wellness of reconstructed complementary view image could indicate how good the predicted depth map is.
Reconstructed image is compared to the real target image in a pixel-wise manner.
See examples/Demo_reconstruction_loss.ipynb for more details.

### Depth map consistency loss
Repeat reconstruction loss computation progress but taking the left & right depth maps predicted as inputs instead.
The goal is to ensure consistency of predicted right & left depth maps, especially for the upper region of captured views where lidar ground truth is not available.
See examples/Demo_consistency_loss.ipynb for more details.

## Run the code
### Dataset preperation
The training code is based on KITTI dataset.
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

### Preprocessing

### Training
> cd workspace/mergedepth <br />
> mkdir train <br />
> python3 train.py --epochs 1 --lr 1e-4 --batch_size 8 --image_output_dir ./train --model_output_dir ./train <br />

### Inference
