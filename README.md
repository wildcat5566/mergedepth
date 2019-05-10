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
See Demo_lidar_gt_loss.ipynb for more details.

### Reconstruction loss
With intrinsic, extrinsic camera parameters & a fine depth prediction map, we can reconstruct the left view image using the corresponding right view image, or vice versa.
Therefore, the wellness of reconstructed complementary view image could indicate how good the predicted depth map is.
Reconstructed image is compared to the real target image in a pixel-wise manner.
See Demo_reconstruction_loss_class.ipynb for more details.

