import numpy as np
import scipy.io as sio
import os
from src.kitti_foundation import Kitti, Kitti_util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import time

"""
crop_lidar:
3d lidar scan --> Project to corresponding left & right views
"""
def crop_lidar(img_path, velo_path, v2c_path, c2c_path, frame_id):
    v_fov, h_fov = (-24.9, 2.0), (-90, 90)
    velo = Kitti(frame=frame_id, velo_path=velo_path)
    frame = velo.velo_file

    res = Kitti_util(frame=frame_id, camera_path=img_path, velo_path=velo_path, \
                v2c_path=v2c_path, c2c_path=c2c_path)

    img, points, depths = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)

    dots = np.zeros((img.shape[0], img.shape[1]))
    for i in range(points.shape[1]):
        x_index = np.int32(points[0][i])
        y_index = np.int32(points[1][i])
        if (0 <= y_index < img.shape[0] and 0 <= x_index < img.shape[1]): #crop range within view
            dots[y_index, x_index] = depths[i]
            
    return np.asarray(dots)

def ground_truth_preprocess(dataset_dir):
    t0 = time.time()
    # Create save dir (left, right)
    save_dir = [os.path.join('groundtruth', dataset_dir[-37:], 'image_02', 'data'),
                os.path.join('groundtruth', dataset_dir[-37:], 'image_03', 'data')]
    print(save_dir)
    
    for i in range(2):
        try:
            os.makedirs(save_dir[i])
            print("Create & save processed gt files to: " + save_dir[i])
        except FileExistsError:
            print("Save processed gt files to: " + save_dir[i])

    l_img_path = os.path.join(dataset_dir, 'image_02', 'data')
    r_img_path = os.path.join(dataset_dir, 'image_03', 'data')
    velo_path = os.path.join(dataset_dir, 'velodyne_points', 'data')
    v2c_path = os.path.join(dataset_dir[:32], 'calib_velo_to_cam.txt')
    c2c_path = os.path.join(dataset_dir[:32], 'calib_cam_to_cam.txt')

    n_files = len([f for f in os.listdir(r_img_path) if os.path.isfile(os.path.join(r_img_path, f))])
    for i in range(n_files):
        d = crop_lidar(l_img_path, velo_path, v2c_path, c2c_path, i)
        np.save(os.path.join(save_dir[0], "{:010}".format(i)+'.npy'), d)
        #np.savetxt(os.path.join(save_dir[0], "{:010}".format(i)+'.csv'), d, delimiter=",")
        #d = crop_lidar(r_img_path, velo_path, v2c_path, c2c_path, i)
        #np.save(os.path.join(save_dir[1], "{:010}".format(i)+'.npy'), d)
        #np.savetxt(os.path.join(save_dir[1], "{:010}".format(i)+'.csv'), d, delimiter=",")
    
    t1 = time.time() - t0
    print('file count: ' + str(n_files) + ', elapsed_time: ' + str(t1) + ' sec')
        
def show_lidar(img, dots):
    canvas = (img*255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV)
    for i in range(dots.shape[0]):
        for j in range(dots.shape[1]):
            if dots[i][j]:
                cv2.circle(canvas, (j,i), 2, (int(dots[i][j]),255,255),-1)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2RGB)
    plt.subplots(1,1, figsize = (18,5)) #13,3
    plt.title('Lidar data')
    plt.imshow(canvas)
    
        