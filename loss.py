import numpy as np
from kitti_foundation import Kitti, Kitti_util
from numpy.linalg import multi_dot, inv

#0926 series
K2 = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
               [0.000000e+00, 9.569251e+02, 2.241806e+02],
               [0.000000e+00, 0.000000e+00, 1.000000e+00]])
K3 = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
               [0.000000e+00, 9.019653e+02, 2.242509e+02],
               [0.000000e+00, 0.000000e+00,1.000000e+00]])

#L2R
R3 = np.eye(3) #No rotation
C3 = np.array([[0.54], [0.], [0.]]) #Baseline length = 54 cm
K3R3_K2 = multi_dot([K3, R3, inv(K2)])
K3R3C3 = multi_dot([K3, R3, C3])

#R2L
R2 = np.eye(3) #No rotation
C2 = np.array([[-0.54], [0.], [0.]]) #Baseline length = 54 cm
K2R2_K3 = multi_dot([K2, R2, inv(K3)])
K2R2C2 = multi_dot([K2, R2, C2])

def remap(p2, Zw, direction):
    if direction == 'L2R':
        p3 = np.dot(K3R3_K2, Zw*p2) - K3R3C3
        return [float(p3[0] / p3[2]), float(p3[1] / p3[2])]
    elif direction == 'R2L':
        p3 = np.dot(K2R2_K3, Zw*p2) - K2R2C2
        return [float(p3[0] / p3[2]), float(p3[1] / p3[2])]

def reconstruct(depth_map, src, direction):
    #Bind original pixels with depth maps
    depths = []
    colors = []
    for x in range(src.shape[1]): #width
        for y in range(src.shape[0]):
            depths.append([x,y,depth_map[y][x]])
            colors.append(src[y][x])

    #remap pixel positions to complementary view (2-D)
    px, py = [], []
    for [x,y,z] in depths:     
        p3_x, p3_y = remap(np.array([[x],[y],[1]]), z, direction)
        px.append(int(p3_x))
        py.append(int(p3_y))

    #map original color of source image pixels
    colors = np.dot(np.array(colors), 255).astype(int)
    canvas = np.zeros_like(src, dtype=np.uint8) #(src*255).astype(np.uint8)*0
    for i in range(len(px)):
        if(0 <= px[i] < src.shape[1] and 0 <= py[i] < src.shape[0]):
            canvas[py[i]][px[i]][0] = colors[i][0]
            canvas[py[i]][px[i]][1] = colors[i][1]
            canvas[py[i]][px[i]][2] = colors[i][2]
        
    return canvas
        
#pixel-wise alignment loss function
def recon_loss(tar, syn):
    loss = 0
    pt = 0
    tar = (np.dot(tar,255)).astype(int)
    syn = syn.astype(int)
    for y in range(tar.shape[0]):
        for x in range(tar.shape[1]): #~460k points
            if sum(syn[y][x]) > 1:
                pt += 1
                loss += ( (tar[y][x][0]-syn[y][x][0])**2
                         +(tar[y][x][1]-syn[y][x][1])**2
                         +(tar[y][x][2]-syn[y][x][2])**2)

    return round(loss / pt, 4)

##############################################################################
def crop_lidar(image_path, velo_path, v2c_filepath, c2c_filepath, frame_id, save_path=None):
    v_fov, h_fov = (-24.9, 2.0), (-90, 90)
    velo = Kitti(frame=frame_id, velo_path=velo_path)
    frame = velo.velo_file

    res = Kitti_util(frame=frame_id, camera_path=image_path, velo_path=velo_path, \
                v2c_path=v2c_filepath, c2c_path=c2c_filepath)

    img, points, depths = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)

    dots = np.zeros((img.shape[0], img.shape[1]))
    for i in range(points.shape[1]):
        x_index = np.int32(points[0][i])
        y_index = np.int32(points[1][i])
        if (0 <= y_index < img.shape[0] and 0 <= x_index < img.shape[1]): #crop range within view
            dots[y_index, x_index] = depths[i]
            
    return np.asarray(dots)

def gt_loss(ref, dots):
    loss = 0
    pt = 0
    for i in range(dots.shape[0]):
        for j in range(dots.shape[1]):
            if dots[i][j]:
                pt += 1
                gs = int(abs(ref[i][j] - dots[i][j]))
                loss += (ref[i][j] - dots[i][j])**2
                
    return round(loss / pt, 4)
            


