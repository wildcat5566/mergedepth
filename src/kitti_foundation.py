""" 2017.07.19
made by changsub Bae
github.com/windowsub0406
"""
import numpy as np
import glob
import cv2

class Kitti:

    """
    frame : specific frame number or 'all' for whole dataset. default = 'all'
    velo_path : velodyne bin file path. default = None
    camera_path : left-camera image file path. default = None
    img_type : image type info 'gray' or 'color'. default = 'gray'
    v2c_path : Velodyne to Camera calibration info file path. default = None
    c2c_path : camera to Camera calibration info file path. default = None
    xml_path : XML file having tracklet info
    """
    def __init__(self, frame='all', velo_path=None, camera_path=None, \
                 img_type='gray', v2c_path=None, c2c_path=None, xml_path=None):
        self.__frame_type = frame
        self.__img_type = img_type
        self.__num_frames = None
        self.__cur_frame = None

        if velo_path is not None:
            self.__velo_path = velo_path
            self.__velo_file = self.__load_from_bin()
        else:
            self.__velo_path, self.__velo_file = None, None

        if camera_path is not None:
            self.__camera_path = camera_path
            self.__camera_file = self.__load_image()
        else:
            self.__camera_path, self.__camera_file = None, None
        if v2c_path is not None:
            self.__v2c_path = v2c_path
            self.__v2c_file = self.__load_velo2cam()
        else:
            self.__v2c_path, self.__v2c_file = None, None
        if c2c_path is not None:
            self.__c2c_path = c2c_path
            self.__c2c_file = self.__load_cam2cam()
        else:
            self.__c2c_path, self.__c2c_file = None, None
        if xml_path is not None:
            self.__xml_path = xml_path
            self.__tracklet_box, self.__tracklet_type = self.__load_tracklet()
        else:
            self.__xml_path = None
            self.__tracklet_box, self.__tracklet_type = None, None

    @property
    def frame_type(self):
        return self.__frame_type

    @property
    def image_type(self):
        return self.__img_type

    @property
    def num_frame(self):
        return self.__num_frames

    @property
    def cur_frame(self):
        return self.__cur_frame

    @property
    def img_size(self):
        return self.__img_size

    @property
    def velo_file(self):
        return self.__velo_file

    @property
    def velo_d_file(self):
        x = self.__velo_file[:, 0]
        y = self.__velo_file[:, 1]
        z = self.__velo_file[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.hstack((self.__velo_file, d[:, None]))

    @property
    def camera_file(self):
        return self.__camera_file

    @property
    def v2c_file(self):
        return self.__v2c_file

    @property
    def c2c_file(self):
        return self.__c2c_file

    @property
    def tracklet_info(self):
        return self.__tracklet_box, self.__tracklet_type

    def __get_velo(self, files):
        """ Convert bin to numpy array for whole dataset"""

        for i in files.keys():
            points = np.fromfile(files[i], dtype=np.float32).reshape(-1, 4)
            self.__velo_file = points[:, :3]
            self.__cur_frame = i
            yield self.__velo_file

    def __get_velo_frame(self, files):
        """ Convert bin to numpy array for one frame """
        points = np.fromfile(files[self.__frame_type], dtype=np.float32).reshape(-1, 4)
        return points[:, :3]

    def __get_camera(self, files):
        """ Return image for whole dataset """

        for i in files.keys():
            self.__camera_file = files[i]
            self.__cur_frame = i
            frame = cv2.imread(self.__camera_file)
            if i == 0:
                self.__img_size = frame.shape
            yield frame

    def __get_camera_frame(self, files):
        """ Return image for one frame """
        frame = cv2.imread(files[self.__frame_type])
        self.__img_size = frame.shape
        return frame

    def __load_from_bin(self):
        """ Return numpy array including velodyne's all 3d x,y,z point cloud """

        velo_bins = glob.glob(self.__velo_path + '/*.bin')
        velo_bins.sort()
        self.__num_frames = len(velo_bins)
        velo_files = {i: velo_bins[i] for i in range(len(velo_bins))}

        if self.__frame_type in velo_files:
            velo_xyz = self.__get_velo_frame(velo_files)
        else:
            velo_xyz = self.__get_velo(velo_files)

        return velo_xyz

    def __load_image(self):
        """ Return camera image """

        image_path = glob.glob(self.__camera_path + '/*.png')
        image_path.sort()
        self.__num_frames = len(image_path)
        image_files = {i: image_path[i] for i in range(len(image_path))}

        if self.__frame_type in image_files:
            image = self.__get_camera_frame(image_files)
        else:
            image = self.__get_camera(image_files)

        return image

    def __load_velo2cam(self):
        """ load Velodyne to Camera calibration info file """
        with open(self.__v2c_path, "r") as f:
            file = f.readlines()
            return file

    def __load_cam2cam(self):
        """ load Camera to Camera calibration info file """
        with open(self.__c2c_path, "r") as f:
            file = f.readlines()
            return file

    def __del__(self):
        pass

class Kitti_util(Kitti):

    def __init__(self, frame='all', velo_path=None, camera_path=None, \
                 img_type='gray', v2c_path=None, c2c_path=None, xml_path=None):
        super(Kitti_util, self).__init__(frame, velo_path, camera_path, img_type, v2c_path, c2c_path, xml_path)
        self.__h_min, self.__h_max = -180, 180
        self.__v_min, self.__v_max = -24.9, 2.0
        self.__v_res, self.__h_res = 0.42, 0.35
        self.__x, self.__y, self.__z, self.__d = None, None, None, None
        self.__h_fov, self.__v_fov = None, None
        self.__x_range, self.__y_range, self.__z_range = None, None, None
        self.__get_sur_size, self.__get_top_size = None, None

    @property
    def surround_size(self):
        return self.__get_sur_size

    @property
    def topview_size(self):
        return self.__get_top_size

    def __calib_velo2cam(self):
        """
        get Rotation(R : 3x3), Translation(T : 3x1) matrix info
        using R,T matrix, we can convert velodyne coordinates to camera coordinates
        """
        if self.v2c_file is None:
            raise NameError("calib_velo_to_cam file isn't loaded.")

        for line in self.v2c_file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
                ######
                #T[0] += (-0.47310 - 0.05957)
                #T[1] += (0.005551 - 0.00029)
                #T[2] += (-0.0052509 - 0.002577)
                ######
        return R, T

    def __calib_cam2cam(self):
        """
        If your image is 'rectified image' :
            get only Projection(P : 3x4) matrix is enough
        but if your image is 'distorted image'(not rectified image) :
            you need undistortion step using distortion coefficients(5 : D)

        In this code, only P matrix info is used for rectified image
        """
        if self.c2c_file is None:
            raise NameError("calib_velo_to_cam file isn't loaded.")

        mode = '00' if self.image_type == 'gray' else '02'

        for line in self.c2c_file:
            (key, val) = line.split(':', 1)
            ######
            #mode = '03'
            ######
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
        return P_

    def __upload_points(self, points):
        self.__x = points[:, 0]
        self.__y = points[:, 1]
        self.__z = points[:, 2]
        self.__d = np.sqrt(self.__x ** 2 + self.__y ** 2 + self.__z ** 2)

    def __point_matrix(self, points):
        """ extract points corresponding to FOV setting """

        # filter in range points based on fov, x,y,z range setting
        self.__points_filter(points)

        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((self.__x[:, None], self.__y[:, None], self.__z[:, None]))
        xyz_ = xyz_.T

        # stack (1,n) arrays filled with the number 1
        one_mat = np.full((1, xyz_.shape[1]), 1)
        xyz_ = np.concatenate((xyz_, one_mat), axis=0)

        # need dist info for points color
        color = self.__normalize_data(self.__d, min=1, max=70, scale=120, clip=True)

        return xyz_, color

    def __normalize_data(self, val, min, max, scale, depth=False, clip=False):
        """ Return normalized data """
        if clip:
            # limit the values in an array
            np.clip(val, min, max, out=val)
        if depth:
            """
            print 'normalized depth value'
            normalize values to (0 - scale) & close distance value has high value. (similar to stereo vision's disparity map)
            """
            return (((max - val) / (max - min)) * scale).astype(np.uint8)
        else:
            """
            print 'normalized value'
            normalize values to (0 - scale) & close distance value has low value.
            """
            return (((val - min) / (max - min)) * scale).astype(np.uint8)

    def __hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """

        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                  np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                  np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def __3d_in_range(self, points):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return points[np.logical_and.reduce((self.__x > self.__x_range[0], self.__x < self.__x_range[1], \
                                             self.__y > self.__y_range[0], self.__y < self.__y_range[1], \
                                             self.__z > self.__z_range[0], self.__z < self.__z_range[1]))]

    def __points_filter(self, points):
        """
        filter points based on h,v FOV and x,y,z distance range.
        x,y,z direction is based on velodyne coordinates
        1. azimuth & elevation angle limit check
        2. x,y,z distance limit
        """

        # upload current points
        self.__upload_points(points)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if self.__h_fov is not None and self.__v_fov is not None:
            if self.__h_fov[1] == self.__h_max and self.__h_fov[0] == self.__h_min and \
                            self.__v_fov[1] == self.__v_max and self.__v_fov[0] == self.__v_min:
                pass
            elif self.__h_fov[1] == self.__h_max and self.__h_fov[0] == self.__h_min:
                con = self.__hv_in_range(d, z, self.__v_fov, fov_type='v')
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
            elif self.__v_fov[1] == self.__v_max and self.__v_fov[0] == self.__v_min:
                con = self.__hv_in_range(x, y, self.__h_fov, fov_type='h')
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
            else:
                h_points = self.__hv_in_range(x, y, self.__h_fov, fov_type='h')
                v_points = self.__hv_in_range(d, z, self.__v_fov, fov_type='v')
                con = np.logical_and(h_points, v_points)
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
        else:
            pass

        if self.__x_range is None and self.__y_range is None and self.__z_range is None:
            pass
        elif self.__x_range is not None and self.__y_range is not None and self.__z_range is not None:
            # extract in-range points
            temp_x, temp_y = self.__3d_in_range(self.__x), self.__3d_in_range(self.__y)
            temp_z, temp_d = self.__3d_in_range(self.__z), self.__3d_in_range(self.__d)
            self.__x, self.__y, self.__z, self.__d = temp_x, temp_y, temp_z, temp_d
        else:
            raise ValueError("Please input x,y,z's min, max range(m) based on velodyne coordinates. ")

    def __velo_2_img_projection(self, points):
        """ convert velodyne coordinates to camera image coordinates """

        # rough velodyne azimuth range corresponding to camera horizontal fov
        if self.__h_fov is None:
            self.__h_fov = (-50, 50)
        if self.__h_fov[0] < -50:
            self.__h_fov = (-50,) + self.__h_fov[1:]
        if self.__h_fov[1] > 50:
            self.__h_fov = self.__h_fov[:1] + (50,)

        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        R_vc, T_vc = self.__calib_velo2cam()

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        P_ = self.__calib_cam2cam()

        """
        xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
        c_    - color value(HSV's Hue vaule) corresponding to distance(m)

                 [x_1 , x_2 , .. ]
        xyz_v =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
                 [ 1  ,  1  , .. ]
        """
        xyz_v, c_ = self.__point_matrix(points)

        """
        RT_ - rotation matrix & translation matrix
            ( velodyne coordinates -> camera coordinates )

                [r_11 , r_12 , r_13 , t_x ]
        RT_  =  [r_21 , r_22 , r_23 , t_y ]
                [r_31 , r_32 , r_33 , t_z ]
        """
        RT_ = np.concatenate((R_vc, T_vc), axis=1)

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

        """
        xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                 [x_1 , x_2 , .. ]
        xyz_c =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
        """
        xyz_c = np.delete(xyz_v, 3, axis=0)

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

        """
        xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
        ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                 [s_1*x_1 , s_2*x_2 , .. ]
        xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
                 [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
        """
        xy_i = xyz_c[::] / xyz_c[::][2]
        ans = np.delete(xy_i, 2, axis=0)

        return ans, c_

    def velo_projection(self, h_fov=None, v_fov=None, x_range=None, y_range=None, z_range=None):
        """ print velodyne 3D points corresponding to camera 2D image """

        self.__v_fov, self.__h_fov = v_fov, h_fov
        self.__x_range, self.__y_range, self.__z_range = x_range, y_range, z_range
        velo_gen, cam_gen = self.velo_file, self.camera_file

        if velo_gen is None:
            raise ValueError("Velo data is not included in this class")
        if cam_gen is None:
            raise ValueError("Cam data is not included in this class")
        for frame, points in zip(cam_gen, velo_gen):
            res, c_ = self.__velo_2_img_projection(points)
            yield [frame, res, c_]

    def velo_projection_frame(self, h_fov=None, v_fov=None, x_range=None, y_range=None, z_range=None):
        """ print velodyne 3D points corresponding to camera 2D image """

        self.__v_fov, self.__h_fov = v_fov, h_fov
        self.__x_range, self.__y_range, self.__z_range = x_range, y_range, z_range
        velo_gen, cam_gen = self.velo_file, self.camera_file

        if velo_gen is None:
            raise ValueError("Velo data is not included in this class")
        if cam_gen is None:
            raise ValueError("Cam data is not included in this class")
        res, c_ = self.__velo_2_img_projection(velo_gen)
        return cam_gen, res, c_

    def __del__(self):
        pass


