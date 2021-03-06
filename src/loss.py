import numpy as np
from numpy.linalg import multi_dot, inv
import torch
import cv2

def gt_loss(depth_map, dots, gt_augmentation=True, depth_max=120, threshold=0.01):
    """
    {Inputs}
    depth_map: Depth map prediction generated by NN.
    dims     : 1*h*w
    val&type : [0, 1], dtype=torch.FloatTensor
    
    dots     : Ground truth depths
    dims     : H*W
    val&type : [0, depth_max], dtype=numpy.ndarray
    
    {Outputs}
    loss     : Point-wise loss between predicted and ground truth depths, where ground truth data is available.
    dims     : 1
    val&type : [0, 1], dtype=torch.FloatTensor
    """
    if gt_augmentation:
        dots = random_sample_drop(dots)
        
    dots = torch.tensor(cv2.resize(dots, (depth_map.shape[2], depth_map.shape[1])) / depth_max).cuda()
    depth_map = depth_map.squeeze()

    dev = (depth_map - dots) * ((dots!=0).float()) #double
    loss = torch.sum(torch.mul(dev, dev)) / (torch.sum((dots!=0).float()))
    accuracy = 1 - (torch.sum(dev>threshold).float() / torch.sum(dots!=0).float())

    return loss, accuracy

def random_sample_drop(array, prob=.2, kernel_size=3):
    padding = int(kernel_size / 2)
    for x in range(padding, array.shape[1] - padding, kernel_size):
        for y in range(padding, array.shape[0] - padding, kernel_size):
            #locate kernel center = array[y][x]
            for i in range(x-padding, x+padding+1):
                for j in range(y-padding, y+padding+1):
                    p = np.random.random()
                    if p < prob:
                        array[j][i] = 0
                        
    return array
            
class Reconstruction():
    def __init__(self, date, scaling=1):
        self.date = date
        self.scaling = scaling
        self.K2R2_K3 = None
        self.K3R3C3 = None
        self.K3R3_K2 = None
        self.K2R2C2 = None
        
        self._get_transformations()
        
    def _get_transformations(self):
        if self.date=='2011_09_26':
            self.K2 = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
                               [0.000000e+00, 9.569251e+02, 2.241806e+02],
                               [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                               [0.000000e+00, 9.019653e+02, 2.242509e+02],
                               [0.000000e+00, 0.000000e+00,1.000000e+00]])
            self.R2 = np.array([[ 9.999758e-01, -5.267463e-03, -4.552439e-03],
                                [ 5.251945e-03,  9.999804e-01, -3.413835e-03],
                                [ 4.570332e-03,  3.389843e-03,  9.999838e-01]])
            self.t2 = np.array([[ 5.956621e-02],
                                [ 2.900141e-04],
                                [ 2.577209e-03]])
            self.R3 = np.array([[ 9.995599e-01,  1.699522e-02, -2.431313e-02],
                                [-1.704422e-02,  9.998531e-01, -1.809756e-03],
                                [ 2.427880e-02,  2.223358e-03,  9.997028e-01]])
            self.t3 = np.array([[-4.731050e-01], 
                                [ 5.551470e-03],
                                [-5.250882e-03]])
  
        elif self.date=='2011_09_28':
            self.K2 = np.array([[9.569475e+02, 0.000000e+00, 6.939767e+02],
                                [0.000000e+00, 9.522352e+02, 2.386081e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.011007e+02, 0.000000e+00, 6.982947e+02],
                                [0.000000e+00, 8.970639e+02, 2.377447e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999838e-01, -5.012736e-03, -2.710741e-03],
                                [ 5.002007e-03,  9.999797e-01, -3.950381e-03],
                                [ 2.730489e-03,  3.936758e-03,  9.999885e-01]])
            self.t2 = np.array([[ 5.989688e-02],
                                [-1.367835e-03],
                                [ 4.637624e-03]])
            self.R3 = np.array([[ 9.995054e-01,  1.665288e-02, -2.667675e-02],
                                [-1.671777e-02,  9.998578e-01, -2.211228e-03],
                                [ 2.663614e-02,  2.656110e-03,  9.996417e-01]])
            self.t3 = np.array([[-4.756270e-01],
                                [ 5.296617e-03],
                                [-5.437198e-03]])
        
        elif self.date=='2011_09_29':
            self.K2 = np.array([[9.607501e+02, 0.000000e+00, 6.944288e+02],
                                [0.000000e+00, 9.570051e+02, 2.363374e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.047872e+02, 0.000000e+00, 6.946163e+02], 
                                [0.000000e+00, 9.017079e+02, 2.353088e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999807e-01, -5.053665e-03, -3.619905e-03],
                                [ 5.036396e-03,  9.999760e-01, -4.764072e-03],
                                [ 3.643894e-03,  4.745749e-03,  9.999821e-01]])
            self.t2 = np.array([[ 5.948968e-02], 
                                [-8.603063e-04], 
                                [ 2.662728e-03]])
            self.R3 = np.array([[ 9.995851e-01,  1.666283e-02, -2.349366e-02],
                                [-1.674297e-02,  9.998546e-01, -3.218496e-03],
                                [ 2.343662e-02,  3.610514e-03,  9.997188e-01]])
            self.t3 = np.array([[-4.732167e-01],
                                [ 5.830806e-03],
                                [-4.405247e-03]])
            
        elif self.date=='2011_09_30':
            self.K2 = np.array([[9.591977e+02, 0.000000e+00, 6.944383e+02],
                                [0.000000e+00, 9.529324e+02, 2.416793e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.035972e+02, 0.000000e+00, 6.979803e+02],
                                [0.000000e+00, 8.979356e+02, 2.392935e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999805e-01, -4.971067e-03, -3.793081e-03],
                                [ 4.954076e-03,  9.999777e-01, -4.475856e-03],
                                [ 3.815246e-03,  4.456977e-03,  9.999828e-01]])
            self.t2 = np.array([[ 6.030222e-02],
                                [-1.293125e-03],
                                [ 5.900421e-03]])
            self.R3 = np.array([[ 9.994995e-01,  1.667420e-02, -2.688514e-02],
                                [-1.673122e-02,  9.998582e-01, -1.897204e-03],
                                [ 2.684969e-02,  2.346075e-03,  9.996367e-01]])
            self.t3 = np.array([[-4.747879e-01],
                                [ 5.631988e-03],
                                [-5.233709e-03]])
            
        elif self.date=='2011_10_03':
            self.K2 = np.array([[9.601149e+02, 0.000000e+00, 6.947923e+02],
                                [0.000000e+00, 9.548911e+02, 2.403547e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.049931e+02, 0.000000e+00, 6.957698e+02],
                                [0.000000e+00, 9.004945e+02, 2.389820e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999788e-01, -5.008404e-03, -4.151018e-03],
                                [ 4.990516e-03,  9.999783e-01, -4.308488e-03],
                                [ 4.172506e-03,  4.287682e-03,  9.999821e-01]])
            self.t2 = np.array([[ 5.954406e-02], 
                                [-7.675338e-04], 
                                [ 3.582565e-03]])
            self.R3 = np.array([[ 9.995578e-01,  1.656369e-02, -2.469315e-02],
                                [-1.663353e-02,  9.998582e-01, -2.625576e-03],
                                [ 2.464616e-02,  3.035149e-03,  9.996916e-01]])
            self.t3 = np.array([[-4.738786e-01],
                                [ 5.991982e-03],
                                [-3.215069e-03]])

        #Dollhouse scaling
        self.K3 = self.K3 * self.scaling
        self.K2 = self.K2 * self.scaling
        self.K3[2][2] = 1
        self.K2[2][2] = 1
        self.t2 = self.t2 * self.scaling
        self.t3 = self.t3 * self.scaling
        
        #L2R
        dR = np.eye(3) #No rotation
        self.K3R3_K2 = multi_dot([self.K3, dR, inv(self.K2)])
        self.K3R3C3 = multi_dot([self.K3, dR, (self.t2 - self.t3)])

        #R2L
        self.K2R2_K3 = multi_dot([self.K2, dR, inv(self.K3)])
        self.K2R2C2 = multi_dot([self.K2, dR, (self.t3 - self.t2)])
        
        #cuda
        self.K3R3_K2 = torch.tensor(self.K3R3_K2).float().cuda()
        self.K2R2_K3 = torch.tensor(self.K2R2_K3).float().cuda()
        self.K3R3C3 = torch.tensor(self.K3R3C3).float().cuda()
        self.K2R2C2 = torch.tensor(self.K2R2C2).float().cuda()
        
    def _remap(self, p2, Zw, direction):
        """
        {Inputs}
        p2       : Pixel location of interested point in source image (Homogeneous).
        dims     : 3*1
        val&type : [[0,w], [0,h], [1]], dtype=torch.floatTensor
        
        Zw       : Predicted depth of interested point in source image.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        
        direction: Flag, indicating transformation direction between left & right views.
        dims     : None
        val&type : 'L2R' or 'R2L', dtype=string
        
        {Outputs}
        [x, y]   : Mapped location of interested point to target view.
        dims     : 1*2
        val&type : [[0,w], [0,h]]
        """
        p2 = p2.float().cuda()
        m2 = (Zw*p2)
        
        if direction == 'L2R':
            m1 = self.K3R3_K2
            p3 = torch.mm(m1, m2) - self.K3R3C3
            return (p3[0]/p3[2]).long(), (p3[1]/p3[2]).long()
            
        elif direction == 'R2L':
            m1 = torch.tensor(self.K2R2_K3).float().cuda()
            p3 = torch.mm(m1, m2) - torch.tensor(self.K2R2C2).float().cuda()
            return (p3[0]/p3[2]).long(), (p3[1]/p3[2]).long()
    
    def _reconstruct(self, depth_map, src_image, direction, depth_max=120.0):
        """
        {Inputs}
        depth_map: Depth map prediction generated by NN.
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        src_image: Reconstruction source view.
        dims     : 3*H*W --> 3*h*w
        val&type : [0, 1], dtype=torch.floatTensor
        
        direction: Flag, indicating transformation direction between left & right views.
        dims     : None
        val&type : 'L2R' or 'R2L', dtype=string
        
        depth_max: Max depth (of ground truth data), for scaling
        dims     : 1
        val&type : 120.0 * scaling factor
        
        {outputs}
        canvas   : Reconstructed complementary view of src_image.
        dims     : 3*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        """
        
        #Only resize source image
        src_np = np.transpose(src_image.cpu().numpy(), (1,2,0))
        src_np = cv2.resize(src_np, (depth_map.shape[2], depth_map.shape[1]))#3,96,320
        src_resize = torch.tensor(np.transpose(src_np, (2,0,1))).float().cuda()

        canvas = torch.stack([depth_map.squeeze(), depth_map.squeeze(), depth_map.squeeze()])#*0
        for x in range(src_resize.shape[2]): #width
            for y in range(src_resize.shape[1]):
                ptz = torch.tensor(depth_map[0][y][x])
                scale = torch.tensor(depth_max*self.scaling).cuda()
                p3_x, p3_y = self._remap(p2=torch.tensor([[x], [y], [1]]), 
                                         Zw=torch.tensor(ptz * scale),
                                         direction=direction)

                xloc, yloc = int(p3_x.cpu()), int(p3_y.cpu())
                if(0 <= xloc < src_resize.shape[2] and 0 <= yloc < src_resize.shape[1]):
                    canvas[0][yloc][xloc] = src_resize[0][y][x]
                    canvas[1][yloc][xloc] = src_resize[1][y][x]
                    canvas[2][yloc][xloc] = src_resize[2][y][x]

        return canvas

    def _recon_loss(self, tar, syn):
        """
        {Inputs}
        tar      : Reconstruction target view.
        dims     : 3*H*W --> 3*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        syn      : Reconstructed complementary view of src_image.
        dims     : 3*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        
        {Outputs}
        loss     : Alignment loss between original & synthesized target view.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        
        """
        tar_np = np.transpose(tar.cpu().numpy(), (1,2,0))
        tar_np = cv2.resize(tar_np, (syn.shape[2], syn.shape[1]))
        tar = torch.tensor(np.transpose(tar_np, (2,0,1))).float().cuda()

        dev = (tar - syn)
        loss = torch.sum(dev*dev) / (tar.shape[2]*tar.shape[1]*3)
        
        return loss
    
    def _recon_loss_weighted(self, tar, syn):
        """
        {Inputs}
        tar      : Reconstruction target view.
        dims     : 3*H*W --> 3*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        syn      : Reconstructed complementary view of src_image.
        dims     : 3*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        
        {Outputs}
        loss     : Alignment loss between original & synthesized target view.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        
        """
        
        tar_np = np.transpose(tar.cpu().numpy(), (1,2,0))
        tar_np = cv2.resize(tar_np, (syn.shape[2], syn.shape[1]))
        tar = torch.tensor(np.transpose(tar_np, (2,0,1))).float().cuda()

        #[2.718, 2.690, 2.662, ...1.032, 1.021, 1.010] -1
        m = np.transpose([[np.exp(1-y/tar.shape[1])-1 for y in range(tar.shape[1])] for x in range(tar.shape[2])])
        mask = torch.stack([torch.tensor(m), torch.tensor(m), torch.tensor(m)]).float().cuda()

        dev = (tar - syn) #weighted losses
        loss = torch.sum(mask*(dev*dev)) / (tar.shape[2]*tar.shape[1]*3)
        
        return loss
    
    #Public function
    def compute_loss(self, depth_map, src_image, tar_image, direction, weighting=False, return_img=False):
        """
        h*w =  96*320
        H*W = 192*640
        
        {Inputs}
        depth_map: Depth map prediction generated by NN.
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        src_image: Reconstruction source view.
        dims     : 3*H*W
        val&type : [0, 1], dtype=torch.FloatTensor
        
        tar_image: Reconstruction target view.
        dims     : 3*H*W
        val&type : [0, 1], dtype=torch.FloatTensor
        
        direction: Flag, indicating transformation direction between left & right views.
        dims     : None
        val&type : 'L2R' or 'R2L', dtype=string
        
        {Outputs}
        syn_image: Reconstructed complementary view of src_image.
        dims     : 3*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        
        loss     : Alignment loss between original & synthesized target view.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        """

        syn_image = self._reconstruct(depth_map=depth_map, src_image=src_image, direction=direction)
        if weighting==False:
            loss = self._recon_loss(tar=tar_image, syn=syn_image)
        else:
            loss = self._recon_loss_weighted(tar=tar_image, syn=syn_image)
            
        if return_img:
            return loss, syn_image
        else:
            return loss

##########
class Consistency():
    def __init__(self, date, scaling=1, device=0):
        self.date = date
        self.scaling = scaling
        self.device = device
        
        self.K2R2_K3 = None
        self.K3R3C3 = None
        self.K3R3_K2 = None
        self.K2R2C2 = None
        self._get_transformations()
        
        
    def _get_transformations(self):
        if self.date=='2011_09_26':
            self.K2 = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
                               [0.000000e+00, 9.569251e+02, 2.241806e+02],
                               [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                               [0.000000e+00, 9.019653e+02, 2.242509e+02],
                               [0.000000e+00, 0.000000e+00,1.000000e+00]])
            self.R2 = np.array([[ 9.999758e-01, -5.267463e-03, -4.552439e-03],
                                [ 5.251945e-03,  9.999804e-01, -3.413835e-03],
                                [ 4.570332e-03,  3.389843e-03,  9.999838e-01]])
            self.t2 = np.array([[ 5.956621e-02],
                                [ 2.900141e-04],
                                [ 2.577209e-03]])
            self.R3 = np.array([[ 9.995599e-01,  1.699522e-02, -2.431313e-02],
                                [-1.704422e-02,  9.998531e-01, -1.809756e-03],
                                [ 2.427880e-02,  2.223358e-03,  9.997028e-01]])
            self.t3 = np.array([[-4.731050e-01], 
                                [ 5.551470e-03],
                                [-5.250882e-03]])
  
        elif self.date=='2011_09_28':
            self.K2 = np.array([[9.569475e+02, 0.000000e+00, 6.939767e+02],
                                [0.000000e+00, 9.522352e+02, 2.386081e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.011007e+02, 0.000000e+00, 6.982947e+02],
                                [0.000000e+00, 8.970639e+02, 2.377447e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999838e-01, -5.012736e-03, -2.710741e-03],
                                [ 5.002007e-03,  9.999797e-01, -3.950381e-03],
                                [ 2.730489e-03,  3.936758e-03,  9.999885e-01]])
            self.t2 = np.array([[ 5.989688e-02],
                                [-1.367835e-03],
                                [ 4.637624e-03]])
            self.R3 = np.array([[ 9.995054e-01,  1.665288e-02, -2.667675e-02],
                                [-1.671777e-02,  9.998578e-01, -2.211228e-03],
                                [ 2.663614e-02,  2.656110e-03,  9.996417e-01]])
            self.t3 = np.array([[-4.756270e-01],
                                [ 5.296617e-03],
                                [-5.437198e-03]])
        
        elif self.date=='2011_09_29':
            self.K2 = np.array([[9.607501e+02, 0.000000e+00, 6.944288e+02],
                                [0.000000e+00, 9.570051e+02, 2.363374e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.047872e+02, 0.000000e+00, 6.946163e+02], 
                                [0.000000e+00, 9.017079e+02, 2.353088e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999807e-01, -5.053665e-03, -3.619905e-03],
                                [ 5.036396e-03,  9.999760e-01, -4.764072e-03],
                                [ 3.643894e-03,  4.745749e-03,  9.999821e-01]])
            self.t2 = np.array([[ 5.948968e-02], 
                                [-8.603063e-04], 
                                [ 2.662728e-03]])
            self.R3 = np.array([[ 9.995851e-01,  1.666283e-02, -2.349366e-02],
                                [-1.674297e-02,  9.998546e-01, -3.218496e-03],
                                [ 2.343662e-02,  3.610514e-03,  9.997188e-01]])
            self.t3 = np.array([[-4.732167e-01],
                                [ 5.830806e-03],
                                [-4.405247e-03]])
            
        elif self.date=='2011_09_30':
            self.K2 = np.array([[9.591977e+02, 0.000000e+00, 6.944383e+02],
                                [0.000000e+00, 9.529324e+02, 2.416793e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.035972e+02, 0.000000e+00, 6.979803e+02],
                                [0.000000e+00, 8.979356e+02, 2.392935e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999805e-01, -4.971067e-03, -3.793081e-03],
                                [ 4.954076e-03,  9.999777e-01, -4.475856e-03],
                                [ 3.815246e-03,  4.456977e-03,  9.999828e-01]])
            self.t2 = np.array([[ 6.030222e-02],
                                [-1.293125e-03],
                                [ 5.900421e-03]])
            self.R3 = np.array([[ 9.994995e-01,  1.667420e-02, -2.688514e-02],
                                [-1.673122e-02,  9.998582e-01, -1.897204e-03],
                                [ 2.684969e-02,  2.346075e-03,  9.996367e-01]])
            self.t3 = np.array([[-4.747879e-01],
                                [ 5.631988e-03],
                                [-5.233709e-03]])
            
        elif self.date=='2011_10_03':
            self.K2 = np.array([[9.601149e+02, 0.000000e+00, 6.947923e+02],
                                [0.000000e+00, 9.548911e+02, 2.403547e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.K3 = np.array([[9.049931e+02, 0.000000e+00, 6.957698e+02],
                                [0.000000e+00, 9.004945e+02, 2.389820e+02],
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
            self.R2 = np.array([[ 9.999788e-01, -5.008404e-03, -4.151018e-03],
                                [ 4.990516e-03,  9.999783e-01, -4.308488e-03],
                                [ 4.172506e-03,  4.287682e-03,  9.999821e-01]])
            self.t2 = np.array([[ 5.954406e-02], 
                                [-7.675338e-04], 
                                [ 3.582565e-03]])
            self.R3 = np.array([[ 9.995578e-01,  1.656369e-02, -2.469315e-02],
                                [-1.663353e-02,  9.998582e-01, -2.625576e-03],
                                [ 2.464616e-02,  3.035149e-03,  9.996916e-01]])
            self.t3 = np.array([[-4.738786e-01],
                                [ 5.991982e-03],
                                [-3.215069e-03]])
            
        #Dollhouse scaling
        self.K3 = self.K3 * self.scaling
        self.K2 = self.K2 * self.scaling
        self.K3[2][2] = 1
        self.K2[2][2] = 1
        self.t2 = self.t2 * self.scaling
        self.t3 = self.t3 * self.scaling
        #L2R
        dR = np.eye(3) #No rotation
        self.K3R3_K2 = multi_dot([self.K3, dR, inv(self.K2)])
        self.K3R3C3 = multi_dot([self.K3, dR, (self.t2 - self.t3)])
        #R2L
        self.K2R2_K3 = multi_dot([self.K2, dR, inv(self.K3)])
        self.K2R2C2 = multi_dot([self.K2, dR, (self.t3 - self.t2)])
        #cuda
        self.K3R3_K2 = torch.tensor(self.K3R3_K2).float().cuda(self.device)
        self.K2R2_K3 = torch.tensor(self.K2R2_K3).float().cuda(self.device)
        self.K3R3C3 = torch.tensor(self.K3R3C3).float().cuda(self.device)
        self.K2R2C2 = torch.tensor(self.K2R2C2).float().cuda(self.device)

    def _remap(self, p2, Zw, direction):
        """
        {Inputs}
        p2       : Pixel location of interested point in source image (Homogeneous).
        dims     : 3*1
        val&type : [[0,w], [0,h], [1]], dtype=torch.floatTensor
        
        Zw       : Predicted depth of interested point in source image.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        
        direction: Flag, indicating transformation direction between left & right views.
        dims     : None
        val&type : 'L2R' or 'R2L', dtype=string
        
        {Outputs}
        [x, y]   : Mapped location of interested point to target view.
        dims     : 1*2
        val&type : [[0,w], [0,h]]
        """
        
        p2 = p2.float().cuda(self.device)
        m2 = (Zw.cuda(self.device)*p2)
        
        if direction == 'L2R':
            m1 = self.K3R3_K2
            p3 = torch.mm(m1, m2) - self.K3R3C3
            return (p3[0]/p3[2]).long(), (p3[1]/p3[2]).long()
            
        elif direction == 'R2L':
            m1 = torch.tensor(self.K2R2_K3).float().cuda(self.device)
            p3 = torch.mm(m1, m2) - torch.tensor(self.K2R2C2).float().cuda(self.device)
            return (p3[0]/p3[2]).long(), (p3[1]/p3[2]).long()
    
    def _reconstruct(self, depth_map, direction, depth_max=120.0):
        """
        {Inputs}
        depth_map: Depth map prediction generated by NN.
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor

        direction: Flag, indicating transformation direction between left & right view depths.
        dims     : None
        val&type : 'L2R' or 'R2L', dtype=string
        
        depth_max: Max depth (of ground truth data), for scaling
        dims     : 1
        val&type : 120.0 * scaling factor
        
        {outputs}
        canvas   : Reconstructed complementary view of depth_map.
        dims     : 1*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        """
        
        src = depth_map.clone()
        canvas = depth_map.clone()
        
        for x in range(depth_map.shape[2]): #width
            for y in range(depth_map.shape[1]):
                ptz = torch.tensor(depth_map[0][y][x]).float().cuda(self.device)
                scale = torch.tensor(depth_max*self.scaling).cuda(self.device)
                p3_x, p3_y = self._remap(p2=torch.tensor([[x], [y], [1]]), 
                                         Zw=torch.tensor(ptz * scale),
                                         direction=direction)

                xloc, yloc = int(p3_x.cpu()), int(p3_y.cpu())
                if(0 <= xloc < depth_map.shape[2] and 0 <= yloc < depth_map.shape[1]):
                    canvas[0][yloc][xloc] = src[0][y][x]

        return canvas
       
    def _recon_loss(self, tar, syn):
        """
        {Inputs}
        tar      : Reconstruction target view.
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        syn      : Reconstructed complementary view of depth_map.
        dims     : 1*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        
        {Outputs}
        loss     : Alignment loss between original & synthesized target depths.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        
        """

        dev = (tar.float() - syn.float())
        loss = torch.sum(dev*dev) / (tar.shape[2]*tar.shape[1])
        
        return loss
    
    def _recon_loss_weighted(self, tar, syn):
        """
        {Inputs}
        tar      : Reconstruction target view.
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        syn      : Reconstructed complementary view of depth_map.
        dims     : 1*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        
        {Outputs}
        loss     : Alignment loss between original & synthesized target depths.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        
        """

        m = np.transpose([[np.exp(1-y/tar.shape[1])-1 for y in range(tar.shape[1])] for x in range(tar.shape[2])])
        mask = torch.tensor(m).float().cuda(self.device).unsqueeze(0)

        dev = (tar.float() - syn.float()) #tar_cl
        loss = torch.sum(mask*(dev*dev)) / (tar.shape[2]*tar.shape[1])
        
        return loss

    def compute_loss(self, depth_map, tar_image, direction, weighting=False):
        """
        h*w =  96*320
        
        {Inputs}
        depth_map: Depth map prediction generated by NN. Also serving as "mapping source".
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        tar_image: Predicted depth map of other view. Serving as target view.
        dims     : 1*h*w
        val&type : [0, 1], dtype=torch.FloatTensor
        
        direction: Flag, indicating transformation direction between left & right views.
        dims     : None
        val&type : 'L2R' or 'R2L', dtype=string
        
        {Outputs}
        syn_image: Reconstructed complementary view of depth_map.
        dims     : 1*h*w 
        val&type : [0, 1], dtype=torch.floatTensor
        
        loss     : Alignment loss between original & synthesized target depths.
        dims     : 1
        val&type : [0, 1], dtype=torch.floatTensor
        """
        depth_map = depth_map.cuda(self.device)
        tar_image = tar_image.cuda(self.device)

        syn_image = self._reconstruct(depth_map=depth_map, direction=direction)
        if weighting==False:
            loss = self._recon_loss(tar=tar_image, syn=syn_image)
        else:
            loss = self._recon_loss_weighted(tar=tar_image, syn=syn_image)
            
        #back to main gpu
        loss = loss.cuda(0)
        return loss
    