import torch
from torch import nn, optim
import torch.nn.functional as F

class ResidualType2(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualType2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                               kernel_size=1, stride=stride, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, stride=stride, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out1 = F.relu(self.batchnorm1(self.conv1(x)))
        out1 = F.relu(self.batchnorm2(self.conv2(out1)))
        out1 = self.batchnorm3(self.conv3(out1))
        out2 = self.batchnorm4(self.conv4(x))

        out = F.relu(out1 + out2)
        
        return out

class ResidualType1(nn.Module):
    def __init__(self, channels): #in_channels = out_channels
        super(ResidualType1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = F.relu(self.batchnorm2(self.conv2(out)))
        out = self.batchnorm3(self.conv3(out))
        
        out = F.relu(x + out)
        
        return out
    
class Upproject(nn.Module):
    def __init__(self, in_channels, out_channels): #residual preserves map sizes, reduces channels
        super(Upproject, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=5, stride=1, padding=2, bias=False) #p=0
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False) #p=1
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=5, stride=1, padding=2, bias=False) #p=0
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out1 = F.relu(self.batchnorm1(self.conv1(x)))
        out1 = self.batchnorm2(self.conv2(out1))
        out2 = self.batchnorm3(self.conv3(x))
        
        out = F.relu(out1 + out2)
        return out

class Upconv(nn.Module): 
    def __init__(self, in_channels, out_channels): #only upsamples map sizes.
        super(Upconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                         kernel_size=2, stride=2, padding=0, bias=False)
        
    def forward(self, x):
        out = self.deconv(x)
        return out
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        #residual blocks 1-4
        self.res1 = ResidualType2(64, 256, stride=1)
        self.res2 = ResidualType1(256)
        self.res3 = ResidualType1(256)
        self.res4 = ResidualType2(256, 512, stride=2)
        
        #residual blocks 5-8
        self.res5 = ResidualType1(512)
        self.res6 = ResidualType1(512)
        self.res7 = ResidualType1(512)
        self.res8 = ResidualType2(512, 1024, stride=2)
        
        #residual blocks 9-14
        self.res9 = ResidualType1(1024)
        self.res10 = ResidualType1(1024)
        self.res11 = ResidualType1(1024)
        self.res12 = ResidualType1(1024)
        self.res13 = ResidualType1(1024)
        self.res14 = ResidualType2(1024, 2048, stride=2)
        
        #residual blocks 15-16
        self.res15 = ResidualType1(2048)
        self.res16 = ResidualType1(2048)
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.batchnorm2 = nn.BatchNorm2d(1024)
        
        #upproject
        self.upconv1 = Upconv(1024,1024)
        self.up1 = Upproject(1024,512)
        
        self.upconv2 = Upconv(512,512)
        self.upconv_r13 = Upconv(1024,512)
        self.up2 = Upproject(512,256)

        self.upconv3 = Upconv(256,256)
        self.upconv_r7 = Upconv(512,256)
        self.up3 = Upproject(256,128)

        self.upconv4 = Upconv(128,128)
        self.upconv_r3 = Upconv(256,128)
        self.up4 = Upproject(128,64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        #self.conv3.bias.data.fill_(.5)
        
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x))) #torch.Size([n, 64, 192, 640])
        x = self.maxpool1(x) #torch.Size([n, 64, 96, 320])
            
        x = self.res1(x) #torch.Size([n, 256, 96, 320])
        x = self.res2(x)
        x = self.res3(x)
        r3_out = x
        x = self.res4(x) #torch.Size([n, 512, 48, 160])
            
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        r7_out = x
        x = self.res8(x) #torch.Size([n, 1024, 24, 80])

        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        r13_out = x
        x = self.res14(x) #torch.Size([n, 2048, 12, 40])

        x = self.res15(x)
        x = self.res16(x)
        x = F.relu(self.batchnorm2(self.conv2(x))) #torch.Size([n, 1024, 12, 40])

        x = self.upconv1(x)
        x = self.up1(x) #torch.Size([n, 512, 24, 80])
        
        x = self.upconv2(x)
        s = self.upconv_r13(r13_out)
        x = self.up2(x + s) #torch.Size([n, 256, 48, 160])

        x = self.upconv3(x)
        s = self.upconv_r7(r7_out)
        x = self.up3(x + s) #torch.Size([n, 128, 96, 320])

        x = self.upconv4(x)
        s = self.upconv_r3(r3_out)
        x = self.up4(x + s) #torch.Size([n, 64, 192, 640])

        #x = self.conv3(x)
        x = F.sigmoid(self.conv3(x)) #torch.Size([n, 1, 192, 640])

        return x