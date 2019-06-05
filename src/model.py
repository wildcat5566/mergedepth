import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models

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
        self.skip_outputs = []

        resnet50 = models.resnet50(pretrained=True)
        modules=list(resnet50.children())[:-2] #remove final fc layer & avgpool layer
        self.encoder=nn.Sequential(*modules)
        
        self.encoder[4].register_forward_hook(self.hook)
        self.encoder[5].register_forward_hook(self.hook)
        self.encoder[6].register_forward_hook(self.hook)

        #Decoder
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_drop = nn.Dropout2d(p=0.25) ######
        self.batchnorm2 = nn.BatchNorm2d(1024)
        
        self.upconv1 = Upconv(1024,1024)
        self.up1 = Upproject(1024,512)
        
        self.upconv2 = Upconv(512,512)
        self.upconv_skip2 = Upconv(1024,512)
        self.up2 = Upproject(512,256)

        self.upconv3 = Upconv(256,256)
        self.upconv_skip3 = Upconv(512,256)
        self.up3 = Upproject(256,128)

        self.upconv4 = Upconv(128,128)
        self.upconv_skip4 = Upconv(256,128)
        self.up4 = Upproject(128,64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_drop = nn.Dropout2d(p=0.25) ######
        
    def hook(self, module, input, output):
        self.skip_outputs.append(output)

    def forward(self, x):
        self.skip_outputs = [] #clear every time

        x = self.encoder(x)
        x = F.relu(self.batchnorm2(self.conv2_drop(self.conv2(x)))) #torch.Size([n, 1024, 12, 40])

        x = self.upconv1(x)
        x = self.up1(x) #torch.Size([n, 512, 24, 80])

        x = self.upconv2(x)
        s = self.upconv_skip2(self.skip_outputs[2])
        x = self.up2(x + s) #torch.Size([n, 256, 48, 160])

        x = self.upconv3(x)
        s = self.upconv_skip3(self.skip_outputs[1])
        x = self.up3(x + s) #torch.Size([n, 128, 96, 320])

        x = self.upconv4(x)
        s = self.upconv_skip4(self.skip_outputs[0])
        x = self.up4(x + s) #torch.Size([n, 64, 192, 640])

        x = F.sigmoid(self.conv3_drop(self.conv3(x))) #torch.Size([n, 1, 192, 640]) #-3~+2 approximately without sigmoid
        return x
    