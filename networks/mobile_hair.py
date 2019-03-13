"""
Implementation of "Real-time deep hair matting on mobile devices(2018)"
"""

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import cv2

def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class GreenBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GreenBlock,self).__init__()

        self.dconv = nn.Sequential(
            SeparableConv2d(in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.dconv(input)
        x = self.conv(x)

        return x

class YellowBlock(nn.Module):
    def __init__(self):
        super(YellowBlock,self).__init__()
    
    def forward(self, input):
        return F.interpolate(input, scale_factor=2)

class OrangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,stride=1,padding=0,dilation=1, bias=False):
        super(OrangeBlock,self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size),
            nn.ReLU()
        )
    
    def forward(self, input):
        return self.conv(input)
        

class MobileMattingFCN(nn.Module):
    # https://github.com/marvis/pytorch-mobilenet modified
    def __init__(self):
        super(MobileMattingFCN, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1), # skip1 1 
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1), # skip2 3 
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), # skip3 5
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1), # skip4 11
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),

            conv_dw(1024, 1024, 1), # added
            #nn.AvgPool2d(7),
        )

        self.upsample0 = YellowBlock()
        self.o0 = OrangeBlock(1024+512, 64)

        self.upsample1 = YellowBlock()
        self.o1 = OrangeBlock(64+256, 64)
        self.upsample2 = YellowBlock()
        self.o2 = OrangeBlock(64+128, 64)
        self.upsample3 = YellowBlock()
        self.o3 = OrangeBlock(64+64, 64)
        self.upsample4 = YellowBlock()
        self.o4 = OrangeBlock(64, 64)

        self.red = nn.Sequential(
            nn.Conv2d(64, 1, 1)
        )

        #self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        skips = []
        #x = self.model(x)
        
        for i, model in enumerate(self.model):
            x = model(x)
            if i in {1,3,5,11}:
                skips.append(x)
        
        x = self.upsample0(x)
        x = torch.cat((x, skips[-1]), dim=1)
        x = self.o0(x)

        x = self.upsample1(x)
        x = torch.cat((x, skips[-2]), dim=1)
        x = self.o1(x)

        x = self.upsample2(x)
        x = torch.cat((x, skips[-3]), dim=1)
        x = self.o2(x)

        x = self.upsample3(x)
        x = torch.cat((x, skips[-4]), dim=1)
        x = self.o3(x)
        x = self.upsample4(x)
        x = self.o4(x)

        #x = self.fc(x)
        return self.red(x)
    
    def load_pretrained_model(self):
        pass
        # hell baidu - https://github.com/marvis/pytorch-mobilenet

class HairMattingLoss(nn.modules.loss._Loss):
    def __init__(self, ratio_of_Gradient=0.0, add_gradient=False):
        super(HairMattingLoss, self).__init__()
        self.ratio_of_gradient = ratio_of_Gradient
        self.add_gradient = add_gradient
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, true, image):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss2 = None
        if self.ratio_of_gradient > 0:
            sobel_kernel_x = torch.Tensor(
                        [[1.0, 0.0, -1.0],
                        [2.0, 0.0, -2.0],
                        [1.0, 0.0, -1.0]]).to(device)
            sobel_kernel_x = sobel_kernel_x.view((1,1,3,3))

            I_x = F.conv2d(image, sobel_kernel_x)
            G_x = F.conv2d(pred, sobel_kernel_x)

            sobel_kernel_y = torch.Tensor(
                        [[1.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [-1.0, -2.0, -1.0]]).to(device)
            sobel_kernel_y = sobel_kernel_y.view((1,1,3,3))

            I_y = F.conv2d(image, sobel_kernel_y)
            G_y = F.conv2d(pred, sobel_kernel_y)

            G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
            
            rang_grad = 1 - torch.pow(I_x*G_x + I_y*G_y,2)
            rang_grad = range_grad if rang_grad > 0 else 0

            loss2 = torch.sum(torch.mul(G, rang_grad))/torch.sum(G) + 1e-6
        
        if self.add_gradient:
            loss = (1-self.ratio_of_gradient)*self.bce_loss(pred, true) + loss2*self.ratio_of_gradient
        else:
            loss = self.bce_loss(pred, true)

        return loss
