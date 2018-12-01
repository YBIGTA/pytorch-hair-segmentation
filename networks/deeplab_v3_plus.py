# Reference = https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class Sep_Conv(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = False):
    super(Sep_Conv, self).__init__()
    
    self.depth_conv = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, dilation, groups = input_channels, bias = bias)
    self.point_conv = nn.Conv2d(input_channels, output_channels, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = bias)
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU(inplace = True)
    
    
  def forward(self, input):
    x = self.depth_conv(input)
    x = self.point_conv(x)
    x = self.bn(x)
    x = self.relu(x)
    
    return x
  

class Block(nn.Module):
  def __init__(self, input_channels, output_channels, repeats, stride_change, skip_connect, kernel_size = 3, stride = 2):
    super(Block, self).__init__()
    
    
    repeat = []
    
    # First Conv
    repeat.append(
        Sep_Conv(input_channels, output_channels)
    )
    
    # Middle Conv
    for i in range(repeats - 2):
      repeat.append(
          Sep_Conv(output_channels, output_channels)
      )
    
    # Last Conv
    if stride_change:
      repeat.append(
          Sep_Conv(output_channels, output_channels, stride = 2)
      )
      
    else:
      repeat.append(
          Sep_Conv(output_channels, output_channels, stride = 1)
      )
    
    
    self.repeat_block = nn.Sequential(*repeat)
    self.skip_conv = nn.Conv2d(input_channels, output_channels, kernel_size = 1, stride = 2)
    self.skipbn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU(inplace = True)

    
  def forward(self, input):
    
    if self.skip_connect:
      skip = self.skip_conv(input)
      skip = self.skipbn(skip)
      skip = self.relu(skip)
    else:
      skip = None
      
    x = self.repeat_block(input)
    
    x = skip + x
    return x


class Xception(nn.Module):
  def __init__(self, input_channels = 3):
    super(Xception, self).__init__()
    
    
    # Entry flow
    self.layer1 = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size = 3, stride = 2, padding = 1, bias = False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True)
    )
    
    self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(64)
    )
    
    self.block1 = Block(64, 128, repeats = 3, stride_change = True, skip_connect = True)
    self.block2 = Block(128, 256, repeats = 3, stride_change = True, skip_connect = True)
    self.block3 = Block(256, 728, repeats = 3, stride_change = True, skip_connect = True)
    
    
    # Middle flow
    self.block4  = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block5  = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block6  = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block7  = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block8  = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block9  = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block10 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block11 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block12 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block13 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block14 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block15 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block16 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block17 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block18 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    self.block19 = Block(728, 728, repeats = 3, stride_change = False, skip_connect = True)
    
    
    # Exit flow
    self.block20 = Block(728, 1024, repeats = 3, stride_change = True, skip_connect = True)
    self.layer3 = nn.Sequential(
        Sep_Conv(1024, 1536, dilation = 2),
        nn.BatchNorm2d(1536)
    )
    self.layer4 = nn.Sequential(
        Sep_Conv(1536, 1536, dilation = 2),
        nn.BatchNorm2d(1536)
    )
    self.layer5 = nn.Sequential(
        Sep_Conv(1536, 2048, dilation = 2),
        nn.BatchNorm2d(2048)
    )
  
  def forward(self, input):
    # Entry flow
    x = self.layer1(input)
    x = self.layer2(x)
    
    x = self.block1(x)
    low_level_features = x
    x = self.block2(x)
    x = self.block3(x)
    
    # Middle flow
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)
    x = self.block13(x)
    x = self.block14(x)
    x = self.block15(x)
    x = self.block16(x)
    x = self.block17(x)
    x = self.block18(x)
    x = self.block19(x)
    
    # Exit flow
    x = self.block20(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    
    return x, low_level_features


class ASPP(nn.Module):
  def __init__(self, input_channels, output_channels, rate):
    super(ASPP, self).__init__()
    
    if rate == 1:
      kernel_size = 1
      padding = 0
    else:
      kernel_size = 3
      padding = rate
    
    self.atrous_conv = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size,
                                stride = 1, padding = padding, dilation = rate, bias = False)
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU(inplace = True)
    
  def forward(self, input):
    x = self.atrous_conv(input)
    x = self.bn(x)
    x = self.relu(x)
    
    return x


class Encoder(nn.Module):
  def __init__(self, input_channels):
    super(Encoder, self).__init__()
    
    self.dcnn = Xception(input_channels = input_channels)
    
    self.aspp1 = ASPP(2048, 256, rate = 1)
    self.aspp2 = ASPP(2048, 256, rate = 6)
    self.aspp3 = ASPP(2048, 256, rate = 12)
    self.aspp4 = ASPP(2048, 256, rate = 18)
    
    self.pooling = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(2048, 256, kernel_size = 1, stride = 1, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True)
    )
    
    self.conv = nn.Conv2d(256 * 5, 256, kernel_size = 1, bias = False)
    self.bn = nn.BatchNorm2d(256)
    self.relu = nn.ReLU(inplace = True)
    
    
  def forward(self, input):
    x, low_ = Xception(input)
    
    aspp1 = self.aspp1(x)
    aspp2 = self.aspp2(x)
    aspp3 = self.aspp3(x)
    aspp4 = self.aspp4(x)
    pool = self.pooling(x)
    
    x = torch.cat([aspp1, aspp2, aspp3, aspp4, pool], dim = 1)
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    
    return x, low_level_features


class Decoder(nn.Module):
  def __init__(self, n_classes):
    super(Decoder, self).__init__()
    
    low_level_channels = 128
    
    self.conv1 = nn.Conv2d(low_level_channels, 48, kernel_size = 1, bias = False)
    self.bn = nn.BatchNorm2d(48)
    self.relu = nn.ReLU(inplace = True)
    
    self.conv2 = nn.Sequential(
        nn.Conv2d(304, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True),
        nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True),
        nn.Conv2d(256, n_classes, kernel_size = 3, stride = 1, padding = 1, bias = False),
    )
    
  def forward(self, x, low_level_features):
    low_level_features = self.conv1(low_level_features)
    low_level_features = self.bn(low_level_features)
    low_level_features = self.relu(low_level_features)
    
    x = F.interpolate(x, size = low_level_features.size()[2:3], mode = 'bilinear')
    x = torch.cat([x, low_level_features], dim = 1)
    x = self.conv2(x)
    
    return x


class Deeplab_v3_plus(nn.Module):
  def __init__(self, input_channels, n_classes):
    super(Deeplab_v3_plus, self).__init__()
    
    self.encoder = Encoder(input_channels)
    self.decoder = Decoder(n_classes)
    
  def forward(self, input):
    x, low_level_features = Encoder(input)
    x = Decoder(x, low_level_features)
    x = F.interpolate(x, size = input.size()[2:], mode = 'bilinear')
    
    return x
