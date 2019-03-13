import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import squeezenet1_1, resnet101
from torch.nn.init import xavier_normal_

"""
Referenced from https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
"""


class ResNet101Extractor(nn.Module):
    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        model = resnet101(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:7])
    def forward(self, x):
        return self.features(x)

class SqueezeNetExtractor(nn.Module):
    def __init__(self):
        super(SqueezeNetExtractor, self).__init__()
        model = squeezenet1_1(pretrained=True)
        features = model.features
        self.feature1 = features[:2]
        self.feature2 = features[2:5]
        self.feature3 = features[5:8]
        self.feature4 = features[8:]

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        return f4


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        pyramid_levels = len(sizes)
        out_channels = in_channels // pyramid_levels

        pooling_layers = nn.ModuleList()
        for size in sizes:
            layers = [nn.AdaptiveAvgPool2d(size), nn.Conv2d(in_channels, out_channels, kernel_size=1)]
            pyramid_layer = nn.Sequential(*layers)
            pooling_layers.append(pyramid_layer)

        self.pooling_layers = pooling_layers

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        for pooling_layer in self.pooling_layers:
            # pool with different sizes
            pooled = pooling_layer(x)

            # upsample to original size
            upsampled = F.upsample(pooled, size=(h, w), mode='bilinear')

            features.append(upsampled)

        return torch.cat(features, dim=1)


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size=None):
        super().__init__()
        self.upsample_size = upsample_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = 2 * x.size(2), 2 * x.size(3)
        f = F.upsample(x, size=size, mode='bilinear')
        return self.conv(f)


class PSPNet(nn.Module):
    def __init__(self, num_class=1, sizes=(1, 2, 3, 6), base_network='resnet101'):
        super(PSPNet, self).__init__()
        base_network = base_network.lower()
        if base_network == 'resnet101':
            self.base_network = ResNet101Extractor()
            feature_dim = 1024
        elif base_network == 'squeezenet':
            self.base_network = SqueezeNetExtractor()
            feature_dim = 512
        else:
            raise ValueError
        self.psp = PyramidPoolingModule(in_channels=feature_dim, sizes=sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = UpsampleLayer(2*feature_dim, 256)
        self.up_2 = UpsampleLayer(256, 64)
        self.up_3 = UpsampleLayer(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_class, kernel_size=1)
        )

        self._init_weight()

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        f = self.base_network(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        if (p.size(2) != h) or (p.size(3) != w):
            p = F.interpolate(p, size=(h, w), mode='bilinear')

        p = self.drop_2(p)

        return self.final(p)

    def _init_weight(self):
        layers = [self.up_1, self.up_2, self.up_3, self.final]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                xavier_normal_(layer.weight.data)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

