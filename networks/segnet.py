import torch
from torch import nn
from torchvision import models


class ConvLayerBn(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3,
                 stride=1, padding=1, bias=False, relu=True):
        super(ConvLayerBn,self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_features,
                                out_features,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias))

        layers.append(nn.BatchNorm2d(out_features))

        if relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class SegNet(nn.Module):
    def __init__(self, num_class):
        super(SegNet, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)

        encoder = list()
        decoder = list()
        layers = list()

        # make encoder layers
        for layer in list(vgg.features):
            layers.append(layer)

            if isinstance(layer, nn.MaxPool2d):
                layers[-1] = nn.MaxPool2d(2, 2, return_indices=True)
                encoder.append(nn.Sequential(*layers))
                layers = list()

        # make decoder layers
        # specs: (num_layers, in_features, out_features)
        layer_specs = [(3, 512, 512), (3, 512, 256), (3, 256, 128), (2, 128, 64)]

        for spec in layer_specs:
            num_layer, in_features, out_features = spec
            for i in range(num_layer):
                if i == 0:
                    layers.append(ConvLayerBn(in_features, out_features))
                else:
                    layers.append(ConvLayerBn(out_features, out_features))
            decoder.append(nn.Sequential(*layers))
            layers = list()

        layers.append(ConvLayerBn(64, 64))
        layers.append(ConvLayerBn(64, num_class, relu=False))
        decoder.append(nn.Sequential(*layers))

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

    def forward(self,x):
        indices = list()
        for layer in self.encoder:
            x, idx = layer(x)
            indices.append(idx)

        indices.reverse()
        for idx, layer in zip(indices, self.decoder):
            x = nn.MaxUnpool2d(2,2)(x,idx)
            x = layer(x)
        return x
