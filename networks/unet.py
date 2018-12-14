import torch
from torch import nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvLayer, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm2d(out_features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_features, out_features, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm2d(out_features))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class Unet(nn.Module):
    def __init__(self, num_class):
        super(Unet, self).__init__()
        ndim = 64
        input_dim = 3

        encoder_conv_layers = [ConvLayer(input_dim, ndim)]
        decoder_conv_layers = []
        upconv_layers = []

        for _ in range(4):
            encoder_conv_layers.append(ConvLayer(ndim, ndim*2))
            upconv_layers.append(nn.ConvTranspose2d(ndim*2, ndim, kernel_size=2, stride=2))
            decoder_conv_layers.append((ConvLayer(ndim*2, ndim)))
            ndim *= 2

        ndim = 64
        self.encoder_conv_layers = nn.ModuleList(encoder_conv_layers)
        self.upconv_layers = nn.ModuleList(upconv_layers)
        self.decoder_conv_layers = nn.ModuleList(decoder_conv_layers)
        self.conv1x1 = nn.Conv2d(ndim, num_class, kernel_size=1, stride=1)

    def forward(self, x):
        features = []
        input_size = x.size()

        for i in range(4):
            x = self.encoder_conv_layers[i](x)
            features.append(x.clone())
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.encoder_conv_layers[4](x)

        for i in range(3, -1, -1):
            x = self.upconv_layers[i](x)
            x = torch.cat([x, F.upsample(features[i], size=x.size()[2:], mode='bilinear')], dim=1)
            x = self.decoder_conv_layers[i](x)

        x = self.conv1x1(x)
        out = F.upsample(x, size = input_size[2:], mode='bilinear')
        return out
