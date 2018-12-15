import torch
from torch import nn
from torchvision import models


class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(ConvLayer, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_features,
                                out_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False))

        if relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class TernausNet(nn.Module):
    def __init__(self, num_class):
        super(TernausNet, self).__init__()
        vgg = models.vgg11(pretrained=True)
        encoder_conv_layers = list()
        decoder_conv_layers = list()
        upconv_layers = list()

        layers = list()
        for layer in list(vgg.features):
            layers.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                encoder_conv_layers.append(nn.Sequential(*layers[:-1]))
                layers = list()

        upsample_spec = [(512, 256), (512, 256), (512, 128), (256, 64), (128, 32)]
        decoder_conv_spec = [(768, 512), (768, 512), (384, 256), (192, 128), (96, num_class)]

        for i in range(5):
            upconv_layers.append(nn.ConvTranspose2d(upsample_spec[i][0],
                                                    upsample_spec[i][1],
                                                    kernel_size=2,
                                                    stride=2))
            relu = i != 4
            decoder_conv_layers.append(ConvLayer(decoder_conv_spec[i][0],
                                                 decoder_conv_spec[i][1],
                                                 relu=relu))

        self.encoder_conv_layers = nn.ModuleList(encoder_conv_layers)
        self.upconv_layers = nn.ModuleList(upconv_layers)
        self.decoder_conv_layers = nn.ModuleList(decoder_conv_layers)
        self.mid_conv_layer = ConvLayer(512, 512)

    def forward(self, x):
        features = list()
        for i, layer in enumerate(self.encoder_conv_layers):
            x = layer(x)
            features.append(x.clone())
            x = nn.MaxPool2d(2, 2)(x)
        x = self.mid_conv_layer(x)
        features.reverse()
        for feature, upconv_layer, decoder_conv_layer in zip(
                features, self.upconv_layers, self.decoder_conv_layers):
            x = upconv_layer(x)
            x = torch.cat([x, feature], dim=1)
            x = decoder_conv_layer(x)
        return x