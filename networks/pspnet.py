import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, squeezenet1_1


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
        return f4, f3


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), output_size=(256, 256)):
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


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size=None):
        super().__init__()
        self.upsample_size = upsample_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = 2 * x.size(2), 2 * x.size(3)
        if self.upsample_size is not None:
            size = self.upsample_size
        p = F.upsample(x, size=size, mode='bilinear')
        return self.conv(p)


class PSPNetWithSqueezeNet(nn.Module):
    def __init__(self, num_class=1, deep_features_size=256, img_size=256,
            sizes=(1, 2, 3, 6), auxiliary_loss=False):
        super().__init__()
        self.auxiliary_loss = auxiliary_loss
        self.base_network = SqueezeNetExtractor()
        self.psp = PyramidPoolingModule(in_channels=512, sizes=sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64, img_size)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_class, kernel_size=1),
        )

        if auxiliary_loss:
            self.classifier = nn.Sequential(
                nn.Linear(deep_features_size, 256),
                nn.ReLU(),
                nn.Linear(256, num_class)
            )

    def forward(self, x):
        f, class_f = self.base_network(x) 
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        if self.auxiliary_loss:
            auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
            return self.final(p), self.classifier(auxiliary)
        return self.final(p)

