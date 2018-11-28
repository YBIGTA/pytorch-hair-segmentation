from .segnet import SegNet
from .unet import Unet
from .ternausnet import TernausNet
from .deeplab_v3 import 


def get_network(name, num_class):
    name = name.lower()
    if name == 'segnet':
        return SegNet(num_class)
    elif name == 'unet':
        return Unet(num_class)
    elif name == 'ternausnet':
        return TernausNet(num_class)
    elif name == 'deeplabv3+':
        return Deeplab_v3_plus(input_channels = 3, n_classes = num_class)
    raise ValueError

