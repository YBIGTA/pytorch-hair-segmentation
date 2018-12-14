from .segnet import SegNet
from .unet import Unet
from .ternausnet import TernausNet
from .pspnet import PSPNetWithSqueezeNet


def get_network(name, num_class):
    name = name.lower()
    if name == 'segnet':
        return SegNet(num_class)
    elif name == 'unet':
        return Unet(num_class)
    elif name == 'ternausnet':
        return TernausNet(num_class)
    elif name == 'pspnet':
        return PSPNetWithSqueezeNet(num_class)
    raise ValueError

