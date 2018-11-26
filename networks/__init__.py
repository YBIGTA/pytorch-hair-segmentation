from .segnet import SegNet
from .unet import Unet
from .ternausnet import TernausNet


def get_network(name, num_class):
    name = name.lower()
    if name == 'segnet':
        return SegNet(num_class)
    elif name == 'unet':
        return Unet(num_class)
    elif name == 'ternausnet':
        return TernausNet(num_class)
    raise ValueError

