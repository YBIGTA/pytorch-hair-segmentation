from .segnet import SegNet
from .unet import Unet
from .ternausnet import TernausNet


def get_network(name):
    name = name.lower()
    if name == 'segnet':
        return SegNet(num_class=1)
    elif name == 'unet':
        return Unet(num_class=1)
    elif name == 'ternausnet':
        return TernausNet(num_class=1)
    raise ValueError

