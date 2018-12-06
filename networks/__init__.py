from .segnet import SegNet
from .unet import Unet
from .ternausnet import TernausNet
from .deeplab_v3_plus import DeepLab


def get_network(name, num_class):
    name = name.lower()
    if name == 'segnet':
        return SegNet(num_class)
    elif name == 'unet':
        return Unet(num_class)
    elif name == 'ternausnet':
        return TernausNet(num_class)
    elif name == 'deeplabv3plus':
        return DeepLab(return_with_logits = True)
    raise ValueError

