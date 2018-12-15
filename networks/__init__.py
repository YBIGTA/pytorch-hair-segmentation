from .segnet import SegNet
from .unet import Unet
from .ternausnet import TernausNet
from .deeplab_v3_plus import DeepLab
from .pspnet import PSPNetWithSqueezeNet



def get_network(name):
    name = name.lower()
    if name == 'segnet':
        return SegNet(num_class=1)
    elif name == 'unet':
        return Unet(num_class=1)
    elif name == 'ternausnet':
        return TernausNet(num_class=1)
    elif name == 'deeplabv3plus':
        return DeepLab(return_with_logits = True)
    elif name == 'pspnet':
        return PSPNetWithSqueezeNet(num_class=1)
    raise ValueError
