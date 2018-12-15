from .deeplab_v3_plus import DeepLab
from .pspnet import PSPNetWithSqueezeNet



def get_network(name):
    name = name.lower()
    if name == 'deeplabv3plus':
        return DeepLab(return_with_logits = True)
    elif name == 'pspnet':
        return PSPNetWithSqueezeNet(num_class=1)
    raise ValueError
