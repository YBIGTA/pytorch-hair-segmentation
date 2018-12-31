from .deeplab_v3_plus import DeepLab
from .pspnet import PSPNet
from .mobile_hair import MobileMattingFCN



def get_network(name):
    name = name.lower()
    if name == 'deeplabv3plus':
        return DeepLab(return_with_logits = True)
    elif name == 'pspnet_squeezenet':
        return PSPNet(num_class=1, base_network='squeezenet')
    elif name == 'pspnet_resnet101':
        return PSPNet(num_class=1, base_network='resnet101')
    elif name == 'mobilenet':
        return MobileMattingFCN()
    raise ValueError
