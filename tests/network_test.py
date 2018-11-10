import numpy

import pytest

import torch
import torchsummary

from networks import unet

image_size = (3, 320, 320)

def test_networks_unet():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = unet(3, 1)
    model.to(device)
    
    assert torchsummary.summary(model, image_size)