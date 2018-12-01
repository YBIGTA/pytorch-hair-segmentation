import torchsummary as ts
from mobile_hair import MobileMattingFCN as mm
from mobile_hair import HairMattingLoss as loss1

import numpy as np
import torch

model = mm()

ts.summary(model,(3,224,224))

inputs = torch.Tensor(np.ones((2,3,224,224)))
inputs2 = torch.Tensor(np.ones((2,1,224,224)))
outputs = torch.Tensor(np.ones((2,1,224,224)))

loss = loss1(0.5)

preds = model(inputs)

l = loss(preds, outputs, inputs2)

l.backward()