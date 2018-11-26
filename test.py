from data import get_loader
from utils.loss import CrossEntropyLoss2d

from utils import joint_transforms as jnt_trnsf

import torch
import torchvision.transforms as std_trnsf
from networks import SegNet

model = SegNet(1).cuda()
criterion = CrossEntropyLoss2d()
joint_transforms = jnt_trnsf.Compose([
    jnt_trnsf.Resize(512),
])
image_transforms = std_trnsf.Compose([
    std_trnsf.ToTensor()
])
mask_transforms = std_trnsf.Compose([
    std_trnsf.ToTensor()
])

train_loader = get_loader(dataset='figaro',
                      train=True,
                      joint_transforms=joint_transforms,
                      image_transforms=image_transforms,
                      mask_transforms=mask_transforms,
                      batch_size=4,
                      shuffle=False)

test_loader = get_loader(dataset='figaro',
                      train=False,
                      joint_transforms=joint_transforms,
                      image_transforms=image_transforms,
                      mask_transforms=mask_transforms,
                      batch_size=4,
                      shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

train_iterator = iter(train_loader)
for _ in range(150):
    data, target = next(train_iterator)
    data, target = data.cuda(), target.cuda()
    print(torch.unique(target))
    exit()
    out = model(data)
    loss = criterion(out, target)