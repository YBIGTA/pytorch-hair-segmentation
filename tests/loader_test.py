import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_loader
from utils import joint_transforms as jnt_trnsf

import numpy as np
import torchvision.transforms as std_trnsf

def test_loader():
    print('Start testing loader...')
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
                              batch_size=1,
                              shuffle=False,
                              )
    test_loader = get_loader(dataset='figaro',
                              train=False,
                              joint_transforms=joint_transforms,
                              image_transforms=image_transforms,
                              mask_transforms=mask_transforms,
                              batch_size=1,
                              shuffle=False,
                              )
    train_iterator = iter(train_loader)
    for _ in range(150):
        next(train_iterator)
    img1, mask1, target1 = next(train_iterator)
    test_iterator = iter(test_loader)
    img2, mask2, target2 = next(test_iterator)


    img1 = img1.data.numpy()
    img2 = img2.data.numpy()

    mask1 = mask1.data.numpy()
    mask2 = mask2.data.numpy()

    ans_img1 = np.load('./tests/data/test_loader.img1.npy')
    ans_img2 = np.load('./tests/data/test_loader.img2.npy')
    ans_mask1 = np.load('./tests/data/test_loader.mask1.npy')
    ans_mask2 = np.load('./tests/data/test_loader.mask2.npy')

    assert target1 == 1
    assert target2 == 0

    np.testing.assert_array_almost_equal(img1, ans_img1)
    np.testing.assert_array_almost_equal(img2, ans_img2)
    np.testing.assert_array_almost_equal(mask1, ans_mask1)
    np.testing.assert_array_almost_equal(mask2, ans_mask2)

    print('Loader Test succedded!')

if __name__ == '__main__':
    test_loader()
