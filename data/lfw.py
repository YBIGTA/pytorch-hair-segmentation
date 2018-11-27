import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def parse_name_list(fp):
    with open(fp, 'r') as fin:
        lines = fin.readlines()
    parsed = list()
    for line in lines:
        name, num = line.strip().split(' ')
        num = format(num, '0>4')
        filename = '{}_{}'.format(name, num)
        parsed.append((name, filename))
    return parsed


class LfwDataset(Dataset):
    def __init__(self, root_dir, train=True, joint_transforms=None,
                 image_transforms=None, mask_transforms=None):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
        """

        txt_file = 'parts_train_val.txt' if train else 'parts_test.txt'
        txt_dir = os.path.join(root_dir, txt_file)
        name_list = parse_name_list(txt_dir)
        img_dir = os.path.join(root_dir, 'lfw_funneled')
        mask_dir = os.path.join(root_dir, 'parts_lfw_funneled_gt_images')

        self.img_path_list = [os.path.join(img_dir, elem[0], elem[1]+'.jpg') for elem in name_list]
        self.mask_path_list = [os.path.join(mask_dir, elem[1]+'.ppm') for elem in name_list]
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)
        mask = LfwDataset.rgb2binary(mask)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return img, mask

    def __len__(self):
        return len(self.mask_path_list)

    @staticmethod
    def rgb2binary(mask):
        mask_arr = np.array(mask)
        mask_map = mask_arr == np.array([255, 0, 0])
        mask_map = np.all(mask_map, axis=2).astype(np.float32)
        return Image.fromarray(mask_map)

    @staticmethod
    def rgb2binary_(mask):
        binary_image = Image.new('F', mask.size)
        binary_pixel = binary_image.load()

        mask_pix = mask.load()

        for i in range(mask.size[0]):
            for j in range(mask.size[1]):
                rgb = mask_pix[i, j]
                binary_pixel[i, j] = (rgb == [255, 0, 0])
        return binary_image
