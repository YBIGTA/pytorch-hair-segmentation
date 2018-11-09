import os
from PIL import Image
from torch.utils.data import Dataset

class FigaroDataset(Dataset):
    def __init__(self, root_dir, train=True, joint_transforms=None,
            image_transforms=None, target_transforms=None):
        '''
        Args:
            root_dir (str): root directory of dataset
            transforms (torchvision.transforms.Compose): tranformation on both data and target
        '''
        mode = 'Training' if train else 'Testing'
        img_dir = os.path.join(root_dir,'Original', mode)
        mask_dir = os.path.join(root_dir, 'GT', mode)

        self.img_path_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        self.mask_path_list = [os.path.join(mask_dir, mask) for mask in sorted(os.listdir(mask_dir))]
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __getitem__(self,idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.target_transforms is not None:
            mask = self.target_transforms(mask)

        return img, mask

    def __len__(self):
        return len(self.mask_path_list)


