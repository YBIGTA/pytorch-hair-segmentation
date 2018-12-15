from .figaro import FigaroDataset
from .lfw import LfwDataset
from torch.utils.data import DataLoader


def get_loader(dataset, data_dir='./data/Figaro1k', train=True, batch_size=64, shuffle=True,
        joint_transforms=None, image_transforms=None, mask_transforms=None, num_workers=0, gray_image=False):
    """
    Args:
        dataset (string): name of dataset to use
        data_dir (string): directory to dataset
        train (bool): whether training or not
        batch_size (int): batch size
        joint_transforms (Compose): list of joint transforms both on images and masks
        image_transforms (Compose): list of transforms only on images
        mask_transforms (Compose): list of transforms only on targets (masks)
    """

    if dataset.lower() == 'figaro':
        dset = FigaroDataset(root_dir=data_dir,
                            train=train,
                            joint_transforms=joint_transforms,
                            image_transforms=image_transforms,
                            mask_transforms=mask_transforms,
                            gray_image=gray_image)

    elif dataset.lower() == 'lfw':
        dset = LfwDataset(root_dir=data_dir,
                          train=train,
                          joint_transforms=joint_transforms,
                          image_transforms=image_transforms,
                          mask_transforms=mask_transforms)
    else:
        raise ValueError

    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
