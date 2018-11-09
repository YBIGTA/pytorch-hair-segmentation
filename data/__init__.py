from .figaro import FigaroDataset
from torch.utils.data import DataLoader

def get_loader(dataset, data_dir='./data/Figaro1k', train=True, batch_size=64,
        joint_transforms=None, image_transforms=None, target_transforms=None):
    """
    Args:
        dataset (string): name of dataset to use
        data_dir (string): directory to dataset
        train (bool): whether training or not
        batch_size (int): batch size
        joint_transforms (Compose): list of joint transforms both on images and masks
        image_transforms (Compose): list of transforms only on images
        target_transforms (Compose): list of transforms only on targets (masks)
    """

    shuffle = True if train else False
    if dataset.lower() == 'figaro':
        dset = FigaroDataset(root_dir=data_dir,
                            train=train,
                            joint_transforms=joint_transforms,
                            image_transforms=image_transforms,
                            target_transforms=target_transforms)
    else:
        raise ValueError
    loader = DataLoader(dset, batch_size = batch_size, shuffle = shuffle)

    return loader
