import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, SVHN

from utils.preprocess import get_basic_transform


class RandomData(Dataset):
    def __init__(self, num_samples, is_gaussian=True, transform=None):
        self.num_samples = num_samples
        self.gaussian = is_gaussian
        self.transform = transform
        self.targets = [-1] * self.num_samples
        if self.gaussian:
            self.data = 255 * np.random.randn(self.num_samples, 32, 32, 3) + 255 / 2
            self.data = np.clip(self.data, 0, 255).astype("uint8")
        else:
            self.data = np.random.randint(0, 255, (self.num_samples, 32, 32, 3)).astype("uint8")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_dataloaders(keys, cfg, dataset_cfg, batch_size=256, shuffle=False, num_workers=8):
    # Get preprocessing transforms
    base_val_tr = get_basic_transform(cfg)
    root_dir = dataset_cfg["root"]

    dataloaders = []
    if "train" in keys:
        # Load train dataset
        name = cfg.dataset.id_dataset
        data_dir = dataset_cfg[name]
        train_loader = get_data_loader(root_dir, data_dir, base_val_tr,
                                       batch_size=batch_size, shuffle=shuffle, is_train=True,
                                       num_workers=num_workers)
        dataloaders.append(train_loader)

    if "val" in keys:
        # Load val dataset
        name = cfg.dataset.val_dataset
        data_dir = dataset_cfg[name]
        val_loader = get_data_loader(root_dir, data_dir, base_val_tr,
                                     batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        dataloaders.append(val_loader)

    if "id" in keys:
        # Load ID dataset
        name = cfg.dataset.id_dataset
        data_dir = dataset_cfg[name]
        id_loader = get_data_loader(root_dir, data_dir, base_val_tr,
                                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        dataloaders.append(id_loader)

    if "ood" in keys:
        # near OOD datasets
        near_ood = get_ood_loaders(cfg.dataset.near_ood, dataset_cfg, root_dir, base_val_tr,
                                   batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        far_ood = get_ood_loaders(cfg.dataset.far_ood, dataset_cfg, root_dir, base_val_tr,
                                  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        dataloaders.append(near_ood)
        dataloaders.append(far_ood)

    return dataloaders


def get_data_loader(root_dir, data_dir, transform, batch_size=128, shuffle=True, is_train=False, num_workers=4):
    path = os.path.join(root_dir, data_dir)
    if "cifar100" in data_dir:
        dataset = CIFAR100(root=path, train=is_train, download=True, transform=transform)
    elif "cifar10" in data_dir:
        dataset = CIFAR10(root=path, train=is_train, download=True, transform=transform)
    elif "svhn" in data_dir:
        split = 'train' if is_train else 'test'
        dataset = SVHN(path, transform=transform, split=split, download=True)
    elif "ImageNet" in data_dir and is_train:
        data_dir = data_dir.replace("val", "train")
        path = os.path.join(root_dir, data_dir)
        dataset = ImageFolder(root=path, transform=transform)
    else:
        dataset = ImageFolder(root=path, transform=transform)
    print(f"Load from {path}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def get_ood_loaders(dataset_list, dataset_cfg, root_dir, base_val_tr,
                    batch_size=256, shuffle=False, num_workers=4):
    loaders = {}
    for dataset_name in dataset_list:
        data_dir = dataset_cfg[dataset_name]
        ood_loader = get_data_loader(root_dir, data_dir, base_val_tr,
                                     batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        loaders[dataset_name] = ood_loader

    return loaders


def get_surrogate_data_loader(cfg, is_gaussian):
    # Get preprocessing transforms
    base_val_tr = get_basic_transform(cfg)

    dataset = RandomData(50000, is_gaussian=is_gaussian, transform=base_val_tr)
    ood_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=16)

    return ood_loader


def get_real_ood_val_loader(cfg, dataset_cfg):
    # Get preprocessing transforms
    base_val_tr = get_basic_transform(cfg)
    name = cfg.dataset.val_dataset
    data_dir = dataset_cfg[name]
    ood_loader = get_data_loader(dataset_cfg['root'], data_dir, base_val_tr,
                                 batch_size=128, shuffle=True, is_train=False, num_workers=4)

    return ood_loader
