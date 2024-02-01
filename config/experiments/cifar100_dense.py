from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "CIFAR100"

# Model
cfg.model = "DenseNet"
cfg.model_type = "dense"
cfg.depth = 101
cfg.num_classes = 100
cfg.resume = "./checkpoints/cifar100_densenet_101.tar"

# Preprocessing
cfg.pre_size = 32
cfg.image_size = 32
cfg.normalization_type = "cifar100"

# Dataset:
cfg.dataset = CN()
cfg.dataset.id_dataset = "cifar100"
cfg.dataset.val_dataset = "places365"
cfg.dataset.near_ood = ["cifar10", "tin"]
cfg.dataset.far_ood = ["svhn", "texture", "places365", "lsunc", 'lsunr', 'isun']
