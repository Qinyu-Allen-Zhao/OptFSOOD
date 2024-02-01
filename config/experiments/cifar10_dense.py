from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "CIFAR10"

# Model
cfg.model = "DenseNet"
cfg.model_type = "dense"
cfg.depth = 101
cfg.num_classes = 10
cfg.resume = "./checkpoints/cifar10_densenet_101.tar"

# Preprocessing
cfg.pre_size = 32
cfg.image_size = 32
cfg.normalization_type = "cifar10"

# Dataset:
cfg.dataset = CN()
cfg.dataset.id_dataset = "cifar10"
cfg.dataset.val_dataset = "cifar100"
cfg.dataset.near_ood = ["cifar100", "tin"]
cfg.dataset.far_ood = ["svhn", "texture", "places365", "lsunc", 'lsunr', 'isun']
