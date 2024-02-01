from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "CIFAR10"

# Model
cfg.model = "ViT"
cfg.model_type = "vit_b_cifar10"
cfg.num_classes = 10
cfg.resume = "https://huggingface.co/aaraki/vit_base_patch16_224_in21k_ft_cifar10/resolve/main/pytorch_model.bin"

# Preprocessing
cfg.pre_size = 224
cfg.image_size = 224
cfg.normalization_type = "cifar10"

# Dataset:
cfg.dataset = CN()
cfg.dataset.id_dataset = "cifar10"
cfg.dataset.val_dataset = "cifar100"
cfg.dataset.near_ood = ["cifar100", "tin"]
cfg.dataset.far_ood = ["svhn", "texture", "places365", "lsunc", 'lsunr', 'isun']
