from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "CIFAR100"

# Model
cfg.model = "ViT"
cfg.model_type = "vit_b_cifar100"
cfg.num_classes = 100
cfg.resume = "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin"

# Preprocessing
cfg.pre_size = 224
cfg.image_size = 224
cfg.normalization_type = "cifar100"

# Dataset:
cfg.dataset = CN()
cfg.dataset.id_dataset = "cifar100"
cfg.dataset.val_dataset = "places365"
cfg.dataset.near_ood = ["cifar10", "tin"]
cfg.dataset.far_ood = ["svhn", "texture", "places365", "lsunc", 'lsunr', 'isun']
