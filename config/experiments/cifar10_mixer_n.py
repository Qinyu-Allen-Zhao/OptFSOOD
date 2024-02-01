from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "CIFAR10"

# Model
cfg.model = "MLPMixer"
cfg.model_type = "Mixer-CIFAR10"
cfg.hidden_size = 128
cfg.patch_size = 4
cfg.hidden_c = 512
cfg.hidden_s = 64
cfg.num_layers = 8
cfg.num_classes = 10
cfg.drop_p = 0
cfg.off_act = False
cfg.is_cls_token = False
cfg.resume = "checkpoints/mix_mlp_nano_cifar10.pth"

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
