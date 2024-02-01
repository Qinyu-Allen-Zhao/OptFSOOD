from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "CIFAR100"

# Model
cfg.model = "MLPMixer"
cfg.model_type = "Mixer-CIFAR100"
cfg.hidden_size = 128
cfg.patch_size = 4
cfg.hidden_c = 512
cfg.hidden_s = 64
cfg.num_layers = 8
cfg.num_classes = 100
cfg.drop_p = 0
cfg.off_act = False
cfg.is_cls_token = False
cfg.resume = "checkpoints/mix_mlp_nano_cifar100.pth"

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
