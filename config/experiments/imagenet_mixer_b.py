from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "ImageNet"

# Model
cfg.model_type = "Mixer-B_16"
cfg.num_classes = 1000
cfg.resume = "./checkpoints/Mixer-B_16.npz"

# Preprocessing
cfg.pre_size = 238
cfg.image_size = 224
cfg.interpolation = "BICUBIC"
cfg.normalization_type = "imagenet"

# Dataset:
cfg.dataset = CN()
cfg.dataset.id_dataset = "imagenet"
cfg.dataset.val_dataset = "inaturalist"
cfg.dataset.near_ood = ["species", "inaturalist",
                        "sun", "places",
                        "openimageo", "imageneto"]
cfg.dataset.far_ood = ["texture", "mnist"]
