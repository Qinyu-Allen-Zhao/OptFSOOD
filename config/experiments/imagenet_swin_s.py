from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "ImageNet"

# Model
cfg.model_type = "swin_s"
cfg.num_classes = 1000
cfg.patch_size = [4, 4]
cfg.embed_dim = 96
cfg.depths = [2, 2, 18, 2]
cfg.num_heads = [3, 6, 12, 24]
cfg.window_size = [7, 7]
cfg.stochastic_depth_prob = 0.3
cfg.resume = "./checkpoints/swin_s-5e29d889.pth"

# Preprocessing
cfg.pre_size = 246
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
