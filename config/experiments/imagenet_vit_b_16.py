from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "ImageNet"

# Model
cfg.model_type = "vit_b_16"
cfg.num_classes = 1000

# ViT-B-16
cfg.patch_size = 16
cfg.num_layers = 12
cfg.num_heads = 12
cfg.hidden_dim = 768
cfg.mlp_dim = 3072
cfg.resume = "./checkpoints/vit_b_16-c867db91.pth"

# Preprocessing
cfg.pre_size = 256
cfg.image_size = 224
cfg.normalization_type = "imagenet"

# Dataset:
cfg.dataset = CN()
cfg.dataset.id_dataset = "imagenet"
cfg.dataset.val_dataset = "inaturalist"
cfg.dataset.near_ood = ["species", "inaturalist",
                        "sun", "places",
                        "openimageo", "imageneto"]
cfg.dataset.far_ood = ["texture", "mnist"]
