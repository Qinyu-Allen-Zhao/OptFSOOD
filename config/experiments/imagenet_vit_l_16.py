from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "ImageNet"

# Model
cfg.model_type = "vit_l_16"
cfg.num_classes = 1000
# ViT-L-16
cfg.patch_size = 16
cfg.num_layers = 24
cfg.num_heads = 16
cfg.hidden_dim = 1024
cfg.mlp_dim = 4096
cfg.resume = "./checkpoints/vit_l_16-852ce7e3.pth"

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
