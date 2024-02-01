from yacs.config import CfgNode as CN

# Basic settings
cfg = CN()
cfg.exp_space = "ImageNet"

# Model
cfg.model = "ResNet"
cfg.model_type = "rn50"
cfg.num_classes = 1000
cfg.block = "Bottleneck"
cfg.norm_layer = "BatchNorm2d"
cfg.layers = [3, 4, 6, 3]
cfg.resume = "./checkpoints/imagenet_resnet50.pth"

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
