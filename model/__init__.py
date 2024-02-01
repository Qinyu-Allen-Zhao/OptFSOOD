import os

import numpy as np
import torch

from .densenet import DenseNet

from .resnet import ResNet, Bottleneck, BasicBlock
from .swin import SwinTransformer
from .vit import VisionTransformer
from .vit_cifar import vit_base_patch16_224


def get_model(cfg, resume=None):
    if cfg.model_type == "vit_b_cifar10" or cfg.model_type == "vit_b_cifar100":
        from torch import nn
        model = vit_base_patch16_224()
        model.head = nn.Linear(model.head.in_features, cfg.num_classes)
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resume,
                map_location="cpu",
                file_name=f"vit_base_patch16_224_in21k_ft_cifar{cfg.num_classes}.pth",
            )
        )
        model = model.cuda()
        model.eval()
        return model
    elif cfg.model_type == "rn50":
        model = ResNet(block=eval(cfg.block),
                       layers=cfg.layers,
                       num_classes=cfg.num_classes)
    elif cfg.model_type == "mb":
        from .mobilenet import MobileNetV2
        model = MobileNetV2(num_classes=cfg.num_classes)
    elif cfg.model_type == "dense":
        model = DenseNet(cfg.depth, cfg.num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0,
                         normalizer=None)
    elif cfg.model_type[:3] == 'vit':
        model = VisionTransformer(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            mlp_dim=cfg.mlp_dim,
        )
    elif cfg.model_type[:4] == "swin":
        model = SwinTransformer(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            stochastic_depth_prob=cfg.stochastic_depth_prob,
        )
    elif cfg.model_type == "Mixer-CIFAR10" or cfg.model_type == "Mixer-CIFAR100":
        from .mlp_mixer_cifar import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=cfg.image_size,
            hidden_size=cfg.hidden_size,
            patch_size=cfg.patch_size,
            hidden_c=cfg.hidden_c,
            hidden_s=cfg.hidden_s,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            drop_p=cfg.drop_p,
            off_act=cfg.off_act,
            is_cls_token=cfg.is_cls_token
        )
    elif cfg.model_type[:5] == "Mixer":
        from model.mlp_mixer import MlpMixer, CONFIGS
        config = CONFIGS[cfg.model_type]
        model = MlpMixer(config, cfg.image_size, num_classes=cfg.num_classes, patch_size=16, zero_head=False)
        model.load_from(np.load(resume))
        print(f"=> loading checkpoint '{resume}'")
        model.cuda()
        model.eval()
        return model
    else:
        raise Exception()

    if resume:
        if os.path.isfile(resume):
            print(f"=> loading checkpoint '{resume}'")
            checkpoint = torch.load(resume)
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
            if "module" in list(checkpoint.keys())[0]:
                state_dict_new = {}
                for k, v in checkpoint.items():
                    state_dict_new[k.replace("module.", "")] = v
                checkpoint = state_dict_new
            model.load_state_dict(checkpoint, strict=True)
        else:
            print(f"=> no checkpoint found at '{resume}'")

    model = model.cuda()
    model.eval()

    return model
