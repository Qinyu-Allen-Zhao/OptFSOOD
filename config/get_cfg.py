def get_cfg(experiment, model):
    if experiment == "imagenet" and model == "rn50":
        from config.experiments.imagenet_rn50 import cfg
        return cfg
    elif experiment == "imagenet" and model == "mb":
        from config.experiments.imagenet_mb import cfg
        return cfg
    elif experiment == "imagenet" and model == "vit_b":
        from config.experiments.imagenet_vit_b_16 import cfg
        return cfg
    elif experiment == "imagenet" and model == "vit_l":
        from config.experiments.imagenet_vit_l_16 import cfg
        return cfg
    elif experiment == "imagenet" and model == "swin_s":
        from config.experiments.imagenet_swin_s import cfg
        return cfg
    elif experiment == "imagenet" and model == "swin_b":
        from config.experiments.imagenet_swin_b import cfg
        return cfg
    elif experiment == "imagenet" and model == "mix_b":
        from config.experiments.imagenet_mixer_b import cfg
        return cfg
    elif experiment == "imagenet" and model == "mix_l":
        from config.experiments.imagenet_mixer_l import cfg
        return cfg
    elif experiment == "cifar10" and model == "dense":
        from config.experiments.cifar10_dense import cfg
        return cfg
    elif experiment == "cifar10" and model == "vit_b":
        from config.experiments.cifar10_vit_b import cfg
        return cfg
    elif experiment == "cifar10" and model == "mix_n":
        from config.experiments.cifar10_mixer_n import cfg
        return cfg
    elif experiment == "cifar100" and model == "dense":
        from config.experiments.cifar100_dense import cfg
        return cfg
    elif experiment == "cifar100" and model == "vit_b":
        from config.experiments.cifar100_vit_b import cfg
        return cfg
    elif experiment == "cifar100" and model == "mix_n":
        from config.experiments.cifar100_mixer_n import cfg
        return cfg
    else:
        raise Exception()
