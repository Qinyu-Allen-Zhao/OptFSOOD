from torchvision import transforms

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'cifar100': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
}

interpolation_modes = {
    'nearest': transforms.InterpolationMode.NEAREST,
    'bilinear': transforms.InterpolationMode.BILINEAR,
}


def get_basic_transform(cfg):
    if "bit" in cfg.model_type:
        crop = 480

        return transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    pre_size = cfg.pre_size
    image_size = cfg.image_size
    normalization_type = cfg.normalization_type
    interpolation = interpolation_modes["bilinear"]
    if normalization_type in normalization_dict.keys():
        mean = normalization_dict[normalization_type][0]
        std = normalization_dict[normalization_type][1]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    print("Normalizing with", mean, std)

    # base_train_transform = transforms.Compose([
    #     transforms.Resize(pre_size, interpolation=interpolation),
    #     transforms.CenterCrop(image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(image_size, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std),
    # ])

    base_test_transform = transforms.Compose([
        transforms.Resize(pre_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return base_test_transform
