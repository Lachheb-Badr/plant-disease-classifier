"""Image transforms: augmentation for training, deterministic for evaluation."""
from torchvision import transforms as T


def get_train_transforms(config: dict) -> T.Compose:
    size = tuple(config["data"]["image_size"])  # (H, W)
    tcfg = config["transforms"]
    aug = tcfg["augmentation"]
    cj = aug["color_jitter"]

    ops = [
        T.RandomResizedCrop(
            size,
            scale=tuple(aug["random_resized_crop_scale"]),
            ratio=(1.4, 1.6),  # stay close to the 3:2 aspect ratio
        ),
    ]
    if aug["random_horizontal_flip"]:
        ops.append(T.RandomHorizontalFlip())
    if aug["random_vertical_flip"]:
        ops.append(T.RandomVerticalFlip())
    if aug["random_rotation"]:
        ops.append(T.RandomRotation(aug["random_rotation"]))
    ops += [
        T.ColorJitter(
            brightness=cj["brightness"],
            contrast=cj["contrast"],
            saturation=cj["saturation"],
            hue=cj["hue"],
        ),
        T.ToTensor(),
        T.Normalize(tcfg["normalize_mean"], tcfg["normalize_std"]),
    ]
    return T.Compose(ops)


def get_eval_transforms(config: dict) -> T.Compose:
    """Deterministic transforms for validation, test and inference."""
    size = tuple(config["data"]["image_size"])
    tcfg = config["transforms"]
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(tcfg["normalize_mean"], tcfg["normalize_std"]),
    ])
