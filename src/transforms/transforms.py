from albumentations.pytorch import ToTensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Resize,
    Cutout,
    Normalize,
    Compose,
    GaussNoise,
    IAAAdditiveGaussianNoise,
    RandomContrast,
    RandomGamma,
    RandomRotate90,
    RandomSizedCrop,
    RandomBrightness,
    ShiftScaleRotate,
    MotionBlur,
    MedianBlur,
    Blur,
    OpticalDistortion,
    GridDistortion,
    IAAPiecewiseAffine,
    OneOf)


def get_transforms(phase_config):
    transforms = []

    if phase_config.Resize.p > 0:
        transforms.append(Resize(
            phase_config.Resize.height,
            phase_config.Resize.width))

    if phase_config.HorizontalFlip.p > 0:
        transforms.append(HorizontalFlip(
            p=phase_config.HorizontalFlip.p))

    if phase_config.VerticalFlip.p > 0:
        transforms.append(VerticalFlip(
            p=phase_config.HorizontalFlip.p))

    if phase_config.ShiftScaleRotate.p > 0:
        transforms.append(ShiftScaleRotate(
            p=phase_config.ShiftScaleRotate.p,
            scale_limit=phase_config.ShiftScaleRotate.scale_limit,
            rotate_limit=phase_config.ShiftScaleRotate.rotate_limit,
            shift_limit=phase_config.ShiftScaleRotate.shift_limit,))

    if phase_config.OpticalDistortion.p > 0:
        transforms.append(ShiftScaleRotate(p=1))

    transforms.append(
        #[
            Normalize(mean=phase_config.mean, std=phase_config.std, p=1),
            #ToTensor(),
        #]
    )

    return Compose(transforms)