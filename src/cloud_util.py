import albumentations as albu
import numpy as np
import cv2


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_training_augmentation():
    train_transform = [
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        #albu.RandomGridShuffle(p=0.5, grid=(3,3)),
        #albu.RandomCrop(p=0.5, height=250, width=325),# ADD
        #albu.GridDistortion(p=0.5),# ADD
        # albu.RandomBrightnessContrast(p=0.5),
        #albu.ToGray(p=1.0),
        albu.Resize(320, 480)#640)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        #albu.ToGray(p=1.0),
        #albu.CLAHE(p=1.0),

        albu.Resize(320, 480)#640)
    ]
    return albu.Compose(test_transform)


def get_training_augmentation_large():
    train_transform = [
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        #albu.RandomGridShuffle(p=0.5, grid=(3,3)),
        #albu.RandomCrop(p=0.5, height=250, width=325),# ADD
        #albu.GridDistortion(p=0.5),# ADD
        # albu.RandomBrightnessContrast(p=0.5),
        #albu.ToGray(p=1.0),
        albu.Resize(640, 960)#640)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation_large():
    test_transform = [
        #albu.ToGray(p=1.0),
        #albu.CLAHE(p=1.0),

        albu.Resize(640, 960)#640)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    :param probability:
    :param threshold:
    :param min_size:
    :return: predictions : mask image (shape==(350, 525)), num : is exist mask
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]  # 閾値で0,1に分ける。
    if mask.shape != (350, 525):
        mask = cv2.resize(mask, dsize=(350, 525))

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))  # 繋がっている部分をシーケンシャルにラベリングしていっている。
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:  # どのくらいpixcelが繋がっていたらmaskとするか? 最小値がmin_size
            predictions[p] = 1
            num += 1
    return predictions, num


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)
    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def estimate_dice(masks, valid_masks):
    d = []
    for i, j in zip(masks, valid_masks):
        if (i.sum() == 0) & (j.sum() == 0):
            d.append(1)
        else:
            d.append(dice(i, j))

    return np.mean(d)