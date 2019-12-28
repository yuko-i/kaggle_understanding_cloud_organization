import numpy as np
import cv2
import pandas as pd


def rle_decode(mask_rle: str = '', shape: tuple = (350, 525)):
    """
    :param mask_rle:
    :param shape:
    :return:
    """

    if pd.isnull(mask_rle):
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (350, 525)):
    """
    :param df:
    :param image_name:
    :param shape:
    :return:
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(mask_rle=label, shape=shape)
            masks[:, :, idx] = mask

    return masks