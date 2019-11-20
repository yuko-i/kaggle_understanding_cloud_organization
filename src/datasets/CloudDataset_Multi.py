import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as albu
#from albumentations import torch as AT

import sys
sys.path.append('../')

import torch
from config.config import TRAIN_IMG_PATH, TEST_IMG_PATH


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (350, 525)):
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
    return masks

def make_mask_label_label(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (350, 525)):
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    label_data = df.loc[df['im_id'] == image_name, 'Image_Label'].values[0]
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
    return masks, label_data


def rle_decode(mask_rle: str = '', shape: tuple = (350, 525)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


class CloudDataset_Multi(Dataset):
    def __init__(self,
                 df: pd.DataFrame = None,
                 datatype: str = 'train',
                 img_ids: np.array = None,
                 transforms=None,
                 preprocessing=None):

        self.df = df
        self.datatype = datatype
        if self.datatype != 'test':
            self.data_folder = f"{TRAIN_IMG_PATH}"
        else:
            self.data_folder = f"{TEST_IMG_PATH}"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
       #mask, label = make_mask_label_label(self.df, image_name)

        if self.datatype != 'test':
            mask, label = make_mask_label_label(self.df, image_name)

        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']

        label_ts = torch.zeros(4)
        if self.datatype != 'test':
            for cls in [i for i, x in enumerate(label) if x == 1]:
                label_ts[int(cls)] = 1
        print(label_ts.shape)
        print(img.shape)
        return img, mask, label_ts

    def __len__(self):
        return len(self.img_ids)


