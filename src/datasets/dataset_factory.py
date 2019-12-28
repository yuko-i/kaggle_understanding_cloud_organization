from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import cv2
import os
from util.mask_util import make_mask
from albumentations import Compose
from cloud_util import to_tensor

class CloudDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame = None,
                 data_folder: str = '',
                 img_names: np.array = None,
                 img_shape: tuple = (),
                 transforms: Compose = None):
        self.df = df
        self.data_folder = data_folder
        self.img_names = img_names
        self.img_shape = img_shape
        self.transforms = transforms

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        mask = make_mask(self.df, img_name, shape=self.img_shape)

        image_path = os.path.join(self.data_folder, img_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=img, mask=mask)

        img = np.transpose(augmented['image'], [2, 0, 1])
        mask = np.transpose(augmented['mask'], [2, 0, 1])

        return img, mask

    def __len__(self):
        return len(self.img_names)

def make_dataset(
        data_folder: str,
        df_path: str,
        fold_dir: str,
        phase: str = 'train',
        fold: int = 0,
        img_shape: tuple =(),
        transforms: Compose = None,
        batch_size: int = 16,
        num_workers: int = 16) -> DataLoader:

    df = pd.read_csv(df_path)
    df['im_id'] = df['Image_Label'].map(lambda x: x.split('_')[0])

    df_fold = pd.read_csv(f'{fold_dir}/{phase}_file_fold_{fold}.csv')
    img_names = np.array(df_fold.file_name)

    if phase in ['train', 'valid']:
        print(phase)

        image_dataset = CloudDataset(
            df=df,
            img_names=img_names,
            data_folder=data_folder,
            img_shape=img_shape,
            transforms=transforms)

        is_shuffle = phase == 'train'

    else:
        # Test
        print(phase)

        image_dataset = CloudDataset(
            df=df,
            img_names=img_names,
            data_folder=data_folder,
            img_shape=img_shape,
            transforms=transforms)
        is_shuffle = False

    return DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=is_shuffle,
        )