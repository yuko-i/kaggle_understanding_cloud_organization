import argparse
import cv2
import os
import pandas as pd
import numpy as np
import albumentations as albu
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as model
from sklearn.model_selection import train_test_split
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback
from catalyst.dl.runner import SupervisedRunner
from cloud_util import get_preprocessing, get_validation_augmentation, get_training_augmentation, to_tensor
from optimizer.RAdam import RAdam

TRAIN_IMG_PATH = './data_process/data/train_images_resize/'


def main():
    train = pd.read_csv('./data_process/data/train_flip_aug_resize.csv')

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[-1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    train['img_label'] = train.EncodedPixels.apply(lambda x: 0 if x is np.nan else 1)

    img_label = train.groupby('im_id')['img_label'].agg(list).reset_index()

    image_train, image_val, label_train, label_val = train_test_split(
        np.array(img_label.im_id),
        np.array(img_label.img_label),
        test_size=0.2, random_state=42)

    train_dataset = CloudClassDataset(
        datatype='train',
        img_ids=image_train,
        img_labels=label_train,
        transforms=get_training_augmentation(),
        preprocessing=ort_get_preprocessing()
    )

    valid_dataset = CloudClassDataset(
        datatype='train',
        img_ids=image_val,
        img_labels=label_val,
        transforms=get_validation_augmentation(),
        preprocessing=ort_get_preprocessing()
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8)

    resnet_model = ResNet()

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    logdir = f"./class/segmentation"

    print(logdir)

    optimizer = RAdam([
        {'params': resnet_model.parameters(), 'lr': 1e-2},
    ])

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    criterion = nn.BCEWithLogitsLoss()
    runner = SupervisedRunner()

    runner.train(
        model=resnet_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[EarlyStoppingCallback(patience=5, min_delta=1e-7)],
        logdir=logdir,
        num_epochs=15,
        verbose=1
    )


class CloudClassDataset(Dataset):
    def __init__(self,
                 datatype: str = 'train',
                 img_ids: np.array = None,
                 img_labels: np.array = None,
                 transforms=None,
                 preprocessing=None
                 ):

        if datatype != 'test':
            self.data_folder = f"{TRAIN_IMG_PATH}"
        else:
            self.data_folder = f"{TEST_IMG_PATH}"
        self.img_ids = img_ids
        self.img_labels = img_labels
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        img_label = self.img_labels[idx]

        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(image=img)['image']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']

        target = torch.zeros(4)

        for cls in [i for i, x in enumerate(img_label) if x == 1]:
            target[int(cls)] = 1

        return img, target

    def __len__(self):
        return len(self.img_ids)


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet(nn.Module):
    def __init__(self,
                 num_classes=4,
                 pre_trained=True,
                 dropout=False):
        super().__init__()
        self.net = model.resnet18(pretrained=pre_trained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


def ort_get_preprocessing():
    _transform = [
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cloud data flip augmentation script')
    parser.add_argument('-fp', '--fold_path', type=str, default='./data_process/data/fold_csv')
    parser.add_argument('-tr', '--train_csv', type=argparse.FileType('r'),
                        default='./data_process/data/train_flip_aug_resize.csv')
    parser.add_argument('-sb', '--sub_csv', type=argparse.FileType('r'),
                        default='./data_process/data/sample_submission.csv')
    parser.add_argument('-fn', '--fold_num', type=int, default=0)
    parser.add_argument('-mo', '--model_name', type=str, default='Unet')
    parser.add_argument('-ec', '--encoder', type=str, default='resnet34')
    parser.add_argument('-nw', '--num_workers', type=int, default=16)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-ep', '--num_epochs', type=int, default=25)
    parser.add_argument('-lr', '--learn_late', type=float, default=1e-3)
    args = parser.parse_args()

    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')

    from subprocess import check_call

    check_call(['/home/yuko/kaggle_understanding_cloud_organization/src/SD.sh'], shell=True)
