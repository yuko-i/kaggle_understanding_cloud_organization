import argparse
import os
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
import sys
sys.path.append('../')
from datasets.CloudDataset import CloudDataset
from datasets.CloudDataset_Multi import CloudDataset_Multi

from loss.MixLoss import MixLoss
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from optimizer.NAdam import Nadam
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from cloud_util import get_preprocessing, get_validation_augmentation, get_training_augmentation
from model.Linknet_Classifer import Linknet_resnet18_Classifer, Multi_Loss
from model.ASPP import Linknet_resnet18_ASPP


def main():

    fold_path = args.fold_path
    fold_num = args.fold_num
    model_name = args.model_name
    train_csv = args.train_csv
    sub_csv = args.sub_csv
    encoder = args.encoder
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learn_late = args.learn_late
    attention_type = args.attention_type

    train = pd.read_csv(train_csv)
    sub = pd.read_csv(sub_csv)

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[-1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[-1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))


    train_fold = pd.read_csv(f'{fold_path}/train_file_fold_{fold_num}.csv')
    val_fold = pd.read_csv(f'{fold_path}/val_file_fold_{fold_num}.csv')

    train_ids = np.array(train_fold.file_name)
    valid_ids = np.array(val_fold.file_name)

    encoder_weights = 'imagenet'

    if  model_name ==  'ORG_Link18':
        model = Linknet_resnet18_Classifer()


    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = CloudDataset_Multi(df=train, datatype='train', img_ids=train_ids,
                                 transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

    valid_dataset = CloudDataset_Multi(df=train, datatype='valid', img_ids=valid_ids,
                                 transforms=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    logdir = f"./log/logs_{model_name}_fold_{fold_num}_{encoder}/segmentation"

    print(logdir)

    if model_name ==  'ORG_Link18':
        optimizer = Nadam([
            {'params': model.parameters(), 'lr': learn_late},
        ])
    else:
        optimizer = Nadam([
            {'params': model.decoder.parameters(), 'lr': learn_late},
            {'params': model.encoder.parameters(), 'lr': learn_late},
        ])


    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    criterion = Multi_Loss()

    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[EarlyStoppingCallback(patience=5, min_delta=1e-7)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=1
    )


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
    parser.add_argument('-at', '--attention_type', type=str, default='scse')
    args = parser.parse_args()

    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')

    from subprocess import check_call
    check_call(['/home/yuko/kaggle_understanding_cloud_organization/src/SD.sh'], shell=True)
