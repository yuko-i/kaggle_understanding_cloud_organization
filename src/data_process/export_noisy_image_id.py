import numpy as np
import pandas as pd
import sys
import os
sys.path.append('../')

from datasets.CloudDataset import make_mask
from cloud_util import estimate_dice
from tqdm import tqdm

tqdm.pandas()
PATH = '/home/yuko/kaggle_understanding_cloud_organization/src/val'

def main():

    val_0 = noisye_id(val_csv=f'{PATH}/val_unet_fold_0_resnet18.csv')
    val_1 = noisye_id(val_csv=f'{PATH}/val_unet_fold_1_resnet18.csv')
    val_2 = noisye_id(val_csv=f'{PATH}/val_unet_fold_2_se_resnet50.csv')
    val_3 = noisye_id(val_csv=f'{PATH}/val_unet_fold_3_resnet34.csv')
    val_4 = noisye_id(val_csv=f'{PATH}/val_unet_fold_4_se_resnet50.csv')
    df_noisey = pd.concat([val_0, val_1, val_2, val_3, val_4])
    df_noisey = df_noisey.rename(columns={'im_id_x':'im_id'})
    df_noisey.to_csv('noisye_id.csv')


def noisye_id(val_csv, train_csv='../../src/data_process/data/train_flip_aug_resize.csv'):
    print(f'start export {export_csv} ==================================')
    val = pd.read_csv(val_csv)
    train = pd.read_csv(train_csv)

    val['label'] = val['Image_Label'].apply(lambda x: x.split('_')[-1])
    val['im_id'] = val['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[-1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    train = pd.merge(val[['Image_Label']], train, how='left', on='Image_Label')

    val['val_mask'] = val.im_id.progress_apply(lambda x: make_mask(val, x))
    train['train_mask'] = train.im_id.progress_apply(lambda x: make_mask(train, x))

    df = pd.merge(val, train, how='left', on='Image_Label')
    df_d = df.drop_duplicates(subset=['im_id_x'])

    df_d['dice_Fish'] = df_d.progress_apply(lambda x: estimate_dice([x.val_mask[:, :, 0]], [x.train_mask[:, :, 0]]),
                                            axis=1)
    df_d['dice_Flower'] = df_d.progress_apply(lambda x: estimate_dice([x.val_mask[:, :, 1]], [x.train_mask[:, :, 1]]),
                                              axis=1)
    df_d['dice_Gravel'] = df_d.progress_apply(lambda x: estimate_dice([x.val_mask[:, :, 2]], [x.train_mask[:, :, 2]]),
                                              axis=1)
    df_d['dice_Sugar'] = df_d.progress_apply(lambda x: estimate_dice([x.val_mask[:, :, 3]], [x.train_mask[:, :, 3]]),
                                             axis=1)

    df_d['mean_dice'] = df_d.apply(lambda x: np.mean([x.dice_Fish, x.dice_Flower, x.dice_Gravel, x.dice_Sugar]), axis=1)

    df_d_r = df_d.reset_index()

    return df_d_r[df_d_r['mean_dice'] < 0.3][['im_id_x']]#.to_csv(export_csv)


if __name__ == '__main__':
    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')