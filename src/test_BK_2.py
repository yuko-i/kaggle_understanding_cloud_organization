import argparse
import pandas as pd
import numpy as np
from cloud_util import get_preprocessing, get_validation_augmentation, \
    get_training_augmentation, estimate_dice, post_process, dice, mask2rle

import sys

sys.path.append('../')
from datasets.CloudDataset import CloudDataset

from util.mask_util import rle_decode
import segmentation_models_pytorch as smp
from torch.utils.data import TensorDataset, DataLoader, Dataset
from optimizer.RAdam import RAdam
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl.core import Callback, RunnerState
from tqdm import tqdm
import ttach as tta
import cv2
import os
import torch

sigmoid = lambda x: 1 / (1 + np.exp(-x))

CLASS = 4
IMG_SIZE = (350, 525)


def main():
    fold_path = args.fold_path
    fold_num = args.fold_num
    model_name = args.model_name
    train_csv = args.train_csv
    sub_csv = args.sub_csv
    encoder = args.encoder
    num_workers = args.num_workers
    batch_size = args.batch_size
    log_path = args.log_path
    is_tta = args.is_tta

    print(log_path)

    train = pd.read_csv(train_csv)
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[-1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    val_fold = pd.read_csv(f'{fold_path}/val_file_fold_{fold_num}.csv')
    valid_ids = np.array(val_fold.file_name)

    if model_name == 'Unet':
        encoder_weights = 'imagenet'
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=CLASS,
            activation='softmax',
        )

        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    valid_dataset = CloudDataset(df=train,
                                 datatype='valid',
                                 img_ids=valid_ids,
                                 transforms=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loaders = {"infer": valid_loader}
    runner = SupervisedRunner()

    if is_tta:
        print('TTA')
        checkpoint = torch.load(f"{log_path}/checkpoints/best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
        runner.infer(
            model=model,
            loaders=loaders,
            callbacks=[InferCallback()],
        )
        callbacks_num = 0
    else:
        print('not TTA')
        runner.infer(
            model=model,
            loaders=loaders,
            callbacks=[
                CheckpointCallback(resume=f"{log_path}/checkpoints/best.pth"),
                InferCallback()
            ],
        )
        callbacks_num = 1

    valid_masks = []
    probabilities = np.zeros((valid_dataset.__len__() * CLASS, IMG_SIZE[0], IMG_SIZE[1]))

    # ========
    # val predict
    #

    for batch in tqdm(valid_dataset):  # クラスごとの予測値
        _, mask = batch
        for m in mask:
            m = resize_img(m)
            valid_masks.append(m)

    for i, output in enumerate(tqdm(runner.callbacks[callbacks_num].predictions["logits"])):
        for j, probability in enumerate(output):
            probability = resize_img(probability)  # 各クラスごとにprobability(予測値)が取り出されている。jは0~3だと思う。
            probabilities[i * CLASS + j, :, :] = probability

    # ========
    # search best size and threshold
    #

    class_params = {}
    for class_id in range(CLASS):
        attempts = []
        for threshold in range(20, 90, 5):
            threshold /= 100
            for min_size in [10000, 15000, 20000]:
                masks = class_masks(class_id, probabilities, threshold, min_size)
                dices = class_dices(class_id, masks, valid_masks)
                attempts.append((threshold, min_size, np.mean(dices)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
        attempts_df = attempts_df.sort_values('dice', ascending=False)

        print(attempts_df.head())

        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)

    # ========
    # predict
    #
    sub = pd.read_csv(sub_csv)
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[-1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    test_dataset = CloudDataset(df=sub,
                                datatype='test',
                                img_ids=test_ids,
                                transforms=get_validation_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn))

    encoded_pixels = get_test_encoded_pixels(test_dataset, runner, class_params)
    sub['EncodedPixels'] = encoded_pixels

    if is_tta:
        sub.to_csv(f'./sub/sub_unet_fold_{fold_num}_{encoder}_tta.csv', columns=['Image_Label', 'EncodedPixels'],
                   index=False)
    else:
        sub.to_csv(f'./sub/sub_unet_fold_{fold_num}_{encoder}.csv', columns=['Image_Label', 'EncodedPixels'],
                   index=False)


def resize_img(img):
    if img.shape != IMG_SIZE:
        return cv2.resize(img, dsize=(IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    return img


def class_masks(class_id, probabilities, threshold, min_size):
    masks = []
    for i in range(class_id, len(probabilities), CLASS):
        probability = probabilities[i]
        predict, _ = post_process(sigmoid(probability), threshold, min_size)
        masks.append(predict)
    return masks


def class_dices(class_id, masks, valid_masks):
    dices = []
    for i, j in zip(masks, valid_masks[class_id::CLASS]):
        if (i.sum() == 0) & (j.sum() == 0):
            dices.append(1)
        else:
            dices.append(dice(i, j))
    return dices


def get_test_encoded_pixels(test_dataset, runner, class_params):
    test_loader = DataLoader(test_dataset, batch_size=4,#8,
                             shuffle=False, num_workers=0)
    loaders = {"test": test_loader}

    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(tqdm(loaders['test'])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        for i, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                probability = resize_img(probability)

                predict, is_exist_mask = post_process(sigmoid(probability),
                                                      class_params[image_id % CLASS][0],
                                                      class_params[image_id % CLASS][1])
                if is_exist_mask == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)

                image_id += 1

    return encoded_pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cloud data flip augmentation script')
    parser.add_argument('-fp', '--fold_path', type=str, default='./data_process/data/fold_csv')
    parser.add_argument('-tr', '--train_csv', type=argparse.FileType('r'),
                        default='./data_process/data/train_flip_aug_resize.csv')
    parser.add_argument('-sb', '--sub_csv', type=argparse.FileType('r'),
                        default='./data_process/data/sample_submission.csv')
    parser.add_argument('-l', '--log_path', type=str, default='./log/logs_unet_fold_0_resnet34/segmentation')
    parser.add_argument('-fn', '--fold_num', type=int, default=0)
    parser.add_argument('-mo', '--model_name', type=str, default='Unet')
    parser.add_argument('-ec', '--encoder', type=str, default='resnet34')
    parser.add_argument('-nw', '--num_workers', type=int, default=16)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-tt', '--is_tta', type=int, default=1)
    args = parser.parse_args()

    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')
