import argparse
import pandas as pd
import os
from skimage.data import imread
import numpy as np
import cv2


def main():
    train_images_dir = args.train_images_dir
    infile = args.infile
    outfile = args.outfile

    train = load_train_csv(infile)
    export_flip_augmentation_png(train, train_images_dir)
    export_flip_augmentation_csv(train, outfile)


def load_train_csv(infile):
    train = pd.read_csv(infile)
    train['label'] = train.Image_Label.apply(lambda x: x.split('_')[1])
    train['file_name'] = train.Image_Label.apply(lambda x: x.split('_')[0])
    return train


def flip_image(train_images_dir, file_name, t=0):
    img = imread(f'{train_images_dir}/{file_name}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.flip(img, t)


def export_flip_augmentation_png(train, train_images_dir):
    for file_name in train.file_name.unique():
        f_img = flip_image(train_images_dir, file_name, t=0)
        cv2.imwrite(f'{train_images_dir}/f0_{file_name}', f_img)

    for file_name in train.file_name.unique():
        f_img = flip_image(train_images_dir, file_name, t=1)
        cv2.imwrite(f'{train_images_dir}/f1_{file_name}', f_img)


def export_flip_augmentation_csv(train, outfile):
    import copy

    train_f0 = copy.deepcopy(train)
    train_f1 = copy.deepcopy(train)
    train_f0['EncodedPixels'] = train.EncodedPixels.apply(lambda x: flip_rle(x, t=0))
    train_f1['EncodedPixels'] = train.EncodedPixels.apply(lambda x: flip_rle(x, t=1))
    train_f0['EncodedPixels'] = train_f0.EncodedPixels.apply(lambda x: np.nan if x == '' else x)
    train_f1['EncodedPixels'] = train_f1.EncodedPixels.apply(lambda x: np.nan if x == '' else x)
    train_f0['file_name'] = train.file_name.apply(lambda x: 'f0_' + x)
    train_f1['file_name'] = train.file_name.apply(lambda x: 'f1_' + x)
    train_f0['Image_Label'] = train.Image_Label.apply(lambda x: 'f0_' + x)
    train_f1['Image_Label'] = train.Image_Label.apply(lambda x: 'f1_' + x)
    train_f0 = train_f0.drop(['label', 'file_name'], axis=1)
    train_f1 = train_f1.drop(['label', 'file_name'], axis=1)

    train_aug = pd.concat([train, train_f0, train_f1]).drop(['file_name', 'label'], axis=1)
    train_aug.to_csv(outfile, index=False)


def resize_img_arr(img_arr):
    return cv2.resize(img_arr, dsize=(525, 350), interpolation=cv2.INTER_CUBIC)


def rle_decode(mask_rle, shape=(1400, 2100)):
    """
    文字列をimgの配列にする
    :param mask_rle:
    :param shape:
    :return:
    """
    if mask_rle is np.nan:
        return np.zeros(shape[0] * shape[1], dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def flip_rle(encoded_pixels, t=0):
    """
    EncodedPixelsを元に、反転させたEncodedPixelsを返す
    :param encoded_pixels:
    :param t:
    :return:
    """
    img = rle_decode(encoded_pixels)
    img = cv2.flip(img, t)
    return mask2rle(img)


parser = argparse.ArgumentParser(description='Cloud data flip augmentation script')
parser.add_argument('--train_images_dir', type=str, default='./data/train_images',
                    help='train_images dir')
parser.add_argument('-i', '--infile', type=argparse.FileType('r'), default='./data/train.csv',
                    help='train.csv file path')
parser.add_argument('-o', '--outfile', type=argparse.FileType('w'), default='./data/train_flip_aug.csv',
                    help='train.csv after augmentation')
args = parser.parse_args()

if __name__ == '__main__':
    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')
