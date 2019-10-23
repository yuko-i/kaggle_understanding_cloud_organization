import argparse
import pandas as pd
import os
from skimage.data import imread
import numpy as np
import cv2
from tqdm import tqdm

tqdm.pandas()

RESIZE = (525, 350)


def main():
    in_train_images_dir = args.in_train_images_dir
    out_train_images_dir = args.out_train_images_dir

    in_test_images_dir = args.in_test_images_dir
    out_test_images_dir = args.out_test_images_dir

    infile = args.infile
    outfile = args.outfile

    # resize train images
    export_resize_png(in_train_images_dir, out_train_images_dir)
    # resize test images
    export_resize_png(in_test_images_dir, out_test_images_dir)
    # rewrite　train.csv to resize data
    train = load_train_csv(infile)
    export_resize_csv(train, outfile)


def export_resize_png(in_images_dir, out_images_dir):
    for file_name in tqdm(os.listdir(in_images_dir)):
        img = imread(f'{in_images_dir}/{file_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        LinerImg = cv2.resize(img, RESIZE)
        cv2.imwrite(f'{out_images_dir}/{file_name}', LinerImg)


def load_train_csv(infile):
    train = pd.read_csv(infile)
    return train


def export_resize_csv(train, outfile):
    train['EncodedPixels'] = train.EncodedPixels.progress_apply(lambda x: resize_rle(x))
    train.to_csv(outfile, index=False)


def resize_rle(encoded_pixels):
    if pd.isnull(encoded_pixels):
        return encoded_pixels
    img = rle_decode(encoded_pixels)
    img = cv2.resize(img, RESIZE)
    return mask2rle(img)


def resize_img_arr(img_arr):
    return cv2.resize(img_arr, dsize=RESIZE, interpolation=cv2.INTER_CUBIC)


def rle_decode(mask_rle, shape=(1400, 2100)):
    # 文字列をimgの配列にする

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


parser = argparse.ArgumentParser(description='Cloud data flip augmentation script')

parser.add_argument('-ir', '--in_train_images_dir', type=str, default='./data/train_images',
                    help='train_images dir')
parser.add_argument('-or', '--out_train_images_dir', type=str, default='./data/train_images_resize',
                    help='train_images dir after resize')

parser.add_argument('-ie', '--in_test_images_dir', type=str, default='./data/test_images',
                    help='test_images dir')
parser.add_argument('-oe', '--out_test_images_dir', type=str, default='./data/test_images_resize',
                    help='test_images dir after resize')

parser.add_argument('-i', '--infile', type=argparse.FileType('r'), default='./data/train_flip_aug.csv',
                    help='train.csv file path')
parser.add_argument('-o', '--outfile', type=argparse.FileType('w'), default='./data/train_flip_aug_resize.csv',
                    help='train.csv after resize')

args = parser.parse_args()

if __name__ == '__main__':
    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')
