import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from util.mask_util import rle_decode, make_mask

import sys

sys.path.append('../')

from cloud_util import mask2rle


def get_majority_mask(i):
    s_r_0 = sub_re_0.iloc[i]['mask']
    s_r_1 = sub_re_1.iloc[i]['mask']
    s_r_2 = sub_re_2.iloc[i]['mask']

    stack_arr = np.stack([s_r_0, s_r_1, s_r_2], 2)

    sum_arr = stack_arr.sum(axis=2)

    majority_mask = np.where(sum_arr >= MAJORITY_CNT, 1, 0)
    return majority_mask


sub_re_0 = pd.read_csv('submission_resnet34_fold_0.csv')
sub_re_1 = pd.read_csv('submission_resnet34_fold_1.csv')
sub_re_2 = pd.read_csv('submission_resnet34_fold_2.csv')

sub_re_0['mask'] = sub_re_0.EncodedPixels.progress_apply(lambda x: rle_decode(x))
sub_re_1['mask'] = sub_re_1.EncodedPixels.progress_apply(lambda x: rle_decode(x))
sub_re_2['mask'] = sub_re_2.EncodedPixels.progress_apply(lambda x: rle_decode(x))

sub_re_0 = sub_re_0.reset_index()

MAJORITY_CNT = 2

sub_re_0['majority_mask'] = sub_re_0['index'].progress_apply(lambda x: get_majority_mask(x))
sub_re_0 = sub_re_0.rename(columns={'EncodedPixels': 'EncodedPixels_org'})
sub_re_0['EncodedPixels'] = sub_re_0.majority_mask.progress_apply(lambda x: mask2rle(x))
sub_re_0['EncodedPixels'] = sub_re_0.EncodedPixels.apply(lambda x: np.nan if len(x) == 0 else x)

sub_re_0[['Image_Label', 'EncodedPixels']].to_csv('Unet_resnet34_fold_012_search_threshold.csv', index=False)
