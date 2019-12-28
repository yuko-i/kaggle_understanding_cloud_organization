from util.config_util import load_config
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import argparse

def split_folds(config_file: str):

    config = load_config(config_file)
    df = pd.read_csv(config.data.train_df_path)

    df = df[~df['EncodedPixels'].isnull()]
    df['file_name'] = df['Image_Label'].map(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].map(lambda x: x.split('_')[-1])

    classes = df['label'].unique()
    df_g = df.groupby('file_name')['label'].agg(set).reset_index()

    for class_name in classes:
        df_g[class_name] = df_g['label'].map(
            lambda x: 1 if class_name in x else 0)

    # categorical
    df_g['categorical'] = df_g.apply(
        lambda x: str(x.Fish) + str(x.Flower) +
                  str(x.Sugar) + str(x.Gravel) , axis=1)

    le = preprocessing.LabelEncoder()
    df_g['categorical_num'] = le.fit_transform(df_g['categorical'])

    df = pd.merge(df, df_g[['file_name', 'categorical_num']],
                  how='left', on='file_name')

    kf = StratifiedKFold(n_splits=config.data.params.num_folds)
    for i, (train_id, val_id) in enumerate(
            kf.split(df.file_name, df.categorical_num)):

        train = df.loc[train_id][['file_name']].drop_duplicates()
        train.to_csv(f'{config.data.fold_dir}/'
                     f'{config.data.fold_train_file}_{i}.csv',
                     index=False)

        val = df.loc[val_id][['file_name']].drop_duplicates()
        val.to_csv(f'{config.data.fold_dir}/'
                   f'{config.data.fold_valid_file}_{i}.csv',
                   index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str,
                        default='./config/config_yml/seg1_config.yml')
    return parser.parse_args()

def main():
    args = parse_args()
    split_folds(args.config_file)

if __name__ == '__main__':
    main()
