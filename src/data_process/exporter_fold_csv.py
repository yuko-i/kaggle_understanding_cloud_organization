import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, KFold
import argparse
from sklearn import preprocessing

FOLD = 5

noisy_id = ['1588d4c.jpg','c26c635.jpg','563fc48.jpg','c0306e5.jpg','e04fea3.jpg','fa645da.jpg','449b792.jpg',
            '41f92e5.jpg','ee0ba55.jpg']


def main():
    infile = args.infile
    fold_csv_dir = args.fold_csv_dir

    train = pd.read_csv(infile)
    train = remove_noisy_image(train)
    train_cate = mask_categorical(train)
    train = augmentation_image_filter(train)
    train = pd.merge(train, train_cate, how='left', on='file_name')
    export_fold_csv(train, fold_csv_dir)


def augmentation_image_filter(train):
    train['im_id'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))
    train_data = pd.DataFrame(train.im_id.unique(), columns=['file_name'])
    train_data['org'] = train_data.file_name.apply(lambda x:  0 if x.startswith('f1_') or x.startswith('f0_') else 1)
    return train_data[train_data['org'] == 1]


def img_label(train):
    train['im_id'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[-1])

    classes = train['label'].unique()
    train_df = train.groupby('im_id')['label'].agg(set).reset_index()
    for class_name in classes:
        train_df[class_name] = train_df['label'].map(lambda x: 1 if class_name in x else 0)
    print(train_df.head())
    img_2_ohe_vector = {img: vec for img, vec in zip(train_df['im_id'], train_df.iloc[:, 2:].values)}
    return img_2_ohe_vector


def export_fold_csv(train_data_org, fold_csv_dir):


    kf = StratifiedKFold(n_splits=FOLD)

    for i, (train_id, val_id) in enumerate(kf.split(train_data_org.file_name, train_data_org.categorical_num)):
        print(i)
        df_train_fold = train_data_org.loc[train_id][['file_name']]
        df_train_fold['f1'] = df_train_fold.file_name.apply(lambda x: 'f1_' + x)
        df_train_fold['f0'] = df_train_fold.file_name.apply(lambda x: 'f0_' + x)

        train_fold = list(df_train_fold.file_name)\
                     + list(df_train_fold.f0) \
                     + list(df_train_fold.f1)

        fold_train = pd.DataFrame(train_fold, columns={'file_name'})
        fold_train.to_csv(f'{fold_csv_dir}/train_file_fold_{i}.csv', index=False)

        fold_val = train_data_org.loc[val_id][['file_name']]
        fold_val.to_csv(f'{fold_csv_dir}/val_file_fold_{i}.csv', index=False)


def remove_noisy_image(train):
    noisy_id_f = noisy_id + list(map(lambda x: 'f1_' + x, noisy_id)) \
                          + list(map(lambda x: 'f0_' + x, noisy_id))


    noisy_id_l = list(map(lambda x: x + '_Fish', noisy_id_f)) + list(map(lambda x: x + '_Flower', noisy_id_f)) + \
                 list(map(lambda x: x + '_Gravel', noisy_id_f)) + list(map(lambda x: x + '_Sugar', noisy_id_f))

    train['noisy'] = train.Image_Label.apply(lambda x: 1 if x in noisy_id_l else 0)
    return train[train['noisy'] == 0]


def mask_categorical(train):

    train = train[~train['EncodedPixels'].isnull()]
    train['file_name'] = train['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))
    train['label'] = train['Image_Label'].map(lambda x: x.split('_')[-1])

    classes = train['label'].unique()
    train_df = train.groupby('file_name')['label'].agg(set).reset_index()

    for class_name in classes:
        train_df[class_name] = train_df['label'].map(lambda x: 1 if class_name in x else 0)

    # categorical
    train_df['categorical'] = train_df.apply(lambda x: str(x.Fish) + str(x.Flower) + str(x.Sugar) + str(x.Gravel) , axis=1)
    le = preprocessing.LabelEncoder()
    train_df['categorical_num'] = le.fit_transform(train_df['categorical'])

    return train_df[['file_name', 'categorical_num']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cloud data flip augmentation script')
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), default='./data/train_flip_aug_resize.csv')
    parser.add_argument('-o', '--fold_csv_dir', type=str, default='./data/fold_csv')
    args = parser.parse_args()

    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')
