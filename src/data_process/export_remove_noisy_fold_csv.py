import pandas as pd

FOLD_CSV_PATH = '/home/yuko/kaggle_understanding_cloud_organization/src/data_process/data/fold_csv'

def main():
    noisy = pd.read_csv('/home/yuko/kaggle_understanding_cloud_organization/src/data_process/data/noisy_id.csv')
    noisy = noisy.rename(columns={'im_id': 'file_name'})
    noisy_list = list(noisy.file_name.values)

    for i in range(5):
        print(i)
        export_remove_noisy_fold_csv(noisy_list = noisy_list,
                                     train_fold_file_path = f'{FOLD_CSV_PATH}/train_file_fold_{i}.csv',
                                     export_path =  f'{FOLD_CSV_PATH}/train_file_clean_fold_{i}.csv')

def export_remove_noisy_fold_csv(noisy_list, train_fold_file_path, export_path):

    train_fold = pd.read_csv(train_fold_file_path)
    fold_list = list(train_fold.file_name.values)
    fold_list_c = [x for x in fold_list if x not in noisy_list]
    pd.DataFrame({'file_name': fold_list_c}).to_csv(export_path)

if __name__ == '__main__':
    main()
    print('success')