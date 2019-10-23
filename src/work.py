import pandas as pd

org = pd.read_csv('/home/pridegoodmusic/Kaggle/kaggle_compe/Cloud/Argmentation/train.csv')
new_ = pd.read_csv('/home/yuko/kaggle_understanding_cloud_organization/src/data_process/data/train_aug.csv')

print(org.equals(new_))
print(org == new_)