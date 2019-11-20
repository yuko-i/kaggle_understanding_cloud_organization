import pandas as pd
import numpy as np
from train_classification import CloudClassDataset, ort_get_preprocessing, ResNet
from cloud_util import get_preprocessing, get_validation_augmentation, get_training_augmentation, to_tensor
from catalyst.dl.callbacks import InferCallback
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner
import torch
import os

def main():
    test = pd.read_csv('/home/yuko/kaggle_understanding_cloud_organization/src/data_process/data/sample_submission.csv')

    test['label'] = test['Image_Label'].apply(lambda x: x.split('_')[-1])
    test['im_id'] = test['Image_Label'].apply(lambda x: x.replace('_' + x.split('_')[-1], ''))

    test['img_label'] = test.EncodedPixels.apply(lambda x: 0 if x is np.nan else 1)

    img_label = test.groupby('im_id')['img_label'].agg(list).reset_index()

    test_id = np.array(img_label.im_id)

    test_dataset = CloudClassDataset(
        datatype='test',
        img_ids=test_id,
        transforms=get_validation_augmentation(),
        preprocessing=ort_get_preprocessing()
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16)

    loaders = {"infer": test_loader}

    for fold in range(5):
        runner = SupervisedRunner()
        clf_model = ResNet()

        checkpoint = torch.load(f'/home/yuko/kaggle_understanding_cloud_organization/src/class/segmentation/fold_{fold}/checkpoints/best.pth')
        clf_model.load_state_dict(checkpoint['model_state_dict'])
        clf_model.eval()
        runner.infer(
            model=clf_model,
            loaders=loaders,
            callbacks=[InferCallback()],
        )
        callbacks_num = 0
        pred = runner.callbacks[callbacks_num].predictions["logits"]

        df_pred = pd.DataFrame([pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]]).T
        df_pred.to_csv(f'/home/yuko/kaggle_understanding_cloud_organization/src/class/segmentation/pred_{fold}.csv')

if __name__ == '__main__':
    print(f'{os.path.basename(__file__)}: start main function')
    main()
    print('success')