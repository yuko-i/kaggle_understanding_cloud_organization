from util.config_util import load_config
from transforms.transforms import get_transforms
import argparse
from datasets.dataset_factory import make_dataset
from model.model_factory import make_model
from optimizer.optimizer_factory import get_optimizer
import segmentation_models_pytorch as smp
from util.log_util import get_train_logger
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics.dice import dice_coeff
from runner.Runner import Runner
import os

def run(config_file):
    config = load_config(config_file)
    logger, log_dir = get_train_logger(config_file, config)

    all_transforms = {}
    all_transforms['train'] = get_transforms(config.transforms.train)
    all_transforms['valid'] = get_transforms(config.transforms.test)

    dataloaders = {
        phase: make_dataset(
            data_folder=config.data.train_dir,
            df_path=config.data.train_df_path,
            fold_dir=config.data.fold_dir,
            phase=phase,
            fold=config.data.params.fold,
            img_shape=(config.data.height, config.data.width),
            transforms=all_transforms[phase],
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
        )
        for phase in ['train', 'valid']
    }

    """
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )
    """

    model = make_model(
        model_name=config.model.model_name,
        encoder=config.model.encoder,
        decoder=config.model.decoder,
        class_num=config.data.num_classes
    )

    params = [
        {'params': model.parameters(), 'lr': config.optimizer.params.lr},
        #{'params': model.encoder.parameters(), 'lr': config.optimizer.params.encoder_lr},
    ]

    optimizer = get_optimizer(config.optimizer.name, params)
    criterion = smp.utils.losses.BCEDiceLoss()
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    #if config.train.mixup:
    #    callbacks.append(MixupCallback())

    #if config.train.cutmix:
    #    callbacks.append(CutMixCallback())

    device = torch.device(config.device)
    model.to(device)

    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = config.devices

    runner = Runner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=dataloaders,
        logger=logger,
        metrics=dice_coeff,
        scheduler=scheduler,
        device=device,
        epoch_num=config.train.epoch,
        log_dir = log_dir
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str,
                        default='./config/config_yml/seg1_config.yml')
    return parser.parse_args()

def main():
    args = parse_args()
    run(args.config_file)

if __name__ == '__main__':
    main()