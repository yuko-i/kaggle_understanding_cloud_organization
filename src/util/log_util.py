from logging import getLogger, StreamHandler, DEBUG, FileHandler, Logger
import os
import datetime
from easydict import EasyDict
from util.util import get_file_name, get_now


def get_train_logger(config_file: str, config: EasyDict) -> (Logger, str):
    conf_file = get_file_name(config_file)

    log_dir = f'{config.data.log_dir}/{conf_file}'
    log_path = f'{log_dir}/log.log'

    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(output_file_path=log_path)
    return logger, log_dir


def get_logger(output_file_path: str) -> Logger:
    logger = getLogger(__name__)
    handler = FileHandler(filename=output_file_path)
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger