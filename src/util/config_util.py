import yaml
from easydict import EasyDict as edict
from config import config


def _get_default_config():
    with open(config.BASE_CONFIG_YML,  'r') as fid:
        yaml_base_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))
    return yaml_base_config


def _merge_config(src: edict, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v

def load_config(config_path: str) -> edict:

    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config: edict, file_name: str) -> None:
    with open(file_name, "w") as wf:
        yaml.dump(config, wf)