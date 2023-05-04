import json
import platform
from datetime import datetime
from os.path import join
from pathlib import Path
import random
from typing import Any, Dict, List
import pickle

import numpy as np
import torch
from numpy.random import Generator
from torch.utils.data import TensorDataset, random_split

LOCAL_PLATFORM_NAME = 'PREEMPT_DYNAMIC'
CLUSTER_DATA_DIR = '/mnt'


def load_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def dump_as_pickle(data: Any, output_dir: str, filename: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(join(output_dir, f'{filename}'), 'wb') as file:
        pickle.dump(data, file)


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def dump_as_json_file(data: Dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(obj=data, fp=f)


def append_date(s: str) -> str:
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f'{s}-{date}'


def get_root_dir_based_on_platform() -> str:
    return '' if LOCAL_PLATFORM_NAME in platform.version() else CLUSTER_DATA_DIR


def generate_data_dir(config: Dict) -> str:
    return join(
        get_root_dir_based_on_platform(),
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['data']['output_dir']
    )


def generate_training_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['training']['output_dir']
    )


def generate_xai_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['xai']['output_dir']
    )


def generate_evaluation_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['evaluation']['output_dir']
    )


def set_random_states(seed: int) -> Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)


def create_train_val_split(data: TensorDataset, val_size: float) -> List:
    num_samples = len(data)
    num_val_samples = int(val_size * num_samples)
    num_train_samples = num_samples - num_val_samples
    return random_split(
        dataset=data,
        lengths=[num_train_samples, num_val_samples]
    )
