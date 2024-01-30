import json
import platform
from datetime import datetime
from os.path import join
from pathlib import Path
import random
from typing import Any, Dict, List
import pickle

import pandas as pd
import numpy as np
import torch
from numpy.random import Generator
from torch.utils.data import TensorDataset, random_split

from common import DATASET_ALL, DATASET_SUBJECT, validate_dataset_key

LOCAL_PLATFORM_NAME = '22.04.1-Ubuntu'
LOCAL_DIR = ''


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


def on_local_platform() -> bool:
    return (
        True
        if LOCAL_PLATFORM_NAME in platform.version().split(' ')[0].split('~')[-1]
        else False
    )


def generate_data_dir(config: Dict) -> str:
    return join(
        LOCAL_DIR if on_local_platform() else config['general']['apptainer_data_dir'],
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['data']['output_dir'],
    )


def generate_training_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['training']['output_dir'],
    )


def generate_xai_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['xai']['output_dir'],
    )


def generate_evaluation_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['evaluation']['output_dir'],
    )


def generate_visualization_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['visualization']['output_dir'],
    )


def load_test_data(config: dict) -> dict[pd.DataFrame]:
    data = dict()
    data_dir = generate_data_dir(config=config)
    for dataset in config["xai"]["datasets"]:
        validate_dataset_key(dataset_key=dataset)
        filename_all = config['data']["datasets"][dataset]['output_filenames']['test']
        data[dataset] = load_pickle(file_path=join(data_dir, dataset, filename_all))

    return data


def set_random_states(seed: int) -> Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)


def create_train_val_split(data: TensorDataset, val_size: float) -> List:
    num_samples = len(data)
    num_val_samples = int(val_size * num_samples)
    num_train_samples = num_samples - num_val_samples
    return random_split(dataset=data, lengths=[num_train_samples, num_val_samples])


def determine_dataset_type(dataset_name: str) -> str:
    output = DATASET_ALL
    if DATASET_SUBJECT in dataset_name:
        output = DATASET_SUBJECT
    return output


def load_model(path: str) -> Any:
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    model.zero_grad()
    return model
