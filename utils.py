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
import json
import hashlib
import os

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


def dump_as_jsonl(data: List[Dict], output_dir: str, filename: str) -> None:
    assert filename.endswith('.jsonl')
    assert isinstance(data, list)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(join(output_dir, f'{filename}'), 'w') as file:
        for line in data:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def load_jsonl_as_dict(file_path: str) -> Dict:
    objects = load_jsonl(file_path)

    # Assume all objects have the same keys
    keys = objects[0].keys()
    return {key: [obj[key] for obj in objects] for key in keys}


def load_jsonl_as_df(file_path: str) -> pd.DataFrame:
    return pd.DataFrame(load_jsonl(file_path))


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def dump_as_json_file(data: Dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(obj=data, fp=f)


def today_formatted() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def append_date(s: str) -> str:
    date = today_formatted()
    return f'{s}-{date}'


def filter_train_datasets(config: Dict) -> List[str]:
    tags = config["data"]["tags"]
    return [dataset for dataset in tags.keys() if "train" in tags[dataset]]


def filter_xai_datasets(config: Dict) -> List[str]:
    tags = config["data"]["tags"]
    return [dataset for dataset in tags.keys() if "xai" in tags[dataset]]


def on_local_platform() -> bool:
    return (
        True
        if LOCAL_PLATFORM_NAME in platform.version().split(' ')[0].split('~')[-1]
        else False
    )


def dict_hash(dictionary) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_cache_path(key: str, config: dict) -> str:
    cache_dir = generate_cache_dir(config)
    os.makedirs(cache_dir, exist_ok=True)
    dhash = dict_hash(config)
    return join(cache_dir, f'{key}-{dhash}.pkl')


def load_from_cache(key: str, config: dict):
    path = get_cache_path(key, config)
    if os.path.exists(path):
        return load_pickle(path)
    else:
        return None


def save_to_cache(key: str, data: Any, config: dict):
    path = get_cache_path(key, config)
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def generate_cache_dir(config: Dict) -> str:
    return join(config['general']['base_dir'], "cache")


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
