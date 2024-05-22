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

BERT_MODEL_TYPE = 'bert'
ONE_LAYER_ATTENTION_MODEL_TYPE = 'one_layer_attention'


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


def filter_eval_datasets(config: Dict) -> List[str]:
    tags = config["data"]["tags"]
    return [dataset for dataset in tags.keys() if "eval" in tags[dataset]]


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


def cache_dec(save_path: str, recalc=False):
    def dec_func(func):
        def f(*args, **kwargs):
            if os.path.exists(save_path) and not recalc:
                return load_pickle(save_path)
            else:
                result = func(*args, **kwargs)
                with open(save_path, "wb") as f:
                    pickle.dump(result, f)
                return result

        return f

    return dec_func


def is_hydra():
    return os.environ.get("SLURM_WORKING_CLUSTER", "").startswith("hydra")


def generate_data_dir(config: Dict) -> str:
    base_dir = config["data"]["data_dir"]
    if is_hydra():
        base_dir = "/mnt/data"
    return join(base_dir, config["data"]["data_scenario"])


def generate_artifacts_dir(config: Dict) -> str:
    if is_hydra():
        return "/mnt/artifacts"

    return config["general"]["artifacts_dir"]


def generate_project_dir(config: Dict) -> str:
    if is_hydra():
        return "/workdir"

    return config["general"]["project_dir"]


def generate_cache_dir(config: Dict) -> str:
    return join(generate_artifacts_dir(config), "cache")


def generate_training_dir(config: Dict) -> str:
    return config['training']['output_dir']


def generate_xai_dir(config: Dict) -> str:
    return config['xai']['output_dir']


def generate_evaluation_dir(config: Dict) -> str:
    return config['evaluation']['output_dir']


def generate_bias_dir(config: Dict) -> str:
    return join(
        generate_artifacts_dir(config),
        config['bias']['output_dir'],
    )


def generate_visualization_dir(config: Dict) -> str:
    return join(
        generate_artifacts_dir(config),
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


def determine_model_type(s: str) -> str:
    result = None
    if BERT_MODEL_TYPE in s:
        result = BERT_MODEL_TYPE
    elif ONE_LAYER_ATTENTION_MODEL_TYPE in s:
        result = ONE_LAYER_ATTENTION_MODEL_TYPE
    return result


def cache_dec(save_path: str, recalc: bool = False):
    def dec_func(func):
        def f(*args, **kwargs):
            if os.path.exists(save_path) and not recalc:
                return load_pickle(save_path)
            else:
                result = func(*args, **kwargs)
                with open(save_path, "wb") as f:
                    pickle.dump(result, f)
                return result

        return f

    return dec_func


def cache_dec_key(key: callable, recalc: bool = False):
    def dec_func(func):
        def f(*args, **kwargs):
            save_path = key(*args, **kwargs)

            if os.path.exists(save_path) and not recalc:
                return load_pickle(save_path)
            else:
                result = func(*args, **kwargs)
                with open(save_path, "wb") as f:
                    pickle.dump(result, f)
                return result

        return f

    return dec_func
