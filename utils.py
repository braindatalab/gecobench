import json
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Any, Dict
import pickle


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


def generate_data_dir(config: Dict) -> str:
    return join(
        config['general']['base_dir'],
        config['general']['data_scenario'],
        config['data']['output_dir']
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
