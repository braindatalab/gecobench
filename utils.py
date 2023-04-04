import json
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Any, Dict
import pickle


def dump_as_pickle(data: Any, output_dir: str, file_name: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(join(output_dir, f'{file_name}'), 'wb') as file:
        pickle.dump(data, file)


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def append_date(s: str) -> str:
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f'{s}-{date}'
