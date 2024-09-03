from typing import Dict, List
import string
import os
import shutil
from os.path import join

import pandas as pd

from common import DatasetKeys

SPACE = " "
JOIN_STRING = ""

data_input_dir = os.path.join(os.path.dirname(__file__), "../dataset")


def prepare_gender_data(
    folder_name: str, config: Dict, data_output_dir: str, dataset_key: str
) -> None:
    dataset_config = config["datasets"][dataset_key]

    dataset_input_dir = join(data_input_dir, folder_name)
    dataset_output_dir = join(data_output_dir, dataset_key)
    os.makedirs(dataset_output_dir, exist_ok=True)

    for filename in dataset_config["output_filenames"].values():
        if 'non_binary' in dataset_key:
            shutil.copyfile(
                join(dataset_input_dir, filename), join(dataset_output_dir, filename)
            )
        else:
            input_file_path = join(dataset_input_dir, filename)
            output_file_path = join(dataset_output_dir, filename)

            data = pd.read_json(input_file_path, lines=True)
            filtered_data = data[data['target'].isin([1, 0])]
            filtered_data.to_json(output_file_path, orient='records', lines=True)


def prepare_binary_gender_all_data(config: Dict, data_output_dir: str):
    prepare_gender_data(
        folder_name="all",
        config=config,
        data_output_dir=data_output_dir,
        dataset_key=DatasetKeys.binary_gender_all.value,
    )


def prepare_binary_gender_subj_data(config: Dict, data_output_dir: str):
    prepare_gender_data(
        folder_name="subj",
        config=config,
        data_output_dir=data_output_dir,
        dataset_key=DatasetKeys.binary_gender_subj.value,
    )


def prepare_non_binary_gender_all_data(config: Dict, data_output_dir: str):
    prepare_gender_data(
        folder_name="all",
        config=config,
        data_output_dir=data_output_dir,
        dataset_key=DatasetKeys.non_binary_gender_all.value,
    )


def prepare_non_binary_gender_subj_data(config: Dict, data_output_dir: str):
    prepare_gender_data(
        folder_name="subj",
        config=config,
        data_output_dir=data_output_dir,
        dataset_key=DatasetKeys.non_binary_gender_subj.value,
    )
