import subprocess
from os.path import join
from typing import Dict
import zipfile
import os


def download_kaggle(dataset_output_dir: str, ds_config: Dict):
    """
    Downloads a dataset from Kaggle and unzips it.

    Args:
        dataset_output_dir (str): Outputdir for dataset.
        ds_config (Dict): Dataset configuration dictionary.
    """
    os.makedirs(dataset_output_dir, exist_ok=True)

    full_name = f"{ds_config['user']}/{ds_config['name']}"

    # Download dataset
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-p",
            dataset_output_dir,
            full_name,
        ]
    )

    zip_path = join(dataset_output_dir, ds_config["name"] + ".zip")

    # Unzip dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_output_dir)

    os.remove(zip_path)
