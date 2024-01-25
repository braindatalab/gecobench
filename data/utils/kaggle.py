import subprocess
from os.path import join
from typing import Dict
import zipfile
import os


def download_kaggle(data_output_dir: str, dataset_subdir: str, ds_config: Dict):
    """
    Downloads a dataset from Kaggle and unzips it.

    Args:
        data_output_dir (str): Root Directory for datasets.
        dataset_subdir (str): Subdirectory (name of the dataset e.g. sentiment_twitter) to download the dataset to.
        ds_config (Dict): Dataset configuration dictionary.

    Returns:
        dataset_output_dir (str): Directory of the downloaded dataset.
    """
    dataset_output_dir = join(data_output_dir, dataset_subdir)
    os.makdirs(dataset_output_dir, exist_ok=True)

    full_name = f"{ds_config['user']/ds_config['name']}"

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

    return dataset_output_dir
