from typing import Dict
from os.path import join
import shutil
from data.utils.kaggle import download_kaggle
from common import DatasetsKeys

RAW_TRAIN = "twitter_training.csv"
RAW_TEST = "twitter_validation.csv"


def prepare_twitter_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["data"]["datasets"][DatasetsKeys.sentiment_twitter.value]
    dataset_output_dir = join(data_output_dir, DatasetsKeys.sentiment_twitter.value)

    download_kaggle(
        dataset_output_dir=dataset_output_dir,
        ds_config=ds_config["kaggle"],
    )

    # Move files to correct location
    shutil.move(
        join(dataset_output_dir, RAW_TRAIN),
        join(dataset_output_dir, ds_config["output_filenames"]["train"]),
    )

    shutil.move(
        join(dataset_output_dir, RAW_TEST),
        join(dataset_output_dir, ds_config["output_filenames"]["test"]),
    )
