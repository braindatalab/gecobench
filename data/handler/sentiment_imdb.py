from typing import Dict
import os
import pandas as pd
from os.path import join
from data.utils.kaggle import download_kaggle
from sklearn.model_selection import train_test_split

from common import DatasetKeys

RAW_ALL = "movie.csv"


def prepare_imdb_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["data"]["datasets"][DatasetKeys.sentiment_imdb.value]
    dataset_output_dir = join(data_output_dir, DatasetKeys.sentiment_imdb.value)

    download_kaggle(
        dataset_output_dir=dataset_output_dir,
        ds_config=ds_config["kaggle"],
    )

    df = pd.read_csv(join(dataset_output_dir, RAW_ALL))

    # Drop nan values
    df = df.dropna()

    # Create train and test datasets

    train, test = train_test_split(
        df, test_size=ds_config["test_split"], random_state=config["general"]["seed"]
    )

    # Save datasets
    train.to_csv(
        join(dataset_output_dir, ds_config["output_filenames"]["train"]), index=False
    )
    test.to_csv(
        join(dataset_output_dir, ds_config["output_filenames"]["test"]), index=False
    )

    # Remove raw data
    os.remove(join(dataset_output_dir, RAW_ALL))
