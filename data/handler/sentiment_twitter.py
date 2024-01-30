from typing import Dict
from os.path import join
import os
import pandas as pd
from data.utils.kaggle import download_kaggle
from common import DatasetKeys

RAW_TRAIN = "twitter_training.csv"
RAW_TEST = "twitter_validation.csv"


def load_df(path: str) -> pd.DataFrame:
    labels = ["Positive", "Negative", "Neutral"]
    columns = ["tweet_id", "entity", "sentiment", "text"]
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
    )

    df = df[df["sentiment"] != "Irrelevant"]
    df = df.dropna()

    # Labels to index
    df["label"] = df["sentiment"].apply(lambda x: labels.index(x))

    return df[["text", "label"]]


def prepare_twitter_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["data"]["datasets"][DatasetKeys.sentiment_twitter.value]
    dataset_output_dir = join(data_output_dir, DatasetKeys.sentiment_twitter.value)

    download_kaggle(
        dataset_output_dir=dataset_output_dir,
        ds_config=ds_config["kaggle"],
    )

    train_df = load_df(join(dataset_output_dir, RAW_TRAIN))
    test_df = load_df(join(dataset_output_dir, RAW_TEST))

    # Save datasets
    train_df.to_csv(
        join(dataset_output_dir, ds_config["output_filenames"]["train"]), index=False
    )
    test_df.to_csv(
        join(dataset_output_dir, ds_config["output_filenames"]["test"]), index=False
    )

    # Remove raw data
    os.remove(join(dataset_output_dir, RAW_TRAIN))
    os.remove(join(dataset_output_dir, RAW_TEST))
