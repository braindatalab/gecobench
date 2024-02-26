from typing import Dict
from os.path import join
import os
import pandas as pd
from common import DatasetKeys
from data.utils.sentence import dump_df_as_jsonl

RAW_TEST = "testdata.manual.2009.06.14.csv"
RAW_TRAIN = "training.1600000.processed.noemoticon.csv"
URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"


def load_df(path: str, encoding=None) -> pd.DataFrame:
    # target 0 = negativ mapped to 0
    # target 4 = positiv mapped to 1

    columns = ["target", "tweet_id", "date", "flag", "user", "text"]
    df = pd.read_csv(
        path,
        encoding=encoding,
        header=None,
        names=columns,
    )

    df = df[df["target"] != 2]
    df = df.dropna()

    # Labels to index
    df["target"][df["target"] == 4] = 1

    return df[["text", "target"]]


def prepare_twitter_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["datasets"][DatasetKeys.sentiment_twitter.value]
    dataset_output_dir = join(data_output_dir, DatasetKeys.sentiment_twitter.value)

    # Download dataset
    os.makedirs(dataset_output_dir, exist_ok=True)
    output_path = join(dataset_output_dir, "trainingandtestdata.zip")
    os.system(f"wget {URL} -O {output_path}")

    # Extract dataset
    os.system(f"unzip {output_path} -d {dataset_output_dir}")

    train = load_df(join(dataset_output_dir, RAW_TRAIN), encoding="ISO-8859-1")
    test = load_df(join(dataset_output_dir, RAW_TEST))

    # Save datasets
    dump_df_as_jsonl(
        df=train,
        output_dir=dataset_output_dir,
        filename=ds_config["output_filenames"]["train"],
    )

    dump_df_as_jsonl(
        df=test,
        output_dir=dataset_output_dir,
        filename=ds_config["output_filenames"]["test"],
    )

    # Remove raw data
    os.remove(output_path)
    os.remove(join(dataset_output_dir, RAW_TRAIN))
    os.remove(join(dataset_output_dir, RAW_TEST))
