from typing import Dict
import os
import shutil
import pandas as pd
from os.path import join


from common import DatasetKeys
from data.utils.sentence import dump_df_as_jsonl

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_ROOT_FOLDER = "aclImdb"


def read_data(data_dir: str):
    data = []
    for sentiment, sentiment_id in [("neg", 0), ("pos", 1)]:
        path = join(data_dir, sentiment)
        for filename in os.listdir(path):
            score = filename.split("_")[1].split(".")[0]
            with open(join(path, filename), "r") as f:
                sentence = f.read()

            data.append([sentence, sentiment_id, int(score)])

    return pd.DataFrame(data, columns=["text", "target", "score"])


def prepare_imdb_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["datasets"][DatasetKeys.sentiment_imdb.value]
    dataset_output_dir = join(data_output_dir, DatasetKeys.sentiment_imdb.value)

    # Download data with wget
    os.makedirs(dataset_output_dir, exist_ok=True)
    output_path = join(dataset_output_dir, "aclImdb_v1.tar.gz")
    os.system(f"wget {URL} -O {output_path}")

    # Extract data
    os.system(f"tar -xvf {output_path} -C {dataset_output_dir}")

    # Read in the data from the aclImdb folder
    train = read_data(join(dataset_output_dir, DATASET_ROOT_FOLDER, "train"))
    test = read_data(join(dataset_output_dir, DATASET_ROOT_FOLDER, "test"))

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
    shutil.rmtree(join(dataset_output_dir, DATASET_ROOT_FOLDER))
