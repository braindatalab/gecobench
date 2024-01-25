from typing import Dict
from data.utils.kaggle import download_kaggle


def prepare_twitter_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["data"]["datasets"]["sentiment_twitter"]

    download_kaggle(
        data_output_dir=data_output_dir,
        dataset_subdir="sentiment_twitter",
        ds_config=ds_config["kaggle"],
    )
