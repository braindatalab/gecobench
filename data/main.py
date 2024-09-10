from copy import deepcopy
import os
from os.path import join
from typing import Dict

from common import NAME_OF_DATA_CONFIG, NAME_OF_PROJECT_CONFIG, DatasetKeys
from utils import dump_as_json_file, today_formatted

from .handler.gender import (
    prepare_binary_gender_all_data,
    prepare_binary_gender_subj_data,
    prepare_non_binary_gender_all_data,
    prepare_non_binary_gender_subj_data,
)
from .handler.sentiment_twitter import prepare_twitter_sentiment_data
from .handler.sentiment_imdb import prepare_imdb_sentiment_data

HANDLERS = {
    DatasetKeys.non_binary_gender_all.value: prepare_non_binary_gender_all_data,
    DatasetKeys.non_binary_gender_subj.value: prepare_non_binary_gender_subj_data,
    DatasetKeys.binary_gender_all.value: prepare_binary_gender_all_data,
    DatasetKeys.binary_gender_subj.value: prepare_binary_gender_subj_data,
    DatasetKeys.sentiment_twitter.value: prepare_twitter_sentiment_data,
    DatasetKeys.sentiment_imdb.value: prepare_imdb_sentiment_data,
}


def main(config: Dict) -> None:
    config["created"] = today_formatted()

    data_output_dir = join(
        config["output_dir"], f"{config['dataset_name']}_{config['created']}"
    )
    os.makedirs(data_output_dir, exist_ok=True)

    # Build every dataset specified in config
    datasets = config["datasets"].keys()
    for dataset in datasets:
        HANDLERS[dataset](config=config, data_output_dir=data_output_dir)

    dump_as_json_file(data=config, file_path=join(data_output_dir, NAME_OF_DATA_CONFIG))


if __name__ == '__main__':
    main(config={})
