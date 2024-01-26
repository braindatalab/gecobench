from copy import deepcopy
from os.path import join
from typing import Dict

from common import NAME_OF_PROJECT_CONFIG, DataTargetPair
from utils import (
    append_date,
    generate_data_dir,
    dump_as_json_file,
)

from .handler.gender import prepare_gender_all_data, prepare_gender_subj_data
from .handler.sentiment_twitter import prepare_twitter_sentiment_data
from .handler.sentiment_imdb import prepare_imdb_sentiment_data

HANDLERS = {
    "gender_all": prepare_gender_all_data,
    "gender_subj": prepare_gender_subj_data,
    "sentiment_twitter": prepare_twitter_sentiment_data,
    "sentiment_imdb": prepare_imdb_sentiment_data,
}


def add_date_to_data_scenario_name(config: Dict) -> Dict:
    config_copy = deepcopy(config)
    config_copy['general']['data_scenario'] = append_date(
        s=config_copy['general']['data_scenario']
    )
    return config_copy


def main(config: Dict) -> None:
    config = add_date_to_data_scenario_name(config=config)
    data_output_dir = generate_data_dir(config=config)

    # Build every dataset specified in config
    datasets = config["data"]["datasets"].keys()
    for dataset in datasets:
        HANDLERS[dataset](config=config, data_output_dir=data_output_dir)

    dump_as_json_file(
        data=config, file_path=join(data_output_dir, NAME_OF_PROJECT_CONFIG)
    )


if __name__ == '__main__':
    main(config={})
