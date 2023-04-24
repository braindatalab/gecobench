import string
from copy import deepcopy
from os.path import join
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DataSet, NAME_OF_PROJECT_CONFIG
from utils import load_pickle, dump_as_pickle, append_date, generate_data_dir, dump_as_json_file

SPACE = ' '
JOIN_STRING = ''


def join_punctuation_with_previews_word(words: List) -> List:
    punctuations = string.punctuation
    for k in range(len(words)):
        if words[k][0] in punctuations:
            words[k - 1] = words[k - 1][:-1]
    return words


def preprocess_list_of_words(words: List) -> List:
    words_without_nan = [w for w in words if not pd.isna(w)]
    words_are_strings = [str(w) for w in words_without_nan]
    words_with_spaces = [w + SPACE for w in words_are_strings]
    return join_punctuation_with_previews_word(words=words_with_spaces)


def assemble_sentences(data: pd.DataFrame) -> List:
    sentences = list()
    for k, row in data.iterrows():
        processed_words = preprocess_list_of_words(words=row.tolist())
        sentences += [JOIN_STRING.join(processed_words)]
    return sentences


def create_train_test_split(x: List, y: List, config: Dict) -> DataSet:
    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=config['data']['test_split'],
        random_state=config['general']['seed']
    )

    return DataSet(
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val
    )


def add_date_to_data_scenario_name(config: Dict) -> Dict:
    config_copy = deepcopy(config)
    config_copy['general']['data_scenario'] = append_date(
        s=config_copy['general']['data_scenario']
    )
    return config_copy


def main(config: Dict) -> None:
    config = add_date_to_data_scenario_name(config=config)

    text_data = load_pickle(file_path=config['data']['raw_data'])
    targets = text_data['target'].tolist()
    sentences = assemble_sentences(data=text_data.drop(['target'], axis=1))
    data_record = create_train_test_split(x=sentences, y=targets, config=config)

    data_dir = generate_data_dir(config=config)
    dump_as_pickle(data=data_record, output_dir=data_dir, filename=config['data']['dataset_filename'])
    dump_as_json_file(data=config, file_path=join(data_dir, NAME_OF_PROJECT_CONFIG))


if __name__ == '__main__':
    main(config={})
