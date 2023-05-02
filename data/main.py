import string
from copy import deepcopy
from os.path import join
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

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


def assemble_sentences_from_list_of_words(data: pd.DataFrame) -> List:
    sentences = list()
    for k, words in data.iterrows():
        processed_words = preprocess_list_of_words(words=words.tolist())
        sentences += [JOIN_STRING.join(processed_words)]
    return sentences


def split_train_data_into_train_val_data(x: List, y: List, config: Dict) -> DataSet:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=config['data']['val_split'],
        random_state=config['general']['seed']
    )

    return DataSet(
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )


def add_date_to_data_scenario_name(config: Dict) -> Dict:
    config_copy = deepcopy(config)
    config_copy['general']['data_scenario'] = append_date(
        s=config_copy['general']['data_scenario']
    )
    return config_copy


def is_training_or_test_data(data_name: str) -> bool:
    return 'train' in data_name or 'test' in data_name


def is_ground_truth(data_name: str) -> bool:
    return 'ground_truth' in data_name


def adjust_data_format(data: pd.DataFrame, name: str) -> Tuple:
    if is_training_or_test_data(data_name=name):
        target = data['target']
        sentences = data.drop(['target'], axis=1)
        output_data = (sentences, target)
    elif is_ground_truth(data_name=name):
        columns = data[[0, 1, 2]]
        tmp = data.drop([0, 1, 2], axis=1)
        tmp[['tmp1', 'tmp2', 'tmp3']] = columns
        output_data = ('ground-truth', tmp)
    else:
        raise RuntimeError(f'Data type with name: {name} is not supported.')
    return output_data


def loop_over_raw_datasets(config: Dict, output_dir: str) -> None:
    for data_name, file_path in config['data']['raw_data'].items():
        dataframe = load_pickle(file_path=file_path)
        output_data = adjust_data_format(data=dataframe, name=data_name)
        dump_as_pickle(
            data=output_data,
            output_dir=output_dir,
            filename=config['data']['output_filenames'][data_name]
        )


def main(config: Dict) -> None:
    config = add_date_to_data_scenario_name(config=config)
    data_output_dir = generate_data_dir(config=config)
    loop_over_raw_datasets(config=config, output_dir=data_output_dir)
    dump_as_json_file(data=config, file_path=join(data_output_dir, NAME_OF_PROJECT_CONFIG))


if __name__ == '__main__':
    main(config={})