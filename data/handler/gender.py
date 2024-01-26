from typing import Dict, List, Tuple
import string
from os.path import join

import pandas as pd

from utils import (
    load_pickle,
    dump_as_pickle,
)
from common import DataTargetPair, DatasetsKeys

SPACE = ' '
JOIN_STRING = ''
START_PADDING_COLUMNS = [0, 1, 2]
END_PADDING_COLUMNS = ['tmp1', 'tmp2', 'tmp3']
DATA_TYPE_ALL = 'all'
DATA_TYPE_SUBJECT = 'subject'


def join_punctuation_with_previews_word(words: List) -> List:
    punctuations = string.punctuation
    for k in range(len(words)):
        if words[k][0] in punctuations:
            words[k - 1] = words[k - 1][:-1]
    return words


def preprocess_list_of_words(words: List) -> List:
    words_without_nan = [w for w in words if not pd.isna(w)]
    words_are_strings = [str(w) for w in words_without_nan]
    return words_are_strings


def assemble_list_of_words(data: pd.DataFrame) -> List:
    sentences = list()
    for k, words in data.iterrows():
        processed_words = preprocess_list_of_words(words=words.tolist())
        sentences += [processed_words]
    return sentences


def is_training_or_test_data(data_name: str) -> bool:
    return 'train' in data_name or 'test' in data_name


def is_ground_truth(data_name: str) -> bool:
    return 'ground_truth' in data_name


def adjust_data_format(data: pd.DataFrame) -> Tuple:
    word_list = assemble_list_of_words(data=data.drop(['target'], axis=1))
    target = data['target'].tolist()
    output_data = DataTargetPair(data=word_list, target=target)
    return output_data


def preprocess_training_datasets(dataset_config: Dict, output_dir: str) -> None:
    for data_name, file_path in dataset_config['raw_data'].items():
        if 'train' not in data_name:
            continue
        dataframe = load_pickle(file_path=file_path)
        output_data = adjust_data_format(data=dataframe)
        dump_as_pickle(
            data=output_data,
            output_dir=output_dir,
            filename=dataset_config['output_filenames'][data_name],
        )


def determine_data_type(name: str) -> str:
    return DATA_TYPE_ALL if DATA_TYPE_ALL in name else DATA_TYPE_SUBJECT


def ground_truth_to_list(data: pd.DataFrame, word_list: list) -> list:
    output = list()
    for (k, row), words in zip(data.iterrows(), word_list):
        ground_truth = row.tolist()
        ground_truth_resized = ground_truth[: len(words)]
        output += [ground_truth_resized]
    return output


def reformat_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = data[START_PADDING_COLUMNS]
    tmp = data.drop(START_PADDING_COLUMNS, axis=1)
    tmp[END_PADDING_COLUMNS] = columns
    return tmp


def preprocess_test_datasets(dataset_config: Dict, output_dir: str) -> list:
    paths = list()
    for data_name, file_path in dataset_config['raw_data'].items():
        if 'train' in data_name or 'ground_truth' in data_name:
            continue
        data_type = determine_data_type(name=data_name)
        dataframe = load_pickle(file_path=file_path)
        word_list = assemble_list_of_words(data=dataframe.drop(['target'], axis=1))
        dataframe['sentence'] = word_list
        ground_truth = load_pickle(
            file_path=dataset_config['raw_data'][f'ground_truth_{data_type}']
        )
        ground_truth = reformat_columns(data=ground_truth)
        ground_truth_list = ground_truth_to_list(data=ground_truth, word_list=word_list)
        dataframe['ground_truth'] = ground_truth_list

        filename = dataset_config['output_filenames'][data_name]
        paths += [join(output_dir, filename)]
        dump_as_pickle(data=dataframe, output_dir=output_dir, filename=filename)

    return paths


def stack_and_dump_female_and_male_datasets(
    dataset_config: dict,
    paths: list[str],
) -> None:
    female_male_all = list()
    female_male_subject = list()

    def determine_gender(s: str) -> str:
        g = 'female'
        if '_male_' in s:
            g = 'male'
        return g

    for p in paths:
        filename = p.split('/')[-1]
        dataframe = load_pickle(file_path=p)
        gender = determine_gender(s=filename)
        dataframe['gender'] = [gender] * dataframe.shape[0]
        if DATA_TYPE_ALL in filename:
            female_male_all += [dataframe]
        else:
            female_male_subject += [dataframe]

    data_all = pd.concat(female_male_all, axis=0)
    data_subject = pd.concat(female_male_subject, axis=0)
    output_dir = join(*paths[0].split('/')[:-1])
    dump_as_pickle(
        data=data_all,
        output_dir=output_dir,
        filename=dataset_config['output_filenames']['test_all'],
    )
    dump_as_pickle(
        data=data_subject,
        output_dir=output_dir,
        filename=dataset_config['output_filenames']['test_subject'],
    )


def prepare_gender_data(config: Dict, data_output_dir: str):
    dataset_config = config['data']['datasets'][DatasetsKeys.gender.value]
    dataset_output_dir = join(data_output_dir, DatasetsKeys.gender.value)
    preprocess_training_datasets(
        dataset_config=dataset_config, output_dir=dataset_output_dir
    )
    output_paths = preprocess_test_datasets(
        dataset_config=dataset_config, output_dir=dataset_output_dir
    )
    stack_and_dump_female_and_male_datasets(
        dataset_config=dataset_config, paths=output_paths
    )