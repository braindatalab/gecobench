from typing import Dict, List
import string
from os.path import join

import pandas as pd

from utils import dump_as_jsonl, load_pickle
from common import DatasetKeys

SPACE = ' '
JOIN_STRING = ''
START_PADDING_COLUMNS = [0, 1, 2]
END_PADDING_COLUMNS = ['tmp1', 'tmp2', 'tmp3']


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
    for _, words in data.iterrows():
        processed_words = preprocess_list_of_words(words=words.tolist())
        sentences += [processed_words]
    return sentences


def determine_gender(s: str) -> str:
    g = 'female'
    if '_male' in s:
        g = 'male'
    return g


def preprocess_training_datasets(dataset_config: Dict, output_dir: str) -> None:
    file_path = dataset_config['raw_data']["train"]
    dataframe = load_pickle(file_path=file_path)

    word_list = assemble_list_of_words(data=dataframe.drop(['target'], axis=1))
    target = dataframe['target'].tolist()

    train_set = [
        {"sentence": word_list_item, "target": target_item}
        for word_list_item, target_item in zip(word_list, target)
    ]

    dump_as_jsonl(
        data=train_set,
        output_dir=output_dir,
        filename=dataset_config['output_filenames']["train"],
    )


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
    sets = list()
    for data_name, file_path in dataset_config['raw_data'].items():
        if not data_name.startswith("test_"):
            continue
        print(f'Processing {data_name}')
        gender = determine_gender(s=data_name)
        dataframe = load_pickle(file_path=file_path)
        word_list = assemble_list_of_words(data=dataframe.drop(['target'], axis=1))

        ground_truth = load_pickle(
            file_path=dataset_config['raw_data'][f'ground_truth']
        )
        ground_truth = reformat_columns(data=ground_truth)
        ground_truth_list = ground_truth_to_list(data=ground_truth, word_list=word_list)

        test_set = [
            {
                "sentence": word_list_item,
                "ground_truth": ground_truth_item,
                "gender": gender,
                "target": target_item,
            }
            for word_list_item, ground_truth_item, target_item in zip(
                word_list, ground_truth_list, dataframe['target'].tolist()
            )
        ]
        sets += test_set

    # Save full test set
    dump_as_jsonl(
        data=sets,
        output_dir=output_dir,
        filename=dataset_config['output_filenames']["test"],
    )


def prepare_gender_data(config: Dict, data_output_dir: str, dataset_key: str) -> None:
    dataset_config = config['datasets'][dataset_key]
    dataset_output_dir = join(data_output_dir, dataset_key)
    preprocess_training_datasets(
        dataset_config=dataset_config, output_dir=dataset_output_dir
    )

    preprocess_test_datasets(
        dataset_config=dataset_config, output_dir=dataset_output_dir
    )


def prepare_gender_all_data(config: Dict, data_output_dir: str):
    prepare_gender_data(
        config=config,
        data_output_dir=data_output_dir,
        dataset_key=DatasetKeys.gender_all.value,
    )


def prepare_gender_subj_data(config: Dict, data_output_dir: str):
    prepare_gender_data(
        config=config,
        data_output_dir=data_output_dir,
        dataset_key=DatasetKeys.gender_subj.value,
    )
