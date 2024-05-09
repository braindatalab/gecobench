import numpy as np
import os
from os import listdir
from os.path import join, isfile, join
from typing import Dict
from transformers import BertTokenizer
from common import DataSet, DatasetKeys, validate_dataset_key
from utils import (
    load_jsonl_as_dict,
    load_jsonl_as_df,
    generate_data_dir,
    filter_train_datasets,
    generate_data_dir,
    generate_artifacts_dir,
    generate_training_dir,
    load_model,
)


def load_dataset_raw(config: Dict, dataset_key: str) -> DataSet:
    path = join(generate_data_dir(config), dataset_key, "test.jsonl")
    raw_data = load_jsonl_as_df(path)
    return raw_data


def generate_corpus(config) -> list:
    corpus = []
    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)
        for sentence in dataset['sentence']:
            for word in sentence:
                corpus.append(word.lower().replace(" ", ""))
    return list(set(corpus))


def get_gender_terms(dataset) -> list:
    # TODO: Get gender terms from the data
    return


def co_occurrence_matrix(dataset, corpus, gender_terms, gender) -> None:
    S = np.zeros((len(corpus), len(gender_terms)))
    for x, term in enumerate(gender_terms):
        for target, sentence in zip(dataset['target'], dataset['sentence']):
            elements_to_remove = {',', '.'}
            sentence = [item.lower().replace(" ", "") for item in sentence]
            sentence = [word for word in sentence if word not in elements_to_remove]
            # print(sentence)
            if target == gender:
                if term in sentence:
                    # print(term)
                    # print(sentence)
                    for word in sentence:
                        S[corpus.index(word.lower().replace(" ", "")), x] += 1

    return S


def co_occurrence_matrix_sentence(sentence, corpus, gender_terms) -> None:
    S = np.zeros((len(corpus), len(gender_terms)))
    for x, term in enumerate(gender_terms):
        if term in sentence:
            elements_to_remove = {',', '.'}
            sentence = [word for word in sentence if word not in elements_to_remove]
            sentence = [item.lower().replace(" ", "") for item in sentence]
            for word in sentence:
                S[corpus.index(word.lower().replace(" ", "")), x] += 1
    return S


def main(config: Dict) -> None:
    # male: target == 1, female: target == 0
    corpus = generate_corpus(config)

    male_terms = ['he']
    female_terms = ['she']

    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)

        S_male = co_occurrence_matrix(dataset, corpus, male_terms, 1)
        S_female = co_occurrence_matrix(dataset, corpus, female_terms, 0)

        print(f"Dataset: {name}")
        S_male_sum, S_female_sum = S_male.sum(), S_female.sum()
        print(f"Matrix sum of S_male = {S_male_sum}")
        print(f"Matrix sum of S_female = {S_female_sum}")
        print("")


if __name__ == '__main__':
    main(config={})
