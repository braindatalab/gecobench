import numpy as np
from os.path import join, join
from typing import Dict
from common import DataSet, DatasetKeys, validate_dataset_key
from utils import (
    load_jsonl_as_dict,
    generate_data_dir,
    filter_train_datasets,
)


def load_dataset_raw(config: Dict, dataset_key: str) -> DataSet:
    path = join(generate_data_dir(config), dataset_key, "train.jsonl")
    raw_data = load_jsonl_as_dict(path)
    return raw_data


def generate_corpus(config) -> list:
    corpus = []
    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)
        for sentence in dataset['sentence']:
            for word in sentence:
                corpus.append(word)
    return list(set(corpus))


def get_gender_terms(dataset) -> list:
    # TODO: Get gender terms from the data
    return


def co_occurrence_matrix(dataset, corpus, gender_terms, gender) -> None:
    S = np.zeros((len(corpus), len(gender_terms)))
    for x, term in enumerate(gender_terms):
        for target, sentence in zip(dataset['target'], dataset['sentence']):
            if target == gender:
                if term in sentence:
                    for word in sentence:
                        if word in corpus:
                            S[corpus.index(word), x] += 1

    return S


def main(config: Dict) -> None:
    # male: target == 1, female: target == 0
    corpus = generate_corpus(config)
    male_terms = ['He', 'His', 'his', 'men']
    female_terms = ['She', 'Her', 'her', 'woman']

    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)
        S_male = co_occurrence_matrix(dataset, corpus, male_terms, 1)
        S_female = co_occurrence_matrix(dataset, corpus, female_terms, 0)
        S_male_norm, S_female_norm = np.linalg.norm(S_male), np.linalg.norm(S_female)
        print(f"Matrix norm of S_male = {S_male_norm}")
        print(f"Matrix norm of S_female = {S_female_norm}")
        assert S_male_norm == S_female_norm


if __name__ == '__main__':
    main(config={})
