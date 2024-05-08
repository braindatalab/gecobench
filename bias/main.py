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
from training.bert import create_bert_ids


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


def load_models(trained_models_dir_path: str, models: list) -> list:
    train_files = [
        f
        for f in listdir(trained_models_dir_path)
        if isfile(join(trained_models_dir_path, f)) and f.endswith('.pt')
    ]
    # example model name: gender_subj_bert_only_embedding_0.pt
    trained_models_paths = [
        path for path in train_files if any(model_name in path for model_name in models)
    ]
    trained_models_paths = [
        str(trained_models_dir_path + "/" + path) for path in trained_models_paths
    ]
    return trained_models_paths


models = [
    'gender_all_bert_only_classification',
    'gender_all_bert_only_embedding_classification',
    'gender_all_bert_all',
    'gender_all_bert_only_embedding',
    'gender_all_bert_randomly_init_embedding_classification',
]


def get_bert_tokenizer(config: dict) -> BertTokenizer:
    return BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased',
        revision=config['training']['bert_revision'],
    )


def main(config: Dict) -> None:
    # male: target == 1, female: target == 0
    corpus = generate_corpus(config)

    male_terms = ['he']
    female_terms = ['she']
    
    # his, him
    # her, her

    trained_models_dir_path = join(
        generate_artifacts_dir(config=config),
        generate_training_dir(config=config),
    )
    trained_models_paths = load_models(trained_models_dir_path, models)

    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)

        # idx_list = list(range(0, 50, 1))
        # temp_dataset = dataset[dataset['sentence_idx'].map(lambda x: x in idx_list)]
        # for sentence in temp_dataset['sentence']:
        #     print(" ".join(sentence))

        S_male = co_occurrence_matrix(dataset, corpus, male_terms, 1)
        S_female = co_occurrence_matrix(dataset, corpus, female_terms, 0)

        # print()
        # print(S_male)
        # print(S_female)
        # print(f"Matrices equal: {(S_male == S_female).all()}")

        # S_male_norm, S_female_norm = np.linalg.norm(S_male), np.linalg.norm(S_female)
        # print(f"Matrix norm of S_male = {S_male_norm}")
        # print(f"Matrix norm of S_female = {S_female_norm}")
        # print("")

        S_male_sum, S_female_sum = S_male.sum(), S_female.sum()
        print(f"Matrix sum of S_male = {S_male_sum}")
        print(f"Matrix sum of S_female = {S_female_sum}")
        print("")

        # S_male_sum, S_female_sum = S_male.sum(0), S_female.sum(0)
        # print(f"Matrix sum(0) of S_male = {S_male_sum}")
        # print(f"Matrix sum(0) of S_female = {S_female_sum}")
        # print(f"Matrix sum(0) equal: {(S_male_sum == S_female_sum).all()}")
        # print("")

        # S_male_sum, S_female_sum = S_male.sum(1), S_female.sum(1)
        # print(f"Matrix sum(1) of S_male = {S_male_sum}")
        # print(f"Matrix sum(1) of S_female = {S_female_sum}")
        # print(f"Matrix sum(1) equal: {(S_male_sum == S_female_sum).all()}")

        # print(len(corpus) * len(male_terms))
        # print("Normalized Metric:", S_male.sum() / (len(corpus) * len(male_terms)))

        # # ChatGPT
        # # Find rows where the sum of the row is not the same between both matrices
        # rows_with_different_sums = np.where(S_male_sum != S_female_sum)[0]

        # # Get the actual rows and their indices from both matrices
        # rows_and_indices_matrix1 = [
        #     (i, row) for i, row in enumerate(S_male) if i in rows_with_different_sums
        # ]
        # rows_and_indices_matrix2 = [
        #     (i, row) for i, row in enumerate(S_female) if i in rows_with_different_sums
        # ]

        # print("")
        # # Print or use the rows and their indices as needed
        # print("Rows and Indices from S_male:")
        # for index, row in rows_and_indices_matrix1:
        #     print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

        # print("\nRows and Indices from S_female:")
        # for index, row in rows_and_indices_matrix2:
        #     print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

        # print(dataset)
        dataset_female = dataset[dataset['gender'] == 'female']
        dataset_male = dataset[dataset['gender'] == 'male']
        print(dataset_male)
        print(dataset_female)

        # for sentence in dataset_female['sentence']:
        #     for word in sentence:
        #         if word == "hers":
        #             print(sentence)

        # for sentence_female, sentence_male in zip(
        #     dataset_female['sentence'], dataset_male['sentence']
        # ):

        #     S_male = co_occurrence_matrix_sentence(sentence_male, corpus, male_terms)
        #     S_female = co_occurrence_matrix_sentence(
        #         sentence_female, corpus, female_terms
        #     )

        #     S_male_norm, S_female_norm = np.linalg.norm(S_male), np.linalg.norm(
        #         S_female
        #     )
        #     if abs(S_male_norm - S_female_norm) > 0.5:

        #         S_male_sum, S_female_sum = S_male.sum(1), S_female.sum(1)
        #         rows_with_different_sums = np.where(S_male_sum != S_female_sum)[0]

        #         rows_and_indices_matrix1 = [
        #             (i, row)
        #             for i, row in enumerate(S_male)
        #             if i in rows_with_different_sums
        #         ]
        #         rows_and_indices_matrix2 = [
        #             (i, row)
        #             for i, row in enumerate(S_female)
        #             if i in rows_with_different_sums
        #         ]

        #         print("")
        #         # Print or use the rows and their indices as needed
        #         print("Rows and Indices from S_male:")
        #         for index, row in rows_and_indices_matrix1:
        #             print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

        #         print("\nRows and Indices from S_female:")
        #         for index, row in rows_and_indices_matrix2:
        #             print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

        #         print(
        #             f"Matrix norm of S_male = {S_male_norm}",
        #             " ",
        #             " ".join(sentence_male),
        #         )
        #         print(
        #             f"Matrix norm of S_female = {S_female_norm}",
        #             " ",
        #             " ".join(sentence_female),
        #         )
        #         print("")
        #         exit()

        for model_path in trained_models_paths:
            print(model_path)
            bert_model = load_model(model_path)
            bert_tokenizer = get_bert_tokenizer(config=config)
            bert_ids_val, val_idxs = create_bert_ids(
                data=dataset_female["sentence"],
                tokenizer=bert_tokenizer,
                type=f"test_{name}",
                config=config,
            )
            y_test = [dataset_female["target"][i] for i in val_idxs]



if __name__ == '__main__':
    main(config={})
