import numpy as np
from os.path import join, join
from typing import Dict
from common import DataSet, DatasetKeys, validate_dataset_key
from utils import (
    load_jsonl_as_dict,
    load_jsonl_as_df,
    generate_data_dir,
    filter_train_datasets,
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
    corpus = list(set(corpus))
    # Define the file path where you want to save the list
    file_path = "corpus.txt"
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each element of the list to the file
        for item in corpus:
            file.write(str(item) + "\n")

    # print(corpus)
    # corpus = []
    # # Open the file in read mode
    # with open("corpus.txt", "r") as file:
    #     # Read the entire contents of the file
    #     corpus = [line.strip() for line in file]

    # Print or process the file contents as needed
    # print(corpus)

    # from collections import Counter
    # item_counts = Counter(corpus)
    # print("Counts of each item:")
    # for item, count in item_counts.items():
    #     print(f"{item}: {count}")
    # # Get the count for the specified item
    # count_of_item = item_counts.get("she", 0)
    # print(f"Count of she: {count_of_item}")
    # print(corpus[2038])
    # print(corpus[2123])

    male_terms = ['he']
    female_terms = ['she']

    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)
        # print(name)

        idx_list = list(range(0, 50, 1))
        temp_dataset = dataset[dataset['sentence_idx'].map(lambda x: x in idx_list)]
        # print(temp_dataset['sentence'])
        for sentence in temp_dataset['sentence']:
            print(" ".join(sentence))

        S_male = co_occurrence_matrix(dataset, corpus, male_terms, 1)
        S_female = co_occurrence_matrix(dataset, corpus, female_terms, 0)
        # np.savetxt("S_male.csv", S_male, delimiter=",")
        # np.savetxt("S_female.csv", S_female, delimiter=",")

        print()
        print(S_male)
        print(S_female)
        print(f"Matrices equal: {(S_male == S_female).all()}")

        S_male_norm, S_female_norm = np.linalg.norm(S_male), np.linalg.norm(S_female)
        print(f"Matrix norm of S_male = {S_male_norm}")
        print(f"Matrix norm of S_female = {S_female_norm}")
        print("")

        S_male_sum, S_female_sum = S_male.sum(), S_female.sum()
        print(f"Matrix sum of S_male = {S_male_sum}")
        print(f"Matrix sum of S_female = {S_female_sum}")
        print("")

        S_male_sum, S_female_sum = S_male.sum(0), S_female.sum(0)
        print(f"Matrix sum(0) of S_male = {S_male_sum}")
        print(f"Matrix sum(0) of S_female = {S_female_sum}")
        print(f"Matrix sum(0) equal: {(S_male_sum == S_female_sum).all()}")
        print("")

        S_male_sum, S_female_sum = S_male.sum(1), S_female.sum(1)
        print(f"Matrix sum(1) of S_male = {S_male_sum}")
        print(f"Matrix sum(1) of S_female = {S_female_sum}")
        print(f"Matrix sum(1) equal: {(S_male_sum == S_female_sum).all()}")

        print(len(corpus) * len(male_terms))
        print("Normalized Metric:", S_male.sum() / (len(corpus) * len(male_terms)))

        # ChatGPT
        # Find rows where the sum of the row is not the same between both matrices
        rows_with_different_sums = np.where(S_male_sum != S_female_sum)[0]
        # print(rows_with_different_sums)
        # print(len(rows_with_different_sums))
        # print(len(set(rows_with_different_sums)))

        # Get the actual rows and their indices from both matrices
        rows_and_indices_matrix1 = [
            (i, row) for i, row in enumerate(S_male) if i in rows_with_different_sums
        ]
        rows_and_indices_matrix2 = [
            (i, row) for i, row in enumerate(S_female) if i in rows_with_different_sums
        ]

        print("")
        # Print or use the rows and their indices as needed
        print("Rows and Indices from S_male:")
        for index, row in rows_and_indices_matrix1:
            print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

        print("\nRows and Indices from S_female:")
        for index, row in rows_and_indices_matrix2:
            print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

        print(dataset)
        dataset_female = dataset[dataset['gender'] == 'female']
        dataset_male = dataset[dataset['gender'] == 'male']

        for sentence in dataset_female['sentence']:
            for word in sentence:
                if word == "hers":
                    print(sentence)

        for sentence_female, sentence_male in zip(
            dataset_female['sentence'], dataset_male['sentence']
        ):

            S_male = co_occurrence_matrix_sentence(sentence_male, corpus, male_terms)
            S_female = co_occurrence_matrix_sentence(
                sentence_female, corpus, female_terms
            )

            S_male_norm, S_female_norm = np.linalg.norm(S_male), np.linalg.norm(
                S_female
            )
            if abs(S_male_norm - S_female_norm) > 0.5:

                S_male_sum, S_female_sum = S_male.sum(1), S_female.sum(1)
                rows_with_different_sums = np.where(S_male_sum != S_female_sum)[0]

                rows_and_indices_matrix1 = [
                    (i, row)
                    for i, row in enumerate(S_male)
                    if i in rows_with_different_sums
                ]
                rows_and_indices_matrix2 = [
                    (i, row)
                    for i, row in enumerate(S_female)
                    if i in rows_with_different_sums
                ]

                print("")
                # Print or use the rows and their indices as needed
                print("Rows and Indices from S_male:")
                for index, row in rows_and_indices_matrix1:
                    print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

                print("\nRows and Indices from S_female:")
                for index, row in rows_and_indices_matrix2:
                    print("Corpus Word:", corpus[index], "| Index:", index, "Row:", row)

                print(
                    f"Matrix norm of S_male = {S_male_norm}",
                    " ",
                    " ".join(sentence_male),
                )
                print(
                    f"Matrix norm of S_female = {S_female_norm}",
                    " ",
                    " ".join(sentence_female),
                )
                print("")
                exit()

        break


if __name__ == '__main__':
    main(config={})
