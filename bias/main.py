import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAccuracy,
    BinaryRecall,
    BinarySpecificity,
    BinaryAUROC,
    ConfusionMatrix,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from os.path import join, join
from typing import Dict
from common import DataSet, validate_dataset_key
from utils import (
    load_jsonl_as_df,
    generate_data_dir,
    filter_train_datasets,
    generate_data_dir,
    generate_artifacts_dir,
    generate_evaluation_dir,
    generate_bias_dir,
    load_pickle,
)
import pandas as pd


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


def compute_co_occurrence_matrix_sum(config, corpus):
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
    return


def bias_metrics_summary(prediction_records, bias_dir):
    dataset_types = list(set(prediction_records['dataset_type']))
    model_variants = list(set(prediction_records['model_name']))
    model_repetition_numbers = list(set(prediction_records['model_repetition_number']))
    model_versions = list(set(prediction_records['model_version']))
    gender_types = list(set(prediction_records['target']))

    for dataset in dataset_types:
        dataset_result = prediction_records[
            prediction_records['dataset_type'] == dataset
        ]

        dataset_result_model_version = dataset_result[
            dataset_result['model_version'] == "best"
        ]

        fig, axs = plt.subplots(ncols=len(model_variants), figsize=(16, 8))

        for i, model_variant in enumerate(model_variants):
            dataset_result_model_version_model_variant = dataset_result_model_version[
                dataset_result_model_version['model_name'] == model_variant
            ]
            cm_repetitions = []
            (
                binary_accruacy_repetitions,
                binary_recall_repetitions,
                binary_specificity_repetitions,
                binary_f1_repetitions,
            ) = ([] for i in range(4))

            for repetition_number in model_repetition_numbers:
                dataset_result_model_version_model_variant_repetition_number = (
                    dataset_result_model_version_model_variant[
                        dataset_result_model_version_model_variant[
                            'model_repetition_number'
                        ]
                        == repetition_number
                    ]
                )
                predictions = (
                    dataset_result_model_version_model_variant_repetition_number[
                        'prediction'
                    ].values
                )
                targets = dataset_result_model_version_model_variant_repetition_number[
                    'target'
                ].values

                predictions_tensor = torch.tensor(predictions)
                targets_tensor = torch.tensor(targets)

                label_mapping = {1: 'male', 0: 'female'}
                predictions = [label_mapping[pred] for pred in predictions]
                targets = [label_mapping[pred] for pred in targets]

                cm = confusion_matrix(targets, predictions, labels=['male', 'female'])
                cm_repetitions.append(cm)

                binary_acc_metric = BinaryAccuracy()
                binary_acc_score = binary_acc_metric(predictions_tensor, targets_tensor)

                binary_recall_metric = BinaryRecall()
                binary_recall_score = binary_recall_metric(
                    predictions_tensor, targets_tensor
                )

                binary_specificity_metric = BinarySpecificity()
                binary_specificity_score = binary_specificity_metric(
                    predictions_tensor, targets_tensor
                )

                binary_f1_metric = BinaryF1Score()
                binary_f1_score = binary_f1_metric(predictions_tensor, targets_tensor)

                binary_recall_repetitions.append(binary_recall_score)
                binary_specificity_repetitions.append(binary_specificity_score)
                binary_f1_repetitions.append(binary_f1_score)
                binary_accruacy_repetitions.append(binary_acc_score)

            binary_acc_score_avg_repetitions = sum(binary_accruacy_repetitions) / len(
                binary_accruacy_repetitions
            )
            binary_recall_score_avg_repetitions = sum(binary_recall_repetitions) / len(
                binary_recall_repetitions
            )
            binary_specificity_score_avg_repetitions = sum(
                binary_specificity_repetitions
            ) / len(binary_specificity_repetitions)
            binary_f1_score_avg_repetitions = sum(binary_f1_repetitions) / len(
                binary_f1_repetitions
            )

            cm_repetitions_stacked = np.stack(cm_repetitions, axis=0)

            cm_repetitions_average = np.mean(cm_repetitions_stacked, axis=0)
            cm_repetitions_std = np.std(cm_repetitions_stacked, axis=0)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_repetitions_average,
                display_labels=['male', 'female'],
            )

            disp.plot(
                ax=axs[i],
                xticks_rotation='vertical',
                values_format='.1f',
                colorbar=False,
            )
            model_variants_mapping = {
                "bert_only_embedding_classification": "BERT-CEf",
                "one_layer_attention_classification": "OLA-CEA",
                "bert_randomly_init_embedding_classification": "BERT-CE",
                "bert_all": "BERT-CEfAf",
                "bert_only_classification": "BERT-C",
            }
            model_variant_explicit_name = model_variants_mapping[model_variant]

            formatted_f1 = "{:.2f}".format(binary_f1_score_avg_repetitions * 100)
            formatted_recall = "{:.2f}".format(
                binary_recall_score_avg_repetitions * 100
            )
            formatted_specificity = "{:.2f}".format(
                binary_specificity_score_avg_repetitions * 100
            )
            formatted_acc = "{:.2f}".format(binary_acc_score_avg_repetitions * 100)

            axs[i].set_title(
                f"{model_variant_explicit_name} \n accruacy: {formatted_acc} \n specificity: {formatted_specificity} \n recall: {formatted_recall} \n f1: {formatted_f1}"
            )

        savedir = f"{bias_dir}/confusion_matrix_{dataset}.png"
        plt.tight_layout()
        plt.savefig(savedir, dpi=300, bbox_inches='tight')

    return


def main(config: Dict) -> None:
    # male: target == 1, female: target == 0
    corpus = generate_corpus(config)
    compute_co_occurrence_matrix_sum(config, corpus)

    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_output_dir = generate_evaluation_dir(config=config)
    filename = config["evaluation"]["prediction_records"]
    prediction_records = load_pickle(
        join(artifacts_dir, evaluation_output_dir, filename)
    )
    bias_dir = generate_bias_dir(config=config)
    Path(bias_dir).mkdir(parents=True, exist_ok=True)
    bias_metrics_summary(prediction_records, bias_dir)


if __name__ == '__main__':
    main(config={})
