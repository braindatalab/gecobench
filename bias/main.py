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
    BinaryROC,
)
import statistics
import seaborn as sns
from sklearn import metrics
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
    path = join(generate_data_dir(config), dataset_key, "train.jsonl")
    raw_data = load_jsonl_as_df(path)
    # print(raw_data)
    # print(raw_data['sentence'])
    raw_data['sentence'] = raw_data['sentence'].apply(
        lambda sentence: [word.lower().replace(" ", "") for word in sentence]
    )
    # print(raw_data)
    return raw_data


def generate_corpus(config, gender_terms) -> list:
    corpus = []
    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)
        for sentence in dataset['sentence']:
            for word in sentence:
                if word.lower().replace(" ", "") not in gender_terms:
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
            her_counts = sentence.count("her")
            if target == gender:
                if term in sentence:
                    for word in sentence:
                        if word in corpus:
                            if term == 'her':
                                S[
                                    corpus.index(word.lower().replace(" ", "")), x
                                ] += her_counts
                            else:
                                S[corpus.index(word.lower().replace(" ", "")), x] += 1

    return S


def co_occurrence_matrix_sentence(sentence, corpus, gender_terms) -> None:
    S = np.zeros((len(corpus), len(gender_terms)))
    for x, term in enumerate(gender_terms):
        if term in sentence:
            elements_to_remove = {',', '.'}
            sentence = [word for word in sentence if word not in elements_to_remove]
            sentence = [item.lower().replace(" ", "") for item in sentence]
            her_counts = sentence.count("her")
            for word in sentence:
                if word == 'her':
                    S[corpus.index(word.lower().replace(" ", "")), x] += her_counts
                else:
                    S[corpus.index(word.lower().replace(" ", "")), x] += 1
    return S


def compute_co_occurrence_matrix_sum(config, corpus):
    male_terms = ['he', 'him', 'his']
    female_terms = ['she', 'her']

    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)

        S_male = co_occurrence_matrix(dataset, corpus, male_terms, 1)
        S_female = co_occurrence_matrix(dataset, corpus, female_terms, 0)

        print(f"Dataset: {name}")
        S_male_sum, S_female_sum = S_male.sum(), S_female.sum()

        print(f"Matrix sum of S_male = {S_male_sum}")
        print(
            f"Normalized matrix sum of S_male = {S_male_sum/(S_male_sum+S_female_sum)}"
        )

        print(f"Matrix sum of S_female = {S_female_sum}")
        print(
            f"Normalized matrix sum of S_female = {S_female_sum/(S_male_sum+S_female_sum)}"
        )
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

        sns.set_theme(style="darkgrid")
        fig, axs = plt.subplots(ncols=len(model_variants), figsize=(16, 8))
        fig2, axs2 = plt.subplots(ncols=len(model_variants), figsize=(25, 5))

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
                binary_roc_repetitions,
                fpr_repetitions,
                tpr_repetitions,
            ) = ([] for i in range(7))

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

                predictions_series = (
                    dataset_result_model_version_model_variant_repetition_number[
                        'logits'
                    ]
                )

                predictions_array = np.vstack(predictions_series).astype(float)
                predictions_tensor = torch.tensor(predictions_array)
                prediction_probabilites = torch.nn.functional.softmax(
                    predictions_tensor, dim=1
                )
                prediction_probabilites, _ = torch.max(prediction_probabilites, 1)
                targets_tensor = targets_tensor.int()

                fpr, tpr, thresholds = metrics.roc_curve(
                    targets_tensor.numpy(), prediction_probabilites.numpy()
                )
                roc_auc = metrics.auc(fpr, tpr)
                
                axs2[i].plot(fpr, tpr, label=f"model repetition {repetition_number}")

                # palette = sns.color_palette(palette='pastel')

                # plt.title('Receiver Operating Characteristic')
                # plt.step(
                #     fpr,
                #     tpr,
                #     color=palette[0],
                #     where='mid',
                #     label='AUC = %0.2f' % roc_auc,
                # )
                # plt.legend(loc='lower right')
                # plt.plot([0, 1], [0, 1], 'r--')

                # plt.ylabel('True Positive Rate')
                # plt.xlabel('False Positive Rate')

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
            
            axs2[i].set_xlabel('X-axis')
            axs2[i].set_ylabel('Y-axis')
            axs2[i].set_title(f'Test')
            axs2[i].legend()
        
        savedir = f"{bias_dir}/roc_curve_{dataset}.png"
        plt.tight_layout()
        fig2.savefig(savedir)
        plt.close(fig2)

        savedir = f"{bias_dir}/confusion_matrix_{dataset}.png"
        fig.tight_layout()
        fig.savefig(savedir, dpi=300, bbox_inches='tight')

    return


def main(config: Dict) -> None:
    # male: target == 1, female: target == 0
    gender_terms = ['he', 'him', 'his', 'she', 'her']
    corpus = generate_corpus(config, gender_terms)

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
