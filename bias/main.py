import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAccuracy,
    BinaryRecall,
    BinarySpecificity,
)
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
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


def load_dataset_raw(config: Dict, dataset_key: str) -> DataSet:
    path = join(generate_data_dir(config), dataset_key, "train.jsonl")
    raw_data = load_jsonl_as_df(path)
    raw_data['sentence'] = raw_data['sentence'].apply(
        lambda sentence: [word.lower().replace(" ", "") for word in sentence]
    )
    return raw_data


def generate_corpus(config, male_terms, female_terms) -> list:
    gender_terms = male_terms + female_terms
    corpus = []
    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)
        for sentence in dataset['sentence']:
            for word in sentence:
                if word.lower().replace(" ", "") not in gender_terms:
                    corpus.append(word.lower().replace(" ", ""))
    return list(set(corpus))


def co_occurrence_matrix(dataset, corpus, gender_terms, gender) -> None:
    S = np.zeros((len(corpus), len(gender_terms)))
    for x, gender_group in enumerate(gender_terms):
        for target, sentence in zip(dataset['target'], dataset['sentence']):
            elements_to_remove = {',', '.'}
            sentence = [item.lower().replace(" ", "") for item in sentence]
            sentence = [word for word in sentence if word not in elements_to_remove]

            if target == gender:
                if any(item in gender_group for item in sentence):
                    for word in sentence:
                        if word in corpus:
                            S[corpus.index(word.lower().replace(" ", "")), x] += 1

    return S


def co_occurrence_matrix_sentence(sentence, corpus, gender_terms) -> None:
    # Compute matrix for each sentence
    S = np.zeros((len(corpus), len(gender_terms)))
    for x, gender_group in enumerate(gender_terms):
        elements_to_remove = {',', '.'}
        sentence = [item.lower().replace(" ", "") for item in sentence]
        sentence = [word for word in sentence if word not in elements_to_remove]

        if any(item in gender_group for item in sentence):
            for word in sentence:
                if word in corpus:
                    S[corpus.index(word.lower().replace(" ", "")), x] += 1

    return S


def compute_co_occurrence_matrix_sum(config, corpus, male_terms, female_terms):
    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset_raw(config, name)

        # male: target == 1, female: target == 0
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

    for dataset in dataset_types:
        dataset_result = prediction_records[
            prediction_records['dataset_type'] == dataset
        ]

        dataset_result_model_version = dataset_result[
            dataset_result['model_version'] == "best"
        ]

        sns.set_theme(style="whitegrid")
        palette = sns.color_palette(palette='pastel')

        fig1, axs1 = plt.subplots(ncols=len(model_variants), figsize=(25, 5))
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
                roc_auc_repetitions,
            ) = ([] for i in range(5))

            model_variants_mapping = {
                "bert_only_embedding_classification": "BERT-CEf",
                "one_layer_attention_classification": "OLA-CEA",
                "bert_randomly_init_embedding_classification": "BERT-CE",
                "bert_all": "BERT-CEfAf",
                "bert_only_classification": "BERT-C",
            }
            model_variant_explicit_name = model_variants_mapping[model_variant]

            for j, repetition_number in enumerate(model_repetition_numbers):
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

                # Computation: confusion matrix
                label_mapping = {1: 'male', 0: 'female'}
                predictions_cm = [label_mapping[pred] for pred in predictions]
                targets_cm = [label_mapping[pred] for pred in targets]

                cm = confusion_matrix(
                    targets_cm, predictions_cm, labels=['male', 'female']
                )
                cm_repetitions.append(cm)

                # Computation: accruacy, recall, specificity, f1
                binary_acc_metric = BinaryAccuracy()
                binary_acc_score = binary_acc_metric(
                    torch.tensor(predictions), torch.tensor(targets)
                )

                binary_recall_metric = BinaryRecall()
                binary_recall_score = binary_recall_metric(
                    torch.tensor(predictions), torch.tensor(targets)
                )

                binary_specificity_metric = BinarySpecificity()
                binary_specificity_score = binary_specificity_metric(
                    torch.tensor(predictions), torch.tensor(targets)
                )

                binary_f1_metric = BinaryF1Score()
                binary_f1_score = binary_f1_metric(
                    torch.tensor(predictions), torch.tensor(targets)
                )

                # Computation: roc curve
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

                fpr, tpr, thresholds = metrics.roc_curve(
                    torch.tensor(targets).int().numpy(), prediction_probabilites.numpy()
                )
                roc_auc = metrics.auc(fpr, tpr)

                axs2[i].plot(
                    fpr,
                    tpr,
                    color=palette[j],
                    label=f"model repetition {repetition_number}",
                )

                binary_recall_repetitions.append(binary_recall_score)
                binary_specificity_repetitions.append(binary_specificity_score)
                binary_f1_repetitions.append(binary_f1_score)
                binary_accruacy_repetitions.append(binary_acc_score)
                roc_auc_repetitions.append(roc_auc)

            # Average accruacy, recall, specificity, f1, roc_auc
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
            roc_auc_score_avg_repetitions = sum(roc_auc_repetitions) / len(
                roc_auc_repetitions
            )

            # Average tp, fp, tn, fn
            cm_repetitions_stacked = np.stack(cm_repetitions, axis=0)
            cm_repetitions_average = np.mean(cm_repetitions_stacked, axis=0)
            cm_repetitions_std = np.std(cm_repetitions_stacked, axis=0)

            # Plot: confusion matrix
            sns.heatmap(
                cm_repetitions_average,
                annot=True,
                cmap=palette,
                fmt='g',
                ax=axs1[i],
                xticklabels=['male', 'female'],
                yticklabels=['male', 'female'],
            )

            formatted_f1 = "{:.2f}".format(binary_f1_score_avg_repetitions * 100)
            formatted_recall = "{:.2f}".format(
                binary_recall_score_avg_repetitions * 100
            )
            formatted_specificity = "{:.2f}".format(
                binary_specificity_score_avg_repetitions * 100
            )
            formatted_acc = "{:.2f}".format(binary_acc_score_avg_repetitions * 100)

            axs1[i].set_title(
                f"{model_variant_explicit_name} \n accruacy: {formatted_acc} \n specificity: {formatted_specificity} \n recall: {formatted_recall} \n f1: {formatted_f1}"
            )

            # Plot: ROC curve
            formatted_roc_auc = "{:.2f}".format(roc_auc_score_avg_repetitions)

            axs2[i].set_xlabel('False Positive Rate')
            axs2[i].set_ylabel('True Positive Rate')
            axs2[i].set_title(
                f"{model_variant_explicit_name} \n Average ROC-AUC: {formatted_roc_auc}"
            )
            axs2[i].plot([0, 1], [0, 1], linestyle='--', color='grey')
            if i == 0:
                axs2[i].legend(loc='upper left')

        savedir = f"{bias_dir}/confusion_matrix_{dataset}.png"
        fig1.tight_layout()
        fig1.savefig(savedir, dpi=300, bbox_inches='tight')

        savedir = f"{bias_dir}/roc_curve_{dataset}.png"
        fig2.tight_layout()
        fig2.savefig(savedir, dpi=300, bbox_inches='tight')

    return


def main(config: Dict) -> None:
    male_terms = [['he'], ['him', 'his']]
    female_terms = [['she'], ['her']]

    corpus = generate_corpus(config, male_terms, female_terms)
    compute_co_occurrence_matrix_sum(config, corpus, male_terms, female_terms)

    # artifacts_dir = generate_artifacts_dir(config=config)
    # evaluation_output_dir = generate_evaluation_dir(config=config)
    # filename = config["evaluation"]["prediction_records"]
    # prediction_records = load_pickle(
    #     join(artifacts_dir, evaluation_output_dir, filename)
    # )
    # bias_dir = generate_bias_dir(config=config)
    # Path(bias_dir).mkdir(parents=True, exist_ok=True)
    # bias_metrics_summary(prediction_records, bias_dir)


if __name__ == '__main__':
    main(config={})
