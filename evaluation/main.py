import os
import warnings
from dataclasses import asdict
from os.path import join
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import auc, roc_curve
from sklearn.metrics._ranking import _binary_clf_curve, average_precision_score
from tqdm import tqdm
from transformers import BertTokenizer
import torch

from common import XAIEvaluationResult, ModelEvaluationResult, EvaluationResult
from training.bert import create_bert_ids, create_tensor_dataset
from xai.main import load_test_data
from utils import (
    filter_eval_datasets,
    filter_train_datasets,
    validate_dataset_key,
    generate_data_dir,
    load_jsonl_as_df,
    load_pickle,
    dump_as_pickle,
    generate_xai_dir,
    generate_evaluation_dir,
    generate_training_dir,
    load_model,
    generate_artifacts_dir,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def roc_auc_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    auc_value = auc(x=fpr, y=tpr)
    return auc_value


def precision_curves(y_true, probas_pred, *, pos_label=None, sample_weight=None):
    """
    Minor adaption of corresponding scikit-learn function:
    We added the calculation of the specificity.
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))
    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    specificity = 1 - np.divide(fps, fps[-1], where=(fps[-1] != 0))
    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return (
        np.hstack((precision[sl], 1)),
        np.hstack((recall[sl], 0)),
        thresholds[sl],
        np.hstack((specificity[sl], 1)),
    )


def precision_specificity_score(
    precision: np.ndarray, specificity: np.ndarray, threshold: float = 0.9
) -> float:
    score = 0.0
    if any(threshold < specificity[:-1]):
        score = precision[:-1][threshold < specificity[:-1]][0]
    return score


def top_k_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    top_k_score_indices = set(np.argsort(y_score)[::-1][y_true])
    true_ground_truth_indices = set(np.arange(y_true.shape[0])[y_true])
    intersection = true_ground_truth_indices.intersection(top_k_score_indices)
    return float(len(intersection)) / float(len(true_ground_truth_indices))


def mass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    a = np.sum(y_pred[1 == y_true])
    b = np.sum(y_pred)
    return np.divide(a, b, where=(0 != b))


def compute_precision_based_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    precision, recall, thresholds, specificity = precision_curves(
        y_true=y_true, probas_pred=y_score
    )
    auc_value = auc(x=recall, y=precision)
    avg_precision = average_precision_score(y_true=y_true, y_score=y_score)
    precision_specificity = precision_specificity_score(
        precision=precision, specificity=specificity, threshold=0.5
    )
    top_k_precision = top_k_precision_score(y_true=y_true, y_score=y_score)
    return dict(
        precision_recall_auc=auc_value,
        avg_precision=avg_precision,
        precision_specificity=precision_specificity,
        top_k_precision=top_k_precision,
    )


def calculate_scores(attribution: np.ndarray, ground_truth: np.ndarray) -> Dict:
    result = dict(
        roc_auc=roc_auc_scores(y_true=ground_truth, y_score=attribution),
        mass_accuracy=mass_accuracy(y_true=ground_truth, y_pred=attribution),
    )
    precision_based_scores = compute_precision_based_scores(
        y_true=ground_truth, y_score=attribution
    )
    result.update(precision_based_scores)
    return result


def bundle_evaluation_results(xai_record: pd.Series, scores: dict) -> dict:
    output = asdict(
        XAIEvaluationResult(
            model_name=xai_record.model_name,
            model_version=xai_record.model_version,
            model_repetition_number=xai_record.model_repetition_number,
            dataset_type=xai_record.dataset_type,
            attribution_method=xai_record.attribution_method,
        )
    )
    output.update(scores)
    return output


def evaluate_xai(data: pd.DataFrame) -> list[dict]:
    results = list()
    for k, row in tqdm(data.iterrows(), total=len(data)):
        scores = calculate_scores(
            attribution=np.abs(np.array(row['attribution'])),
            ground_truth=np.array(row['ground_truth']),
        )
        result = bundle_evaluation_results(xai_record=row, scores=scores)
        results += [result]

    return results


def create_bert_tensor_data(data: dict, config: dict) -> dict:
    output = dict(dataset=dict(), sentences=dict())
    for name, dataset in data.items():
        bert_tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            revision=config['training']['bert_revision'],
        )
        sentences, target = dataset['sentence'].tolist(), dataset['target'].tolist()
        logger.info(f"Create tensor data for dataset: {name}")
        bert_ids, valid_idxs = create_bert_ids(
            data=sentences,
            tokenizer=bert_tokenizer,
            type=f"{name}_test_bert_ids",
            config=config,
        )

        target = [target[i] for i in valid_idxs]

        logger.info(f"Created bert ids for dataset: {name}")
        tensor_data = create_tensor_dataset(
            data=bert_ids, target=target, tokenizer=bert_tokenizer, include_idx=True
        )

        output['sentences'][name] = sentences
        output['dataset'][name] = tensor_data
    return output


def create_dataset_with_predictions(
    data: dict, records: list, config: dict
) -> pd.DataFrame:
    data_dict = {
        'model_repetition_number': list(),
        'model_version': list(),
        'model_name': list(),
        'sentence': list(),
        'target': list(),
        'prediction': list(),
        'dataset_type': list(),
        'logits': list(),
    }
    artifacts_dir = generate_artifacts_dir(config=config)
    tensor_data = create_bert_tensor_data(data=data, config=config)
    for trained_on_dataset_name, model_params, model_path, _ in tqdm(
        records, desc='Evaluate model predictions'
    ):

        # If model was trained on a dataset e.g. gender_all, only evaluate on that dataset.
        # Otherwise, e.g. in the case of sentiment analysis, evaluate on all datasets.
        datasets = [trained_on_dataset_name]
        if trained_on_dataset_name not in data:
            datasets = filter_eval_datasets(config)

        logger.info(
            f"Model: {model_params['model_name']}, trained on: {trained_on_dataset_name} - Evaluate on: {datasets}"
        )

        for dataset_name in datasets:
            dataset = tensor_data["dataset"][dataset_name]
            sentences = tensor_data["sentences"][dataset_name]

            model = load_model(path=join(artifacts_dir, model_path))
            model.to(DEVICE)

            with torch.no_grad():
                for batch in torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=256,
                    shuffle=False,
                ):
                    x, attention_mask, target, sentence_idxs = batch
                    x, attention_mask, target = (
                        x.to(DEVICE),
                        attention_mask.to(DEVICE),
                        target.to(DEVICE),
                    )

                    logits = model(x, attention_mask=attention_mask).logits
                    prediction = torch.argmax(logits, dim=1)
                    n = target.shape[0]
                    data_dict['model_repetition_number'] += n * [
                        model_params['repetition']
                    ]
                    data_dict['model_version'] += n * [
                        model_params['save_version'].value
                    ]
                    data_dict['model_name'] += n * [model_params['model_name']]
                    data_dict['sentence'] += [
                        str(sentences[sentence_idx]) for sentence_idx in sentence_idxs
                    ]
                    data_dict['target'] += target.detach().cpu().numpy().tolist()
                    data_dict['prediction'] += (
                        prediction.detach().cpu().numpy().tolist()
                    )
                    data_dict['logits'] += logits.detach().cpu().numpy().tolist()
                    data_dict['dataset_type'] += n * [dataset_name]

    return pd.DataFrame(data_dict)


def load_xai_results(config: dict) -> list:
    output = list()

    artifacts_dir = generate_artifacts_dir(config=config)
    file_path = join(
        artifacts_dir, generate_xai_dir(config=config), config["xai"]["xai_records"]
    )
    xai_result_paths = load_pickle(file_path=file_path)
    for result_path in tqdm(xai_result_paths, desc="Loading XAI results"):
        xai_records = load_pickle(file_path=join(artifacts_dir, result_path))
        for record in xai_records:
            record.sentence = str(record.sentence)
            record.model_version = record.model_version.value
            output += [asdict(record)]

    return output


def get_correctly_classified_samples(
    xai_data: pd.DataFrame, predication_data: pd.DataFrame
) -> pd.DataFrame:

    merge_columns = [
        'model_repetition_number',
        'model_version',
        'model_name',
        'sentence',
        'target',
        'dataset_type',
    ]
    data_for_evaluation = pd.merge(
        xai_data, predication_data, how='outer', on=merge_columns
    )

    correctly_classified_mask = (
        data_for_evaluation['target'] == data_for_evaluation['prediction']
    )
    return data_for_evaluation[correctly_classified_mask]


def create_prediction_data(config: dict) -> pd.DataFrame:
    artifacts_dir = generate_artifacts_dir(config=config)
    training_records_path = join(
        artifacts_dir,
        generate_training_dir(config=config),
        config['training']['training_records'],
    )
    training_records = load_pickle(file_path=training_records_path)
    test_data = load_test_data(config=config)

    logger.info('Compute prediction dataset.')
    data = create_dataset_with_predictions(
        data=test_data, records=training_records, config=config
    )
    return data


def create_xai_data(config: dict) -> pd.DataFrame:
    xai_df = pd.DataFrame(load_xai_results(config=config))
    return xai_df


def load_test_data_for_trained_on_ds(config: dict) -> pd.DataFrame:
    """
    Load the test data for the datasets the model was trained on.
    The datasets with tag "train" in the config.
    """

    data = dict()
    for dataset in filter_train_datasets(config):
        validate_dataset_key(dataset_key=dataset)
        data[dataset] = load_jsonl_as_df(
            file_path=join(generate_data_dir(config), dataset, "test.jsonl")
        )

    return data


def evaluate_model_performance(config: Dict) -> None:
    logger.info("Evaluate model performance.")
    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_output_dir = generate_evaluation_dir(config=config)
    os.makedirs(join(artifacts_dir, evaluation_output_dir), exist_ok=True)

    training_records_path = join(
        artifacts_dir,
        generate_training_dir(config=config),
        config['training']['training_records'],
    )
    training_records = load_pickle(file_path=training_records_path)
    test_data = load_test_data_for_trained_on_ds(config=config)
    logger.info(f"Loaded test data for trained on datasets: {test_data.keys()}")

    predictions_output_path = join(
        artifacts_dir, evaluation_output_dir, "model_performance_predictions.pkl"
    )

    logger.info("Compute prediction dataset.")
    data = create_dataset_with_predictions(
        data=test_data, records=training_records, config=config
    )
    logger.info("Prediction dataset created.")

    # Save predicitons
    data.to_pickle(predictions_output_path)

    logger.info("Evaluate model predictions.")
    results = []
    for (
        model_name,
        model_version,
        model_repetition,
        dataset_type,
    ), group in data.groupby(
        ['model_name', 'model_version', 'model_repetition_number', 'dataset_type']
    ):
        accuracy_score = accuracy(
            y_true=group['target'].values, y_pred=group['prediction'].values
        )

        results.append(
            ModelEvaluationResult(
                model_name=model_name,
                model_version=model_version,
                model_repetition_number=model_repetition,
                dataset_type=dataset_type,
                accuracy=accuracy_score,
            )
        )

    logger.info("Model evaluation finished.")

    return results


def evaluate_xai_performance(config: Dict) -> None:
    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_output_dir = generate_evaluation_dir(config=config)
    os.makedirs(join(artifacts_dir, evaluation_output_dir), exist_ok=True)

    xai_data = create_xai_data(config=config)
    xai_data.to_pickle(join(artifacts_dir, evaluation_output_dir, 'xai_data.pkl'))

    data_with_predictions = create_prediction_data(config=config)
    data_with_predictions.to_pickle(
        join(artifacts_dir, evaluation_output_dir, 'data_with_predictions.pkl')
    )

    evaluation_data = get_correctly_classified_samples(
        xai_data=xai_data, predication_data=data_with_predictions
    )
    logger.info("Calculate evaluation scores.")
    xai_evaluation_results = evaluate_xai(data=evaluation_data)

    return xai_evaluation_results


def main(config: Dict) -> None:
    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_output_dir = generate_evaluation_dir(config=config)

    xai_evaluation_results = evaluate_xai_performance(config=config)
    model_evaluation_results = evaluate_model_performance(config=config)

    evaluation_results = EvaluationResult(
        xai_results=xai_evaluation_results,
        model_results=model_evaluation_results,
    )

    filename = config["evaluation"]["evaluation_records"]
    logger.info(f"Output path: {join(artifacts_dir, evaluation_output_dir, filename)}")
    dump_as_pickle(
        data=evaluation_results,
        output_dir=join(artifacts_dir, evaluation_output_dir),
        filename=filename,
    )


if __name__ == "__main__":
    main(config={})
