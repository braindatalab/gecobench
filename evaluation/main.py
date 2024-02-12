import ast
import json
import json
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

from common import EvaluationResult
from training.bert import create_bert_ids, create_tensor_dataset
from utils import (
    load_pickle,
    dump_as_pickle,
    generate_xai_dir,
    generate_evaluation_dir,
    generate_training_dir,
    load_model,
    load_test_data,
)


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
        EvaluationResult(
            model_name=xai_record.model_name,
            model_repetition_number=xai_record.model_repetition_number,
            dataset_type=xai_record.dataset_type,
            attribution_method=xai_record.attribution_method,
        )
    )
    output.update(scores)
    return output


def evaluate(data: pd.DataFrame) -> list[dict]:
    results = list()
    for k, row in tqdm(data.iterrows()):
        scores = calculate_scores(
            attribution=np.abs(np.array(row['attribution'])),
            ground_truth=np.array(row['ground_truth']),
        )
        result = bundle_evaluation_results(xai_record=row, scores=scores)
        results += [result]

    return results


def create_bert_tensor_data(data: dict, config: dict) -> dict:
    output = dict(tensors=dict(), sentences=dict())
    for name, dataset in data.items():
        bert_tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            revision=config['training']['bert_revision'],
        )
        sentences, target = dataset['sentence'].tolist(), dataset['target'].tolist()
        bert_ids = create_bert_ids(data=sentences, tokenizer=bert_tokenizer)
        tensor_data = create_tensor_dataset(
            data=bert_ids, target=target, tokenizer=bert_tokenizer
        )
        output['sentences'][name] = sentences
        output['tensors'][name] = (
            tensor_data.tensors[0],
            tensor_data.tensors[1],
            tensor_data.tensors[2],
        )
    return output


def create_dataset_with_predictions(
    data: dict, records: list, config: dict
) -> pd.DataFrame:
    data_dict = {
        'model_repetition_number': list(),
        'model_name': list(),
        'sentence': list(),
        'target': list(),
        'prediction': list(),
        'dataset_type': list(),
    }
    tensor_data = create_bert_tensor_data(data=data, config=config)
    for dataset_name, model_params, model_path, _ in tqdm(records):
        x, attention_mask, target = tensor_data['tensors'][dataset_name]
        model = load_model(path=model_path)
        prediction = torch.argmax(model(x, attention_mask=attention_mask).logits, dim=1)
        n = target.shape[0]
        data_dict['model_repetition_number'] += n * [model_params['repetition']]
        data_dict['model_name'] += n * [model_params['model_name']]
        data_dict['sentence'] += [
            str(sentence) for sentence in tensor_data['sentences'][dataset_name]
        ]
        data_dict['target'] += target.detach().numpy().tolist()
        data_dict['prediction'] += prediction.detach().numpy().tolist()
        data_dict['dataset_type'] += n * [dataset_name]

    return pd.DataFrame(data_dict)


def load_xai_results(config: dict) -> list:
    output = list()
    file_path = join(generate_xai_dir(config=config), config["xai"]["xai_records"])
    xai_result_paths = load_pickle(file_path=file_path)
    for result_path in tqdm(xai_result_paths):
        xai_records = load_pickle(file_path=result_path)
        for record in xai_records:
            record.sentence = str(record.sentence)
            output += [asdict(record)]

    return output


def get_correctly_classified_samples(
    xai_data: pd.DataFrame, predication_data: pd.DataFrame
) -> pd.DataFrame:
    merge_columns = [
        'model_repetition_number',
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
    training_records_path = join(
        generate_training_dir(config=config), config['training']['training_records']
    )
    training_records = load_pickle(file_path=training_records_path)
    test_data = load_test_data(config=config)

    logger.info(f'Compute prediction dataset.')
    data = create_dataset_with_predictions(
        data=test_data, records=training_records, config=config
    )
    return data


def create_xai_data(config: dict) -> pd.DataFrame:
    xai_df = pd.DataFrame(load_xai_results(config=config))
    return xai_df


def main(config: Dict) -> None:
    output_dir = generate_evaluation_dir(config=config)
    data_with_predictions = create_prediction_data(config=config)
    data_with_predictions.to_csv(join(output_dir, 'data_with_predictions.csv'))

    xai_data = create_xai_data(config=config)
    xai_data.to_csv(join(output_dir, 'xai_data.csv'))

    evaluation_data = get_correctly_classified_samples(
        xai_data=xai_data, predication_data=data_with_predictions
    )
    logger.info(f"Calculate evaluation scores.")
    evaluation_results = evaluate(data=evaluation_data)
    filename = config["evaluation"]["evaluation_records"]
    logger.info(f"Output path: {join(output_dir, filename)}")
    dump_as_pickle(data=evaluation_results, output_dir=output_dir, filename=filename)


if __name__ == "__main__":
    main(config={})
