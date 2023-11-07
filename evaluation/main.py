import json
import json
import warnings
from dataclasses import asdict
from os.path import join
from pathlib import Path
from typing import Dict, Tuple, List
from uuid import uuid4

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import auc, roc_curve
from sklearn.metrics._ranking import _binary_clf_curve, average_precision_score
from tqdm import tqdm

from common import EvaluationResult, XAIResult
from utils import load_pickle, dump_as_pickle, generate_xai_dir, generate_evaluation_dir


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
    roc_auc = roc_auc_scores(y_true=ground_truth, y_score=attribution)
    result = dict(roc_auc=roc_auc)
    precision_based_scores = compute_precision_based_scores(
        y_true=ground_truth, y_score=attribution
    )
    result.update(precision_based_scores)
    return result


def bundle_evaluation_results(xai_result: XAIResult, scores: dict) -> dict:
    output = asdict(
        EvaluationResult(
            model_name=xai_result.model_name,
            model_repetition_number=xai_result.model_repetition_number,
            dataset_type=xai_result.dataset_type,
            attribution_method=xai_result.attribution_method,
        )
    )
    output.update(scores)
    return output


def evaluate(xai_records_paths: list) -> list[dict]:
    results = list()
    for result_path in tqdm(xai_records_paths):
        xai_records = load_pickle(file_path=result_path)
        for record in xai_records:
            scores = calculate_scores(
                attribution=np.abs(np.array(record.attribution)),
                ground_truth=np.array(record.ground_truth),
            )
            result = bundle_evaluation_results(xai_result=record, scores=scores)
            results += [result]

    return results


def main(config: Dict) -> None:
    xai_dir = generate_xai_dir(config=config)
    xai_result_paths = load_pickle(
        file_path=join(xai_dir, config["xai"]["xai_records"])
    )

    logger.info(f"Calculate evaluation scores.")
    evaluation_results = evaluate(xai_records_paths=xai_result_paths)
    output_dir = generate_evaluation_dir(config=config)
    filename = config["evaluation"]["evaluation_records"]
    logger.info(f"Output path: {join(output_dir, filename)}")
    dump_as_pickle(data=evaluation_results, output_dir=output_dir, filename=filename)


if __name__ == "__main__":
    main(config={})
