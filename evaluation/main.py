import os
import warnings
from dataclasses import asdict
from os.path import join
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import auc, roc_curve
from sklearn.metrics._ranking import _binary_clf_curve, average_precision_score
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer

from common import (
    XAIEvaluationResult,
    ModelEvaluationResult,
    EvaluationResult,
    SaveVersion,
)
from training.bert import create_bert_ids, create_tensor_dataset
from training.bert_zero_shot_utils import (
    determine_gender_type,
    get_zero_shot_prompt_function,
    PROMPT_TEMPLATES,
    zero_shot_prediction,
    transform_predicted_tokens_to_labels,
    extract_original_sentence_from_prompt,
    remove_empty_strings_from_list,
    get_slicing_indices_of_sentence_embedded_in_prompt,
    format_logits,
    extract_attribution_for_original_sentence_from_prompt_attribution,
)
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
    get_num_labels,
    BERT_ZERO_SHOT,
)
from xai.main import load_test_data
from xai.methods import normalize_attributions

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


def mass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.sum(y_pred[1 == y_true])
    b = np.sum(y_pred)
    return np.divide(a, b, where=(0 != b))


def relative_mass_accuracy(ma1: float, ma2: float, epsilon=1e-6) -> float:
    output = np.log(
        1.0 if ma1 == ma2 else ma1 / ma2 - 1 if ma2 > 0 else 1 / epsilon - 1
    )
    return output


def calculate_scores(
    attribution: np.ndarray, attribution_zero_shot: np.ndarray, ground_truth: np.ndarray
) -> Dict:
    roc_auc = roc_auc_scores(y_true=ground_truth, y_score=attribution)
    precision, recall, thresholds, specificity = precision_curves(
        y_true=ground_truth, probas_pred=attribution
    )
    auc_value = auc(x=recall, y=precision)
    avg_precision = average_precision_score(y_true=ground_truth, y_score=attribution)
    precision_specificity = precision_specificity_score(
        precision=precision, specificity=specificity, threshold=0.5
    )
    top_k_precision = top_k_precision_score(y_true=ground_truth, y_score=attribution)

    ma = mass_accuracy(y_true=ground_truth, y_pred=attribution)
    ma_zero_shot = mass_accuracy(y_true=ground_truth, y_pred=attribution_zero_shot)
    relative_ma_to_ma_zero_shot = relative_mass_accuracy(ma1=ma, ma2=ma_zero_shot)

    return dict(
        roc_auc=roc_auc,
        precision_recall_auc=auc_value,
        avg_precision=avg_precision,
        precision_specificity=precision_specificity,
        top_k_precision=top_k_precision,
        mass_accuracy=ma,
        mass_accuracy_zero_shot=ma_zero_shot,
        mass_accuracy_relative=relative_ma_to_ma_zero_shot,
    )


def evaluate_row(
    idx_row: tuple[int, pd.Series], only_correctly_classified: bool
) -> dict:
    _, xai_record = idx_row

    attribution = np.array(xai_record['attribution'])
    if len(attribution) != len(xai_record['attribution_zero_shot_stripped']):
        raise ValueError(
            'Zero-shot attribution length does not match the original sentence length.'
        )

    scores = calculate_scores(
        attribution=np.abs(attribution),
        attribution_zero_shot=np.abs(
            np.array(xai_record['attribution_zero_shot_stripped'])
        ),
        ground_truth=np.array(xai_record['ground_truth']),
    )

    output = asdict(
        XAIEvaluationResult(
            model_name=xai_record.model_name,
            model_version=xai_record.model_version,
            model_repetition_number=xai_record.model_repetition_number,
            dataset_type=xai_record.dataset_type,
            attribution_method=xai_record.attribution_method,
        )
    )
    output["only_correctly_classified"] = only_correctly_classified
    output.update(scores)
    return output


def evaluate_xai(data: pd.DataFrame, only_correctly_classified: bool) -> list[dict]:
    results = []

    for idx_row in tqdm(data.iterrows(), desc="Evaluate XAI results", total=len(data)):
        try:
            results += [evaluate_row(idx_row, only_correctly_classified)]
        except Exception as e:
            logger.error(f"Error evaluating row: {idx_row[0]} - {e}")

    return results


def create_bert_tensor_data(
    data: dict, config: dict, is_zero_shot: bool = False
) -> dict:
    output = dict(dataset=dict(), sentences=dict())
    for name, dataset in data.items():
        bert_tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased',
            revision=config['training']['bert_revision'],
        )
        sentences, target = dataset['sentence'].tolist(), dataset['target'].tolist()
        logger.info(f"Create tensor data for dataset: {name}")
        gender_type_of_dataset = determine_gender_type(dataset_name=name)
        zero_shot_prompt = get_zero_shot_prompt_function(
            prompt_templates=PROMPT_TEMPLATES[gender_type_of_dataset], index=0
        )
        bert_ids, valid_idxs = create_bert_ids(
            data=sentences,
            tokenizer=bert_tokenizer,
            type=f"{name}_test_bert_ids",
            config=config,
            sentence_context=zero_shot_prompt if is_zero_shot else None,
        )

        target = [target[i] for i in valid_idxs]

        logger.info(f"Created bert ids for dataset: {name}")
        tensor_data = create_tensor_dataset(
            data=bert_ids, target=target, tokenizer=bert_tokenizer, include_idx=True
        )

        output['sentences'][name] = sentences
        output['dataset'][name] = tensor_data
    return output


def zero_shot_model_predictions(
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    model: torch.nn.Module,
    config: dict,
    target: Tensor,
    dataset_name: str,
) -> Tuple[Tensor, Tensor]:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased',
        revision=config['training']['bert_revision'],
    )
    with torch.no_grad():
        tokens, token_ids, logits = zero_shot_prediction(
            model=model,
            input_ids=x,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        prediction = transform_predicted_tokens_to_labels(predictions=tokens)
        output_logits = format_logits(
            logits=logits, target=target, token_ids=token_ids, dataset_name=dataset_name
        )
    return torch.tensor(prediction), output_logits


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
        'pred_probs': list(),
    }
    artifacts_dir = generate_artifacts_dir(config=config)
    tensor_data = create_bert_tensor_data(data=data, config=config)
    tensor_data_zero_shot = create_bert_tensor_data(
        data=data, config=config, is_zero_shot=True
    )
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

        is_zero_shot = model_params['model_name'] == BERT_ZERO_SHOT
        for dataset_name in datasets:
            sentences = tensor_data["sentences"][dataset_name]
            if is_zero_shot:
                dataset = tensor_data_zero_shot["dataset"][dataset_name]
            else:
                dataset = tensor_data["dataset"][dataset_name]

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

                    if is_zero_shot:
                        prediction, logits = zero_shot_model_predictions(
                            x=x,
                            attention_mask=attention_mask,
                            model=model,
                            config=config,
                            target=target,
                            dataset_name=dataset_name,
                        )
                    else:
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
                        sentences[sentence_idx] for sentence_idx in sentence_idxs
                    ]
                    data_dict['target'] += target.detach().cpu().numpy().tolist()
                    data_dict['prediction'] += (
                        prediction.detach().cpu().numpy().tolist()
                    )
                    data_dict['logits'] += logits.detach().cpu().numpy().tolist()
                    data_dict['pred_probs'] += (
                        torch.softmax(logits, dim=1).detach().cpu().numpy().tolist()
                    )
                    data_dict['dataset_type'] += n * [dataset_name]

    return pd.DataFrame(data_dict)


def extract_sentence_from_prompt(x: pd.Series) -> list:
    return (
        x['sentence']
        if x['model_name'] != BERT_ZERO_SHOT
        else extract_original_sentence_from_prompt(
            prompt=x['sentence'],
            prompt_templates=PROMPT_TEMPLATES[
                ('non_binary' if 'non_binary' in x['dataset_type'] else 'binary')
            ],
            index=0,
        )
    )


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
            record.model_version = record.model_version.value
            output += [asdict(record)]

    return output


def determine_correctly_classified(data: pd.DataFrame) -> pd.DataFrame:
    correctly_classified = list()

    binary_data = data[~data['dataset_type'].str.contains('non_binary')]
    non_binary_data = data[data['dataset_type'].str.contains('non_binary')]

    for d in [binary_data, non_binary_data]:
        for (d_type, s_id), df in d.groupby(['dataset_type', 'sentence_idx']):
            # predictions = df['pred_probabilities'].map(lambda x: np.argmax(x))
            if 1 == np.prod(df['target'] == df['prediction']):
                correctly_classified += [(d_type, s_id)]

    correctly_classified_mask = data.apply(
        lambda x: (x['dataset_type'], x['sentence_idx']) in correctly_classified, axis=1
    )

    return data[correctly_classified_mask]


def merge_xai_results_with_prediction_data(
    xai_data: pd.DataFrame, predication_data: pd.DataFrame
) -> pd.DataFrame:

    merge_columns = [
        'model_repetition_number',
        'model_version',
        'model_name',
        'sentence_str',
        'target',
        'dataset_type',
    ]

    xai_data['sentence_str'] = xai_data['sentence'].map(
        lambda x: ''.join(x).lower().replace(' ', '')
    )
    predication_data['sentence_str'] = predication_data['sentence'].map(
        lambda x: ''.join(x).lower().replace(' ', '')
    )

    output = pd.merge(xai_data, predication_data, how='outer', on=merge_columns)

    output.rename(
        columns={
            'sentence_x': 'sentence',
        },
        inplace=True,
    )

    return output


def merge_with_zero_shot_model_results(
    data: pd.DataFrame, zero_shot_model_data: pd.DataFrame
) -> pd.DataFrame:
    merge_columns = [
        'model_repetition_number',
        'model_version',
        'attribution_method',
        'sentence_str',
        'target',
        'dataset_type',
    ]
    output = pd.merge(data, zero_shot_model_data, how='inner', on=merge_columns)

    output.rename(
        columns={
            'model_name_x': 'model_name',
            'attribution_x': 'attribution',
            'ground_truth_x': 'ground_truth',
            'sentence_idx_x': 'sentence_idx',
            'sentence_y': 'sentence',
            'original_sentence_x': 'original_sentence',
            'prediction_x': 'prediction',
            'attribution_y': 'attribution_zero_shot',
            'pred_probs_x': 'pred_probs',
        },
        inplace=True,
    )

    return output


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


def evaluate_model_performance(config: Dict) -> list:
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
        artifacts_dir, evaluation_output_dir, config["evaluation"]["prediction_records"]
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


# strip the zero-shot attributions. for the zero-shot model, the sentences got
# wrapped in a prompt template rendering the corresponding attribution arrays
# as longer than the actual sentence. we need to strip the zero-shot attributions
# to match the length of the original sentence.
def strip_zero_shot_attributions(x) -> list:
    a = extract_attribution_for_original_sentence_from_prompt_attribution(
        attribution_prompt=x['attribution_zero_shot'],
        prompt_templates=PROMPT_TEMPLATES[
            ('non_binary' if 'non_binary' in x['dataset_type'] else 'binary')
        ],
        index=0,
    )
    return normalize_attributions(a=np.array(a)[np.newaxis, :]).tolist()


def evaluate_xai_performance(config: Dict) -> Tuple:
    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_output_dir = generate_evaluation_dir(config=config)
    output_dir = join(artifacts_dir, evaluation_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    xai_data = create_xai_data(config=config)
    xai_data.to_pickle(join(output_dir, 'xai_data.pkl'))

    xai_data['original_sentence'] = xai_data.apply(extract_sentence_from_prompt, axis=1)

    data_with_predictions = create_prediction_data(config=config)
    data_with_predictions.to_pickle(
        join(output_dir, config["evaluation"]["data_prediction_records"])
    )

    evaluation_data_all = merge_xai_results_with_prediction_data(
        xai_data=xai_data, predication_data=data_with_predictions
    )

    zero_shot_model_results = evaluation_data_all[
        evaluation_data_all['model_name'] == BERT_ZERO_SHOT
    ]

    evaluation_data_all = merge_with_zero_shot_model_results(
        data=evaluation_data_all,
        zero_shot_model_data=zero_shot_model_results,
    )

    evaluation_data_all['attribution_zero_shot_stripped'] = evaluation_data_all.apply(
        strip_zero_shot_attributions, axis=1
    )

    evaluation_data_all['attribution'] = evaluation_data_all.apply(
        lambda x: (
            x['attribution']
            if x['model_name'] != BERT_ZERO_SHOT
            else x['attribution_zero_shot_stripped']
        ),
        axis=1,
    )

    evaluation_data_all = evaluation_data_all[
        evaluation_data_all["model_version"] == SaveVersion.best.value
    ]

    evaluation_data_correct = determine_correctly_classified(data=evaluation_data_all)

    # Calculate gender difference in prediction & attributions
    difference_config = config["evaluation"]["gender_difference"]
    binary_data = evaluation_data_correct[
        ~evaluation_data_correct['dataset_type'].str.contains('non_binary')
    ]
    # if difference_config["correctly_classified"]:
    #     prepare_difference_data(
    #         df=binary_data,
    #         idxs=difference_config["prediction_idx"],
    #         output_dir=output_dir,
    #     )
    # else:
    #     prepare_difference_data(
    #         df=binary_data,
    #         idxs=difference_config["prediction_idx"],
    #         output_dir=output_dir,
    #     )

    logger.info("Calculate evaluation scores.")
    xai_evaluation_results = evaluate_xai(
        data=evaluation_data_correct, only_correctly_classified=True
    )
    xai_evaluation_results_all = evaluate_xai(
        data=evaluation_data_all, only_correctly_classified=False
    )

    return xai_evaluation_results, xai_evaluation_results_all


def main(config: Dict) -> None:
    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_output_dir = generate_evaluation_dir(config=config)

    xai_evaluation_results, xai_evaluation_results_all = evaluate_xai_performance(
        config=config
    )
    model_evaluation_results = evaluate_model_performance(config=config)

    evaluation_results = EvaluationResult(
        xai_results_correct=xai_evaluation_results,
        xai_results_all=xai_evaluation_results_all,
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
