import os.path
from itertools import chain
from os.path import join
from typing import Dict, Any

import pandas as pd
import torch
from captum.attr import TokenReferenceBase
from joblib import Parallel, delayed
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer

from common import XAIResult, validate_dataset_key
from training.bert import (
    create_bert_ids,
    get_bert_ids,
    get_bert_tokenizer,
)
from utils import (
    filter_xai_datasets,
    generate_training_dir,
    load_json_file,
    load_jsonl_as_df,
    load_pickle,
    dump_as_pickle,
    generate_xai_dir,
    append_date,
    load_model,
    load_jsonl_as_dict,
    generate_data_dir,
    generate_artifacts_dir,
)
from xai.methods import get_captum_attributions

DEVICE = 'cpu'
BERT_MODEL_TYPE = 'bert'
ALL_BUT_CLS_SEP = slice(1, -1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_bert_to_original_token_mapping_from_sentence(
    tokenizer: BertTokenizer, sentence: list[str]
) -> dict:
    output = dict()
    for k, word in enumerate(sentence):
        bert_ids = get_bert_ids(tokenizer=tokenizer, token=word)
        for bert_id in bert_ids:
            key = tokenizer.decode(bert_id).replace(' ', '') + f'{k}'
            output[key] = word + f'{k}'
    return output


def create_bert_to_original_token_mapping(data: list, tokenizer: BertTokenizer) -> list:
    mappings = list()
    for k, sentence in enumerate(data):
        mappings += [
            create_bert_to_original_token_mapping_from_sentence(
                tokenizer=tokenizer, sentence=sentence
            )
        ]
    return mappings


def determine_model_type(s: str) -> str:
    result = None
    if BERT_MODEL_TYPE in s:
        result = BERT_MODEL_TYPE
    return result


def create_bert_reference_tokens(
    bert_tokenizer: BertTokenizer, sequence_length: int
) -> Tensor:
    reference_tokens_pad = TokenReferenceBase(
        reference_token_idx=bert_tokenizer.pad_token_id
    )
    reference_indices = reference_tokens_pad.generate_reference(
        sequence_length=sequence_length, device=DEVICE
    ).unsqueeze(0)
    reference_indices[0, 0] = bert_tokenizer.cls_token_id
    reference_indices[0, -1] = bert_tokenizer.sep_token_id
    return reference_indices


def create_xai_results(
    attributions: dict, row: pd.Series, model_params: dict, dataset_type: str
) -> list:
    results = list()
    for xai_method, attribution in attributions.items():
        results += [
            XAIResult(
                model_repetition_number=model_params['repetition'],
                model_name=model_params['model_name'],
                dataset_type=dataset_type,
                target=row['target'],
                attribution_method=xai_method,
                sentence=row['sentence'],
                raw_attribution=attribution,
                ground_truth=row['ground_truth'],
            )
        ]
    return results


def map_bert_attributions_to_original_tokens(
    model_type: str, result: XAIResult, config: dict
) -> list:
    tokenizer = get_tokenizer[model_type](config)
    token_mapping = create_model_token_to_original_token_mapping[model_type](
        [result.sentence], tokenizer
    )
    original_token_to_attribution_mapping = dict()
    for k, word in enumerate(result.sentence):
        original_token_to_attribution_mapping[word + str(k)] = 0

    bert_token_to_attribution_mapping = dict()
    for word, attribution in zip(
        list(token_mapping[0].keys()), result.raw_attribution[ALL_BUT_CLS_SEP]
    ):
        bert_token_to_attribution_mapping[word] = attribution

    for k, v in bert_token_to_attribution_mapping.items():
        original_token_to_attribution_mapping[token_mapping[0][k]] += v

    return list(original_token_to_attribution_mapping.values())


def map_raw_attributions_to_original_tokens(
    xai_results_paths: list[str], config: dict
) -> list[XAIResult]:
    output = list()
    for path in xai_results_paths:
        results = load_pickle(file_path=path)
        for result in results:
            model_type = determine_model_type(s=result.model_name)
            result.attribution = raw_attributions_to_original_tokens_mapping[
                model_type
            ](model_type, result, config)

        output_dir = generate_xai_dir(config=config)
        filename = append_date(s=config['xai']['intermediate_xai_result_prefix'])
        dump_as_pickle(data=results, output_dir=output_dir, filename=filename)
        output += [join(output_dir, filename)]

    return output


def apply_xai_methods_on_sentence(
    model: Any,
    row: pd.Series,
    dataset_type: str,
    model_params: dict,
    config: dict,
    num_samples: int,
    index: int,
) -> list[XAIResult]:
    logger.info(f'Dataset type: {dataset_type}, sentence: {index} of {num_samples}')
    model_type = determine_model_type(s=model_params['model_name'])
    tokenizer = get_tokenizer[model_type](config)
    token_ids = create_token_ids[model_type]([row['sentence']], tokenizer)[0]
    token_ids = token_ids.to(DEVICE)
    num_ids = token_ids.shape[0]
    reference_tokens = create_reference_tokens[model_type](tokenizer, num_ids)

    attributions = get_captum_attributions(
        model=model,
        model_type=model_type,
        x=token_ids.unsqueeze(0),
        baseline=reference_tokens,
        methods=config['xai']['methods'],
        target=row['target'],
        tokenizer=tokenizer,
    )

    results = create_xai_results(
        attributions=attributions,
        row=row,
        model_params=model_params,
        dataset_type=dataset_type,
    )

    return results


def apply_xai_methods(
    model: Any,
    dataset: pd.DataFrame,
    dataset_type: str,
    model_params: dict,
    config: dict,
) -> list[XAIResult]:
    results = list()
    num_samples = dataset.shape[0]

    results = Parallel(n_jobs=config["xai"]["num_workers"])(
        delayed(apply_xai_methods_on_sentence)(
            model, row, dataset_type, model_params, config, num_samples, k
        )
        for k, (_, row) in enumerate(dataset.iterrows())
    )

    return list(chain.from_iterable(results))


def loop_over_training_records(
    training_records: list, data: dict, config: dict
) -> list[str]:
    output = list()
    torch.set_num_threads(1)
    artifacts_dir = generate_artifacts_dir(config)
    for trained_on_dataset_name, model_params, model_path, _ in tqdm(training_records):
        # If model was trained on a dataset e.g. gender_all, only evaluate on that dataset.
        # Otherwise, e.g. in the case of sentiment analysis, evaluate on all datasets.
        datasets = [trained_on_dataset_name]
        if trained_on_dataset_name not in data:
            datasets = filter_xai_datasets(config)

        for dataset_name in datasets:
            logger.info(
                f'Processing {model_params["model_name"]} trained on {trained_on_dataset_name} with dataset {dataset_name}.'
            )
            dataset = data[dataset_name]
            model = load_model(path=join(artifacts_dir, model_path)).to(DEVICE)

            result = apply_xai_methods(
                model=model,
                dataset=dataset,
                config=config,
                dataset_type=dataset_name,
                model_params=model_params,
            )

            xai_output_dir = generate_xai_dir(config=config)
            filename = f'{append_date(s=config["xai"]["intermediate_raw_xai_result_prefix"])}.pkl'
            dump_as_pickle(
                data=result,
                output_dir=join(artifacts_dir, xai_output_dir),
                filename=filename,
            )
            output += [join(xai_output_dir, filename)]

    return output


def load_test_data(config: dict) -> dict[pd.DataFrame]:
    data = dict()

    for dataset in filter_xai_datasets(config):
        validate_dataset_key(dataset_key=dataset)
        data[dataset] = load_jsonl_as_df(
            file_path=join(generate_data_dir(config), dataset, "test.jsonl")
        )

    return data


get_tokenizer = {'bert': get_bert_tokenizer}

create_token_ids = {'bert': create_bert_ids}

create_model_token_to_original_token_mapping = {
    'bert': create_bert_to_original_token_mapping
}

create_reference_tokens = {'bert': create_bert_reference_tokens}

raw_attributions_to_original_tokens_mapping = {
    'bert': map_bert_attributions_to_original_tokens
}


def main(config: Dict) -> None:
    training_records_path = join(
        generate_training_dir(config=config), config['training']['training_records']
    )
    training_records = load_pickle(file_path=training_records_path)
    test_data = load_test_data(config=config)

    logger.info(f'Generate explanations.')
    intermediate_results_paths = loop_over_training_records(
        training_records=training_records, data=test_data, config=config
    )

    logger.info('Map raw attributions to original words.')
    results = map_raw_attributions_to_original_tokens(
        xai_results_paths=intermediate_results_paths, config=config
    )

    logger.info('Dump path records.')
    output_dir = join(
        generate_artifacts_dir(config=config), generate_xai_dir(config=config)
    )
    dump_as_pickle(
        data=results, output_dir=output_dir, filename=config['xai']['xai_records']
    )


if __name__ == '__main__':
    main(config={})
