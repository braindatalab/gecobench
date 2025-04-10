import json
import os.path
from collections import defaultdict
from itertools import chain
from os.path import join
from typing import Dict, Any, Callable

import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import TokenReferenceBase
from joblib import Parallel, delayed
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer, GPT2Tokenizer

from common import XAIResult, validate_dataset_key

from training.main import get_prompt_template

from training.bert_zero_shot_utils import (
    determine_gender_type,
    get_zero_shot_prompt_function,
)

from training.bert import (
    create_bert_ids,
    get_bert_ids,
    get_bert_tokenizer,
)

from training.gpt2 import (
    create_gpt2_ids,
    get_gpt2_ids,
    get_gpt2_tokenizer,
)

from utils import (
    filter_eval_datasets,
    generate_training_dir,
    load_jsonl_as_df,
    load_pickle,
    dump_as_pickle,
    generate_xai_dir,
    append_date,
    load_model,
    generate_data_dir,
    generate_artifacts_dir,
    BERT_MODEL_TYPE,
    ONE_LAYER_ATTENTION_MODEL_TYPE,
    BERT_ZERO_SHOT,
    GPT2_MODEL_TYPE,
    GPT2_ZERO_SHOT,
    determine_model_type,
    ZERO_SHOT,
    load_json_file,
    dump_as_json_file,
)
from xai.methods import (
    get_captum_attributions,
    calculate_covariance_between_words_target,
    get_covariance_between_words_target,
)

DEVICE = 'cpu'
ALL_BUT_CLS_SEP = slice(1, -1)
ALL_BUT_END_OF_TEXT = slice(0, -1)
SPACE = ' '

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


def create_gpt2_to_original_token_mapping_from_sentence(
    tokenizer: GPT2Tokenizer, sentence: list[str]
) -> dict:
    output = dict()
    for k, word in enumerate(sentence):
        gpt2_ids = get_gpt2_ids(tokenizer=tokenizer, token=word)
        for gpt2_id in gpt2_ids:
            key = tokenizer.decode(gpt2_id).replace(' ', '') + f'{k}'
            output[key] = word + f'{k}'
    return output


def create_gpt2_to_original_token_mapping(data: list, tokenizer: GPT2Tokenizer) -> list:
    mappings = list()
    for k, sentence in enumerate(data):
        mappings += [
            create_gpt2_to_original_token_mapping_from_sentence(
                tokenizer=tokenizer, sentence=sentence
            )
        ]
    return mappings


def create_gpt2_reference_tokens(
    gpt2_tokenizer: GPT2Tokenizer, sequence_length: int
) -> Tensor:
    reference_tokens_pad = TokenReferenceBase(
        reference_token_idx=gpt2_tokenizer.pad_token_id
    )
    reference_indices = reference_tokens_pad.generate_reference(
        sequence_length=sequence_length, device=DEVICE
    ).unsqueeze(0)
    reference_indices[0, -1] = gpt2_tokenizer.eos_token_id
    return reference_indices


def create_xai_results(
    attributions: dict,
    row: pd.Series,
    model_params: dict,
    dataset_type: str,
    pred_probabilities: list = None,
    zero_shot_prompt: Callable = None,
) -> list:
    results = list()
    for xai_method, attribution in attributions.items():
        results += [
            XAIResult(
                model_repetition_number=model_params['repetition'],
                model_name=model_params['model_name'],
                model_version=model_params['save_version'],
                dataset_type=dataset_type,
                target=row['target'],
                attribution_method=xai_method,
                sentence=row['sentence'],
                prompt=(
                    zero_shot_prompt(row['sentence'])
                    if zero_shot_prompt is not None
                    else None
                ),
                raw_attribution=attribution,
                ground_truth=row['ground_truth'],
                sentence_idx=row["sentence_idx"],
                pred_probabilities=pred_probabilities,
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


def map_zero_shot_bert_attributions_to_original_tokens(
    model_type: str, result: XAIResult, config: dict
) -> list:
    tokenizer = get_tokenizer[model_type](config)
    token_mapping = create_model_token_to_original_token_mapping[model_type](
        [result.prompt], tokenizer
    )
    original_token_to_attribution_mapping = dict()
    for k, word in enumerate(result.prompt):
        original_token_to_attribution_mapping[word + str(k)] = 0

    bert_token_to_attribution_mapping = dict()
    for word, attribution in zip(
        list(token_mapping[0].keys()), result.raw_attribution[ALL_BUT_CLS_SEP]
    ):
        bert_token_to_attribution_mapping[word] = attribution

    for k, v in bert_token_to_attribution_mapping.items():
        original_token_to_attribution_mapping[token_mapping[0][k]] += v

    return list(original_token_to_attribution_mapping.values())


def map_gpt2_attributions_to_original_tokens(
    model_type: str, result: XAIResult, config: dict
) -> list:
    tokenizer = get_tokenizer[model_type](config)
    token_mapping = create_model_token_to_original_token_mapping[model_type](
        [result.sentence], tokenizer
    )
    original_token_to_attribution_mapping = dict()
    for k, word in enumerate(result.sentence):
        original_token_to_attribution_mapping[word + str(k)] = 0

    gpt2_token_to_attribution_mapping = dict()
    for word, attribution in zip(
        list(token_mapping[0].keys()), result.raw_attribution[ALL_BUT_END_OF_TEXT]
    ):
        gpt2_token_to_attribution_mapping[word] = attribution

    for k, v in gpt2_token_to_attribution_mapping.items():
        original_token_to_attribution_mapping[token_mapping[0][k]] += v

    return list(original_token_to_attribution_mapping.values())


def map_zero_shot_gpt2_attributions_to_original_tokens(
    model_type: str, result: XAIResult, config: dict
) -> list:
    tokenizer = get_tokenizer[model_type](config)
    token_mapping = create_model_token_to_original_token_mapping[model_type](
        [result.prompt], tokenizer
    )
    original_token_to_attribution_mapping = dict()
    for k, word in enumerate(result.prompt):
        original_token_to_attribution_mapping[word + str(k)] = 0

    gpt2_token_to_attribution_mapping = dict()
    for word, attribution in zip(
        list(token_mapping[0].keys()), result.raw_attribution[ALL_BUT_CLS_SEP]
    ):
        gpt2_token_to_attribution_mapping[word] = attribution

    for k, v in gpt2_token_to_attribution_mapping.items():
        original_token_to_attribution_mapping[token_mapping[0][k]] += v

    return list(original_token_to_attribution_mapping.values())


def map_raw_attributions_to_original_tokens(
    xai_results_paths: list[str], config: dict
) -> list[XAIResult]:
    output = list()
    artifacts_dir = generate_artifacts_dir(config)

    def process_result(result, config):
        model_type = determine_model_type(s=result.model_name)
        result.attribution = raw_attributions_to_original_tokens_mapping[model_type](
            model_type, result, config
        )
        return result

    for path in xai_results_paths:
        results = load_pickle(file_path=join(artifacts_dir, path))
        results = Parallel(n_jobs=1)(
            delayed(process_result)(result, config) for result in tqdm(results)
        )

        output_dir = generate_xai_dir(config=config)
        filename = append_date(s=config['xai']['intermediate_xai_result_prefix'])
        dump_as_pickle(
            data=results, output_dir=join(artifacts_dir, output_dir), filename=filename
        )
        output += [join(output_dir, filename)]

    return output


def apply_xai_methods_on_sentence(
    model: Any,
    row: pd.Series,
    dataset_type: str,
    trained_on_dataset_name: str,
    model_params: dict,
    config: dict,
    num_samples: int,
    covariance_between_words_target: dict,
    index: int,
) -> list[XAIResult]:
    logger.info(f'Dataset type: {dataset_type}, sentence: {index} of {num_samples}')
    model_type = determine_model_type(s=model_params['model_name'])
    tokenizer = get_tokenizer[model_type](config)
    zero_shot_prompt = None
    if ZERO_SHOT in model_type:
        gender_type_of_dataset = determine_gender_type(dataset_type)
        zero_shot_prompt = get_zero_shot_prompt_function(
            prompt_templates=get_prompt_template(
                dataset_type=gender_type_of_dataset, model_type=model_type
            ),
            index=0,
        )

    token_ids = create_token_ids[model_type](
        [row['sentence']],
        tokenizer,
        '',
        None,
        zero_shot_prompt,
    )[0][0]
    token_ids_without_prompt = create_token_ids[model_type](
        [row['sentence']],
        tokenizer,
        '',
        None,
        None,
    )[0][0]
    token_ids = token_ids.to(DEVICE)
    num_ids = token_ids.shape[0]
    reference_tokens = create_reference_tokens[model_type](tokenizer, num_ids)

    # Incase the dataset_type differs from trained_on_dataset_name e.g trained on sentiment, evaluated on gender_all
    pred_probabilities = None
    xai_target = int(row["target"])
    if trained_on_dataset_name != dataset_type:
        logits = model(token_ids.unsqueeze(0))[0]
        probabilities = F.softmax(logits, dim=-1).squeeze()
        xai_target = torch.argmax(probabilities).item()
        pred_probabilities = probabilities.detach().tolist()
        logger.info(f"XAI Target: {pred_probabilities}, {xai_target}")

    attributions = get_captum_attributions(
        model=model,
        model_type=model_type,
        x=token_ids.unsqueeze(0),
        baseline=reference_tokens,
        methods=config['xai']['methods'],
        target=xai_target,
        dataset_type=dataset_type,
        tokenizer=tokenizer,
    )

    if covariance_between_words_target is not None:
        attributions.update(
            get_covariance_between_words_target(
                covariance_between_words_target=covariance_between_words_target,
                token_ids=token_ids_without_prompt,
            )
        )

    results = create_xai_results(
        attributions=attributions,
        row=row,
        model_params=model_params,
        dataset_type=dataset_type,
        pred_probabilities=pred_probabilities,
        zero_shot_prompt=zero_shot_prompt,
    )

    return results


def prepare_data_for_covariance_calculation(
    dataset: pd.DataFrame, model_name: str, config: dict, dataset_type: str
) -> dict:
    model_type = determine_model_type(s=model_name)
    vocabulary = set()
    sentences = list()
    targets = list()
    word_to_token_id = defaultdict(list)
    tokenizer = get_tokenizer[model_type](config)
    for k, row in tqdm(dataset.iterrows(), disable=True):
        token_ids = create_token_ids[model_type](
            [row['sentence']],
            tokenizer,
            '',
            None,
            None,
        )[0][0]
        decoded_words = list()
        for tid in token_ids:
            decoded_word = tokenizer.decode(tid).replace(' ', '')
            decoded_words += [decoded_word]
            word_to_token_id[decoded_word].append(tid.numpy().item())

        vocabulary.update(decoded_words)
        sentences += [SPACE.join(decoded_words)]
        targets += [row['target']]

    return {
        'vocabulary': vocabulary,
        'sentences': sentences,
        'targets': targets,
        'word_to_token_id_mapping': word_to_token_id,
    }


def apply_xai_methods(
    model: Any,
    dataset: pd.DataFrame,
    dataset_type: str,
    trained_on_dataset_name: str,
    model_params: dict,
    config: dict,
) -> list[XAIResult]:
    results = list()
    num_samples = dataset.shape[0]

    prepared_data = prepare_data_for_covariance_calculation(
        dataset=dataset,
        model_name=model_params['model_name'],
        config=config,
        dataset_type=dataset_type,
    )

    covariance_between_words_target = calculate_covariance_between_words_target(
        sentences=prepared_data['sentences'],
        targets=prepared_data['targets'],
        vocabulary=prepared_data['vocabulary'],
        word_to_token_id_mapping=prepared_data['word_to_token_id_mapping'],
    )

    # results = Parallel(n_jobs=config["xai"]["num_workers"])(
    results = Parallel(n_jobs=1)(
        delayed(apply_xai_methods_on_sentence)(
            model,
            row,
            dataset_type,
            trained_on_dataset_name,
            model_params,
            config,
            num_samples,
            covariance_between_words_target,
            k,
        )
        for k, (_, row) in enumerate(dataset.iterrows())
    )

    return list(chain.from_iterable(results))


def loop_over_training_records(
    training_records: list,
    data: dict,
    config: dict,
    tracking_file_path: str,
) -> list[str]:
    output = list()
    torch.set_num_threads(1)
    artifacts_dir = generate_artifacts_dir(config)
    for j, (trained_on_dataset_name, model_params, model_path, _) in tqdm(
        enumerate(training_records), disable=True
    ):
        # If model was trained on a dataset e.g. gender_all, only evaluate on that dataset.
        # Otherwise, e.g. in the case of sentiment analysis, evaluate on all datasets.
        datasets = [trained_on_dataset_name]
        if trained_on_dataset_name not in data:
            datasets = filter_eval_datasets(config)

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
                trained_on_dataset_name=trained_on_dataset_name,
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
            update_tracking_file(
                tracking_file_path=tracking_file_path, record=training_records[j]
            )

    return output


def load_test_data(config: dict) -> dict[pd.DataFrame]:
    data = dict()

    for dataset in filter_eval_datasets(config):
        validate_dataset_key(dataset_key=dataset)
        data[dataset] = load_jsonl_as_df(
            file_path=join(generate_data_dir(config), dataset, "test.jsonl")
        )

    return data


get_tokenizer = {
    BERT_MODEL_TYPE: get_bert_tokenizer,
    ONE_LAYER_ATTENTION_MODEL_TYPE: get_bert_tokenizer,
    BERT_ZERO_SHOT: get_bert_tokenizer,
    GPT2_MODEL_TYPE: get_gpt2_tokenizer,
    GPT2_ZERO_SHOT: get_gpt2_tokenizer,
}

create_token_ids = {
    BERT_MODEL_TYPE: create_bert_ids,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_ids,
    BERT_ZERO_SHOT: create_bert_ids,
    GPT2_MODEL_TYPE: create_gpt2_ids,
    GPT2_ZERO_SHOT: create_gpt2_ids,
}

create_model_token_to_original_token_mapping = {
    BERT_MODEL_TYPE: create_bert_to_original_token_mapping,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_to_original_token_mapping,
    BERT_ZERO_SHOT: create_bert_to_original_token_mapping,
    GPT2_MODEL_TYPE: create_gpt2_to_original_token_mapping,
    GPT2_ZERO_SHOT: create_gpt2_to_original_token_mapping,
}

create_reference_tokens = {
    BERT_MODEL_TYPE: create_bert_reference_tokens,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_reference_tokens,
    BERT_ZERO_SHOT: create_bert_reference_tokens,
    GPT2_MODEL_TYPE: create_gpt2_reference_tokens,
    GPT2_ZERO_SHOT: create_gpt2_reference_tokens,
}

raw_attributions_to_original_tokens_mapping = {
    BERT_MODEL_TYPE: map_bert_attributions_to_original_tokens,
    ONE_LAYER_ATTENTION_MODEL_TYPE: map_bert_attributions_to_original_tokens,
    BERT_ZERO_SHOT: map_bert_attributions_to_original_tokens,
    GPT2_MODEL_TYPE: map_gpt2_attributions_to_original_tokens,
    GPT2_ZERO_SHOT: map_zero_shot_gpt2_attributions_to_original_tokens,
}


def convert_record_to_str(r: tuple) -> str:
    if not isinstance(r[1]['save_version'], str):
        r[1]['save_version'] = r[1]['save_version'].value
    return f'{r[0]}-{json.dumps(r[1])}-{r[2]}-{r[3]}'


def update_tracking_file(tracking_file_path: str, record: tuple) -> list:
    if os.path.exists(tracking_file_path):
        processed_records = list(load_json_file(file_path=tracking_file_path))
    else:
        processed_records = list()
    record_str = convert_record_to_str(r=record)
    if record_str in processed_records:
        raise ValueError(
            f'The record {record_str} already exists in the tracking file {tracking_file_path}.'
        )
    processed_records.append(record_str)
    dump_as_json_file(data=processed_records, file_path=tracking_file_path)

    return processed_records


def load_training_records_with_tracking(
    training_records_path: str, tracking_file_path: str
) -> tuple:
    training_records = load_pickle(file_path=training_records_path)
    if os.path.exists(tracking_file_path):
        processed_records = list(load_json_file(file_path=tracking_file_path))
    else:
        processed_records = list()

    unprocessed_records = list()
    for training_record in training_records:
        train_record_str = convert_record_to_str(training_record)
        if train_record_str not in processed_records:
            unprocessed_records.append(training_record)

    return unprocessed_records, processed_records


def main(config: Dict) -> None:
    training_records_path = join(
        generate_artifacts_dir(config=config),
        generate_training_dir(config=config),
        config['training']['training_records'],
    )
    tracking_file_path = join(
        generate_artifacts_dir(config=config),
        generate_xai_dir(config=config),
        config['xai']['tracking_file'],
    )
    unprocessed_training_records, _ = load_training_records_with_tracking(
        training_records_path=training_records_path,
        tracking_file_path=tracking_file_path,
    )
    test_data = load_test_data(config=config)

    logger.info(f'Generate explanations.')
    intermediate_results_paths = loop_over_training_records(
        training_records=unprocessed_training_records,
        data=test_data,
        config=config,
        tracking_file_path=tracking_file_path,
    )

    logger.info(f'Dump intermediate result paths.')
    output_dir = generate_xai_dir(config=config)
    dump_as_pickle(
        data=intermediate_results_paths,
        output_dir=join(generate_artifacts_dir(config=config), output_dir),
        filename=config['xai']['intermediate_xai_result_paths'],
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
