import os.path
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
from transformers import BertTokenizer

from common import XAIResult, validate_dataset_key

from training.bert_zero_shot_utils import (
    determine_gender_type,
    get_zero_shot_prompt_function,
    PROMPT_TEMPLATES,
)

from training.bert import (
    create_bert_ids,
    get_bert_ids,
    get_bert_tokenizer,
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
    determine_model_type,
    BERT_ZERO_SHOT,
)
from xai.methods import (
    get_captum_attributions,
    calculate_covariance_between_words_target,
    get_covariance_between_words_target,
)

DEVICE = 'cpu'
ALL_BUT_CLS_SEP = slice(1, -1)
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
                prompt=zero_shot_prompt(row['sentence']) if zero_shot_prompt is not None else None,
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


def map_raw_attributions_to_original_tokens(
    xai_results_paths: list[str], config: dict
) -> list[XAIResult]:
    output = list()
    artifacts_dir = generate_artifacts_dir(config)

    def process_result(result, config):
        model_type = determine_model_type(s=result.model_name)
        result.attribution = raw_attributions_to_original_tokens_mapping[
            model_type
        ](model_type, result, config)
        return result

    for path in xai_results_paths:
        results = load_pickle(file_path=join(artifacts_dir, path))
        results = Parallel(n_jobs=4)(
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
    if BERT_ZERO_SHOT == model_type:
        gender_type_of_dataset = determine_gender_type(dataset_type)
        zero_shot_prompt = get_zero_shot_prompt_function(
            prompt_templates=PROMPT_TEMPLATES[gender_type_of_dataset], index=0
        )
    else:
        zero_shot_prompt = None
    token_ids = create_token_ids[model_type](
        [row['sentence']],
        tokenizer,
        '',
        None,
        zero_shot_prompt,
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
                token_ids=token_ids,
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
    word_to_bert_id_mapping = dict()
    tokenizer = get_tokenizer[model_type](config)
    gender_type_of_dataset = determine_gender_type(dataset_type)
    for k, row in tqdm(dataset.iterrows()):
        zero_shot_prompt = get_zero_shot_prompt_function(
            prompt_templates=PROMPT_TEMPLATES[gender_type_of_dataset], index=0
        )
        token_ids = create_token_ids[model_type](
            [row['sentence']],
            tokenizer,
            '',
            None,
            zero_shot_prompt,
        )[0][0]
        decoded_words = list()
        for tid in token_ids:
            decoded_word = tokenizer.decode(tid).replace(' ', '')
            decoded_words += [decoded_word]
            word_to_bert_id_mapping[decoded_word] = tid.numpy().item()

        vocabulary.update(decoded_words)
        sentences += [SPACE.join(decoded_words)]
        targets += [row['target']]

    return {
        'vocabulary': vocabulary,
        'sentences': sentences,
        'targets': targets,
        'word_to_bert_id_mapping': word_to_bert_id_mapping,
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
        word_to_bert_id_mapping=prepared_data['word_to_bert_id_mapping'],
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
}

create_token_ids = {
    BERT_MODEL_TYPE: create_bert_ids,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_ids,
    BERT_ZERO_SHOT: create_bert_ids,
}

create_model_token_to_original_token_mapping = {
    BERT_MODEL_TYPE: create_bert_to_original_token_mapping,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_to_original_token_mapping,
    BERT_ZERO_SHOT: create_bert_to_original_token_mapping,
}

create_reference_tokens = {
    BERT_MODEL_TYPE: create_bert_reference_tokens,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_reference_tokens,
    BERT_ZERO_SHOT: create_bert_reference_tokens,
}

raw_attributions_to_original_tokens_mapping = {
    BERT_MODEL_TYPE: map_bert_attributions_to_original_tokens,
    ONE_LAYER_ATTENTION_MODEL_TYPE: map_bert_attributions_to_original_tokens,
    BERT_ZERO_SHOT: map_bert_attributions_to_original_tokens,
}


def main(config: Dict) -> None:
    training_records_path = join(
        generate_artifacts_dir(config=config),
        generate_training_dir(config=config),
        config['training']['training_records'],
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
