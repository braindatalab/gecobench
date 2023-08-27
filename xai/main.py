from os.path import join
from typing import Dict, Any
from itertools import chain

import pandas as pd
import torch
from IPython.display import display, HTML
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from joblib import Parallel, delayed
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer

from common import DATASET_ALL, DATASET_SUBJECT, XAIResult
from training.bert import (
    create_tensor_dataset,
    create_bert_ids,
    get_bert_ids,
    BERT_CLASSIFICATION,
    BERT_SEPARATION,
)
from utils import (
    generate_training_dir,
    load_pickle,
    generate_data_dir,
    dump_as_pickle,
    generate_xai_dir,
    append_date,
)
from xai.methods import get_captum_attributions

DEVICE = 'cpu'
BERT_MODEL_TYPE = 'bert'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def custom_forward_grad(inputs, model):
    preds = model(inputs)[0]
    y = torch.softmax(preds, dim=1)[0][1].unsqueeze(-1)
    return y


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def determine_dataset_type(dataset_name: str) -> str:
    output = DATASET_ALL
    if DATASET_SUBJECT in dataset_name:
        output = DATASET_SUBJECT
    return output


def load_test_data(config: dict) -> dict:
    data = dict()
    data_dir = generate_data_dir(config=config)
    filename_all = config['data']['output_filenames']['test_all']
    filename_subject = config['data']['output_filenames']['test_subject']
    data[DATASET_ALL] = load_pickle(file_path=join(data_dir, filename_all))
    data[DATASET_SUBJECT] = load_pickle(file_path=join(data_dir, filename_subject))
    return data


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


def create_bert_tensor_data(data: dict) -> dict:
    output = dict()
    for name, dataset in data.items():
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sentences, target = dataset['sentence'].tolist(), dataset['target'].tolist()
        bert_ids = create_bert_ids(data=sentences, tokenizer=bert_tokenizer)
        tensor_data = create_tensor_dataset(
            data=bert_ids, target=target, tokenizer=bert_tokenizer
        )
        x, target = tensor_data.tensors[0], tensor_data.tensors[2]
        output[name] = (x, target)
    return output


def load_model(path: str) -> Any:
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    model.zero_grad()
    return model


def get_intersection_of_correctly_classified_samples(data: dict, records: list) -> dict:
    output = {key: torch.ones((len(data['all'][1]),)) for key in data.keys()}
    for dataset_name, model_params, model_path, _ in tqdm(records):
        dataset_type = determine_dataset_type(dataset_name=dataset_name)
        x, target = data[dataset_type]
        model = load_model(path=model_path)
        prediction = torch.argmax(model(x).logits, dim=1)
        output[dataset_type] *= target == prediction

    return output


def filter_data(data: dict, mask: dict) -> dict:
    output = {key: list() for key in data.keys()}
    for key, dataset in data.items():
        for sentence, target, mask_value in zip(
            dataset[0], dataset[1], mask[key].tolist()
        ):
            if 0 == mask_value:
                continue
            output[key] += [(sentence, target)]
    return output


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
<<<<<<< HEAD
    for xai_method, attribution in attributions.items():
        results += [
            XAIResult(
                model_repetition_number=model_params['repetition'],
                model_name=model_params['model_name'],
                dataset_type=dataset_type,
                target=row['target'],
                attribution_method=xai_method,
                sentence=row['sentence'],
                correct_classified_intersection=row[
                    'correctly_classified_intersection'
                ],
                raw_attribution=attribution,
                ground_truth=row['ground_truth'],
            )
        ]
=======
    num_samples = dataset.shape[0]
    for k, (_, row) in enumerate(dataset.iterrows()):
        logger.info(f'Dataset type: {dataset_type}, sentence: {k} of {num_samples}')
        model_type = determine_model_type(s=model_params['model_name'])
        tokenizer = get_tokenizer[model_type]()
        token_ids = create_token_ids[model_type]([row['sentence']], tokenizer)
        num_ids = token_ids[0].shape[0]
        reference_tokens = create_reference_tokens[model_type](tokenizer, num_ids)
        attributions = get_captum_attributions(
            model=model,
            model_type=model_type,
            x=token_ids[0].unsqueeze(0),
            baseline=reference_tokens,
            methods=config['xai']['methods'],
            target=row['target']
        )

        results += [XAIResultPerSentence(
            model_name=model_params['model_name'],
            dataset_type=dataset_type,
            target=row['target'],
            attribution_method=config['xai']['methods'],
            sentence=row['sentence'],
            correct_classified_intersection=row['correctly_classified_intersection'],
            raw_attributions_all_xai_methods=attributions,
            ground_truth=row['ground_truth']
        )]

        print("XAIResultPerSentence")
        print(results)
        print(attributions.keys())
        print(attributions)

        #if 1 < k:
        #    break

>>>>>>> a1df1a7 (deleted comment)
    return results


def get_bert_tokenizer(path: str = None) -> BertTokenizer:
    path_or_name = 'bert-base-uncased' if path is None else path
    return BertTokenizer.from_pretrained(pretrained_model_name_or_path=path_or_name)


def map_bert_attributions_to_original_tokens(
    model_type: str, result: XAIResult
) -> list:
    tokenizer = get_tokenizer[model_type]()
    token_mapping = create_model_token_to_original_token_mapping[model_type](
        [result.sentence], tokenizer
    )
    original_token_to_attribution_mapping = dict()
    for k, word in enumerate(result.sentence):
        original_token_to_attribution_mapping[word + str(k)] = 0

    bert_token_to_attribution_mapping = dict()
    for word, attribution in zip(
        list(token_mapping[0].keys()), result.raw_attribution[1:-1]
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
            ](model_type, result)

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
    tokenizer = get_tokenizer[model_type]()
    token_ids = create_token_ids[model_type]([row['sentence']], tokenizer)
    num_ids = token_ids[0].shape[0]
    reference_tokens = create_reference_tokens[model_type](tokenizer, num_ids)
    attributions = get_captum_attributions(
        model=model,
        model_type=model_type,
        x=token_ids[0].unsqueeze(0),
        baseline=reference_tokens,
        methods=config['xai']['methods'],
        target=row['target'],
    )

    results = create_xai_results(
        attributions=attributions,
        row=row,
        model_params=model_params,
        dataset_type=dataset_type,
    )

#     return results


def apply_xai_methods(
    model: Any,
    dataset: pd.DataFrame,
    dataset_type: str,
    model_params: dict,
    config: dict,
) -> list[XAIResult]:
    results = list()
    num_samples = dataset.shape[0]

    results = Parallel(n_jobs=4)(
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
    for dataset_name, model_params, model_path, _ in tqdm(training_records):
        dataset_type = determine_dataset_type(dataset_name=dataset_name)
        dataset = data[dataset_type]
        model = load_model(path=model_path)

        result = apply_xai_methods(
            model=model,
            dataset=dataset,
            config=config,
            dataset_type=dataset_type,
            model_params=model_params,
        )

        output_dir = generate_xai_dir(config=config)
        filename = (
            f'{append_date(s=config["xai"]["intermediate_raw_xai_result_prefix"])}.pkl'
        )
        dump_as_pickle(data=result, output_dir=output_dir, filename=filename)
        output += [join(output_dir, filename)]

    return output


def preprocess_artifacts(config: dict) -> tuple:
    training_records_path = join(
        generate_training_dir(config=config), config['training']['training_records']
    )
    training_records = load_pickle(file_path=training_records_path)
    test_data = load_test_data(config=config)
    tensor_data = create_bert_tensor_data(data=test_data)

    logger.info(f'Compute intersection dataset.')
    correctly_classified_mask = get_intersection_of_correctly_classified_samples(
        data=tensor_data, records=training_records
    )

    test_data['all']['correctly_classified_intersection'] = correctly_classified_mask[
        'all'
    ]
    test_data['subject'][
        'correctly_classified_intersection'
    ] = correctly_classified_mask['subject']

    return training_records, test_data


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
    training_records, test_data = preprocess_artifacts(config=config)

    logger.info(f'Generate explanations.')
    intermediate_results_paths = loop_over_training_records(
        training_records=training_records, data=test_data, config=config
    )

    logger.info('Map raw attributions to original words.')
    results = map_raw_attributions_to_original_tokens(
        xai_results_paths=intermediate_results_paths, config=config
    )

    logger.info('Dump path records.')
    output_dir = generate_xai_dir(config=config)
    dump_as_pickle(
        data=results, output_dir=output_dir, filename=config['xai']['xai_records']
    )


if __name__ == '__main__':
    main(config={})
