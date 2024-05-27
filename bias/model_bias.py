from itertools import chain
from itertools import chain
from os.path import join
from typing import Dict, Any
from time import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

from common import validate_dataset_key, SaveVersion
from training.bert import (
    create_bert_ids,
    get_bert_tokenizer,
)
from utils import (
    filter_eval_datasets,
    generate_training_dir,
    load_jsonl_as_df,
    load_pickle,
    dump_as_pickle,
    append_date,
    load_model,
    generate_data_dir,
    generate_artifacts_dir,
    BERT_MODEL_TYPE,
    ONE_LAYER_ATTENTION_MODEL_TYPE,
    determine_model_type,
    generate_bias_dir,
    generate_visualization_dir,
)
from visualization.common import (
    MODEL_NAME_MAP,
    DATASET_NAME_MAP,
    METRIC_NAME_MAP,
    MODEL_ORDER,
    ROW_ORDER,
    GENDER,
)

DEVICE = 'cpu'
ALL_BUT_CLS_SEP = slice(1, -1)
SPACE = ' '

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_bias_results(
    row: pd.Series,
    model_params: dict,
    dataset_type: str,
    mlm_score: float,
) -> dict:
    results = dict(
        mlm_score=mlm_score,
        model_repetition_number=model_params['repetition'],
        model_name=model_params['model_name'],
        model_version=model_params['save_version'],
        dataset_type=dataset_type,
        target=row['target'],
        sentence=row['sentence'],
        ground_truth=row['ground_truth'],
        sentence_idx=row["sentence_idx"],
    )
    return results


def calculate_model_bias_metrics_on_sentence(
    generator_model: Any,
    row: pd.Series,
    dataset_type: str,
    model_params: dict,
    config: dict,
) -> list[dict]:
    model_type = determine_model_type(s=model_params['model_name'])
    tokenizer = get_tokenizer[model_type](config)

    non_ground_truth_mask = 0 == np.array(row['ground_truth'])
    sentence_batch = np.array([row['sentence']] * np.sum(non_ground_truth_mask))
    masktoken_mask = np.zeros(sentence_batch.shape, dtype=bool)

    k = 0
    for j in range(non_ground_truth_mask.shape[0]):
        if True == non_ground_truth_mask[j]:
            masktoken_mask[k, j] = 1
            k += 1

    sentence_batch[masktoken_mask] = tokenizer.mask_token
    masked_sentences = list()
    for sentence in sentence_batch:
        masked_sentences += [' '.join(sentence)]

    token_ids = tokenizer(masked_sentences, return_tensors='pt', padding=True).to(
        DEVICE
    )
    token_logits = generator_model(**token_ids).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(token_ids['input_ids'] == tokenizer.mask_token_id)[1]
    softmax_mask_token_logits = torch.softmax(token_logits, dim=2)
    mlm_score = 0.0
    for i, k in enumerate(mask_token_index):
        # Pick the [MASK] candidates with the highest softmax
        max_softmax_mask_token_logit = torch.max(softmax_mask_token_logits[i, k, :])
        mlm_score += (-1) * torch.log(
            max_softmax_mask_token_logit
        ).detach().cpu().numpy()

    results = create_bias_results(
        mlm_score=mlm_score,
        row=row,
        model_params=model_params,
        dataset_type=dataset_type,
    )

    return results


def calculate_model_bias_metrics(
    model: Any,
    dataset: pd.DataFrame,
    dataset_type: str,
    trained_on_dataset_name: str,
    model_params: dict,
    config: dict,
) -> list[dict]:
    generator_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    generator_model.bert = model.bert
    generator_model = generator_model.to(DEVICE)

    results = []
    for _, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc=f"Processing sentences for {dataset_type} for model trained on {trained_on_dataset_name}.",
    ):
        results.append(
            calculate_model_bias_metrics_on_sentence(
                generator_model,
                row,
                dataset_type,
                model_params,
                config,
            )
        )

    return results


def loop_over_training_records(
    training_records: list, data: dict, config: dict
) -> list[str]:
    output = list()
    torch.set_num_threads(1)
    artifacts_dir = generate_artifacts_dir(config)
    for trained_on_dataset_name, model_params, model_path, _ in tqdm(training_records):
        # If model was trained on a dataset e.g. gender_all, only evaluate on that dataset.
        # Otherwise, e.g. in the case of sentiment analysis, evaluate on all datasets.
        if model_params['model_name'] == 'one_layer_attention_classification':
            continue
        datasets = [trained_on_dataset_name]
        if trained_on_dataset_name not in data:
            datasets = filter_eval_datasets(config)

        for dataset_name in datasets:
            logger.info(
                f'Processing {model_params["model_name"]} trained on {trained_on_dataset_name} with dataset {dataset_name}.'
            )
            dataset = data[dataset_name]
            model = load_model(path=join(artifacts_dir, model_path)).to(DEVICE)

            result = calculate_model_bias_metrics(
                model=model,
                dataset=dataset,
                config=config,
                dataset_type=dataset_name,
                trained_on_dataset_name=trained_on_dataset_name,
                model_params=model_params,
            )

            bias_output_dir = generate_bias_dir(config=config)
            filename = f'{append_date(s="model_bias")}.pkl'
            dump_as_pickle(
                data=result,
                output_dir=join(artifacts_dir, bias_output_dir),
                filename=filename,
            )
            output += [join(bias_output_dir, filename)]

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
}

create_token_ids = {
    BERT_MODEL_TYPE: create_bert_ids,
    ONE_LAYER_ATTENTION_MODEL_TYPE: create_bert_ids,
}


def create_results(config: Dict, artifacts_dir: str, output_dir: str):
    training_records_path = join(
        artifacts_dir,
        generate_training_dir(config=config),
        config['training']['training_records'],
    )
    training_records = load_pickle(file_path=training_records_path)
    training_records = list(
        filter(
            lambda x: x[1]["save_version"] == SaveVersion.best
            and x[1]['model_name'] != 'one_layer_attention_classification',
            training_records,
        )
    )

    test_data = load_test_data(config=config)
    logger.info(f'Generate explanations.')
    intermediate_results_paths = loop_over_training_records(
        training_records=training_records, data=test_data, config=config
    )

    logger.info('Dump path records.')
    dump_as_pickle(
        data=intermediate_results_paths,
        output_dir=output_dir,
        filename='bias_records.pkl',
    )


def visualize_results(config: Dict, artifacts_dir: str, output_dir: str):
    logger.info('Visualize model bias results.')
    data_paths = load_pickle(file_path=join(output_dir, 'bias_records.pkl'))
    list_of_dicts = list()
    for p in data_paths:
        d = load_pickle(file_path=join(artifacts_dir, p))
        list_of_dicts += d

    logger.info('Loaded data.')

    data = pd.DataFrame(list_of_dicts)

    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    data['dataset_type'] = data['dataset_type'].map(lambda x: DATASET_NAME_MAP[x])
    data['target'] = data['target'].map(lambda x: GENDER[x])

    data = data.rename(
        columns={
            "mapped_model_name": "Model",
            "dataset_type": "Dataset",
            "mlm_score": "MLM Score",
            "target": "Class",
        }
    )

    logger.info('Creating plot')
    g = sns.catplot(
        data=data,
        x='Model',
        y='MLM Score',
        order=list(
            filter(
                lambda x: x != MODEL_NAME_MAP["one_layer_attention_classification"],
                MODEL_ORDER,
            )
        ),
        col_order=ROW_ORDER,
        hue='Class',
        col='Dataset',
        kind='box',
        palette=sns.color_palette(palette='pastel'),
        fill=True,
        # height=height,
        fliersize=0,
        estimator='median',
        # aspect=9.5 / height,
        legend_out=False,
        legend=True,
    )

    sns.move_legend(g, 'upper center', bbox_to_anchor=(0.935, 0.95), ncol=1)

    g.set_titles(
        row_template='Dataset: {row_name}', col_template='Dataset: {col_name}', size=12
    )
    for ax in g.axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('MLM Score', fontsize=12)

        for tick in ax.get_xticklabels():
            tick.set_fontsize(12)

    # Set legend font size
    g._legend.set_title('Class', prop={'size': 12})
    for t in g._legend.texts:
        t.set_fontsize(12)

    visualization_dir = generate_visualization_dir(config)
    plt.savefig(
        join(visualization_dir, 'model_bias.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def main(config: Dict) -> None:
    rc('text', usetex=True)
    artifacts_dir = generate_artifacts_dir(config=config)
    output_dir = join(artifacts_dir, generate_bias_dir(config=config))

    create_results(config=config, artifacts_dir=artifacts_dir, output_dir=output_dir)
    visualize_results(config=config, artifacts_dir=artifacts_dir, output_dir=output_dir)


if __name__ == '__main__':
    main(config={})
