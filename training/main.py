from os.path import join
from typing import List, Dict

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from common import DataSet, DatasetKeys, validate_dataset_key
from training.bert import train_bert
from training.bert_zero_shot import train_bert_zero_shot
from training.simple_model import train_simple_attention_model
from training.gpt2 import train_gpt2
from utils import (
    dump_as_pickle,
    load_json_file,
    load_jsonl_as_dict,
    filter_train_datasets,
    generate_training_dir,
    generate_data_dir,
    generate_artifacts_dir,
)


def split_train_data_into_train_val_data(
    x: pd.DataFrame | List, y: pd.DataFrame | List, config: Dict
) -> DataSet:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config['training']['val_size'],
        random_state=config['general']['seed'],
    )

    return DataSet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def extract_name_of_dataset(s: str) -> str:
    return s.split('/')[-1].split('.')[0]


def train_models(config: Dict) -> List:
    records = list()

    dataset_config = load_json_file(join(generate_data_dir(config), "data_config.json"))

    for name in filter_train_datasets(config):
        validate_dataset_key(name)
        dataset = load_dataset(config, name)
        num_labels = dataset_config['datasets'][name]['num_labels']
        for model_name, params in config['training']['models'].items():
            train_func = TrainModel.get(model_name, None)
            if train_func is None:
                continue
            records += train_func(dataset, name, num_labels, params, config)

    return records


def load_dataset(config: Dict, dataset_key: str) -> DataSet:
    path = join(generate_data_dir(config), dataset_key, "train.jsonl")

    raw_data = load_jsonl_as_dict(path)

    dataset = split_train_data_into_train_val_data(
        x=raw_data["sentence"], y=raw_data["target"], config=config
    )

    return dataset


TrainModel = {
    'bert_only_classification': train_bert,
    'bert_only_embedding_classification': train_bert,
    'bert_all': train_bert,
    'bert_only_embedding': train_bert,
    'bert_randomly_init_embedding_classification': train_bert,
    'bert_zero_shot': train_bert_zero_shot,
    'one_layer_attention': train_simple_attention_model,
    'gpt2_only_classification': train_gpt2,
    'gpt2_only_embedding_classification': train_gpt2,
    'gpt2_all': train_gpt2,
    'gpt2_only_embedding': train_gpt2,
    'gpt2_randomly_init_embedding_classification': train_gpt2,
    'gpt2_zero_shot': train_gpt2
}


def main(config: Dict) -> None:
    training_records = train_models(config=config)
    training_dir = generate_training_dir(config=config)
    artifacts_dir = generate_artifacts_dir(config=config)
    dump_as_pickle(
        data=training_records,
        output_dir=join(artifacts_dir, training_dir),
        filename=config['training']['training_records'],
    )


if __name__ == '__main__':
    main(config={})
