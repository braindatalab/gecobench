from os.path import join
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DataSet
from training.bert import train_bert
from training.simple_model import train_simple_attention_model
from utils import dump_as_pickle, load_pickle, generate_data_dir, generate_training_dir


def split_train_data_into_train_val_data(
    x: pd.DataFrame, y: pd.DataFrame, config: Dict
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


def train_models(data_paths: List, config: Dict) -> List:
    records = list()
    for path in data_paths:
        name = extract_name_of_dataset(s=path)
        raw_data = load_pickle(file_path=path)
        dataset = split_train_data_into_train_val_data(
            x=raw_data.data, y=raw_data.target, config=config
        )
        for model_name, params in config['training']['models'].items():
            records += TrainModel[model_name](dataset, name, params, config)

    return records


def generate_data_paths(config: Dict) -> List:
    output_paths = list()
    data_dir = generate_data_dir(config=config)
    output_paths += [join(data_dir, config['data']['output_filenames']['train_all'])]
    output_paths += [
        join(data_dir, config['data']['output_filenames']['train_subject'])
    ]
    return output_paths


TrainModel = {
    'bert_only_classification': train_bert,
    'bert_only_embedding_classification': train_bert,
    'bert_all': train_bert,
    'bert_only_embedding': train_bert,
    'simple_model': train_simple_attention_model,
}


def main(config: Dict) -> None:
    paths = generate_data_paths(config=config)
    training_records = train_models(data_paths=paths, config=config)
    output_dir = generate_training_dir(config=config)
    dump_as_pickle(
        data=training_records,
        output_dir=output_dir,
        filename=config['training']['training_records'],
    )


if __name__ == '__main__':
    main(config={})
