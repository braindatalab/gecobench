from os.path import join
from typing import List, Dict

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DataSet, DatasetKeys, validate_dataset_key
from training.bert import train_bert
from training.simple_model import train_simple_attention_model
from utils import dump_as_pickle, load_pickle, generate_data_dir, generate_training_dir


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
    for name in config['training']['datasets']:
        validate_dataset_key(name)
        dataset = DatasetHandler[name](config=config, dataset_key=name)
        for model_name, params in config['training']['models'].items():
            records += TrainModel[model_name](dataset, name, params, config)

    return records


def load_gender_dataset(config: Dict, dataset_key: str) -> DataSet:
    data_dir = generate_data_dir(config=config)
    path = join(
        data_dir,
        dataset_key,
        config['data']['datasets'][dataset_key]['output_filenames']['train'],
    )

    raw_data = load_pickle(file_path=path)
    dataset = split_train_data_into_train_val_data(
        x=raw_data.data, y=raw_data.target, config=config
    )

    return dataset


def load_sentiment_datset(config: Dict, dataset_key: str) -> DataSet:
    logger.info(f'Loading dataset {dataset_key}')
    data_dir = generate_data_dir(config=config)
    path = join(
        data_dir,
        dataset_key,
        config['data']['datasets'][dataset_key]['output_filenames']['train'],
    )

    logger.info(f'Loading dataset from {path}')
    df = pd.read_csv(path)
    x = df['text'].apply(lambda x: x.split(" ")).tolist()
    y = df['label'].tolist()

    logger.info(f'Splitting dataset into train and val')
    dataset = split_train_data_into_train_val_data(x=x, y=y, config=config)
    logger.info(f'Dataset loaded')

    return dataset


TrainModel = {
    'bert_only_classification': train_bert,
    'bert_only_embedding_classification': train_bert,
    'bert_all': train_bert,
    'bert_only_embedding': train_bert,
    'bert_randomly_init_embedding_classification': train_bert,
    'simple_model': train_simple_attention_model,
}

DatasetHandler = {
    DatasetKeys.gender_all.value: load_gender_dataset,
    DatasetKeys.gender_subj.value: load_gender_dataset,
    DatasetKeys.sentiment_twitter.value: load_sentiment_datset,
    DatasetKeys.sentiment_imdb.value: load_sentiment_datset,
}


def main(config: Dict) -> None:
    training_records = train_models(config=config)
    output_dir = generate_training_dir(config=config)
    dump_as_pickle(
        data=training_records,
        output_dir=output_dir,
        filename=config['training']['training_records'],
    )


if __name__ == '__main__':
    main(config={})
