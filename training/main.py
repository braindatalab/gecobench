from os.path import join
from typing import List, Tuple, Dict

import torch

from common import DataSet
from training.bert import train_bert
from utils import dump_as_pickle, load_pickle, generate_data_dir, generate_training_dir


def train_models(dataset: DataSet, config: Dict) -> List:
    records = list()
    for model_name, params in config['training']['models'].items():
        records += [TrainModel[model_name](dataset, params, config)]

    return records


TrainModel = {
    'bert': train_bert
}


def main(config: Dict) -> None:
    data_dir = generate_data_dir(config=config)
    dataset = load_pickle(
        file_path=join(data_dir, config['data']['dataset_filename'])
    )

    training_records = train_models(dataset=dataset, config=config)
    output_dir = generate_training_dir(config=config)
    dump_as_pickle(
        data=training_records,
        output_dir=output_dir,
        filename=config['training']['training_records']
    )


if __name__ == '__main__':
    main(config={})
