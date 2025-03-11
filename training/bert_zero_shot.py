from copy import deepcopy

import numpy as np
import torch
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

from common import DataSet, SaveVersion
from training.bert import (
    get_bert_tokenizer,
    create_bert_ids,
    create_tensor_dataset,
    save_model,
    dump_history,
)
from training.bert_zero_shot_utils import (
    determine_gender_type,
    get_zero_shot_prompt_function,
    PROMPT_TEMPLATES,
    zero_shot_prediction,
    accuracy_zero_shot,
)
from utils import set_random_states

BERT_PADDING = '[PAD]'
BERT_CLASSIFICATION = '[CLS]'
BERT_SEPARATION = '[SEP]'
BERT_MASK = '[MASK]'
NUM_SPECIAL_BERT_TOKENS = 2
MAX_TOKEN_LENGTH = 512


def accuracy_over_data_loader(
    model: Module, data_loder: DataLoader, tokenizer: BertTokenizer
) -> float:
    accuracies = list()
    for batch in tqdm(data_loder, desc='Training'):
        input_ids = batch[0].to(torch.long)
        attention_mask = batch[1].to(torch.long)
        labels = batch[2].to(torch.long)
        predicted_tokens, _, _ = zero_shot_prediction(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            num_labels=torch.unique(labels).shape[0],
        )

        accuracies += [
            accuracy_zero_shot(
                prediction=predicted_tokens, labels=labels.detach().numpy()
            )
        ]

    return np.mean(accuracies)


def train_model(
    config: dict,
    dataset_name: str,
    training_params: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: BertTokenizer,
    idx: int,
) -> list:
    set_random_states(seed=config['general']['seed'] + idx)
    logger.info(
        f"BERT Training, repetition {idx + 1} of "
        f"{config['training']['num_training_repetitions']}, "
        f"dataset: {dataset_name}, and "
        f"model name: {training_params['model_name']}"
    )

    training_history = {
        'train_loss': list(),
        'val_loss': list(),
        'train_acc': list(),
        'val_acc': list(),
    }

    model = BertForMaskedLM.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased",
        revision=config['training']['bert_revision'],
    )
    # model.to(config['training']['device'])
    model.eval()

    train_acc = accuracy_over_data_loader(
        model=model, data_loder=train_loader, tokenizer=tokenizer
    )
    val_acc = accuracy_over_data_loader(
        model=model, data_loder=val_loader, tokenizer=tokenizer
    )

    training_history['train_loss'] = [-1]
    training_history['train_acc'] = [train_acc]
    training_history['val_loss'] = [-1]
    training_history['val_acc'] = [val_acc]

    logger.info(
        f"AVG Training Loss:{training_history['train_loss'][-1]:.2f}, "
        f"AVG Val Loss:{training_history['val_loss'][-1]:.2f}, "
        f"AVG Training Acc {training_history['train_acc'][-1]:.2f}, "
        f"AVG Val Acc {training_history['val_acc'][-1]:.2f}"
    )

    model_path = save_model(
        model=model,
        config=config,
        model_name=f'{dataset_name}_{training_params["model_name"]}_{idx}_best.pt',
    )
    history_path = dump_history(
        history=training_history,
        config=config,
        history_name=f'{dataset_name}_{training_params["model_performance"]}_{idx}_best.pkl',
    )

    output_params = deepcopy(training_params)
    output_params['repetition'] = idx
    output_params['save_version'] = SaveVersion.best
    records = [(dataset_name, output_params, model_path, history_path)]

    output_params_last = deepcopy(training_params)
    output_params_last['repetition'] = idx
    output_params_last['save_version'] = SaveVersion.last

    model_path_last = save_model(
        model=model,
        config=config,
        model_name=f'{dataset_name}_{training_params["model_name"]}_{idx}_last.pt',
    )
    history_path_last = dump_history(
        history=training_history,
        config=config,
        history_name=f'{dataset_name}_{training_params["model_performance"]}_{idx}_last.pkl',
    )
    records += [(dataset_name, output_params_last, model_path_last, history_path_last)]

    return records


def train_bert_zero_shot(
    dataset: DataSet, dataset_name: str, num_labels: int, params: dict, config: dict
):

    output = list()
    logger.info(f'Creating BERT datasets')
    gender_type_of_dataset = determine_gender_type(dataset_name)
    zero_shot_prompt = get_zero_shot_prompt_function(
        prompt_templates=PROMPT_TEMPLATES[gender_type_of_dataset], index=0
    )
    bert_tokenizer = get_bert_tokenizer(config=config)
    bert_ids_train, train_idxs = create_bert_ids(
        data=dataset.x_train,
        tokenizer=bert_tokenizer,
        type=f"train_{dataset_name}",
        config=config,
        sentence_context=zero_shot_prompt,
    )
    bert_ids_val, val_idxs = create_bert_ids(
        data=dataset.x_test,
        tokenizer=bert_tokenizer,
        type=f"test_{dataset_name}",
        config=config,
        sentence_context=zero_shot_prompt,
    )
    # Keep valid train targets
    y_train = [dataset.y_train[i] for i in train_idxs]
    y_test = [dataset.y_test[i] for i in val_idxs]

    logger.info(f'Creating BERT datasets')
    train_data = create_tensor_dataset(
        data=bert_ids_train, target=y_train, tokenizer=bert_tokenizer
    )
    val_data = create_tensor_dataset(
        data=bert_ids_val, target=y_test, tokenizer=bert_tokenizer
    )

    logger.info(f'Creating BERT data loaders')
    train_loader = DataLoader(train_data, shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=params['batch_size'])

    logger.info(f'Begin zero shot training BERT model')
    for k in range(config['training']['num_training_repetitions']):
        output += train_model(
            config=config,
            dataset_name=dataset_name,
            training_params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=bert_tokenizer,
            idx=k,
        )

    return output
