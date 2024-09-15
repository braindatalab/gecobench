from copy import deepcopy
from typing import Tuple, List, Callable

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

from common import DataSet, SaveVersion
from evaluation.main import accuracy
from training.bert import (
    get_bert_tokenizer,
    create_bert_ids,
    create_tensor_dataset,
    save_model,
    dump_history,
)
from utils import set_random_states

BERT_PADDING = '[PAD]'
BERT_CLASSIFICATION = '[CLS]'
BERT_SEPARATION = '[SEP]'
BERT_MASK = '[MASK]'
NUM_SPECIAL_BERT_TOKENS = 2
MAX_TOKEN_LENGTH = 512


LABEL_MAP = {'female': 0, 'male': 1, 'neutral': 2}

PROMPT_TEMPLATES = dict(
    binary=[
        # '{sentence} male or female: [MASK] .', # 0.48
        # '{sentence} : male or female: [MASK] .',  # 0.67
        # '{sentence} | male or female: [MASK] .', # 0.39
        # '{sentence} : female or male: [MASK] .', # 0.49
        # '{sentence} Decide: male or female: [MASK] .', # 0.56
        # '{sentence} Pronouns: male or female: [MASK] .', # 0.40
        # 'Pronouns of {sentence} are male or female: [MASK] .', # 0.49
        # 'Pronouns of "{sentence}" are male or female: [MASK] .', # 0.48
        # 'Female or male: {sentence} [MASK] .', # 0.57
        'Female or male: [MASK]. {sentence}',  # 0.82
    ],
    non_binary=['Female, male or neutral: [MASK]. {sentence}'],  # 0.57
)


def get_zero_shot_prompt_function(
    prompt_templates: List[str], index: int
) -> Callable[[List[str]], List[str]]:
    def zero_shot_prompt(sentence: List[str]) -> List[str]:
        sentence_as_str = ' '.join(sentence)
        prompt = prompt_templates[index].format(sentence=sentence_as_str)
        return prompt.split(' ')

    return zero_shot_prompt


def get_first_token_that_coincides_with_label(predictions: List[str]) -> str:
    output = None
    for prediction in predictions:
        if prediction in LABEL_MAP.keys():
            output = prediction
            break
    return output


def predict_tokens(
    topk_predicted_tokens_ids: Tensor, tokenizer: BertTokenizer
) -> tuple:
    topk_predicted_tokens = tokenizer.convert_ids_to_tokens(topk_predicted_tokens_ids)
    ids_and_tokens_map = {
        token: token_id
        for token, token_id in zip(topk_predicted_tokens, topk_predicted_tokens_ids)
    }
    prediction = get_first_token_that_coincides_with_label(topk_predicted_tokens)
    if prediction is None:
        prediction = topk_predicted_tokens[0]

    return prediction, ids_and_tokens_map[prediction]


def zero_shot_prediction(
    model: Module, input_ids: Tensor, attention_mask: Tensor, tokenizer: BertTokenizer
) -> Tuple[List[str], List[int]]:
    predicted_tokens = list()
    predicted_token_ids = list()
    output = model(input_ids, attention_mask=attention_mask).logits
    mask_token_ids = input_ids == tokenizer.mask_token_id
    _, topk_predicted_tokens_ids = output[mask_token_ids].topk(5, axis=-1)
    for k in range(topk_predicted_tokens_ids.shape[0]):
        prediction, predicted_id = predict_tokens(
            topk_predicted_tokens_ids=topk_predicted_tokens_ids[k], tokenizer=tokenizer
        )
        predicted_tokens += [prediction]
        predicted_token_ids += [predicted_id]

    return predicted_tokens, predicted_token_ids


def accuracy_over_data_loader(
    model: Module, data_loder: DataLoader, tokenizer: BertTokenizer
) -> float:
    accuracies = list()
    for batch in tqdm(data_loder, desc='Training'):
        input_ids = batch[0].to(torch.long)
        attention_mask = batch[1].to(torch.long)
        labels = batch[2].to(torch.long)
        predicted_tokens, _ = zero_shot_prediction(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        accuracies += [
            accuracy(
                [LABEL_MAP.get(token, -1) for token in predicted_tokens],
                labels.detach().numpy(),
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

    training_history['train_loss'] = training_params['epochs'] * [-1]
    training_history['train_acc'] = training_params['epochs'] * [train_acc]
    training_history['val_loss'] += training_params['epochs'] * [-1]
    training_history['val_acc'] += training_params['epochs'] * [val_acc]

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


def determine_gender_type(dataset_name: str) -> str:
    output = 'binary'
    if 'non_binary' in dataset_name:
        output = 'non_binary'
    return output


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
        output += [
            train_model(
                config=config,
                dataset_name=dataset_name,
                training_params=params,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=bert_tokenizer,
                idx=k,
            )
        ]

    return output
