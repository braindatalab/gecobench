from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Tuple, Dict, List, Any

import torch
from loguru import logger
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

from common import DataSet
from utils import set_random_states, create_train_val_split, generate_training_dir, dump_as_pickle

BERT_PADDING = '[PAD]'
BERT_CLASSIFICATION = '[CLS]'
BERT_SEPARATION = '[SEP]'
NUM_SPECIAL_BERT_TOKENS = 2


def train_epoch(
        model: BertForSequenceClassification,
        data_loader: DataLoader,
        loss: CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: str
) -> Tuple:
    train_loss, train_acc = 0.0, 0

    for step, batch in enumerate(data_loader):
        input_ids = batch[0].to(torch.long).to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(torch.long).to(device)

        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = output.logits
        l = loss(logits, labels)

        l.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # for exploding gradients

        train_loss += l.item()
        scores, predictions = torch.max(logits, 1)
        train_acc += (torch.sum(
            (predictions == labels),
            dtype=torch.float
        ).item() / float(labels.shape[0]))

    return train_loss, train_acc


def validate_epoch(
        model: BertForSequenceClassification,
        data_loader: DataLoader,
        loss: CrossEntropyLoss,
        device: str
) -> Tuple:
    val, val_correct = 0.0, 0
    model.eval()
    for step, batch in enumerate(data_loader):
        input_ids = batch[0].to(torch.long).to(device)
        attention_mask = batch[1].to(device)
        lables = batch[2].to(torch.long).to(device)

        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = output.logits
        l = loss(logits, lables)

        val += l.item()
        scores, predictions = torch.max(logits, 1)
        val_correct += (torch.sum(
            (predictions == lables),
            dtype=torch.float
        ).item() / float(lables.shape[0]))

    return val, val_correct


def get_bert_ids(tokenizer: BertTokenizer, token: str) -> Tensor:
    return tokenizer(
        text=token, padding=True,
        truncation=True, return_tensors='pt',
        add_special_tokens=False
    )['input_ids'].flatten()


def create_bert_ids_from_sentence(tokenizer: BertTokenizer, sentence: List[str]) -> Tensor:
    tokens = torch.tensor([])
    classification_id = get_bert_ids(tokenizer=tokenizer, token=BERT_CLASSIFICATION)
    separation_id = get_bert_ids(tokenizer=tokenizer, token=BERT_SEPARATION)
    tokens = torch.cat((tokens, classification_id), dim=0)
    for k, word in enumerate(sentence):
        word_id = get_bert_ids(tokenizer=tokenizer, token=word)
        tokens = torch.cat((tokens, word_id), dim=0)

    tokens = torch.cat((tokens, separation_id), dim=0)
    return tokens


def add_padding_if_necessary(
        tokenizer: BertTokenizer,
        ids: Tensor,
        max_sentence_length: int
) -> Tensor:
    difference = max_sentence_length - ids.shape[0]
    output = ids
    if difference > 0:
        padding_id = get_bert_ids(tokenizer=tokenizer, token=BERT_PADDING)
        padding = torch.tensor([padding_id.numpy()[0]] * difference)
        output = torch.cat((ids, padding), dim=0)

    return output


def create_attention_mask_from_bert_ids(tokenizer: BertTokenizer, ids: Tensor) -> Tensor:
    attention_mask = torch.ones_like(ids)
    padding_id = get_bert_ids(tokenizer=tokenizer, token=BERT_PADDING)
    padding_mask = (ids == padding_id)
    attention_mask[padding_mask] = 0
    return attention_mask


def create_tensor_dataset(data: List, target: List, tokenizer: BertTokenizer) -> TensorDataset:
    max_sentence_length = max([len(ids) for ids in data])
    tokens = torch.zeros(size=(len(data), max_sentence_length))
    attention_mask = torch.zeros(size=(len(data), max_sentence_length))
    for k, ids in enumerate(data):
        tokens[k, :] = add_padding_if_necessary(
            tokenizer=tokenizer, ids=ids,
            max_sentence_length=max_sentence_length
        )
        attention_mask[k, :] = create_attention_mask_from_bert_ids(
            tokenizer=tokenizer, ids=tokens[k, :],
        )

    return TensorDataset(tokens, attention_mask, torch.tensor(target))


def save_model(model: Any, model_name: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = join(output_dir, model_name)
    torch.save(model, model_path)
    return model_path


def dump_history(history: Dict, output_dir: str, history_name: str) -> str:
    dump_as_pickle(
        data=history, output_dir=output_dir,
        filename=history_name
    )
    return join(output_dir, history_name)


def create_bert_ids(data: List, tokenizer: BertTokenizer) -> List:
    bert_ids = list()
    for k, sentence in enumerate(data):
        bert_ids += [create_bert_ids_from_sentence(tokenizer=tokenizer, sentence=sentence)]
    return bert_ids


def train_bert_only_classification(
        dataset: DataSet,
        dataset_name: str,
        params: Dict,
        config: Dict
) -> List[Tuple]:
    output = list()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_ids_train = create_bert_ids(data=dataset.x_train, tokenizer=bert_tokenizer)
    bert_ids_val = create_bert_ids(data=dataset.x_test, tokenizer=bert_tokenizer)
    train_data = create_tensor_dataset(
        data=bert_ids_train, target=dataset.y_train, tokenizer=bert_tokenizer
    )
    val_data = create_tensor_dataset(
        data=bert_ids_val, target=dataset.y_test, tokenizer=bert_tokenizer
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=params['batch_size'])
    for k in range(config['training']['num_training_repetitions']):
        set_random_states(seed=config['general']['seed'] + k)
        logger.info(f"BERT Training, repetition {k + 1} of "
                    f"{config['training']['num_training_repetitions']}, and"
                    f"dataset: {dataset_name}")

        history = {
            'train_loss': list(), 'val_loss': list(),
            'train_acc': list(), 'val_acc': list()
        }

        num_epochs = params['epochs']
        learning_rate = params['learning_rate']

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        )

        model.to(config['training']['device'])
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss = CrossEntropyLoss()

        for name, param in model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model=model,
                data_loader=train_loader,
                loss=loss,
                optimizer=optimizer,
                device=config['training']['device']
            )
            val_loss, val_acc = validate_epoch(
                model=model,
                data_loader=val_loader,
                loss=loss,
                device=config['training']['device']
            )
            history['train_loss'] += [train_loss / float(len(train_loader))]
            history['train_acc'] += [train_acc / float(len(train_loader))]
            history['val_loss'] += [val_loss / float(len(val_loader))]
            history['val_acc'] += [val_acc / float(len(val_loader))]

            logger.info(
                f"Epoch:{epoch}/{num_epochs},"
                f"AVG Training Loss:{history['train_loss'][-1]:.2f}, "
                f"AVG Val Loss:{history['val_loss'][-1]:.2f}, "
                f"AVG Training Acc {history['train_acc'][-1]:.2f}, "
                f"AVG Val Acc {history['val_acc'][-1]:.2f}"
            )

        model_path = save_model(
            model=model, output_dir=generate_training_dir(config=config),
            model_name=f'{dataset_name}_{params["model_name"]}_{k}.pt'
        )
        history_path = dump_history(
            history=history, output_dir=generate_training_dir(config=config),
            history_name=f'{dataset_name}_{params["model_performance"]}_{k}.pkl'
        )
        output_params = deepcopy(params)
        output_params['repetition'] = k
        output += [(dataset_name, output_params, model_path, history_path)]

    return output


def train_bert_only_embedding_classification(
        dataset: DataSet,
        dataset_name: str,
        params: Dict,
        config: Dict
) -> List[Tuple]:
    output = list()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_ids_train = create_bert_ids(data=dataset.x_train, tokenizer=bert_tokenizer)
    bert_ids_val = create_bert_ids(data=dataset.x_test, tokenizer=bert_tokenizer)
    train_data = create_tensor_dataset(
        data=bert_ids_train, target=dataset.y_train, tokenizer=bert_tokenizer
    )
    val_data = create_tensor_dataset(
        data=bert_ids_val, target=dataset.y_test, tokenizer=bert_tokenizer
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=params['batch_size'])
    for k in range(config['training']['num_training_repetitions']):
        set_random_states(seed=config['general']['seed'] + k)
        logger.info(f"BERT Training, repetition {k + 1} of "
                    f"{config['training']['num_training_repetitions']}, and"
                    f"dataset: {dataset_name}")

        history = {
            'train_loss': list(), 'val_loss': list(),
            'train_acc': list(), 'val_acc': list()
        }

        num_epochs = params['epochs']
        learning_rate = params['learning_rate']

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        )

        model.to(config['training']['device'])
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss = CrossEntropyLoss()

        layers_to_train = ["bert.embeddings","classifier"]

        for name, param in model.named_parameters():
            if name.startswith(tuple(layers_to_train)):
                param.requires_grad = True
            else:
                param.requires_grad = False

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model=model,
                data_loader=train_loader,
                loss=loss,
                optimizer=optimizer,
                device=config['training']['device']
            )
            val_loss, val_acc = validate_epoch(
                model=model,
                data_loader=val_loader,
                loss=loss,
                device=config['training']['device']
            )
            history['train_loss'] += [train_loss / float(len(train_loader))]
            history['train_acc'] += [train_acc / float(len(train_loader))]
            history['val_loss'] += [val_loss / float(len(val_loader))]
            history['val_acc'] += [val_acc / float(len(val_loader))]

            logger.info(
                f"Epoch:{epoch}/{num_epochs},"
                f"AVG Training Loss:{history['train_loss'][-1]:.2f}, "
                f"AVG Val Loss:{history['val_loss'][-1]:.2f}, "
                f"AVG Training Acc {history['train_acc'][-1]:.2f}, "
                f"AVG Val Acc {history['val_acc'][-1]:.2f}"
            )

        model_path = save_model(
            model=model, output_dir=generate_training_dir(config=config),
            model_name=f'{dataset_name}_{params["model_name"]}_{k}.pt'
        )
        history_path = dump_history(
            history=history, output_dir=generate_training_dir(config=config),
            history_name=f'{dataset_name}_{params["model_performance"]}_{k}.pkl'
        )
        output_params = deepcopy(params)
        output_params['repetition'] = k
        output += [(dataset_name, output_params, model_path, history_path)]

    return output

def train_bert_all(
        dataset: DataSet,
        dataset_name: str,
        params: Dict,
        config: Dict
) -> List[Tuple]:
    output = list()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_ids_train = create_bert_ids(data=dataset.x_train, tokenizer=bert_tokenizer)
    bert_ids_val = create_bert_ids(data=dataset.x_test, tokenizer=bert_tokenizer)
    train_data = create_tensor_dataset(
        data=bert_ids_train, target=dataset.y_train, tokenizer=bert_tokenizer
    )
    val_data = create_tensor_dataset(
        data=bert_ids_val, target=dataset.y_test, tokenizer=bert_tokenizer
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=params['batch_size'])
    for k in range(config['training']['num_training_repetitions']):
        set_random_states(seed=config['general']['seed'] + k)
        logger.info(f"BERT Training, repetition {k + 1} of "
                    f"{config['training']['num_training_repetitions']}, and"
                    f"dataset: {dataset_name}")

        history = {
            'train_loss': list(), 'val_loss': list(),
            'train_acc': list(), 'val_acc': list()
        }

        num_epochs = params['epochs']
        learning_rate = params['learning_rate']

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        )

        model.to(config['training']['device'])
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss = CrossEntropyLoss()


        #for name, param in model.named_parameters():
        #    if param.requires_grad == True:
        #        print(name)

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model=model,
                data_loader=train_loader,
                loss=loss,
                optimizer=optimizer,
                device=config['training']['device']
            )
            val_loss, val_acc = validate_epoch(
                model=model,
                data_loader=val_loader,
                loss=loss,
                device=config['training']['device']
            )
            history['train_loss'] += [train_loss / float(len(train_loader))]
            history['train_acc'] += [train_acc / float(len(train_loader))]
            history['val_loss'] += [val_loss / float(len(val_loader))]
            history['val_acc'] += [val_acc / float(len(val_loader))]

            logger.info(
                f"Epoch:{epoch}/{num_epochs},"
                f"AVG Training Loss:{history['train_loss'][-1]:.2f}, "
                f"AVG Val Loss:{history['val_loss'][-1]:.2f}, "
                f"AVG Training Acc {history['train_acc'][-1]:.2f}, "
                f"AVG Val Acc {history['val_acc'][-1]:.2f}"
            )

        model_path = save_model(
            model=model, output_dir=generate_training_dir(config=config),
            model_name=f'{dataset_name}_{params["model_name"]}_{k}.pt'
        )
        history_path = dump_history(
            history=history, output_dir=generate_training_dir(config=config),
            history_name=f'{dataset_name}_{params["model_performance"]}_{k}.pkl'
        )
        output_params = deepcopy(params)
        output_params['repetition'] = k
        output += [(dataset_name, output_params, model_path, history_path)]

    return output