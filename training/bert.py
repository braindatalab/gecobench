from copy import deepcopy
from os.path import join
from typing import Tuple, Dict, List, Any

import torch
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

from common import DataSet
from utils import set_random_states, create_train_val_split, generate_training_dir, dump_as_pickle


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


def create_tokenized_tensor_dataset(dataset: DataSet) -> TensorDataset:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_encoding = bert_tokenizer(
        text=dataset.x_train, padding=True,
        truncation=True, return_tensors='pt'
    )

    return TensorDataset(
        bert_encoding['input_ids'],
        bert_encoding['attention_mask'],
        torch.tensor(dataset.y_train)
    )


def train_bert(
        dataset: DataSet,
        params: Dict,
        config: Dict
) -> List[Tuple]:
    output = list()
    for k in range(config['training']['num_training_repetitions']):
        logger.info(f"BERT Training, repetition {k + 1} of "
                    f"{config['training']['num_training_repetitions']}")
        set_random_states(seed=config['general']['seed'] + k)
        tensor_data = create_tokenized_tensor_dataset(dataset=dataset)
        train_data, val_data = create_train_val_split(
            data=tensor_data, val_size=config['training']['val_size']
        )

        train_loader = DataLoader(
            train_data, shuffle=True,
            batch_size=params['batch_size']
        )
        val_loader = DataLoader(
            val_data, shuffle=True,
            batch_size=params['batch_size']
        )

        history = {
            'train_loss': list(),
            'val_loss': list(),
            'train_acc': list(),
            'val_acc': list()
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

        output_dir = generate_training_dir(config=config)
        model_path = join(output_dir, f'{params["model_name"]}_{k}.pt')
        torch.save(model, model_path)
        history_path = join(output_dir, f'{params["model_performance"]}_{k}')
        dump_as_pickle(
            data=history, output_dir=output_dir,
            filename=f'bert_history_{k}'
        )
        output_params = deepcopy(params)
        output_params['repetition'] = k
        output += [output_params, model_path, history_path]

    return output
