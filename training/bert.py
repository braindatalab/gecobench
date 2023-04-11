import random
import string
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from utils import dump_as_pickle, load_pickle

SEED = 0
DEVICE = 'cuda'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SPACE = ' '
JOIN_STRING = ''


def train_epoch(
        model: BertForSequenceClassification,
        data_loader: DataLoader,
        loss: CrossEntropyLoss,
        optimizer: torch.optim.Optimizer
) -> Tuple:
    train_loss, train_correct = 0.0, 0

    for step, batch in enumerate(data_loader):
        input_ids = batch[0].to(torch.long).to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        labels = batch[2].to(torch.long).to(DEVICE)

        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = output.logits
        l = loss(logits, labels)

        l.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # for exploding gradients

        train_loss += l.item()
        scores, predictions = torch.max(logits, 1)
        train_correct += torch.sum((predictions == labels).long()).item()

    return train_loss, train_correct


def validate_epoch(
        model: BertForSequenceClassification,
        data_loader: DataLoader,
        loss: CrossEntropyLoss,
) -> Tuple:
    val, val_correct = 0.0, 0
    model.eval()
    for step, batch in enumerate(data_loader):
        input_ids = batch[0].to(torch.long).to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        lables = batch[2].to(torch.long).to(DEVICE)

        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = output.logits
        l = loss(logits, lables)

        val += l.item()
        scores, predictions = torch.max(logits, 1)
        val_correct += torch.sum((predictions == lables).long()).item()

    return val, val_correct


def train(
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
) -> Tuple:
    history = {
        'train_loss': list(),
        'test_loss': list(),
        'train_acc': list(),
        'test_acc': list()
    }

    num_epochs = 20
    learning_rate = 0.001

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,  # binary classification
        output_attentions=False,
        output_hidden_states=False
    )

    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss = CrossEntropyLoss()

    for name, param in model.named_parameters():
        if 'classifier' not in name:  # classifier layer
            param.requires_grad = False

    for epoch in range(num_epochs):
        train_loss, train_correct = train_epoch(model=model, data_loader=train_loader, loss=loss, optimizer=optimizer)
        test_loss, test_correct = validate_epoch(model=model, data_loader=val_loader, loss=loss)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_loader) * 100
        test_loss = test_loss / len(val_loader)
        test_acc = test_correct / len(val_loader) * 100

        logger.info(
            "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                epoch + 1,
                num_epochs,
                train_loss,
                test_loss,
                train_acc,
                test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    return history, model


def join_punctuation_with_previews_word(words: List) -> List:
    punctuations = string.punctuation
    for k in range(len(words)):
        if words[k][0] in punctuations:
            words[k - 1] = words[k - 1][:-1]
    return words


def preprocess_list_of_words(words: List) -> List:
    words_without_nan = [w for w in words if not pd.isna(w)]
    words_are_strings = [str(w) for w in words_without_nan]
    words_with_spaces = [w + SPACE for w in words_are_strings]
    return join_punctuation_with_previews_word(words=words_with_spaces)


def assemble_sentences(data: pd.DataFrame) -> List:
    sentences = list()
    for k, row in data.iterrows():
        processed_words = preprocess_list_of_words(words=row.tolist())
        sentences += [JOIN_STRING.join(processed_words)]
    return sentences


def create_train_test_split(data: TensorDataset, test_size: float) -> List:
    num_samples = len(data)
    num_val_samples = int(test_size * num_samples)
    num_train_samples = num_samples - num_val_samples
    return random_split(
        dataset=data,
        lengths=[num_train_samples, num_val_samples]
    )


def main():
    val_size = 0.2
    batch_size = 16

    text_data = load_pickle(file_path='../data/data_all_same.pkl')
    targets = text_data['target'].tolist()
    sentences = assemble_sentences(data=text_data.drop(['target'], axis=1))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_encoding = bert_tokenizer(text=sentences, padding=True, truncation=True, return_tensors='pt')

    tensor_data = TensorDataset(bert_encoding['input_ids'], bert_encoding['attention_mask'], torch.tensor(targets))
    train_data, val_data = create_train_test_split(data=tensor_data, test_size=val_size)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    history, model = train(train_loader=train_loader, val_loader=val_loader)

    torch.save(model, 'bert_model.pt')
    dump_as_pickle(data=history, output_dir='', file_name='history_of_model_performance.pkl')


if __name__ == '__main__':
    main()
