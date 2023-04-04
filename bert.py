import string
import pickle
from typing import Dict

import pandas as pd
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.optim import Adam, SGD

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from transformers import BertTokenizer

from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertConfig
from transformers import DistilBertTokenizer, TFDistilBertModel

import matplotlib.pyplot as plt

from utils import dump_as_pickle

SEED = 0
DEVICE = 'cpu'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def create_vocab(dataset):
    # Identifies the different words in all the sentences in a dataset and adds (the different ones) to a vocabulary (cf. output vocab)
    #
    # input
    # dataset: is a dataframe where each row is a sentence and each column a word of that sentence
    #
    # output
    # vocab: set of words (words that exist on the dataset without repetition)
    # word_dict: dictionary where each different word of the dataset sentences correspond to an integer (token)
    #        special tokens for BERT are added to this volab: [PAD]:0, [CLS]:1,[SEP]:2,[MASK]:3

    """ data is a dataframe with target and tokens"""
    vocab = []
    for phrase_idx in range(len(dataset)):
        vocab += [str(w) for w in list(dataset.iloc[phrase_idx])]

    print(f'len of the vocabulary with repeated words: {len(vocab)}')
    vocab = set(vocab)
    print(f'len of the vocabulary without repeated words: {len(vocab)}')
    list(vocab).sort()
    #     vocab.remove('[PAD]')

    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(vocab):
        word_dict[w] = i + 4
        number_dict = {i: w for i, w in enumerate(vocab)}
        vocab_size = len(vocab)

    return vocab, word_dict


def create_strings(phrase):
    # gets a sentence from a row of a dataframe where each column is a word (or punctuatuion) of the sentence
    #
    # input
    # phrase: line of the dataframe (obtained with data.loc[i])
    #
    # output
    # s: sentence with the words of a line in the dataset separated by a white space

    result = string.punctuation

    text = [str(w) for w in phrase if type(w) != float]
    for w in range(len(text)):
        text[w] = text[w] + ' '

        if text[w][0] in result:
            text[w - 1] = text[w - 1][:-1]

    s = ''.join(text)
    return s


# Creating the datasets:
def phrase_datasets(dataframe, target):
    # transforms the entries of the dataframe into strings and adds them to a list alongside the corresponding target
    #
    # input
    # dataframe : dataframe of sentences with words separated by columns
    #
    # output
    # datalist : list of sentences and targets

    data = dataframe.copy()
    data_list = []
    for i in list(data.index):
        data_list.append([create_strings(data.loc[i]), target.loc[i]])

    return data_list


def create_encoded_dataset(tokenizer, encoded_sents, max_phrase_len):
    # uses the tokenizer to encode the sentences in dataset and padded these encoded senteces to max_phrase_len
    #
    # input
    # tokenizer: BERT tokenizer used to create tokens from a sentence
    # dataset: list of lists: [[sentence,target],...]
    # max_phrase_len: int that represents the value until which the phrases are padded
    #    (> than maximum phrase length of dataset - trunctation is not implemented)
    #
    # output
    # padded_phrases : tokenized phrases padded to max_phrase_len
    # attention_mask : for BERT - list of 1 and 0 where 1 corresponds to tokens of words and 0 to padded tokens

    # padding
    padded_phrases = []
    for s in encoded_sents:
        size = len(s)
        padding = np.zeros(max_phrase_len - size)
        padd_tokens = np.concatenate((s, padding), axis=0)
        padded_phrases.append(padd_tokens)

    attention_masks = []
    for sent in padded_phrases:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return padded_phrases, attention_masks


def update_tokenizer(tokenizer, dataset):
    # adds the words that did not exist in the tokenizer to it
    #
    # input
    # tokenizer : BERT_tokenizer
    # dataset : list of lists: [[sentence,target],...]
    #
    # output
    # BERT tokenizer with new words added (words that were not present in the tokenizer initially)

    words_list = []
    for words in dataset:
        words_list += words[0]

    words = set(words_list)  # words without repetition
    print('len list words: ', len(words_list), ' len unique words: ', len(set(words_list)))

    print('adding words')
    tokenizer.add_tokens(list(words))

    return tokenizer


def dataset_pre_processing(dataframe, tokenizer, n_example=None, max_phrase_len=None):
    # pre-processes the dataframe and created the volcab dictionary and a dataset of encoded sentences
    #
    # input
    # dataframe:
    #
    # output
    # vocab: set of words (words that exist on the dataset without repetition)
    # target: correct label
    # word_dict: dictionary where each different word of the dataset sentences correspond to an integer (token)
    #        special tokens for BERT are added to this volab: [PAD]:0, [CLS]:1,[SEP]:2,[MASK]:3
    # new_tokenizer: tokenizer with new words added (words that were not present in the tokenizer initially)
    # encoded_train_phrases: tokenized phrases padded to max_phrase_len
    # train_attention_masks: for BERT - list of 1 and 0 where 1 corresponds to tokens of words and 0 to padded tokens

    target = dataframe.target
    dataframe = dataframe.drop(['target'], axis=1)
    #     dataframe=dataframe.fillna('[PAD]')

    vocab, word_dict = create_vocab(dataframe)

    # list of sentences and targets
    dataset = phrase_datasets(dataframe, target)
    print(dataset[5][0])

    # updating tokenizer to include all the words in the dataframe
    print('len tokenizer before: ', len(tokenizer))
    # new_tokenizer = update_tokenizer(tokenizer, dataset)
    # print('len tokenizer after: ', len(new_tokenizer))

    # encoding and padding the phrases

    # encoding
    sentences = [s[0] for s in dataset]
    encoded_sents = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        encoded_sents.append(encoded_sent)

    if not max_phrase_len:
        max_phrase_len = max([len(sen) for sen in encoded_sents]) + 1
        print(f'Max sentence length: {max_phrase_len - 1}')

    print(max_phrase_len)
    # encoded_train_phrases,train_attention_masks=create_encoded_dataset(new_tokenizer,encoded_sents,max_phrase_len)
    encoded_train_phrases, train_attention_masks = create_encoded_dataset(tokenizer, encoded_sents, max_phrase_len)

    if n_example:
        print('\n\n')
        n = n_example
        print(f'original: {dataset[n][0]}')
        print(f'encoded: {encoded_train_phrases[n]}')
        print(f'shape encoded: {encoded_train_phrases[n].shape}')
        print(f'attention mask: {train_attention_masks[n]}')
        # print(f'decoded phrase: \n {new_tokenizer.decode(encoded_train_phrases[n])}')

    new_tokenizer = ''
    return vocab, list(target), word_dict, new_tokenizer, encoded_train_phrases, train_attention_masks


def acc(pred, labels):
    # calculated the accuracy between pred and labels
    #
    # input
    # pred: labels predicted by a model
    # labels: correct labels
    #
    # output: accuracy

    pred = np.argmax(pred, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred == labels_flat) / len(labels_flat)


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler):
    # training the model based on the dataloader examples (1 eopoch)
    # returns loss and the number of correct labels in an epoch
    train_loss, train_correct = 0.0, 0
    # model.train()

    for step, batch in enumerate(dataloader):
        # umpack dataloader
        batch_input_ids = batch[0].to(torch.long).to(DEVICE)
        batch_input_mask = batch[1].to(DEVICE)
        batch_labels = batch[2].to(torch.long).to(DEVICE)

        # obtaining predictions and loss
        out = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)
        logits = out.logits
        loss = loss_fn(logits, batch_labels)

        # backpropagating loss
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # for exploding gradients
        # scheduler.step()

        #
        train_loss += loss.item()
        scores, predictions = torch.max(logits, 1)
        train_correct += torch.sum((predictions == batch_labels).long()).item()

    return train_loss, train_correct


def valid_epoch(model, dataloader, loss_fn, optimizer, scheduler):
    # validating (no backpropagation) the model based on the examples in the dataloader (1 epoch)
    # returns loss and the number of correct labels in an epoch
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for step, batch in enumerate(dataloader):
        # umpacking dataloader
        batch_input_ids = batch[0].to(torch.long).to(DEVICE)
        batch_input_mask = batch[1].to(DEVICE)
        batch_labels = batch[2].to(torch.long).to(DEVICE)

        # obtaining the predictions and loss
        out = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)
        logits = out.logits
        loss = loss_fn(logits, batch_labels)

        #
        valid_loss += loss.item()
        scores, predictions = torch.max(logits, 1)
        val_correct += torch.sum((predictions == batch_labels).long()).item()

    return valid_loss, val_correct


def train(dataset, lr, num_epochs, k, batch_size):
    # training loop over num_epochs epochs using k-fold cross validation

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    # get the indexes for each fold train and validation examples
    splits = KFold(n_splits=k, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        # create dataloaders with the different splits for each new fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # initiate a model for each new fold
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        )

        for name, param in model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

        model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        for epoch in range(num_epochs):
            # training and validating the model
            train_loss, train_correct = train_epoch(model, train_loader, loss_fn, optimizer, scheduler)
            test_loss, test_correct = valid_epoch(model, val_loader, loss_fn, optimizer, scheduler)

            # loss and accuracies
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(val_loader.sampler)
            test_acc = test_correct / len(val_loader.sampler) * 100

            print(
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


def main():
    with open('/mnt/data_all_same.pkl', 'rb') as f:
        data_all_same = pickle.load(f)

    tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')

    vocab_all, target_all, word_dict_all, new_tokenizer_all, encoded_train_phrases_all, train_attention_masks_all = dataset_pre_processing(
        data_all_same, tokenizer1, n_example=5)

    train_df = torch.tensor(encoded_train_phrases_all)
    train_labels = torch.tensor(target_all)
    train_masks = torch.tensor(train_attention_masks_all)

    print(len(train_df), len(train_masks))

    BATCH = 16

    # train loaders
    train_data = TensorDataset(train_df, train_masks, train_labels)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH)

    num_epochs = 20
    k = 2
    lr = 0.001
    batch_size = BATCH
    history, model_all = train(train_data, lr, num_epochs, k, batch_size)

    torch.save(model_all, 'bert_model.pt')
    dump_as_pickle(
        data=history,
        output_dir='',
        file_name='history_of_model_performance.pkl'
    )


if __name__ == '__main__':
    main()
