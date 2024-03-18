from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Tuple, Dict, List, Any

import torch
from loguru import logger
from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
import wandb

from common import DataSet
from utils import (
    set_random_states,
    generate_training_dir,
    dump_as_pickle,
    load_from_cache,
    save_to_cache,
    generate_artifacts_dir,
)

BERT_PADDING = '[PAD]'
BERT_CLASSIFICATION = '[CLS]'
BERT_SEPARATION = '[SEP]'
NUM_SPECIAL_BERT_TOKENS = 2
MAX_TOKEN_LENGTH=512


class Trainer:
    def __init__(
        self,
        config: Dict,
        model: BertForSequenceClassification,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss: CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: str,
        run_name: str,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.logger_enabled = "wandb" in self.config['general']
        self.run_name = run_name

        self.init_logger()

    def init_logger(self):
        if self.logger_enabled:
            wandb.init(
                project=self.config['general']['wandb']['project'],
                entity=self.config['general']['wandb']['entity'],
                name=self.run_name,
                config=self.config,
            )

    def finish_run(self):
        if self.logger_enabled:
            wandb.finish()

    def log_dict(self, log_dict):
        if self.logger_enabled:
            wandb.log(log_dict)

    def log_val(self, step):
        if (
            self.logger_enabled
            and step % self.config['general']['wandb']['validate_every_n_steps'] == 0
            and step > 0
        ):
            val_loss, val_acc = self.validate_epoch()
            self.log_dict(
                {
                    'val_loss': val_loss / len(self.val_loader),
                    'val_acc': val_acc / len(self.val_loader),
                }
            )

    def train_epoch(self) -> Tuple:
        train_loss, train_acc = 0.0, 0

        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, batch in bar:
            input_ids = batch[0].to(torch.long).to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(torch.long).to(self.device)

            output = self.model(
                input_ids, token_type_ids=None, attention_mask=attention_mask
            )
            logits = output.logits
            l = self.loss(logits, labels)

            l.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )  # for exploding gradients

            train_loss += l.item()
            scores, predictions = torch.max(logits, 1)
            batch_train_acc = torch.sum(
                (predictions == labels), dtype=torch.float
            ).item() / float(labels.shape[0])

            train_acc += batch_train_acc

            self.log_dict(
                {
                    'train_loss': l.item(),
                    'train_acc': batch_train_acc,
                }
            )

            # Validate every n steps, as specified in the config
            self.log_val(step)

            bar.set_description(
                f"Train Loss:{train_loss / float(step + 1):.2f}, "
                f"Train Acc:{train_acc / float(step + 1):.2f}"
            )

        return train_loss, train_acc

    def validate_epoch(self) -> Tuple:
        val, val_correct = 0.0, 0
        self.model.eval()
        bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for step, batch in bar:
            input_ids = batch[0].to(torch.long).to(self.device)
            attention_mask = batch[1].to(self.device)
            lables = batch[2].to(torch.long).to(self.device)

            output = self.model(
                input_ids, token_type_ids=None, attention_mask=attention_mask
            )
            logits = output.logits
            l = self.loss(logits, lables)

            val += l.item()
            scores, predictions = torch.max(logits, 1)
            val_correct += torch.sum(
                (predictions == lables), dtype=torch.float
            ).item() / float(lables.shape[0])

            bar.set_description(
                f"Val Loss:{val / float(step + 1):.2f}, "
                f"Val Acc:{val_correct / float(step + 1):.2f}"
            )

        return val, val_correct


def get_bert_ids(tokenizer: BertTokenizer, token: str) -> Tensor:
    return tokenizer(
        text=token,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=False,
    )['input_ids'].flatten()


def create_bert_ids_from_sentence(
    tokenizer: BertTokenizer, sentence: List[str]
) -> Tensor:
    tokens = torch.tensor([])
    classification_id = get_bert_ids(tokenizer=tokenizer, token=BERT_CLASSIFICATION)
    separation_id = get_bert_ids(tokenizer=tokenizer, token=BERT_SEPARATION)
    tokens = torch.cat((tokens, classification_id), dim=0)
    for k, word in enumerate(sentence):
        word_id = get_bert_ids(tokenizer=tokenizer, token=word)
        tokens = torch.cat((tokens, word_id), dim=0)

    tokens = torch.cat((tokens, separation_id), dim=0)
    return tokens.type(torch.long)


def add_padding_if_necessary(
    tokenizer: BertTokenizer, ids: Tensor, max_sentence_length: int
) -> Tensor:
    difference = max_sentence_length - ids.shape[0]
    output = ids
    if difference > 0:
        padding_id = get_bert_ids(tokenizer=tokenizer, token=BERT_PADDING)
        padding = torch.tensor([padding_id.numpy()[0]] * difference)
        output = torch.cat((ids, padding), dim=0)

    return output


def create_attention_mask_from_bert_ids(
    tokenizer: BertTokenizer, ids: Tensor
) -> Tensor:
    attention_mask = torch.ones_like(ids)
    padding_id = get_bert_ids(tokenizer=tokenizer, token=BERT_PADDING)
    padding_mask = ids == padding_id
    attention_mask[padding_mask] = 0
    return attention_mask


def create_tensor_dataset(
    data: List, target: List, tokenizer: BertTokenizer
) -> TensorDataset:
    max_sentence_length = max([len(ids) for ids in data])
    tokens = torch.zeros(size=(len(data), max_sentence_length))
    attention_mask = torch.zeros(size=(len(data), max_sentence_length))
    for k, ids in enumerate(data):
        tokens[k, :] = add_padding_if_necessary(
            tokenizer=tokenizer, ids=ids, max_sentence_length=max_sentence_length
        )
        attention_mask[k, :] = create_attention_mask_from_bert_ids(
            tokenizer=tokenizer,
            ids=tokens[k, :],
        )

    return TensorDataset(tokens.type(torch.long), attention_mask, torch.tensor(target))


def save_model(model: Any, model_name: str, config: dict) -> str:
    base_output_dir = generate_artifacts_dir(config)
    training_dir = generate_training_dir(config)
    output_dir = join(base_output_dir, training_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model, join(output_dir, model_name))
    return join(training_dir, model_name)


def dump_history(history: Dict, config: dict, history_name: str) -> str:
    base_output_dir = generate_artifacts_dir(config)
    training_dir = generate_training_dir(config)
    output_dir = join(base_output_dir, training_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dump_as_pickle(data=history, output_dir=output_dir, filename=history_name)
    return join(training_dir, history_name)


def create_bert_ids(
    data: List, tokenizer: BertTokenizer, type: str = "", config: dict = None
) -> List:
    should_cache = type != "" and config is not None

    if should_cache:
        cache_key = f"bert_ids_{type}"
        cache_entry = load_from_cache(cache_key, config)
        if cache_entry is not None:
            return cache_entry

    bert_ids = list()
    valid_idxs = list()
    for k, sentence in enumerate(data):
        cur = create_bert_ids_from_sentence(tokenizer=tokenizer, sentence=sentence)
        
        if len(cur) <= MAX_TOKEN_LENGTH:
            bert_ids.append(cur)
            valid_idxs.append(k)
        
    if should_cache:
        save_to_cache(cache_key, (bert_ids, valid_idxs), config)

    return bert_ids, valid_idxs


def initialize_embedding(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(module.weight)


def train_model(
    config: dict,
    dataset_name: str,
    training_params: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_labels: int,
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

    num_epochs = training_params['epochs']
    learning_rate = training_params['learning_rate']

    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased",
        revision=config['training']['bert_revision'],
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.to(config['training']['device'])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss = CrossEntropyLoss()

    for name, param in model.named_parameters():
        if name.startswith(tuple(training_params['layers_to_train'])) or 0 == len(
            training_params['layers_to_train']
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if 'bert_randomly_init_embedding_classification' == training_params['model_name']:
        model.apply(initialize_embedding)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        optimizer=optimizer,
        device=config['training']['device'],
        run_name=f'{dataset_name}_{training_params["model_name"]}_{idx}',
    )
    lowest_loss_so_far = 1e7
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate_epoch()

        training_history['train_loss'] += [train_loss / float(len(train_loader))]
        training_history['train_acc'] += [train_acc / float(len(train_loader))]
        training_history['val_loss'] += [val_loss / float(len(val_loader))]
        training_history['val_acc'] += [val_acc / float(len(val_loader))]

        logger.info(
            f"Epoch:{epoch}/{num_epochs},"
            f"AVG Training Loss:{training_history['train_loss'][-1]:.2f}, "
            f"AVG Val Loss:{training_history['val_loss'][-1]:.2f}, "
            f"AVG Training Acc {training_history['train_acc'][-1]:.2f}, "
            f"AVG Val Acc {training_history['val_acc'][-1]:.2f}"
        )

        trainer.log_dict(
            {
                'train_loss': training_history['train_loss'][-1],
                'val_loss': training_history['val_loss'][-1],
                'train_acc': training_history['train_acc'][-1],
                'val_acc': training_history['val_acc'][-1],
            }
        )

        if lowest_loss_so_far > val_loss:
            logger.info(f'Save model weights at epoch: {epoch}')
            lowest_loss_so_far = val_loss
            model_path = save_model(
                model=model,
                config=config,
                model_name=f'{dataset_name}_{training_params["model_name"]}_{idx}.pt',
            )
            history_path = dump_history(
                history=training_history,
                config=config,
                history_name=f'{dataset_name}_{training_params["model_performance"]}_{idx}.pkl',
            )
    output_params = deepcopy(training_params)
    output_params['repetition'] = idx

    trainer.finish_run()

    return [(dataset_name, output_params, model_path, history_path)]


def get_bert_tokenizer(config: dict) -> BertTokenizer:
    return BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased',
        revision=config['training']['bert_revision'],
    )


def train_bert(
    dataset: DataSet, dataset_name: str, num_labels: int, params: Dict, config: Dict
) -> List[Tuple]:
    output = list()
    bert_tokenizer = get_bert_tokenizer(config=config)
    bert_ids_train, train_idxs = create_bert_ids(
        data=dataset.x_train,
        tokenizer=bert_tokenizer,
        type=f"train_{dataset_name}",
        config=config,
    )
    bert_ids_val, val_idxs = create_bert_ids(
        data=dataset.x_test,
        tokenizer=bert_tokenizer,
        type=f"test_{dataset_name}",
        config=config,
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

    logger.info(f'Begin training BERT model')
    for k in range(config['training']['num_training_repetitions']):
        output += train_model(
            config=config,
            dataset_name=dataset_name,
            training_params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            num_labels=num_labels,
            idx=k,
        )

    return output
