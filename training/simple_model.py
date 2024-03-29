from copy import deepcopy
from typing import Tuple, Dict, List

import torch
from einops import rearrange
from loguru import logger
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from common import DataSet, SaveVersion
from training.bert import (
    get_bert_tokenizer,
    create_bert_ids,
    create_tensor_dataset,
    save_model,
    dump_history,
    Trainer,
)
from utils import (
    set_random_states,
)

BERT_PADDING = '[PAD]'
BERT_CLASSIFICATION = '[CLS]'
BERT_SEPARATION = '[SEP]'
NUM_SPECIAL_BERT_TOKENS = 2
MAX_TOKEN_LENGTH = 512


def masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor = -1e6
):
    x = x.masked_fill(mask.unsqueeze(2) == 0, mask_value) if mask is not None else x
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, p: float = 0.0):
        super().__init__()
        self.dim = dim
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=p)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        s = 1 / (self.dim**0.5)
        key_transposed = rearrange(tensor=key, pattern='b t n -> b n t')
        attention_scores = s * torch.einsum(
            'b t m, b m s -> b t s', query, key_transposed
        )
        attention_weights = masked_softmax(attention_scores, mask=attention_mask)
        attention = torch.einsum(
            'b s t, b t m -> b s m', self.dropout(attention_weights), value
        )

        return attention, attention_weights


class AttentionPooler(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_token = x[:, 0, :]
        x = self.dense(first_token)
        x = self.activation(x)
        return x


class VanillaAttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, emd_dim: int, num_labels: int, p: float = 0.0):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emd_dim)
        self.attention = SelfAttention(dim=emd_dim, p=p)
        self.pooler = AttentionPooler(dim=emd_dim)
        self.classifier = nn.Linear(emd_dim, num_labels)

    def forward(
        self,
        x: Tensor = None,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
        embeddings: Tensor = None,
    ) -> SequenceClassifierOutput:
        if embeddings is not None:
            x = embeddings
        else:
            x = self.embeddings(x)
        attention, attention_weights = self.attention(x, attention_mask)
        first_token = self.pooler(attention)
        logits = self.classifier(first_token)
        return SequenceClassifierOutput(logits=logits, attentions=attention_weights)


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
        f"Simple Attention Model Training, repetition {idx + 1} of "
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

    bert_model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased",
        revision=config['training']['bert_revision'],
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )

    vocab_size = bert_model.config.vocab_size
    del bert_model
    model = VanillaAttentionClassifier(
        vocab_size=vocab_size,
        emd_dim=training_params['embedding_dim'],
        num_labels=num_labels,
    )

    model.to(config['training']['device'])
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-1, eps=1e-5)
    loss = CrossEntropyLoss()

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

    # Best validiation accuracy model
    output_params = deepcopy(training_params)
    output_params['repetition'] = idx
    output_params['save_version'] = SaveVersion.last
    records = [(dataset_name, output_params, model_path, history_path)]

    # Last epoch model
    output_params_last = deepcopy(training_params)
    output_params_last['repetition'] = idx
    output_params_last['save_version'] = SaveVersion.best

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

    trainer.finish_run()

    return records 



def train_simple_attention_model(
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

    logger.info(f'Creating VanillaAttention datasets')
    train_data = create_tensor_dataset(
        data=bert_ids_train, target=y_train, tokenizer=bert_tokenizer
    )
    val_data = create_tensor_dataset(
        data=bert_ids_val, target=y_test, tokenizer=bert_tokenizer
    )

    logger.info(f'Creating VanillaAttention data loaders')
    train_loader = DataLoader(train_data, shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=params['batch_size'])

    logger.info(f'Begin training VanillaAttention model')
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
