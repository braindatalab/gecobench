from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from transformers import BertTokenizer

from common import LABEL_MAP
from utils import get_num_labels

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOKEN_ID_FEMALE = 2931
TOKEN_ID_MALE = 3287
TOKEN_ID_NEUTRAL = 8699

TOKEN_SUBSET = {
    2: [TOKEN_ID_FEMALE, TOKEN_ID_MALE],
    3: [TOKEN_ID_FEMALE, TOKEN_ID_MALE, TOKEN_ID_NEUTRAL],
}


def determine_gender_type(dataset_name: str) -> str:
    output = 'binary'
    if 'non_binary' in dataset_name:
        output = 'non_binary'
    return output


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
        # 'Female or male: [MASK]. {sentence}',  # 0.82
        ['Female', 'or', 'male', ':', '[MASK]', '.', '{sentence}'],
        # val accruacy: 0.87
        # val accruacy: 0.72
        # [
        #     'He',
        #     'continues',
        #     'to',
        #     'pursue',
        #     'him',
        #     '.',
        #     'male,',
        #     'She',
        #     'continues',
        #     'to',
        #     'pursue',
        #     'her',
        #     '.',
        #     'female,',
        #     'Female',
        #     'or',
        #     'male',
        #     ':',
        #     '[MASK]',
        #     '.',
        #     '{sentence}',
        # ],
        # val accruacy: 0.85
        # val accruacy: 0.69
    ],
    non_binary=[
        # [
        #     'Female',
        #     ',',
        #     'male',
        #     'or',
        #     'neutral',
        #     ':',
        #     '[MASK]',
        #     '.',
        #     '{sentence}',
        # ],
        # val accruacy: 0.60
        # val accruacy: 0.50
        [
            'He',
            'proceeds',
            'to',
            'his',
            'place',
            '.',
            'male,',
            'She',
            'proceeds',
            'to',
            'her',
            'place',
            '.',
            'female,',
            'They',
            'proceed',
            'to',
            'their',
            'place',
            '.',
            'neutral,',
            'Female,',
            'male',
            'or',
            'neutral:',
            '[MASK].',
            '{sentence}',
        ]
        # val accruacy: 0.66
        # val accruacy: 0.60
    ],
)


def remove_empty_strings_from_list(s: list[str]) -> list[str]:
    return [w for w in s if w != '']


def strip_spaces_from_words(s: list[str]) -> list[str]:
    return [w.strip() for w in s]


def concatenate_sentence(s: list[str]) -> str:
    return ' '.join(strip_spaces_from_words(s=s))


def split_sentence(s: str) -> list[str]:
    return s.split(' ')


def get_zero_shot_prompt_function(
    prompt_templates: list[str], index: int
) -> Callable[[list[str]], list[str]]:
    def zero_shot_prompt(sentence: list[str]) -> list[str]:
        template = prompt_templates[index]
        k = template.index('{sentence}')
        return template[:k] + sentence + template[k + 1 :]

    return zero_shot_prompt


def extract_original_sentence_from_prompt(
    prompt: list[str], prompt_templates: list[str], index: int
) -> list[str]:
    k = prompt_templates[index].index('{sentence}')
    num_template_words_before_sentence = len(prompt_templates[:k])
    num_template_words_after_sentence = len(prompt_templates[k + 1 :])

    original_sentence = prompt[
        num_template_words_before_sentence : len(prompt)
        - num_template_words_after_sentence
    ]

    return original_sentence


def extract_attribution_for_original_sentence_from_prompt_attribution(
    attribution_prompt: list[float], prompt_templates: list[str], index: int
) -> list[float]:
    k = prompt_templates[index].index('{sentence}')
    num_template_words_before_sentence = len(prompt_templates[index][:k])
    num_template_words_after_sentence = len(prompt_templates[index][k + 1 :])

    attribution_original_sentence = attribution_prompt[
        num_template_words_before_sentence : len(attribution_prompt)
        - num_template_words_after_sentence
    ]

    return attribution_original_sentence


def get_slicing_indices_of_sentence_embedded_in_prompt(
    sentence: list[str], prompt_templates: list[str], index: int
) -> slice:
    empty_template = prompt_templates[index].format(sentence='').strip()
    num_template_words = len(empty_template.split())
    len_sentence = len(sentence)

    if concatenate_sentence(s=sentence).startswith(
        empty_template.split('{sentence}')[0].strip()
    ):
        start_index = num_template_words
        end_index = len_sentence
    else:
        start_index = 0
        end_index = len_sentence - num_template_words

    return slice(start_index, end_index)


def predict_masked_tokens(
    logits: Tensor, mask_token_ids: Tensor, tokenizer: BertTokenizer, num_labels: int
) -> Tuple[Tensor, list[str], Tensor]:
    predicted_tokens = list()
    predicted_token_ids = list()
    predicted_logits = list()
    logits_of_mask_tokens = logits[mask_token_ids].detach()
    token_logits, token_indices = (
        logits_of_mask_tokens[:, TOKEN_SUBSET[num_labels]],
        torch.tensor(TOKEN_SUBSET[num_labels])
        * torch.ones((logits.shape[0], num_labels), dtype=torch.long).to(DEVICE),
    )

    n = 0
    for k, m in enumerate(mask_token_ids.any(dim=-1)):
        if m:
            predicted_id = token_indices[n, F.softmax(token_logits[n]).argmax()]
            prediction = tokenizer.convert_ids_to_tokens(predicted_id.unsqueeze(-1))[0]

            predicted_tokens += [prediction]
            predicted_token_ids += [predicted_id]
            predicted_logits += [token_logits[n, F.softmax(token_logits[n]).argmax()]]
            n += 1
        else:
            predicted_tokens += ['']
            predicted_token_ids += [-1]
            predicted_logits += [-1]

    return (
        torch.tensor(predicted_token_ids).to(DEVICE),
        predicted_tokens,
        torch.tensor(predicted_logits).to(DEVICE),
    )


def create_prediction_mask(
    logits: Tensor, mask_token_ids: Tensor, tokenizer: BertTokenizer, num_labels: int
) -> Tensor:
    prediction_mask = torch.zeros_like(logits, dtype=torch.bool)

    # Get the top 5 predicted token IDs for the masked positions
    logits_of_mask_tokens = logits[mask_token_ids].detach()
    # _, topk_predicted_tokens_ids = output_mask_tokens.topk(5, axis=-1)
    token_logits, token_indices = (
        logits_of_mask_tokens[:, TOKEN_SUBSET[num_labels]],
        torch.tensor(TOKEN_SUBSET[num_labels])
        * torch.ones((logits.shape[0], num_labels), dtype=torch.long).to(DEVICE),
    )

    n = 0
    for k, m in enumerate(mask_token_ids.any(dim=-1)):
        if m:
            # # Get the first token that coincides with the labels
            # topk_predicted_tokens = tokenizer.convert_ids_to_tokens(
            #     topk_predicted_tokens_ids[n]
            # )
            # prediction = next(
            #     (token for token in topk_predicted_tokens if token in LABEL_MAP.keys()),
            #     topk_predicted_tokens[0],
            # )
            # predicted_id = tokenizer.convert_tokens_to_ids(prediction)

            predicted_id = token_indices[k][F.softmax(token_logits[k]).argmax()]

            # Update the mask tensor
            prediction_mask[k, mask_token_ids[k, :], predicted_id] = True
            n += 1

    return prediction_mask


def predict_max_logit_tokens(
    logits: Tensor, input_ids: Tensor, tokenizer: BertTokenizer
) -> Tuple[Tensor, list[str], Tensor]:
    predicted_logits = logits.max(dim=-1)[0].max(dim=-1)[0].to(DEVICE)
    predicted_token_ids = logits.argmax(dim=-1).argmax(dim=-1).to(DEVICE)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids.tolist())
    return predicted_token_ids, predicted_tokens, predicted_logits


def zero_shot_prediction(
    model: Module,
    tokenizer: BertTokenizer,
    input_ids: Tensor = None,
    attention_mask: Tensor = None,
    input_embeddings: Tensor = None,
    num_labels: int = 2,
) -> Tuple[list[str], Tensor, Tensor]:
    '''
    Predicts the tokens of the input_ids using the model.
    If input_embeddings is not None, then the input_ids are not used.
    And the general prediction strategy is to predict the token with the highest logit
    and if there is a mask token, then the prediction is based on the maximum logits
    among the pre-defined tokens female or male (or neutral).
    Of these top 5 logits, the first token that coincides with the labels is selected.
    Otherwise, we fall back to the token with the highest logit.
    '''
    if (
        input_ids is not None
        and input_embeddings is not None
        and input_ids.shape[0] != input_embeddings.shape[0]
    ):
        input_ids = input_ids.repeat(input_embeddings.shape[0], 1)

    output = model(
        input_ids=None if input_embeddings is not None else input_ids,
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
    ).logits.to(DEVICE)

    predicted_token_ids, predicted_tokens, predicted_logits = predict_max_logit_tokens(
        logits=output, input_ids=input_ids, tokenizer=tokenizer
    )

    mask_token_ids = (input_ids == tokenizer.mask_token_id).to(DEVICE)
    if mask_token_ids.any(dim=-1).any(dim=-1):
        mask_token_mask = create_prediction_mask(
            logits=output,
            mask_token_ids=mask_token_ids,
            tokenizer=tokenizer,
            num_labels=num_labels,
        )
        predicted_mask_ids, predicted_mask_tokens, _ = predict_masked_tokens(
            logits=output,
            mask_token_ids=mask_token_ids,
            tokenizer=tokenizer,
            num_labels=num_labels,
        )
        predicted_logits = (mask_token_mask * output).max(dim=-1)[0].max(dim=-1)[0].to(
            DEVICE
        ) + ~mask_token_ids.any(dim=-1) * predicted_logits
        predicted_token_ids = (
            mask_token_ids.any(dim=-1) * predicted_mask_ids
            + ~mask_token_ids.any(dim=-1) * predicted_token_ids
        )

        # Update tokens
        counter = 0
        for k, has_mask_token in enumerate(mask_token_ids.any(dim=-1)):
            if has_mask_token:
                predicted_tokens[k] = predicted_mask_tokens[counter]
                counter += 1

    return (
        predicted_tokens,
        predicted_token_ids,
        predicted_logits,
    )


def format_logits(
    token_ids: Tensor,
    logits: Tensor,
    target: int | Tensor,
    dataset_name: str,
    input_embeddings: Tensor = None,
) -> Tensor:
    n = token_ids.shape[0]
    if input_embeddings is not None and token_ids.shape[0] != input_embeddings.shape[0]:
        n = input_embeddings.shape[0]
    mask = torch.zeros(
        size=(n, get_num_labels(dataset_name=dataset_name)),
    ).to(DEVICE)
    if isinstance(target, int):
        mask[:, target] = 1.0
    else:
        one_hot_target = torch.nn.functional.one_hot(
            target, num_classes=get_num_labels(dataset_name=dataset_name)
        ).to(torch.bool)
        mask[one_hot_target] = 1.0
    return mask * logits.unsqueeze(1)


def transform_predicted_tokens_to_labels(predictions: list[str]) -> list[int]:
    return [LABEL_MAP.get(token, -1) for token in predictions]


def accuracy_zero_shot(prediction: list[str], labels: list[int]) -> float:
    p = np.array(transform_predicted_tokens_to_labels(predictions=prediction))
    return np.mean(p == labels)
