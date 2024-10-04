from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from transformers import BertTokenizer

LABEL_MAP = {'female': 0, 'male': 1, 'neutral': 2}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        ['Female', 'or', 'male', ':', '[MASK]', '.', '{sentence}'],  # 0.82
    ],
    non_binary=[
        'Female',
        ',',
        'male',
        'or',
        'neutral',
        ':',
        '[MASK]',
        '.',
        '{sentence}',
    ],  # 0.57
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


def get_first_token_that_coincides_with_label(predictions: list[str]) -> str:
    output = None
    for prediction in predictions:
        if prediction in LABEL_MAP.keys():
            output = prediction
            break
    return output


def predict_first_token_coinciding_with_labels(
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


def predict_masked_tokens(
    logits: Tensor, mask_token_ids: Tensor, tokenizer: BertTokenizer
) -> Tuple[Tensor, list[str], Tensor]:
    predicted_tokens = list()
    predicted_token_ids = list()
    predicted_logits = list()
    output_mask_tokens = logits[mask_token_ids]
    _, topk_predicted_tokens_ids = output_mask_tokens.topk(5, axis=-1)
    for j in range(topk_predicted_tokens_ids.shape[0]):
        prediction, predicted_id = predict_first_token_coinciding_with_labels(
            topk_predicted_tokens_ids=topk_predicted_tokens_ids[j],
            tokenizer=tokenizer,
        )
        predicted_tokens += [prediction]
        predicted_token_ids += [predicted_id]
        predicted_logits += [output_mask_tokens[j, predicted_id]]

    return (
        torch.tensor(predicted_token_ids).to(DEVICE),
        predicted_tokens,
        torch.tensor(predicted_logits).to(DEVICE),
    )


def predict_max_logit_tokens(
    logits: Tensor, input_ids: Tensor, tokenizer: BertTokenizer
) -> Tuple[Tensor, list[str], Tensor]:
    predicted_logits = logits.max(dim=-1)[0].max(dim=-1)[0]
    predicted_token_ids = logits.argmax(dim=-1).argmax(dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids.tolist())
    return predicted_token_ids, predicted_tokens, predicted_logits


def zero_shot_prediction(
    model: Module,
    tokenizer: BertTokenizer,
    input_ids: Tensor = None,
    attention_mask: Tensor = None,
    input_embeddings: Tensor = None,
) -> Tuple[list[str], Tensor, Tensor]:
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
    ).logits

    predicted_token_ids, predicted_tokens, predicted_logits = predict_max_logit_tokens(
        logits=output, input_ids=input_ids, tokenizer=tokenizer
    )

    mask_token_ids = input_ids == tokenizer.mask_token_id
    if mask_token_ids.any(dim=-1).any(dim=-1):
        predicted_mask_ids, predicted_mask_tokens, predicted_mask_logits = (
            predict_masked_tokens(
                logits=output, mask_token_ids=mask_token_ids, tokenizer=tokenizer
            )
        )
        # Update logits and token ids
        predicted_logits[mask_token_ids.any(dim=-1)] = predicted_mask_logits
        predicted_token_ids[mask_token_ids.any(dim=-1)] = predicted_mask_ids

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


def transform_predicted_tokens_to_labels(predictions: list[str]) -> list[int]:
    return [LABEL_MAP.get(token, -1) for token in predictions]


def accuracy_zero_shot(prediction: list[str], labels: list[int]) -> float:
    p = np.array(transform_predicted_tokens_to_labels(predictions=prediction))
    return np.mean(p == labels)
