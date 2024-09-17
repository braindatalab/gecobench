from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from transformers import BertTokenizer

LABEL_MAP = {'female': 0, 'male': 1, 'neutral': 2}


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
        'Female or male: [MASK]. {sentence}',  # 0.82
    ],
    non_binary=['Female, male or neutral: [MASK]. {sentence}'],  # 0.57
)


def get_zero_shot_prompt_function(
    prompt_templates: list[str], index: int
) -> Callable[[list[str]], list[str]]:
    def zero_shot_prompt(sentence: list[str]) -> list[str]:
        sentence_as_str = ' '.join(sentence)
        prompt = prompt_templates[index].format(sentence=sentence_as_str)
        return prompt.split(' ')

    return zero_shot_prompt


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


def zero_shot_prediction(
    model: Module,
    tokenizer: BertTokenizer,
    input_ids: Tensor = None,
    attention_mask: Tensor = None,
    input_embeddings: Tensor = None,
) -> Tuple[list[str], list[int], Tensor]:
    predicted_tokens = list()
    predicted_token_ids = list()

    if (input_ids is not None and input_embeddings is not None) and (
        input_ids.shape[0] != input_embeddings.shape[0]
    ):
        input_ids = input_ids.repeat(input_embeddings.shape[0], 1)

    output = model(
        input_ids=None if input_embeddings is not None else input_ids,
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
    ).logits
    mask_token_ids = input_ids == tokenizer.mask_token_id

    predicted_logits = torch.zeros(output.shape[0])
    for k in range(mask_token_ids.shape[0]):
        if not mask_token_ids[k].any():
            predicted_logits[k] *= output[k].sum()
        else:
            _, topk_predicted_tokens_ids = output[k, mask_token_ids[k]].topk(5, axis=-1)
            for j in range(topk_predicted_tokens_ids.shape[0]):
                prediction, predicted_id = predict_first_token_coinciding_with_labels(
                    topk_predicted_tokens_ids=topk_predicted_tokens_ids[j],
                    tokenizer=tokenizer,
                )
                predicted_tokens += [prediction]
                predicted_token_ids += [predicted_id]
                predicted_logits[k] = output[k, mask_token_ids[k], predicted_id]

    return (
        predicted_tokens,
        predicted_token_ids,
        predicted_logits,
    )


def accuracy_zero_shot(prediction: list[str], labels: list[int]) -> float:
    p = np.array([LABEL_MAP.get(token, -1) for token in prediction])
    return np.mean(p == labels)
