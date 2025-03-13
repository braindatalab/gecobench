from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from transformers import GPT2Tokenizer

from common import LABEL_MAP
from utils import get_num_labels


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKEN_ID_FEMALE = 24724
TOKEN_ID_MALE = 22606
TOKEN_ID_NEUTRAL = 29797

TOKEN_SUBSET = {
    2: [TOKEN_ID_FEMALE, TOKEN_ID_MALE],
    3: [TOKEN_ID_FEMALE, TOKEN_ID_MALE, TOKEN_ID_NEUTRAL],
}

TOKEN_LABEL_MAP = {
    TOKEN_ID_FEMALE: 0,
    TOKEN_ID_MALE: 1,
    TOKEN_ID_NEUTRAL: 2,
}


def determine_gender_type(dataset_name: str) -> str:
    output = 'binary'
    if 'non_binary' in dataset_name:
        output = 'non_binary'
    return output


PROMPT_TEMPLATES = dict(
    binary=[
        # Using natural language prompts that work well with GPT2's autoregressive nature
        # ['The', 'gender', 'in', 'this', 'text', 'is', '{sentence}', 'Answer:', '[PRED]', '.', 'Only', 'reply', 'with', 'one', 'word', 'and', 'do', 'not', 'include', 'any', 'punctuation', 'or', 'capitalization', '.'],
        # ['Your', 'task', 'is', 'to', 'predict', 'the', 'gender', 'in', 'this', 'text', ':', '{sentence}', 'Prediction:'],
        [
            'Your',
            'task',
            'is',
            'to',
            'predict',
            'the',
            'grammatical',
            'gender',
            'in',
            'this',
            'text',
            ':',
            '{sentence}',
            'Gender',
            'prediction',
            '(female,',
            'male)',
            ':',
        ],
        # ['Text', ':', '{sentence}', 'Classification', '(', 'female,', 'male', ')', ':'],
        # ['The', 'gender', 'in', 'this', 'text', 'is', '{sentence}', 'Answer:', '[PRED]', '.'],
        # [
        #   "You", "are", "a", "highly", "accurate", "grammatical", "gender", "classifier", ".",
        #   "Your", "task", "is", "to", "read", "a", "given", "sentence", "and", "determine", "the", "grammatical", "gender", "of", "its", "primary", "subject", "or", "reference", ".",
        #   "The", "output", "must", "be", "one", "of", "the", "following", "options", ":", "\"", "female", "\"", ",", "\"", "male", "\"", ",", "or", "\"", "neutral", "\"", ".",
        #   "Instructions", ":",
        #   "1", ".", "Analyze", "the", "sentence", "carefully", ".",
        #   "2", ".", "Identify", "the", "primary", "subject", "or", "the", "relevant", "grammatical", "element", ".",
        #   "3", ".", "Classify", "its", "gender", "strictly", "as", "\"", "female", "\"", ",", "\"", "male", "\"", ",", "or", "\"", "neutral", "\"", "based", "solely", "on", "grammatical", "cues", ".",
        #   "4", ".", "Provide", "only", "the", "single", "word", "answer", "(", "do", "not", "include", "additional", "commentary", ")", ".",
        #   "5", ".", "Ensure", "that", "your", "output", "does", "not", "include", "the", "word", "\"", "the", "\"", "under", "any", "circumstance", ".",
        #   "Example", ":",
        #   "Input", "Sentence", ":", "\"", "The", "actress", "gracefully", "accepted", "her", "award", ".", "\"",
        #   "Output", ":", "female",
        #   "Now", ",", "analyze", "the", "following", "sentence", ":",
        #   "Input", "Sentence", ":", "\"", "{sentence}", "\"",
        #   "Output", ":"
        # ]
    ],
    non_binary=[
        # ['The', 'gender', '(female,', 'male,', 'or', 'neutral)', 'in', 'this', 'text', 'is', '{sentence}', 'Answer:', '[PRED]', '.','Only', 'reply', 'with', 'one', 'word', 'and', 'do', 'not', 'include', 'any', 'punctuation', 'or', 'capitalization', '.'],
        # ['The', 'gender', '(female,', 'male,', 'or', 'neutral)', 'in', 'this', 'text', 'is', '{sentence}', 'Answer:', '[PRED]', '.'],
        # ['Your', 'task', 'is', 'to', 'predict', 'the', 'gender', '(female,', 'male,', 'or', 'neutral)', 'in', 'this', 'text', ':', '{sentence}', 'Prediction:'],
        [
            'Your',
            'task',
            'is',
            'to',
            'predict',
            'the',
            'grammatical',
            'gender',
            '(female,',
            'male,',
            'or',
            'neutral)',
            'in',
            'this',
            'text',
            ':',
            '{sentence}',
            'Gender',
            'prediction',
            '(female,',
            'male,',
            'or',
            'neutral)',
            ':',
        ],
        # [
        #     'Text',
        #     ':',
        #     '{sentence}',
        #     'Classification',
        #     '(',
        #     'female,',
        #     'male',
        #     'or',
        #     'neutral',
        #     ')',
        #     ':',
        # ],
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
        if prediction.lower() in LABEL_MAP.keys():
            output = prediction.lower()
            break
    return output


def predict_next_token(
    logits: Tensor,
    tokenizer: GPT2Tokenizer,
    num_labels: int,
) -> Tuple[Tensor, list[str], Tensor]:
    token_scores = torch.nn.functional.softmax(logits[:, -1, :])
    mask = torch.zeros_like(token_scores, dtype=torch.bool).to(DEVICE)
    mask[:, TOKEN_SUBSET[num_labels]] = True
    masked_token_scores = mask * token_scores


    predicted_tokens = []
    predicted_token_ids = []
    predicted_scores = []
    for batch_idx in range(logits.shape[0]):
        predicted_id = masked_token_scores[batch_idx].argmax()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_id.unsqueeze(0))[0]
        predicted_score = masked_token_scores[batch_idx].max(dim=-1)
        predicted_tokens.append(predicted_token)
        predicted_token_ids.append(predicted_id)
        predicted_scores.append(predicted_score)

    return (
        torch.tensor(predicted_token_ids).to(DEVICE),
        predicted_tokens,
        masked_token_scores,
    )


def zero_shot_prediction(
    model: Module,
    tokenizer: GPT2Tokenizer,
    input_ids: Tensor = None,
    attention_mask: Tensor = None,
    input_embeddings: Tensor = None,
    num_labels: int = 2,
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
    ).logits.to(DEVICE)

    pred_token_ids, pred_tokens, pred_logits = predict_next_token(
        logits=output,
        tokenizer=tokenizer,
        num_labels=num_labels,
    )

    return pred_tokens, pred_token_ids, pred_logits


def format_scores_gpt2(
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
        mask[
            (
                one_hot_target.unsqueeze(dim=0)
                if 1 == len(one_hot_target.shape)
                else one_hot_target
            )
        ] = 1.0
    return mask * logits.max(dim=-1).values.unsqueeze(-1)



def transform_predicted_tokens_to_labels(predictions: list[str]) -> list[int]:
    return [LABEL_MAP.get(token.lower(), -1) for token in predictions]


def accuracy_zero_shot(prediction: list[str], labels: list[int]) -> float:
    p = np.array(transform_predicted_tokens_to_labels(predictions=prediction))
    return np.mean(p == labels)
