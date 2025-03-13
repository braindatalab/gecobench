from typing import Dict, Callable

import lime
import numpy as np
import torch
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr import (
    Saliency,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    GuidedBackprop,
    Deconvolution,
    ShapleyValueSampling,
    LimeBase,
    KernelShap,
    LayerLRP,
    FeaturePermutation,
    LayerIntegratedGradients,
    InputXGradient,
)
from lime.lime_text import LimeTextExplainer
from loguru import logger
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import BertTokenizer, GPT2Tokenizer

from training.bert import (
    create_bert_ids,
    add_padding_if_necessary,
    create_attention_mask_from_bert_ids,
)
from training.bert_zero_shot_utils import (
    zero_shot_prediction as bert_zero_shot_prediction,
    format_logits,
)
from training.gpt2 import (
    create_gpt2_ids,
    add_padding_if_necessary as add_padding_if_necessary_gpt2,
    create_attention_mask_from_gpt2_ids,
)
from training.gpt2_zero_shot_utils import (
    zero_shot_prediction as gpt2_zero_shot_prediction,
    format_scores_gpt2,
    TOKEN_LABEL_MAP,
)
from utils import (
    determine_model_type,
    BERT_MODEL_TYPE,
    ONE_LAYER_ATTENTION_MODEL_TYPE,
    BERT_ZERO_SHOT,
    GPT2_MODEL_TYPE,
    GPT2_ZERO_SHOT,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

BERT = 'bert'
GPT2 = 'gpt2'
ALL_BUT_CLS_SEP = slice(1, -1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRADIENT_BASED_METHODS = [
    "Saliency",
    "InputXGradient",
    "Guided Backprop",
    "Deconvolution",
    "DeepLift",
]


class SkippingEmbedding(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        tokenizer: BertTokenizer | GPT2Tokenizer,
        dataset_type: str,
        target: int,
        input_ids: Tensor = None,
    ):
        super().__init__()
        self.model = model
        self.model_type = determine_model_type(s=model_type)
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.dataset_type = dataset_type
        self.target = target

    def forward(self, inputs: torch.Tensor, attention_mask: Tensor = None):
        if BERT_ZERO_SHOT == self.model_type:
            _, predicted_token_ids, logits = bert_zero_shot_prediction(
                model=self.model,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                input_embeddings=inputs,
                input_ids=self.input_ids,
            )
            pred = logits.argmax(dim=-1)
            x = format_logits(
                token_ids=self.input_ids,
                input_embeddings=inputs,
                logits=logits,
                target=pred,
                dataset_name=self.dataset_type,
            )
        elif GPT2_ZERO_SHOT == self.model_type:
            _, predicted_token_ids, scores = gpt2_zero_shot_prediction(
                model=self.model,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                input_embeddings=inputs,
                input_ids=self.input_ids,
            )
            pred = torch.tensor(
                [TOKEN_LABEL_MAP[s.item()] for s in scores.argmax(dim=-1)]
            )
            x = format_scores_gpt2(
                token_ids=self.input_ids,
                input_embeddings=inputs,
                logits=scores,
                target=self.target,
                dataset_name=self.dataset_type,
            )
        elif BERT_MODEL_TYPE == self.model_type:
            x = self.model(input_ids=None, inputs_embeds=inputs)[0]
        elif ONE_LAYER_ATTENTION_MODEL_TYPE == self.model_type:
            x = self.model(embeddings=inputs)[0]
        elif GPT2_MODEL_TYPE == self.model_type:
            x = self.model(input_ids=None, inputs_embeds=inputs)[0]
        return x


def normalize_attributions(a: np.ndarray) -> np.ndarray:
    a = np.abs(a)
    a = a.squeeze(0)
    a /= np.sum(a)
    return a


def check_availability_of_xai_methods(methods: list) -> None:
    for method_name in methods:
        if method_name not in methods_dict:
            raise Exception(
                f'Method "{method_name}" is either not (yet) implemented or misspelt!'
            )


def get_captum_attributions(
    model: torch.nn.Module,
    model_type: str,
    x: Tensor,
    baseline: Tensor,
    methods: list,
    target: list,
    dataset_type: str,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> Dict:
    attributions = dict()
    check_availability_of_xai_methods(methods=methods)

    def forward_function(inputs: Tensor, attention_mask: Tensor = None) -> Tensor:
        if BERT_ZERO_SHOT == model_type:
            _, predicted_token_ids, logits = bert_zero_shot_prediction(
                model=model,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                input_embeddings=None,
                input_ids=inputs,
            )
            pred = logits.argmax(dim=-1)
            formated_logits = format_logits(
                token_ids=inputs,
                logits=logits,
                target=pred,
                dataset_name=dataset_type,
            )
            forward_function_output = torch.softmax(formated_logits, dim=1)
        elif GPT2_ZERO_SHOT == model_type:
            _, predicted_token_ids, scores = gpt2_zero_shot_prediction(
                model=model,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                input_embeddings=None,
                input_ids=inputs,
            )
            pred = torch.tensor(
                [TOKEN_LABEL_MAP[s.item()] for s in scores.argmax(dim=-1)]
            )
            formated_logits = format_scores_gpt2(
                token_ids=inputs,
                logits=scores,
                target=target,
                dataset_name=dataset_type,
            )
            forward_function_output = torch.softmax(formated_logits, dim=1)
        else:
            output = (
                model(inputs)
                if attention_mask is None
                else model(input_ids=inputs, attention_mask=attention_mask)
            )
            forward_function_output = torch.softmax(output.logits, dim=1)
        return forward_function_output

    if BERT in model_type:
        embeddings = model.base_model.embeddings
    elif GPT2 in model_type:
        embeddings = model.transformer.wte
    else:
        embeddings = model.embeddings

    for method_name in methods:
        # logger.info(method_name)

        if method_name in GRADIENT_BASED_METHODS:
            a = methods_dict.get(method_name)(
                forward_function=SkippingEmbedding(
                    model=model,
                    model_type=model_type,
                    tokenizer=tokenizer,
                    input_ids=x,
                    target=target,
                    dataset_type=dataset_type,
                ),
                baseline=baseline,
                data=embeddings(x),
                model=model,
                target=target,
            )
        elif method_name == "Gradient SHAP":
            a = methods_dict.get(method_name)(
                forward_function=SkippingEmbedding(
                    model=model,
                    model_type=model_type,
                    tokenizer=tokenizer,
                    input_ids=x,
                    target=target,
                    dataset_type=dataset_type,
                ),
                baseline=embeddings(baseline),
                data=embeddings(x),
                target=target,
            )
        elif 'Covariance' not in method_name:
            a = methods_dict.get(method_name)(
                forward_function=forward_function,
                baseline=baseline,
                data=x,
                model=embeddings,
                target=target,
                tokenizer=tokenizer,
            )
        else:
            continue

        attributions[method_name] = normalize_attributions(a=a.detach().cpu().numpy())

    return attributions


def get_integrated_gradients_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer,
) -> torch.tensor:
    explainer = LayerIntegratedGradients(forward_function, model)
    explanations = explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=int(target),
        n_steps=200,
        return_convergence_delta=False,
    )
    return explanations.sum(dim=2)


def get_saliency_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    explainer = Saliency(forward_function)
    explanations = explainer.attribute(
        inputs=data, target=int(target), abs=True, additional_forward_args=None
    )
    return explanations.sum(dim=2)


def get_deeplift_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    explainer = DeepLift(forward_function, multiply_by_inputs=None, eps=1e-10)
    # : UserWarning: Setting forward, backward hooks and attributes on non-linear activations.
    # The hooks and attributes will be removed after the attribution is finished
    explanations = explainer.attribute(
        inputs=data,
        baselines=torch.cat(
            ([torch.unsqueeze(baseline, dim=2)] * data.shape[2]), axis=2
        ),
        target=int(target),
        additional_forward_args=None,
        return_convergence_delta=False,
        custom_attribution_func=None,
    )
    return explanations.sum(dim=2)


def get_deepshap_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer,
) -> torch.tensor:
    # Will throw the same error w.r.t. shape of baseline vs input as with gradient shap
    return DeepLiftShap(model, multiply_by_inputs=None).attribute(
        inputs=data, target=target, baselines=baseline
    )


def get_gradient_shap_attributions(
    data: torch.Tensor, baseline: Tensor, forward_function: Callable, target: list
) -> torch.tensor:
    explanations = GradientShap(forward_function).attribute(
        inputs=data, baselines=baseline, target=target
    )

    return explanations.sum(dim=2)


def get_guided_backprop_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    # UserWarning: Setting backward hooks on ReLU activations.The hooks will be removed after the attribution is finished
    explainer = GuidedBackprop(forward_function)
    explanations = explainer.attribute(
        inputs=data, target=int(target), additional_forward_args=None
    )
    return explanations.sum(dim=2)


def get_deconvolution_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    explainer = Deconvolution(forward_function)
    explanations = explainer.attribute(
        inputs=data, target=int(target), additional_forward_args=None
    )
    return explanations.sum(dim=2)


def get_shapley_sampling_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer,
) -> torch.tensor:
    explainer = ShapleyValueSampling(forward_function)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=int(target),
        additional_forward_args=None,
        feature_mask=None,
        n_samples=25,
        perturbations_per_eval=1,
        show_progress=False,
    )


def create_tensor_dataset_gpt2(
    data: list, target: list, tokenizer: GPT2Tokenizer
) -> TensorDataset:
    max_sentence_length = max([ids.shape[0] for ids in data])
    tokens = torch.zeros(size=(len(data), max_sentence_length))
    attention_mask = torch.zeros(size=(len(data), max_sentence_length))

    for k, ids in enumerate(data):
        tokens[k, :] = add_padding_if_necessary_gpt2(
            tokenizer=tokenizer, ids=ids, max_sentence_length=max_sentence_length
        )
        attention_mask[k, :] = create_attention_mask_from_gpt2_ids(
            tokenizer=tokenizer,
            ids=tokens[k, :],
        )

    return TensorDataset(tokens.type(torch.long), attention_mask)


def create_tensor_dataset(data: list, tokenizer: BertTokenizer) -> TensorDataset:
    """Create a tensor dataset for BERT models."""
    max_sentence_length = max([ids.shape[0] for ids in data])
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

    return TensorDataset(tokens.type(torch.long), attention_mask)


def assemble_lime_explanations(
    explanation: lime.explanation.Explanation,
    x: np.ndarray,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> Tensor:
    explanations_per_word = [
        (
            explanation.domain_mapper.indexed_string.word(exp[0]),
            explanation.domain_mapper.indexed_string.string_position(exp[0]),
            exp[1],
        )
        for exp in explanation.local_exp[1]
    ]

    explanations = np.zeros(x.shape[0])
    for k, token_id in enumerate(x):
        word = tokenizer.decode(token_id).replace(' ', '')
        for exp_word in explanations_per_word:
            if word in exp_word[0]:
                explanations[k] = exp_word[2]
                break

    return torch.tensor(explanations).unsqueeze(0)


def get_lime_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> torch.tensor:
    def new_forward_function(text_input: list) -> np.ndarray:
        output = list()
        list_of_ids = list()
        for sentence in text_input:
            if isinstance(tokenizer, BertTokenizer):
                ids = create_bert_ids(data=[sentence.split()], tokenizer=tokenizer)[0][
                    0
                ]
            else:  # GPT2Tokenizer
                ids = create_gpt2_ids(data=[sentence.split()], tokenizer=tokenizer)[0][
                    0
                ]
            list_of_ids += [ids]

        if isinstance(tokenizer, BertTokenizer):
            dataset = create_tensor_dataset(data=list_of_ids, tokenizer=tokenizer)
        else:  # GPT2Tokenizer
            dataset = create_tensor_dataset_gpt2(
                data=list_of_ids, target=[0] * len(list_of_ids), tokenizer=tokenizer
            )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)),
        )
        for batch in tqdm(dataloader, disable=True):
            output += [forward_function(*batch).cpu().detach().numpy()]

        return np.concatenate(output)

    x = data.flatten().detach()
    text = tokenizer.decode(x[1:-1])
    class_names = [str(k) for k in np.unique(target)]
    explainer = LimeTextExplainer(class_names=class_names, char_level=False)
    lime_explanation = explainer.explain_instance(
        text,
        new_forward_function,
        num_features=x.shape[0],
        num_samples=int(1e2),
    )

    explanations = assemble_lime_explanations(
        explanation=lime_explanation,
        x=x,
        tokenizer=tokenizer,
    )

    return explanations


def get_lime_attributions2(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> torch.tensor:
    # Adopted from: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    def exponential_cosine_similarity(
        original_input: Tensor, perturbed_input: Tensor, _, **kwargs
    ):
        if isinstance(tokenizer, BertTokenizer):
            embedding_layer = model.word_embeddings
        else:  # GPT2Tokenizer
            embedding_layer = model.transformer.wte

        average_embedding = torch.mean(embedding_layer.weight, dim=0)
        mask = 0 == perturbed_input
        original_emb = 1 * (0 != original_input).float()
        perturbed_emb = 1 * (0 != perturbed_input).float()
        distance = 1 - torch.nn.functional.cosine_similarity(
            original_emb, perturbed_emb, dim=1
        )
        similarity = distance
        return similarity

    def bernoulli_perturbation(x: Tensor, **kwargs):
        probs = torch.ones_like(x) * 0.5
        output = torch.bernoulli(probs).long()
        return output

    def interpretable_to_input(
        interpretable_sample: Tensor, original_input: Tensor, **kwargs
    ):
        output = torch.zeros_like(original_input)
        output[interpretable_sample.bool()] = original_input[
            interpretable_sample.bool()
        ]
        return output

    def new_forward_function(inputs: Tensor) -> Tensor:
        return torch.argmax(forward_function(inputs)).unsqueeze(-1).unsqueeze(-1)

    lasso_lime_base = LimeBase(
        new_forward_function,
        interpretable_model=SkLearnLasso(alpha=0.08),
        similarity_func=exponential_cosine_similarity,
        perturb_func=bernoulli_perturbation,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interpretable_to_input,
        to_interp_rep_transform=None,
    )
    explanations = lasso_lime_base.attribute(
        data,
        target=int(target),
        n_samples=int(1e2),
        show_progress=False,
    )

    return explanations


def get_kernel_shap_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> torch.tensor:
    explainer = KernelShap(forward_function)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=int(target),
        additional_forward_args=None,
        feature_mask=None,
        n_samples=25,
        perturbations_per_eval=1,
        return_input_shape=True,
        show_progress=False,
    )


def get_lrp_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> torch.tensor:
    return LayerLRP(forward_function, model).attribute(inputs=data)


def get_pfi_attributions(
    data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module
) -> torch.tensor:
    return FeaturePermutation(model).attribute(data, target=target)


def get_uniform_random_attributions(
    data: torch.Tensor,
    target: torch.Tensor,
    model: torch.nn.Module,
    baseline: Tensor,
    forward_function: Callable,
    tokenizer: BertTokenizer | GPT2Tokenizer,
) -> torch.tensor:
    return torch.rand(data.shape)


def get_input_x_gradient(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    explainer = InputXGradient(forward_function)
    explanations = explainer.attribute(
        inputs=data, target=int(target), additional_forward_args=None
    )
    return explanations.sum(dim=2)


def calculate_covariance_between_words_target(
    sentences: list,
    targets: list,
    vocabulary: set,
    word_to_bert_id_mapping: dict,
) -> dict:
    pipeline = Pipeline(
        [
            ('count', CountVectorizer(vocabulary=vocabulary)),
            ('tfidf', TfidfTransformer()),
        ]
    )
    pipeline.fit(sentences)
    x = pipeline.transform(sentences)

    ret = dict()
    for word in pipeline.named_steps['count'].get_feature_names_out():
        word_representation = (
            x[:, pipeline.named_steps['count'].vocabulary_[word]].toarray().flatten()
        )

        c = np.cov(word_representation, targets)[0, 1]
        ret[word_to_bert_id_mapping[word]] = 0.0 if np.isnan(c) else c

    return ret


def get_covariance_between_words_target(
    covariance_between_words_target: dict,
    token_ids: Tensor,
) -> dict:
    a = list()
    for tid in token_ids:
        a += [covariance_between_words_target[tid.cpu().numpy().item()]]
    return {'Covariance': normalize_attributions(a=np.array(a)[np.newaxis, :])}


# https://captum.ai/api/
methods_dict = {
    'Integrated Gradients': get_integrated_gradients_attributions,
    'Saliency': get_saliency_attributions,
    'DeepLift': get_deeplift_attributions,
    'DeepSHAP': get_deepshap_attributions,
    'Gradient SHAP': get_gradient_shap_attributions,
    'Guided Backprop': get_guided_backprop_attributions,
    'Deconvolution': get_deconvolution_attributions,
    'Shapley Value Sampling': get_shapley_sampling_attributions,
    'LIME': get_lime_attributions,
    'Kernel SHAP': get_kernel_shap_attributions,
    'LRP': get_lrp_attributions,
    'PFI': get_pfi_attributions,
    'Uniform random': get_uniform_random_attributions,
    'InputXGradient': get_input_x_gradient,
    'Covariance': get_covariance_between_words_target,
}
