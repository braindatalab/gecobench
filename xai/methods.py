from typing import Dict, Callable

import lime
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
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
from torch import Tensor
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from scipy.stats import cov


from training.bert import (
    create_bert_ids,
    add_padding_if_necessary,
    create_attention_mask_from_bert_ids,
)
from utils import determine_model_type, BERT_MODEL_TYPE, ONE_LAYER_ATTENTION_MODEL_TYPE

BERT = 'bert'
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
    def __init__(self, model: torch.nn.Module, model_type: str):
        super().__init__()
        self.model = model
        self.model_type = determine_model_type(s=model_type)

    def forward(self, inputs: torch.tensor):
        if BERT_MODEL_TYPE == self.model_type:
            x = self.model(input_ids=None, inputs_embeds=inputs)[0]
        elif ONE_LAYER_ATTENTION_MODEL_TYPE == self.model_type:
            x = self.model(embeddings=inputs)[0]
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
    tokenizer: BertTokenizer,
) -> Dict:
    attributions = dict()
    check_availability_of_xai_methods(methods=methods)

    def forward_function(inputs: Tensor, attention_mask: Tensor = None) -> Tensor:
        output = (
            model(inputs) if attention_mask is None else model(inputs, attention_mask)
        )
        forward_function_output = torch.softmax(output.logits, dim=1)
        return forward_function_output

    if BERT in model_type:
        embeddings = model.base_model.embeddings
    else:
        embeddings = model.embeddings

    for method_name in methods:
        logger.info(method_name)

        if method_name in GRADIENT_BASED_METHODS:
            a = methods_dict.get(method_name)(
                forward_function=SkippingEmbedding(model, model_type),
                baseline=baseline,
                data=embeddings(x),
                model=model,
                target=target,
            )
        elif method_name == "Gradient SHAP":
            a = methods_dict.get(method_name)(
                forward_function=SkippingEmbedding(model, model_type),
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
    return torch.abs(explanations).sum(dim=2)


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
    return torch.abs(explanations).sum(dim=2)


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
    return torch.abs(explanations).sum(dim=2)


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

    return torch.abs(explanations).sum(dim=2)


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
    return torch.abs(explanations).sum(dim=2)


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
    return torch.abs(explanations).sum(dim=2)


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
        show_progress=True,
    )


def create_tensor_dataset(data: list, tokenizer: BertTokenizer) -> TensorDataset:
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
    explanation: lime.explanation.Explanation, x: np.ndarray, tokenizer: BertTokenizer
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
    for k, bert_id in enumerate(x):
        word = tokenizer.decode(bert_id).replace(' ', '')
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
    tokenizer: BertTokenizer,
) -> torch.tensor:
    def new_forward_function(text_input: list) -> np.ndarray:
        output = list()
        list_of_bert_ids = list()
        for sentence in text_input:
            bert_ids = create_bert_ids(data=[sentence.split()], tokenizer=tokenizer)[0][
                0
            ]
            list_of_bert_ids += [bert_ids]

        dataset = create_tensor_dataset(data=list_of_bert_ids, tokenizer=tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)),
        )
        for batch in tqdm(dataloader):
            output += [forward_function(*batch).cpu().detach().numpy()]

        return np.concatenate(output)

    x = data.flatten().detach()
    text = tokenizer.decode(x[1:-1])
    explainer = LimeTextExplainer(class_names=['0', '1'], char_level=False)
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
    tokenizer: BertTokenizer,
) -> torch.tensor:
    # Adopted from: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    def exponential_cosine_similarity(
        original_input: Tensor, perturbed_input: Tensor, _, **kwargs
    ):
        embedding_model = model
        average_embedding = torch.mean(embedding_model.word_embeddings.weight, dim=0)
        mask = 0 == perturbed_input
        # perturbed_input[mask] = 0
        # original_emb = embedding_model(original_input)
        original_emb = 1 * (0 != original_input).float()
        # perturbed_emb = embedding_model(perturbed_input)
        perturbed_emb = 1 * (0 != perturbed_input).float()
        # perturbed_emb[mask] = average_embedding
        distance = 1 - torch.nn.functional.cosine_similarity(
            original_emb, perturbed_emb, dim=1
        )
        # similarity = torch.exp(-1 * (distance**2) / 2)
        similarity = distance
        # return torch.mean(similarity, axis=1)
        return similarity

    def bernoulli_perturbation(x: Tensor, **kwargs):
        probs = torch.ones_like(x) * 0.5
        output = torch.bernoulli(probs).long()
        return output

    def interpretable_to_input(
        interpretable_sample: Tensor, original_input: Tensor, **kwargs
    ):
        # m = torch.sum(interpretable_sample.bool()).int()
        output = torch.zeros_like(original_input)
        output[interpretable_sample.bool()] = original_input[
            interpretable_sample.bool()
        ]
        # ].view(original_input.size(0), -1)
        return output

    def new_forward_function(inputs: Tensor) -> Tensor:
        return torch.argmax(forward_function(inputs)).unsqueeze(-1).unsqueeze(-1)

    lasso_lime_base = LimeBase(
        new_forward_function,
        interpretable_model=SkLearnLasso(alpha=0.08),
        # interpretable_model=SkLearnLasso(),
        similarity_func=exponential_cosine_similarity,
        perturb_func=bernoulli_perturbation,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interpretable_to_input,
        to_interp_rep_transform=None,
    )
    explanations = lasso_lime_base.attribute(
        # test_text.unsqueeze(0), # add batch dimension for Captum
        data,
        target=int(target),
        # additional_forward_args=(test_offsets,),
        n_samples=int(1e2),
        show_progress=True,
    )

    # l = Lime(forward_function)
    # explanations = l.attribute(
    #     inputs=data,
    #     target=int(target),
    #     n_samples=25,
    #     perturbations_per_eval=1,
    #     return_input_shape=True,
    #     show_progress=True,
    # )

    return explanations


def get_kernel_shap_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
    tokenizer: BertTokenizer,
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
        show_progress=True,
    )


def get_lrp_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    tokenizer: BertTokenizer,
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
    tokenizer: BertTokenizer,
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
    return torch.abs(explanations).sum(dim=2)


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

        c = cov(word_representation, targets)[0]

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
