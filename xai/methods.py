from typing import List, Dict, Tuple, Callable

import numpy as np
import torch
from captum.attr import (
    IntegratedGradients,
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
from sklearn import linear_model
from loguru import logger
from torch import Tensor
import copy

BERT = 'bert'

GRADIENT_BASED_METHODS = [
    "Saliency",
    "InputXGradient",
    "Guided Backprop",
    "Deconvolution",
    "DeepLift",
]


class SkippingEmbedding(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.tensor):
        return self.model(input_ids=None, inputs_embeds=inputs)[0]


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
) -> Dict:
    attributions = dict()
    check_availability_of_xai_methods(methods=methods)

    if BERT in model_type:

        def forward_function(inputs: Tensor) -> Tensor:
            output = model(inputs)
            forward_function_output = torch.softmax(output.logits, dim=1)
            return forward_function_output

        bert_embeddings = model.base_model.embeddings

    for method_name in methods:
        logger.info(method_name)

        if method_name in GRADIENT_BASED_METHODS:
            a = methods_dict.get(method_name)(
                forward_function=SkippingEmbedding(model),
                baseline=baseline,
                data=bert_embeddings(x),
                model=model,
                target=target,
            )
        else:
            a = methods_dict.get(method_name)(
                forward_function=forward_function,
                baseline=baseline,
                data=x,
                model=model if 'Gradient SHAP' == method_name else bert_embeddings,
                target=target,
            )

        attributions[method_name] = normalize_attributions(a=a.detach().numpy())

    return attributions


def get_integrated_gradients_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
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
) -> torch.tensor:
    # Will throw the same error w.r.t. shape of baseline vs input as with gradient shap
    return DeepLiftShap(model, multiply_by_inputs=None).attribute(
        inputs=data, target=target, baselines=baseline
    )


def get_gradient_shap_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    return GradientShap(forward_function).attribute(inputs=data, baselines=baseline)


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


def get_lime_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
) -> torch.tensor:
    # encode text indices into latent representations & calculate cosine similarity
    # Source: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        print("")
        print("exp_embedding_cosine_distance():")
        print("original", original_inp, original_inp.shape)
        print("perturbed", perturbed_inp, perturbed_inp.shape)
        embedding_model = model
        original_emb = embedding_model(original_inp)
        perturbed_emb = embedding_model(perturbed_inp)
        print("original_emb", original_emb.shape)
        print("perturbed_emb", perturbed_emb.shape)

        distance = 1 - torch.nn.functional.cosine_similarity(
            original_emb, perturbed_emb, dim=1
        )
        return torch.exp(-1 * (distance**2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    # Source: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    def bernoulli_perturb(text, **kwargs):
        print("")
        print("bernoulli_perturb():")
        print("text:")
        print(text, text.shape)
        probs = torch.ones_like(text) * 0.5
        output = torch.bernoulli(probs).long()
        print("bernoulli", output, output.shape)

        return torch.bernoulli(probs).long()

    # remove absenst token based on the intepretable representation sample
    # Source: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    def interp_to_input(interp_sample, original_input, **kwargs):
        print("")
        print("interp_to_input()")
        print("interp_sample", interp_sample, interp_sample.shape)
        print("original_input", original_input, original_input.shape)
        output = original_input[interp_sample.bool()].view(original_input.size(0), -1)
        print("output", output, output.shape)
        return original_input[interp_sample.bool()].view(original_input.size(0), -1)

    lasso_lime_base = LimeBase(
        forward_function,
        interpretable_model=linear_model.Lasso(alpha=0.08),
        similarity_func=exp_embedding_cosine_distance,
        perturb_func=bernoulli_perturb,
        perturb_interpretable_space=False,
        from_interp_rep_transform=interp_to_input,
        to_interp_rep_transform=None,
    )
    print("data", data)
    explanations = lasso_lime_base.attribute(
        # test_text.unsqueeze(0), # add batch dimension for Captum
        data,
        target=int(target),
        # additional_forward_args=(test_offsets,),
        # n_samples=,
        show_progress=True,
    ).squeeze(0)

    return explanations


def get_kernel_shap_attributions(
    data: torch.Tensor,
    baseline: Tensor,
    model: torch.nn.Module,
    forward_function: Callable,
    target: list,
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
}
