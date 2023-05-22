from typing import List, Dict, Tuple, Callable

import numpy as np
import torch
from captum.attr import (
    IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop,
    Deconvolution, ShapleyValueSampling, Lime, KernelShap, LRP, FeaturePermutation, LayerIntegratedGradients
)
from loguru import logger
from torch import Tensor

BERT = 'bert'


def check_availability_of_xai_methods(methods: list) -> None:
    for method_name in methods:
        if method_name not in methods_dict:
            raise Exception(f'Method "{method_name}" is either not (yet) implemented or misspelt!')


def get_captum_attributions(
        model: torch.nn.Module,
        model_type: str,
        x: Tensor,
        baseline: Tensor,
        methods: list,
) -> Dict:
    attributions = dict()
    check_availability_of_xai_methods(methods=methods)
    if BERT in model_type:
        def forward_function(inputs: Tensor) -> Tensor:
            output = model(inputs)
            return torch.max(torch.softmax(output.logits, dim=1)).unsqueeze(-1)

        input_model = model.base_model.embeddings

    else:
        forward_function = None
        input_model = model

    for method_name in methods:
        # logger.info(method_name)
        a = methods_dict.get(method_name)(
            forward_function=forward_function,
            baseline=baseline,
            data=x, model=input_model
        )

        if BERT in model_type:
            a = a.sum(dim=2).squeeze(0)
            a = a / torch.norm(a)
            a = a.cpu().detach().numpy()

        attributions[method_name] = a

    return attributions


def get_integrated_gradients_attributions(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable
) -> torch.tensor:
    explainer = LayerIntegratedGradients(forward_function, model)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        n_steps=200,
        return_convergence_delta=False,
    )


def get_saliency_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return Saliency(model).attribute(data, target=target)


def get_deeplift_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return DeepLift(model).attribute(data, target=target)


def get_deepshap_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return DeepLiftShap(model).attribute(data, target=target, baselines=torch.zeros(data.shape))


def get_gradient_shap_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return GradientShap(model).attribute(data, target=target, baselines=torch.zeros(data.shape))


def get_guided_backprop_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return GuidedBackprop(model).attribute(data, target=target)


def get_deconvolution_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return Deconvolution(model).attribute(data, target=target)


def get_shapley_sampling_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return ShapleyValueSampling(model).attribute(data, target=target)


def get_lime_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return Lime(model).attribute(data, target=target)


def get_kernel_shap_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return KernelShap(model).attribute(data, target=target)


def get_lrp_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return LRP(model).attribute(data, target=target)


def get_pfi_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return FeaturePermutation(model).attribute(data, target=target)


def get_pfi_scikit_learn_attribution(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    result = permutation_importance(
        estimator=ScikitLearnModelAdapter(model=model), random_state=329841,
        X=tensor_to_numpy(data), y=tensor_to_numpy(target), n_repeats=1, n_jobs=1,
    )

    return np.array([result.importances[:, 0]] * data.shape[0])


def get_uniform_random_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return torch.rand(data.shape)


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
    'PFI Scikit-learn': get_pfi_scikit_learn_attribution,
    'Uniform random': get_uniform_random_attributions
}