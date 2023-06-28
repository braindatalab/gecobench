from typing import List, Dict, Tuple, Callable

import numpy as np
import torch
from captum.attr import (
    IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop,
    Deconvolution, ShapleyValueSampling, Lime, KernelShap, LayerLRP, FeaturePermutation, LayerIntegratedGradients, InputXGradient
)
from loguru import logger
from torch import Tensor
import copy

BERT = 'bert'


class SkippingEmbedding(torch.nn.Module):
    def __init__(self,bertmodel):
        super(SkippingEmbedding, self).__init__()
        self.bertmodel = bertmodel
    def forward(self,inputs: torch.tensor):
        return self.bertmodel(input_ids=None,inputs_embeds=inputs)[0]


def check_availability_of_xai_methods(methods: list) -> None:
    for method_name in methods:
        if method_name not in methods_dict:
            raise Exception(
                f'Method "{method_name}" is either not (yet) implemented or misspelt!')


def get_captum_attributions(
        model: torch.nn.Module,
        model_type: str,
        x: Tensor,
        baseline: Tensor,
        methods: list,
        target: list
) -> Dict:
    attributions = dict()
    check_availability_of_xai_methods(methods=methods)
    print(methods)
    if BERT in model_type:
        def forward_function(inputs: Tensor) -> Tensor:
            output = model(inputs)
            forward_function_output = torch.softmax(output.logits, dim=1)
            return forward_function_output

        embedding_model = model.base_model.embeddings
 
    for method_name in methods:
        logger.info(method_name)

        gradient_based_methods = ["Saliency", "InputXGradient", "Guided Backprop", "Deconvolution", "DeepLift"]
        if method_name in gradient_based_methods:
            a = methods_dict.get(method_name)(
                forward_function=SkippingEmbedding(model),
                baseline=baseline,
                data=embedding_model(x), 
                model=embedding_model, 
                target=target
            )
        else:
            a = methods_dict.get(method_name)(
                forward_function=forward_function,
                baseline=baseline,
                data=x, 
                model=embedding_model, 
                target=target
            )

        if BERT in model_type and method_name:
            a = a.squeeze(0)
            a = a / torch.norm(a)
            a = a.cpu().detach().numpy()

        attributions[method_name] = a

    return attributions


def get_integrated_gradients_attributions(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable,
        target: list
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


def get_saliency_attributions(data: torch.Tensor,
                              baseline: Tensor,
                              model: torch.nn.Module,
                              forward_function: Callable,
                              target: list
) -> torch.tensor:
    explainer = Saliency(forward_function)
    explanations = explainer.attribute(
        inputs=data,
        target=int(target),
        abs=True,
        additional_forward_args=None
    )
    return explanations.sum(dim=2)


def get_deeplift_attributions(data: torch.Tensor,
                              baseline: Tensor,
                              model: torch.nn.Module,
                              forward_function: Callable,
                              target: list
) -> torch.tensor:
    explainer = DeepLift(forward_function, multiply_by_inputs=None, eps=1e-10)
    # : UserWarning: Setting forward, backward hooks and attributes on non-linear activations. 
    # The hooks and attributes will be removed after the attribution is finished
    explanations = explainer.attribute(
        inputs=data,
       # baselines=baseline,
        target=int(target),
        additional_forward_args=None,
        return_convergence_delta=False,
        custom_attribution_func=None
    )
    return explanations.sum(dim=2)


def get_deepshap_attributions(data: torch.Tensor, baseline: Tensor, model: torch.nn.Module,forward_function: Callable) -> torch.tensor:
     # Will throw the same error w.r.t. shape of baseline vs input as with gradient shap
    return DeepLiftShap(model, multiply_by_inputs=None).attribute(inputs=data)


def get_gradient_shap_attributions(data: torch.Tensor, baseline: Tensor, model: torch.nn.Module, forward_function: Callable, target: list) -> torch.tensor:
    return GradientShap(forward_function).attribute(inputs=data)


def get_guided_backprop_attributions(data: torch.Tensor, 
                                     baseline: Tensor,
                                     model: torch.nn.Module,
                                     forward_function: Callable,
                                     target: list
) -> torch.tensor:
    # UserWarning: Setting backward hooks on ReLU activations.The hooks will be removed after the attribution is finished
    explainer = GuidedBackprop(forward_function)
    explanations = explainer.attribute(
        inputs=data,
        target=int(target),
        additional_forward_args=None
    )
    return explanations.sum(dim=2)


def get_deconvolution_attributions(data: torch.Tensor, 
                                   baseline: Tensor,
                                   model: torch.nn.Module,
                                   forward_function: Callable,
                                   target: list
) -> torch.tensor:
    explainer = Deconvolution(forward_function)
    explanations = explainer.attribute(
        inputs=data,
        target=int(target),
        additional_forward_args=None
    )
    return explanations.sum(dim=2)


def get_shapley_sampling_attributions(data: torch.Tensor, 
                                      baseline: Tensor,
                                      model: torch.nn.Module,
                                      forward_function: Callable,
                                      target: list
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
        show_progress=True
    )


def get_lime_attributions(data: torch.Tensor,
                          baseline: Tensor,
                          model: torch.nn.Module,
                          forward_function: Callable,
                          target: list
                          ) -> torch.tensor:
    # Difference between captum.attr.Lime and captum.attr.LimeBase?
    # Default interpretable_model = SkLearnLasso(alpha=0.01)
    explainer = Lime(forward_function)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=int(target),
        additional_forward_args=None,
        feature_mask=None,
        n_samples=50,
        perturbations_per_eval=1,
        return_input_shape=True,
        show_progress=True
    )


def get_kernel_shap_attributions(data: torch.Tensor,
                                 baseline: Tensor,
                                 model: torch.nn.Module,
                                 forward_function: Callable,
                                 target: list
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
        show_progress=True
    )


def get_lrp_attributions(data: torch.Tensor, baseline: Tensor, model: torch.nn.Module, forward_function: Callable) -> torch.tensor:
    return LayerLRP(forward_function, model).attribute(inputs=data)


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


def get_input_x_gradient(data: torch.Tensor,
                         baseline: Tensor,
                         model: torch.nn.Module,
                         forward_function: Callable,
                         target: list
                         ) -> torch.tensor:
    explainer = InputXGradient(forward_function)
    explanations = explainer.attribute(
        inputs=data,
        target=int(target),
        additional_forward_args=None
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
    'PFI Scikit-learn': get_pfi_scikit_learn_attribution,
    'Uniform random': get_uniform_random_attributions,
    'InputXGradient': get_input_x_gradient
}
