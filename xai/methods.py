from typing import List, Dict, Tuple, Callable

import numpy as np
import torch
from captum.attr import (
    IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop,
    Deconvolution, ShapleyValueSampling, Lime, KernelShap, LayerLRP, FeaturePermutation, LayerIntegratedGradients
)
from loguru import logger
from torch import Tensor

BERT = 'bert'


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
) -> Dict:
    attributions = dict()
    check_availability_of_xai_methods(methods=methods)
    print(methods)
    if BERT in model_type:
        def forward_function(inputs: Tensor) -> Tensor:
            # print("forward_function")
            # print(inputs, type(inputs))
            # for name, param in model.named_parameters():
            #    print(name)

            # Output:
            # SequenceClassifierOutput(loss=None, logits=tensor([[-0.4450, -0.5829]]), hidden_states=None, attentions=None)
            output = model(inputs)
            # print("forward_function after output = model(inputs)")
            # print(output[0].shape)
            # print("after classification layer")
            # last_layer = torch.max(torch.softmax(output.logits, dim=1)).unsqueeze(-1)
            # print(last_layer.shape)
            return torch.max(torch.softmax(output.logits, dim=1)).unsqueeze(-1)

        input_model = model.base_model.embeddings

        # input_model:
        # word_embeddings.weight
        # position_embeddings.weight
        # token_type_embeddings.weight
        # LayerNorm.weight
        # LayerNorm.bias

        # print(model.base_model)
        # for name, param in input_model.named_parameters():
        #    print(name)

    else:
        forward_function = None
        input_model = model

    for method_name in methods:
        logger.info(method_name)

        if method_name == "Guided Backprop" or method_name == "Deconvolution":
            input_model = model

        a = methods_dict.get(method_name)(
            forward_function=forward_function,
            baseline=baseline,
            data=x, model=input_model
        )

        # print("a:")
        # print(a.shape)
        # print(a)

        if BERT in model_type and method_name:
            a = a.squeeze(0)
            a = a / torch.norm(a)
            a = a.cpu().detach().numpy()

        # print("a after summation:")
        # print(a.shape)

        attributions[method_name] = a

    return attributions


def get_integrated_gradients_attributions(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable
) -> torch.tensor:
    explainer = LayerIntegratedGradients(forward_function, model)
    explanations = explainer.attribute(
        inputs=data,
        baselines=baseline,
        n_steps=200,
        return_convergence_delta=False,
    )
    return explanations.sum(dim=2)


def get_saliency_attributions(data: torch.Tensor,
                              baseline: Tensor,
                              model: torch.nn.Module,
                              forward_function: Callable
) -> torch.tensor:
    # method not working yet
    explainer = Saliency(forward_function)
    # print("forward_function:")
    # print(forward_function(data))
    # print(data.shape, type(data), data.dtype)
    # print(data)
    # data = data.double()
    # print(data.dtype)

    # UserWarning: Input Tensor 0 has a dtype of torch.int64. 
    # Gradients cannot be activated for these data types.
    # RuntimeError: One of the differentiated Tensors does not require grad
    return explainer.attribute(
        inputs=data,
        target=None,
        abs=True,
        additional_forward_args=None
    )


def get_deeplift_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return DeepLift(model).attribute(data, target=target)


def get_deepshap_attributions(data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return DeepLiftShap(model).attribute(data, target=target, baselines=torch.zeros(data.shape))


def get_gradient_shap_attributions(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable
) -> torch.tensor:
    # method not working ydt
    explainer = GradientShap(forward_function, model)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        n_samples=5,  # default
        stdevs=0.0,  # default
        return_convergence_delta=True
    )


def get_guided_backprop_attributions(data: torch.Tensor, 
                                     baseline: Tensor,
                                     model: torch.nn.Module,
                                     forward_function: Callable
) -> torch.tensor:
    print("get_guided_backprop_attributions")
    #for name, param in model.named_parameters():
    #     print(name)
    explainer = GuidedBackprop(model)

    print("model(data):")
    print(model(data)[0])

    # Output of model is a SequenceClassifierOutput where we need to access the logits
    # 'SequenceClassifierOutput' object has no attribute 'shape'
    return explainer.attribute(
        inputs=data,
        target=[0,1],
        additional_forward_args=None
    )

def get_deconvolution_attributions(data: torch.Tensor, 
                                   baseline: Tensor,
                                   model: torch.nn.Module,
                                   forward_function: Callable
) -> torch.tensor:
    explainer = Deconvolution(model)
    # Same issue with output of Bert model as with guided_backprop
    return explainer.attribute(
        inputs=data,
        target=[0,1],
        additional_forward_args=None
    )

def get_shapley_sampling_attributions(data: torch.Tensor, 
                                      baseline: Tensor,
                                      model: torch.nn.Module,
                                      forward_function: Callable
) -> torch.tensor:
    explainer = ShapleyValueSampling(forward_function)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=None,
        additional_forward_args=None,
        feature_mask=None,
        n_samples=25,
        perturbations_per_eval=1,
        show_progress=True
    )

def get_lime_attributions(data: torch.Tensor,
                          baseline: Tensor,
                          model: torch.nn.Module,
                          forward_function: Callable
                          ) -> torch.tensor:
    # Difference between captum.attr.Lime and captum.attr.LimeBase?
    # Default interpretable_model = SkLearnLasso(alpha=0.01)
    explainer = Lime(forward_function)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=None,
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
                                 forward_function: Callable
                                 ) -> torch.tensor:
    explainer = KernelShap(forward_function)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=None,
        additional_forward_args=None,
        feature_mask=None,
        n_samples=25,
        perturbations_per_eval=1,
        return_input_shape=True,
        show_progress=True
    )


def get_lrp_attributions(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable
) -> torch.tensor:
    # method not working yet
    # No LRP rule for 'torch.nn.modules.sparse.Embedding'
    # input_model_layers = []
    # for name, param in model.named_parameters():
    #    input_model_layers.append(name)
    #    print(name)
    # layers_to_explain = [model.word_embeddings]
    # print("layers_to_explain:",layers_to_explain)'

    explainer = LayerLRP(forward_function, model)
    return explainer.attribute(
        inputs=data
    )
    # return LRP(model).attribute(data, target=target)
    # return GradientShap(model).attribute(data, target=target, baselines=torch.zeros(data.shape))
    # https://arxiv.org/pdf/2101.00196.pdf
    # https://github.com/frankaging/BERT-LRP
    # https://proceedings.mlr.press/v162/ali22a/ali22a.pdf
    # https://github.com/ameenali/xai_transformers


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
