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

# class BertLogits(torch.nn.module)
# init function: our bert model
# forward function: return model.logits

class BertLogits(torch.nn.Module):
    def __init__(self,bertmodel) -> None:
        super(BertLogits, self).__init__()
        print("BertLogits initalized")
        self.model = bertmodel

    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.model(input)[0]

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
            # print("forward_function")
            # print(inputs, type(inputs))
            # for name, param in model.named_parameters():
            #    print(name)

            # Output:
            # SequenceClassifierOutput(loss=None, logits=tensor([[-0.4450, -0.5829]]), hidden_states=None, attentions=None)

            # ToDo 22.06.:
            # Cut off embedding layer from Bert model
            # Use this model and provide embeddings which we have
            # Probably create a Wrapper
            # Probably create a new forward function

            #print("model:")
            #print(type(model))
            #print(model)
            #print(model.layer[0])

            #print(model.bert.encoder)

            #print("model.bert.encoder[0]")
            #layer_list = model.bert.encoder.layer
            #print(layer_list[0])

            #print("####################################")
            #print("Iterate over model children:")
            #for idx,module in enumerate(model.children()):
            #    print(idx,module)

            #print("####################################")

            #input_model = model.base_model.embeddings
            #print(input_model)
            #embeddings = input_model(x)


            #print(model.bert.encoder)
            #print(model.bert.encoder(embeddings))
            
            #print(model.bert.pooler(inputs))
            
            #print("####################################")
            model_children = list(model.children())
            #print(model_children[0].encoder)
            #print(model_children[0].pooler) 

            print("####################################")
            
            model_without_embedding_layer = []
            model_without_embedding_layer.append(model_children[0].encoder)
            model_without_embedding_layer.append(model_children[0].pooler)
            model_without_embedding_layer.append(model_children[1])
            model_without_embedding_layer.append(model_children[2])

            new_model = torch.nn.Sequential(*model_without_embedding_layer)
            #print(new_model)

            #input_model = model.base_model.embeddings
            #print(input_model)
            #embeddings = input_model(x)

            #print(new_model(embeddings))
            
            #print("####################################")
            #model_without_embedding_layer = torch.nn.ModuleList()
            #model_without_embedding_layer.append(model_children[0].encoder)
            #model_without_embedding_layer.append(model_children[0].pooler)
            #model_without_embedding_layer.append(model_children[1])
            #model_without_embedding_layer.append(model_children[2])
            #print(model_without_embedding_layer)

            #copyOfModel = copy(model)
            #print("copyOfModel", copyOfModel)
            


            # copyOfModel.bert.encoder.layer = model_without_embedding_layer
            # print(copyOfModel)

            # print("####################################")
            # print(copyOfModel(inputs))

            #print("Iterate over model modules:")
            #for idx,module in enumerate(model.modules()):
            #    print(idx,module)
            

            output = model(inputs)
            # print("forward_function after output = model(inputs)")
            # print(output[0].shape)
            # print("after classification layer")
            # last_layer = torch.max(torch.softmax(output.logits, dim=1)).unsqueeze(-1)
            # print(last_layer.shape)
            #return torch.max(torch.softmax(output.logits, dim=1)).unsqueeze(-1)
            forward_function_output = torch.softmax(output.logits, dim=1)
            print(forward_function_output)
            return forward_function_output

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

        def forward_function_saliency(inputs: Tensor) -> Tensor:
            print("forward_function_saliency")
            
            model_children = list(model.children())
            model_without_embedding_layer = []
            model_without_embedding_layer.append(model_children[0].encoder)
            model_without_embedding_layer.append(model_children[0].pooler)
            model_without_embedding_layer.append(model_children[1])
            model_without_embedding_layer.append(model_children[2])
            model = torch.nn.Sequential(*model_without_embedding_layer)

            return model

    else:
        forward_function = None
        input_model = model
    




    for method_name in methods:
        logger.info(method_name)

        if method_name == "Saliency":
            print(input_model)
            x = input_model(x)
            print("EMEBDINNGS:",x, x.shape)
            #forward_function = forward_function_saliency
            
        a = methods_dict.get(method_name)(
            forward_function=forward_function,
            baseline=baseline,
            data=x, model=input_model, target=target
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
        forward_function: Callable,
        target: list
) -> torch.tensor:
    # Model = embedding layers of bert
    # Works because compute IG only up to the embedding layer
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
                              forward_function: Callable,
                              target: list
) -> torch.tensor:
    print("data type:",data.dtype, data)  
    print(model)
      
    explainer = Saliency(forward_function)
    # UserWarning: Input Tensor 0 has a dtype of torch.int64. 
    # Gradients cannot be activated for these data types.
    # RuntimeError: One of the differentiated Tensors does not require grad
    print(int(target))
    return explainer.attribute(
        inputs=data,
        target=int(target),
        abs=True,
        additional_forward_args=None
    )


def get_deeplift_attributions(data: torch.Tensor,
                              baseline: Tensor,
                              model: torch.nn.Module,
                              forward_function: Callable
) -> torch.tensor:
    explainer = DeepLift(model, multiply_by_inputs=None, eps=1e-10)
    # AssertionError: Target list length does not match output!
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=[0,1],
        additional_forward_args=None,
        return_convergence_delta=False,
        custom_attribution_func=None
    )


def get_deepshap_attributions(data: torch.Tensor,
                              baseline: Tensor,
                              model: torch.nn.Module,
                              forward_function: Callable
) -> torch.tensor:
    explainer = DeepLiftShap(model, multiply_by_inputs=None)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        target=[0,1],
        additional_forward_args=None,
        return_convergence_delta=None,
        custom_attribution_func=None
    )


def get_gradient_shap_attributions(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable,
        target: list
) -> torch.tensor:
    # RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: 
    # Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
    # print(data.dtype)
    explainer = GradientShap(model)
    return explainer.attribute(
        inputs=data,
        baselines=baseline,
        n_samples=5,  
        stdevs=0.0,
        target=target,  
        return_convergence_delta=True
    )


def get_guided_backprop_attributions(data: torch.Tensor, 
                                     baseline: Tensor,
                                     model: torch.nn.Module,
                                     forward_function: Callable,
                                     target: list
) -> torch.tensor:    
    print("get_guided_backprop_attributions")
    print(model)
    explainer = GuidedBackprop(model)
    # Input Tensor 0 has a dtype of torch.int64. Gradients cannot be activated for these data types.
    # RuntimeError: One of the differentiated Tensors does not require grad
    this_is_target = int(target)
    print(this_is_target)
    return explainer.attribute(
        inputs=data,
        target=this_is_target,
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


def get_input_x_gradient(
        data: torch.Tensor,
        baseline: Tensor,
        model: torch.nn.Module,
        forward_function: Callable
) -> torch.tensor:
    explainer = InputXGradient(forward_function)
    # RuntimeError: One of the differentiated Tensors does not require grad
    # UserWarning: Input Tensor 0 has a dtype of torch.int64. Gradients cannot be activated for these data types.
    # If traget = int then AssertionError: Cannot choose target column with output shape torch.Size([1]).
    return explainer.attribute(
        inputs=data,
        target=0,
        additional_forward_args=None
    )

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
