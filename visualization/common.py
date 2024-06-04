MODEL_NAME_MAP = dict(
    bert_only_classification='\\textit{BERT}-C',
    bert_randomly_init_embedding_classification='\\textit{BERT}-CE',
    bert_only_embedding_classification='\\textit{BERT}-CEf',
    bert_all='\\textit{BERT}-CEfAf',
    one_layer_attention_classification='\\textit{OLA}-CEA',
    bert_only_embedding='\\textit{BERT}-Ef',
)

MODEL_NAME_HTML_MAP = dict(
    bert_only_classification='<p><i>BERT</i>-C</p>',
    bert_randomly_init_embedding_classification='<p><i>BERT</i>-CE</p>',
    bert_only_embedding_classification='<p><i>BERT</i>-CEf</p>',
    bert_all='<p><i>BERT</i>-CEfAf</p>',
    one_layer_attention_classification='<p><i>OLA</i>-CEA</p>',
    bert_only_embedding='<p><i>BERT</i>-Ef</p>',
)

METRIC_NAME_MAP = dict(
    roc_auc='ROC AUC',
    precision_recall_auc='Precision-Recall AUC',
    avg_precision='Precision',
    precision_specificity='Precision-Specificity',
    top_k_precision='Top-K Precision',
    mass_accuracy='Mass Accuracy (MA)',
    mass_accuracy_method_grouped="Mass Accuracy (MA)",
)

METHOD_NAME_MAP = dict(Covariance="Pattern Variant")

MODEL_ORDER = [
    MODEL_NAME_MAP["bert_only_classification"],
    MODEL_NAME_MAP["bert_only_embedding_classification"],
    MODEL_NAME_MAP["bert_randomly_init_embedding_classification"],
    MODEL_NAME_MAP["bert_all"],
    MODEL_NAME_MAP["one_layer_attention_classification"],
]


HUE_ORDER = [
    'Uniform random',
    'Saliency',
    'Kernel SHAP',
    'Guided Backprop',
    'DeepLift',
    'InputXGradient',
    'LIME',
    'Gradient SHAP',
    'Integrated Gradients',
    'Pattern Variant',
]

DATASET_NAME_MAP = dict(
    gender_subj='$\mathcal{D}_{S}$',
    gender_all='$\mathcal{D}_{A}$',
    subj='$\mathcal{D}_{S}$',
    all='$\mathcal{D}_{A}$',
    imdb='IMDB',
)
ROW_ORDER = [DATASET_NAME_MAP['gender_all'], DATASET_NAME_MAP['gender_subj']]

GENDER = {0.0: 'female', 1.0: 'male'}
