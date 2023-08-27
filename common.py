from collections import namedtuple
from dataclasses import dataclass, field

NAME_OF_PROJECT_CONFIG = 'project_config.json'
CLUSTER_PLATFORM_NAME = '#102-Ubuntu'
CLUSTER_DATA_DIR = '/mnt'
DATASET_ALL = 'all'
DATASET_SUBJECT = 'subject'

DataTargetPair = namedtuple('DataTargetPair', 'data target')
DataSet = namedtuple('DataSet', 'x_train y_train x_test y_test')


# XAIResult = namedtuple(
#     'XAIResult',
#     'model_name dataset_name sentence target '
#     'correct_classified attribution_method '
#     'raw_attribution attribution ground_truth'
# )
@dataclass
class XAIResultPerSentence:
    model_name: str = None
    dataset_type: str = None
    target: float = None
    correct_classified_intersection: float = None
    attribution_method: list = field(default_factory=list)
    sentence: list = field(default_factory=list)
    raw_attributions_all_xai_methods: list = field(default_factory=list)
    attribution: list = field(default_factory=list)
    ground_truth: list = field(default_factory=list)

class XAIResult:
    model_name: str = None
    model_repetition_number: int = None
    dataset_type: str = None
    target: float = None
    correct_classified_intersection: float = None
    attribution_method: str = None
    sentence: list = field(default_factory=list)
    raw_attribution: list = field(default_factory=list)
    attribution: list = field(default_factory=list)
    ground_truth: list = field(default_factory=list)


@dataclass
class EvaluationResult:
    model_name: str = None
    dataset_type: str = None
    model_repetition_number: int = None
    attribution_method: str = None
    # Here, we append evaluation metrics: ROC-AUC, precision, etc
