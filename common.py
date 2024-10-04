from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum

NAME_OF_PROJECT_CONFIG = 'project_config.json'
NAME_OF_DATA_CONFIG = 'data_config.json'
CLUSTER_PLATFORM_NAME = '#102-Ubuntu'
CLUSTER_DATA_DIR = '/mnt'
DATASET_ALL = 'all'
DATASET_SUBJECT = 'subject'

DataTargetPair = namedtuple('DataTargetPair', 'data target')
DataSet = namedtuple('DataSet', 'x_train y_train x_test y_test')


class SaveVersion(Enum):
    last = "last"
    best = "best"


class DatasetKeys(Enum):
    binary_gender_all = "binary_gender_all"
    binary_gender_subj = "binary_gender_subj"
    non_binary_gender_all = "non_binary_gender_all"
    non_binary_gender_subj = "non_binary_gender_subj"
    sentiment_twitter = "sentiment_twitter"
    sentiment_imdb = "sentiment_imdb"


def validate_dataset_key(dataset_key: str):
    if dataset_key not in DatasetKeys.__members__:
        raise ValueError(f'Invalid dataset key: {dataset_key}')


# XAIResult = namedtuple(
#     'XAIResult',
#     'model_name dataset_name sentence target '
#     'correct_classified attribution_method '
#     'raw_attribution attribution ground_truth'
# )
@dataclass
class XAIResult:
    model_name: str = None
    model_version: SaveVersion = None
    model_repetition_number: int = None
    model_version: SaveVersion = None
    dataset_type: str = None
    target: float = None
    attribution_method: str = None
    sentence: list = field(default_factory=list)
    prompt: list = field(default_factory=list)
    raw_attribution: list = field(default_factory=list)
    attribution: list = field(default_factory=list)
    ground_truth: list = field(default_factory=list)
    sentence_idx: int = None
    pred_probabilities: list = None


@dataclass
class XAIEvaluationResult:
    model_name: str = None
    model_version: str = None
    model_repetition_number: int = None
    dataset_type: str = None
    attribution_method: str = None
    # Here, we append evaluation metrics: ROC-AUC, precision, etc


@dataclass
class ModelEvaluationResult:
    model_name: str = None
    model_version: str = None
    model_repetition_number: int = None
    dataset_type: str = None
    accuracy: float = None


@dataclass
class EvaluationResult:
    xai_results_correct: list[XAIEvaluationResult] = None
    xai_results_all: list[XAIEvaluationResult] = None
    model_results: list[ModelEvaluationResult] = None
