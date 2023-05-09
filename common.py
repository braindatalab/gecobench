from collections import namedtuple

NAME_OF_PROJECT_CONFIG = 'project_config.json'
CLUSTER_PLATFORM_NAME = '#102-Ubuntu'
CLUSTER_DATA_DIR = '/mnt'
DATASET_ALL = 'all'
DATASET_SUBJECT = 'subject'


DataTargetPair = namedtuple('DataTargetPair', 'data target')
DataSet = namedtuple('DataSet', 'x_train y_train x_test y_test')
