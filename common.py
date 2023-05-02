from collections import namedtuple

NAME_OF_PROJECT_CONFIG = 'project_config.json'

DataTargetPair = namedtuple('DataPair', 'data target')._make([list(), list()])
DataSet = namedtuple('DataSet', 'x_train y_train x_test y_test')
