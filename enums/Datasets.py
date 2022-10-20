from enum import Enum

class AvailableDatasets(Enum):
    MNIST = 'MNIST'

class DataSetType(Enum):
    valid = 10
    train = 20
    test = 30