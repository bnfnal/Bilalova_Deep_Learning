# -*- coding: utf-8 -*-
""" main.py """
import json

from configs.config import CFG
from dataloader.dataloader import DataLoader
from datasets.mnist_dataset import MNISTDataset
from enums.Datasets import DataSetType
from utils.config import Config
from ops.normalize import Normalize
from ops.view import View

def run():
    """Builds model, loads data, trains and evaluates"""

    cfg = Config()

    dataset = MNISTDataset(dataset_type = DataSetType.train,
                                 transforms = [Normalize(),View()],
                                 nrof_classes = cfg.data.nrof_classes)

    dataload = DataLoader(dataset,
                          cfg.train.batch_size,
                          None,
                          cfg.train.nrof_epoch)
    dataset.read_data()

    next(dataload.batch_generator())
    dataload.show_batch()

if __name__ == '__main__':
    run()

