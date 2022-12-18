# -*- coding: utf-8 -*-
""" main.py """

from training.trainer import Trainer
from utils.config import Config


def run():
    """Builds model, loads datasets, trains and evaluates"""

    cfg = Config()

    trainer = Trainer(cfg)

    #trainer.overfit_on_batch()
    trainer.fit()

if __name__ == '__main__':
    run()

