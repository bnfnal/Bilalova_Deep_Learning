# -*- coding: utf-8 -*-
"""Config class"""

from asyncio.windows_events import NULL
import json

from configs.config import CFG
from utils.config_models.config_data import ConfigData
from utils.config_models.config_model import ConfigModel
from utils.config_models.config_train import ConfigTrain


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self):
        self.from_json()

    @classmethod
    def from_json(self):
        """Creates config from json"""

        self.data = ConfigData(CFG['data'])
        self.train = ConfigTrain(CFG['train'])
        self.model = ConfigModel(CFG['model'])
