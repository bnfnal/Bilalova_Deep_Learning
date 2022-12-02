# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod
from utils.config import Config

"""Используя абстракцию, мы можем объявить желаемые функции, не разбираясь в том, как они будут реализованы. 
Это похоже на договор о том, как должен выглядеть код. Таким образом, вы можете сначала принять решение 
о реализации на высоком уровне, а затем подробно рассмотреть каждую часть. 
Основной способ абстракции в Python - использование библиотеки ABC."""
class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


