from abc import ABC, abstractmethod

class BaseConfigSection(ABC):

    def __init__(self, section):
        self._fill_model_from_json(section)

    @abstractmethod
    def _fill_model_from_json(self, section):
        pass