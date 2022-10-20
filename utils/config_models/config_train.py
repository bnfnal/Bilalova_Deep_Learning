from asyncio.windows_events import NULL
from utils.config_models.base_config_section import BaseConfigSection

class ConfigTrain(BaseConfigSection):

    def __init__(self, section):
        self.__batch_size = NULL
        self.__nrof_epoch = NULL
        self.__optimizer = {}
        self.__metrics = []
        super().__init__(section)

    @property
    def batch_size(self):
        return self.__batch_size
    @property
    def nrof_epoch(self):
        return self.__nrof_epoch
    @property
    def optimizer(self):
        return self.__optimizer
    @property
    def metrics(self):
        return self.__metrics


    def _fill_model_from_json(self, section):
        self.__batch_size = int(section['batch_size'])
        self.__nrof_epoch = section['nrof_epoch']
        self.__optimizer = self.__optimizer | section['optimizer']
        for metrics in section['metrics']:
            self.__metrics.append(metrics)