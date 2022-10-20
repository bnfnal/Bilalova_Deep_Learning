from asyncio.windows_events import NULL
from utils.config_models.base_config_section import BaseConfigSection


class ConfigData(BaseConfigSection):

    def __init__(self, section):
        self.__path = NULL
        self.__image_size = NULL
        self.__load_with_info = NULL
        self.__nrof_classes = NULL
        self.__shuffle = NULL
        super().__init__(section)

    @property
    def path(self):
        return self.__path


    @property
    def image_size(self):
        return self.__image_size
    @property
    def load_with_info(self):
        return self.__load_with_info
    @property
    def nrof_classes(self):
        return self.__nrof_classes
    @property
    def shuffle(self):
        return self.__shuffle


    def _fill_model_from_json(self, section):

        self.__path = section['path']
        self.__image_size = section['image_size']
        self.__load_with_info = section['load_with_info']
        self.__nrof_classes = section['nrof_classes']
        self.__shuffle = section['shuffle']
