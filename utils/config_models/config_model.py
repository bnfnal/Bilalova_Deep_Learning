from asyncio.windows_events import NULL
from utils.config_models.base_config_section import BaseConfigSection

class ConfigModel(BaseConfigSection):

    def __init__(self, section):
        self.__input = []
        self.__up_stack = {}
        self.__acivation_function = NULL
        self.__output = NULL
        self.parameters = []
        super().__init__(section)


    @property
    def up_stack(self):
        return self.__up_stack
    @property
    def acivation_function(self):
        return self.__acivation_function
    @property
    def output(self):
        return self.__output


    def _fill_model_from_json(self, section):
        # self.__acivation_function = section['acivation_function']
        # self.__output = section['output']
        # self.__up_stack = self.__up_stack | section['up_stack']
        # for inputs in section['input']:
        #     self.__input.append(inputs)
        pass