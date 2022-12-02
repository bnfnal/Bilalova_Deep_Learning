import math

import numpy as np

from layers.registry_utils import REGISTRY_TYPE
from layers.base_layer_class import BaseLayerClass


@REGISTRY_TYPE.register_module
class FullyConnected(BaseLayerClass):
    def __init__(self, input_size, output_size):   # гиперпараметры слоя
        """
        инициализация обучаемых параметров нейронной сети случайным образом по формуле из презентации
        """
        super().__init__()
        self.bias = np.zeros(output_size)
        self.weight = np.random.normal(0, math.sqrt(2/output_size), size = (input_size, output_size))   # иницализируем матрицу


    def __call__(self, x, phase):  # forward  фаза обучения или валидации
        self.x = x   # батч (небольшое множество) картинок
        return np.dot(self.x, self.weight) + self.bias     # вычислить матрично

    @property
    def trainable(self):
        """
        свойство, является ли слой обучаемым
        """
        return True

    def get_grad(self):   # dg_df
        df_dx = self.weight
        df_dw = self.x
        df_db = 1
        self.grads = [df_dx, df_dw, df_db]
        return self.grads

    def backward(self, dy):   # градиенты по текущему слою относ целевой функции  df_dx
        self.get_grad()
        self.x_grad = np.dot(self.grads[0], dy)
        self.weight_grad = np.dot(self.grads[1], dy)
        self.bias_grad = self.grads[2] * dy
        return self.x_grad

    def update_weights(self, update_func):
        """
        обновление обучаемых параметров, если они есть, иначе ничего
        :param update_func: функция обновления, указано в презентации
        """
        self.weight = self.weight - update_func(self.weight_grad)
        self.weight = self.bias - update_func(self.bias_grad)

