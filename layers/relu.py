import numpy as np
from layers.registry_utils import REGISTRY_TYPE
from layers.base_layer_class import BaseLayerClass


@REGISTRY_TYPE.register_module
class ReLU(BaseLayerClass):

    def __call__(self, x, phase):   # forward
        self.x = x
        return np.max(0, x)

    def get_grad(self):
        # np.dot, np.zeros_like/np.ones_like, np.zeros(), np.random...,
        self.grads = np.zeros_like(self.x)  # np.zeros((self.x.shape[0], self.x.shape[1]))
        self.grads[self.x > 0] = 1  # np.array
        # x>0: 1
        # x<=: 0
        return self.grads

    def backward(self, dy):
        return dy * self.grads