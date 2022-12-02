import numpy as np


class CrossEntripyLoss:
    def __call__(self, logits, labels):
        """
        вычисление значения целевой функции
        :param logits: выход последнего полносвязного слоя/выход слоя softmax
        :param labels: ground-truth метки
        :return: значение целевой функции
        """
        self.labels = labels
        self.logits = self.softmax(logits)
        #return -np.sum(labels * np.log(self.logits))
        return -np.sum(np.log(self.logits)[np.arange(len(self.labels)), self.labels])

    def softmax(self, x):
        self.x = x
        out = np.exp(x - np.max(x))
        return out / np.max(out)

    def get_grad(self):
        """
        вычисление градиентов по целевой функции
        """
        self.grads = self.logits
        self.grads[np.arange(len(self.labels)), self.labels] -= 1
        return self.grads

    def backward(self, dy=1):
        """
        вычисление градиентов итоговых для передачи дальше
        :param dy: значение градиента от следующего слоя
        :return: значение градиента текущего слоя
        """
        self.get_grad()
        return dy * self.grads

    def update_weights(self, update_func):
        """
        ничего не делать
        :param update_func:
        :return:
        """
        pass