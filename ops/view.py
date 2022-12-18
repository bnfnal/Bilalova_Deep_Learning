import numpy as np
from ops.transformation import Transformation


class View(Transformation):
    def __init__(self):
        """
        reshape image to vector
        """
        pass

    def __call__(self, images):
        # for image in images:
        #     image = np.array(image, dtype='uint8')
        #     image = image.reshape((28, 28))

        return images.reshape((images.shape[0], -1))
        # возвращает список, вытянутый в один вектор
        # 9*28*28 -> 9*724