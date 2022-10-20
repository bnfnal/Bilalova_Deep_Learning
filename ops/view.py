import numpy as np
from ops.transformation import Transformation


class View(Transformation):
    def __init__(self):
        """
        reshape image to vector
        """
        pass

    def __call__(self, images):
        for image in images:
            image = np.array(image, dtype='uint8')
            image = image.reshape((28, 28))
        return images