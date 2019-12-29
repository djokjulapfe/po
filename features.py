import cv2
import numpy as np


class Feature:
    def get_feature(self, x: np.ndarray):
        return x


class RectSelect(Feature):

    def __init__(self, corner: tuple, shape: tuple = (1, 1)):
        self.corner = corner
        self.shape = shape

    def get_feature(self, x: np.ndarray):
        return x[
               self.corner[0]: self.corner[0] + self.shape[0],
               self.corner[1]: self.corner[1] + self.shape[1],
               ]


class Sum(Feature):

    def get_feature(self, x: np.ndarray):
        return np.sum(x)


class LinearFilter(Feature):

    def __init__(self, kernel: np.ndarray):
        super().__init__()
        self.kernel = kernel

    def get_feature(self, x: np.ndarray):
        return cv2.filter2D(x, -1, self.kernel)
