import cv2
import numpy as np


class Feature:
    def get_feature(self, x: np.ndarray):
        return x


class RectSelect(Feature):

    def __init__(self, up, down, left, right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def get_feature(self, x: np.ndarray):
        return x[self.up: self.down, self.left: self.right]


class Sum(Feature):

    def get_feature(self, x: np.ndarray):
        return np.sum(x)


class Multiply(Feature):

    def __init__(self, factor: float):
        self.factor = factor

    def get_feature(self, x: np.ndarray):
        return x * self.factor


class Mean(Feature):

    def get_feature(self, x: np.ndarray):
        return np.mean(x)


class LinearFilter(Feature):

    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel

    def get_feature(self, x: np.ndarray):
        return cv2.filter2D(x, -1, self.kernel)


class Stack(Feature):
    def __init__(self, sub_extractors: list):
        self.sub_extractors = sub_extractors

    def get_feature(self, x: np.ndarray):
        ret = x
        for extractor in self.sub_extractors:
            ret = extractor.get_feature(ret)
        return ret
