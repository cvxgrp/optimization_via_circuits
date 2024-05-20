import numpy as np
import sympy as sp

from ciropt.expression import Expression
from ciropt.utils import *
from ciropt.sympy_parsing import *
from ciropt.sympy_to_solvers import *

"""
Most of the code is borrowed from the PEPit https://github.com/PerformanceEstimation/PEPit
"""


class Point(object):
    # point or a gradient
    # count the number of points needed to linearly generate the others
    counter = 0
    list_of_leaf_points = list()

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None):
        self._is_leaf = is_leaf
        self._value = None
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Point.counter
            Point.counter += 1
            Point.list_of_leaf_points.append(self)
        else:
            assert type(decomposition_dict) == dict
            self.decomposition_dict = decomposition_dict
            self.counter = None

    def get_is_leaf(self):
        return self._is_leaf

    def __add__(self, other):
        assert isinstance(other, Point)
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        merged_decomposition_dict = prune_dict(merged_decomposition_dict)
        return Point(is_leaf=False, decomposition_dict=merged_decomposition_dict)

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, sp.Basic):
            new_decomposition_dict = dict()
            for key, value in self.decomposition_dict.items():
                # new_decomposition_dict[key] = linearize_expression(value * other)
                new_decomposition_dict[key] = sp.simplify(value * other)
            return Point(is_leaf=False, decomposition_dict=new_decomposition_dict)
        elif isinstance(other, Point):
            decomposition_dict = multiply_dicts(self.decomposition_dict, other.decomposition_dict)
            return Expression(is_leaf=False, decomposition_dict=decomposition_dict)
        else:
            raise TypeError("Points can be multiplied by scalar constants and other points only!"
                            "Got {}".format(type(other)))

    def __mul__(self, other):
        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        return self.__rmul__(1 / denominator)

    def __pow__(self, power):
        assert power == 2
        return self.__rmul__(self)

    def eval(self):
        if self._value is None:
            if self._is_leaf:
                raise ValueError("The CircuitOpt must be solved to evaluate Points!")
            else:
                value = np.zeros(Point.counter)
                for point, weight in self.decomposition_dict.items():
                    value += weight * point.eval()
                self._value = value
        return self._value

