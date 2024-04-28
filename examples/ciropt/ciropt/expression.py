import casadi as ca
import numpy as np
import sympy as sp

from ciropt.utils import *
from ciropt.sympy_parsing import *
from ciropt.sympy_to_solvers import *


"""
Most of the code is borrowed from the PEPit https://github.com/PerformanceEstimation/PEPit
"""


class Expression(object):
    # linear combination of functions values, inner products of points and / or gradients
    counter = 0
    list_of_leaf_expressions = list()

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None):
        self._is_leaf = is_leaf
        self._value = None
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Expression.counter
            Expression.counter += 1
            Expression.list_of_leaf_expressions.append(self)
        else:
            assert type(decomposition_dict) == dict
            self.decomposition_dict = decomposition_dict
            self.counter = None

    def get_is_leaf(self):
        return self._is_leaf

    def __add__(self, other):
        if isinstance(other, Expression):
            merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        elif isinstance(other, int) or isinstance(other, float):
            merged_decomposition_dict = merge_dict(self.decomposition_dict, {1: other})
        else:
            raise TypeError("Expression can be added only to other expression or scalar values!"
                            "Got {}".format(type(other)))
        return Expression(is_leaf=False, decomposition_dict=merged_decomposition_dict)

    def __radd__(self, other):
        return self.__add__(other=other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return - self.__sub__(other=other)

    def __neg__(self):
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        assert isinstance(other, int) or isinstance(other, float) or isinstance(other, sp.Basic)
        new_decomposition_dict = dict()
        for key, value in self.decomposition_dict.items():
            # new_decomposition_dict[key] = linearize_expression(sp.simplify(value * other))
            new_decomposition_dict[key] = sp.simplify(value * other)
        return Expression(is_leaf=False, decomposition_dict=new_decomposition_dict)

    def __mul__(self, other):
        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        return self.__rmul__(other=1 / denominator)

    def __hash__(self):
        return super().__hash__()

    def eval(self):
        if self._value is None:
            if self._is_leaf:
                raise ValueError("The CircuitOpt must be solved to evaluate Expressions!")
            else:
                value = 0
                for key, weight in self.decomposition_dict.items():
                    if type(key) == Expression:
                        assert key.get_is_leaf()
                        value += weight * key.eval()
                    elif type(key) == tuple:
                        point1, point2 = key
                        assert point1.get_is_leaf()
                        assert point2.get_is_leaf()
                        value += weight * np.dot(point1.eval(), point2.eval())
                    elif key == 1:
                        value += weight
                    else:
                        raise TypeError("Expressions are made of function values, inner products and constants only!"
                                        "Got {}".format(type(key)))
                self._value = value
        return self._value
