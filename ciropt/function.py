import casadi as ca
import numpy as np
import sympy as sp

from ciropt.point import Point
from ciropt.expression import Expression
from ciropt.utils import *


"""
Most of the code is borrowed from the PEPit https://github.com/PerformanceEstimation/PEPit
"""

class Function(object):
    # It counts the number of functions defined from scratch.
    counter = 0
    list_of_functions = list()

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        self._is_leaf = is_leaf
        self.reuse_gradient = reuse_gradient
        Function.list_of_functions.append(self)

        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Function.counter
            Function.counter += 1
        else:
            assert type(decomposition_dict) == dict
            self.decomposition_dict = decomposition_dict
            self.counter = None

        self.list_of_stationary_points = list()
        self.list_of_points = list()
        self.list_of_class_constraints = list()

    def get_is_leaf(self):
        return self._is_leaf

    def __add__(self, other):
        assert isinstance(other, Function)
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        return Function(is_leaf=False,
                        decomposition_dict=merged_decomposition_dict,
                        reuse_gradient=self.reuse_gradient and other.reuse_gradient)

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, sp.Basic)
        new_decomposition_dict = dict()
        for key, value in self.decomposition_dict.items():
            new_decomposition_dict[key] = linearize_expression(value * other)
        return Function(is_leaf=False,
                        decomposition_dict=new_decomposition_dict,
                        reuse_gradient=self.reuse_gradient)

    def __mul__(self, other):
        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        return self.__rmul__(other=1 / denominator)

    def add_class_constraints(self):
        raise NotImplementedError("This method must be overwritten in children classes")

    def _is_already_evaluated_on_point(self, point):
        for triplet in self.list_of_points:
            if triplet[0].decomposition_dict == point.decomposition_dict:
                return triplet[1:]
        return None

    def _separate_leaf_functions_regarding_their_need_on_point(self, point):
        list_of_functions_which_need_nothing = list()
        list_of_functions_which_need_gradient_only = list()
        list_of_functions_which_need_gradient_and_function_value = list()
        for function, weight in self.decomposition_dict.items():
            if function._is_already_evaluated_on_point(point=point):
                if function.reuse_gradient:
                    list_of_functions_which_need_nothing.append((function, weight))
                else:
                    list_of_functions_which_need_gradient_only.append((function, weight))
            else:
                list_of_functions_which_need_gradient_and_function_value.append((function, weight))
        return list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value

    def add_point(self, triplet):
        point, g, f = triplet
        assert isinstance(point, Point) and  isinstance(g, Point) and isinstance(f, Expression)
        for element in triplet:
            element.decomposition_dict = prune_dict(element.decomposition_dict)
        self.list_of_points.append(triplet)
        if g.decomposition_dict == dict():
            self.list_of_stationary_points.append(triplet)
        if not self._is_leaf:
            self.decomposition_dict = prune_dict(self.decomposition_dict)
            tuple_of_lists_of_functions = self._separate_leaf_functions_regarding_their_need_on_point(point=point)
            list_of_functions_which_need_nothing = tuple_of_lists_of_functions[0]
            list_of_functions_which_need_something = tuple_of_lists_of_functions[1] + tuple_of_lists_of_functions[2]
            if list_of_functions_which_need_something != list():
                total_number_of_involved_leaf_functions = len(self.decomposition_dict.keys())
                gradient_of_last_leaf_function = g
                value_of_last_leaf_function = f
                number_of_currently_computed_gradients_and_values = 0
                # enforce g = \sum_i w_i * g_i 
                #         f = \sum_i w_i * f_i
                for function, weight in list_of_functions_which_need_nothing + list_of_functions_which_need_something:
                    if number_of_currently_computed_gradients_and_values < total_number_of_involved_leaf_functions - 1:
                        grad, val = function.oracle(point)
                        gradient_of_last_leaf_function = gradient_of_last_leaf_function - weight * grad
                        value_of_last_leaf_function = value_of_last_leaf_function - weight * val
                        number_of_currently_computed_gradients_and_values += 1
                    else: # The latest function must receive fully conditioned gradient and function value
                        gradient_of_last_leaf_function = gradient_of_last_leaf_function / weight
                        value_of_last_leaf_function = value_of_last_leaf_function / weight
                        function.add_point((point, gradient_of_last_leaf_function, value_of_last_leaf_function))

    def oracle(self, point):
        assert isinstance(point, Point)
        associated_grad_and_function_val = self._is_already_evaluated_on_point(point=point)
        if associated_grad_and_function_val and self.reuse_gradient:
            return associated_grad_and_function_val
        if associated_grad_and_function_val and not self.reuse_gradient:
            f = associated_grad_and_function_val[-1]
        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value = \
                                        self._separate_leaf_functions_regarding_their_need_on_point(point=point)
        if associated_grad_and_function_val is None:
            if list_of_functions_which_need_gradient_and_function_value == list():
                f = Expression(is_leaf=False, decomposition_dict=dict())
                for function, weight in self.decomposition_dict.items():
                    f += weight * function.value(point=point)
            else: 
                f = Expression(is_leaf=True, decomposition_dict=None)
        if list_of_functions_which_need_gradient_and_function_value == list() and list_of_functions_which_need_gradient_only == list():
            g = Point(is_leaf=False, decomposition_dict=dict())
            for function, weight in self.decomposition_dict.items():
                g += weight * function.gradient(point=point)
        else:
            g = Point(is_leaf=True, decomposition_dict=None)
        self.add_point(triplet=(point, g, f))
        return g, f

    def gradient(self, point):
        return self.subgradient(point)

    def subgradient(self, point):
        assert isinstance(point, Point)
        g, _ = self.oracle(point)
        return g

    def value(self, point):
        assert isinstance(point, Point)
        associated_grad_and_function_val = self._is_already_evaluated_on_point(point=point)
        if associated_grad_and_function_val:
            f = associated_grad_and_function_val[-1]
        else:
            _, f = self.oracle(point)
        return f

    def __call__(self, point):
        return self.value(point=point)

    def stationary_point(self, return_gradient_and_function_value=False):
        point = Point(is_leaf=True, decomposition_dict=None)
        # equivalent to having g = 0
        g = Point(is_leaf=False, decomposition_dict=dict())
        f = Expression(is_leaf=True, decomposition_dict=None)
        self.add_point((point, g, f))
        if return_gradient_and_function_value:
            return point, g, f
        else:
            return point


def proximal_step(x0, func, gamma):
    # (I + gamma * \partial f)x = x0
    gx = Point()
    fx = Expression()
    x = x0 - gamma * gx
    func.add_point((x, gx, fx))
    return x, gx, fx
        


class ConvexFunction(Function):
    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        super().__init__(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)

    def add_class_constraints(self):
        for point_i in self.list_of_points:
            xi, gi, fi = point_i
            for point_j in self.list_of_points:
                xj, gj, fj = point_j
                if point_i != point_j:
                    # Interpolation conditions of convex functions class
                    self.list_of_class_constraints.append( fj - fi +  gj * (xi - xj) )



class SmoothStronglyConvexFunction(Function):
    def __init__(self,
                 mu,
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        super().__init__(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=True)
        self.mu = mu
        self.L = L
        assert self.L < np.inf, print("use the class StronglyConvexFunction")

    def add_class_constraints(self):
        for point_i in self.list_of_points:
            xi, gi, fi = point_i
            for point_j in self.list_of_points:
                xj, gj, fj = point_j
                if point_i != point_j:
                    # Interpolation conditions of smooth strongly convex functions class
                    self.list_of_class_constraints.append(fj - fi +
                                        gj * (xi - xj)
                                        + 1/(2*self.L) * (gi - gj) ** 2
                                        + self.mu / (2 * (1 - self.mu / self.L)) * (xi - xj - 1/self.L * (gi - gj))**2)
                    


class StronglyConvexFunction(Function):
    def __init__(self,
                 mu,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        super().__init__(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        self.mu = mu

    def add_class_constraints(self):
        for point_i in self.list_of_points:
            xi, gi, fi = point_i
            for point_j in self.list_of_points:
                xj, gj, fj = point_j
                if point_i != point_j:
                    # Interpolation conditions of smooth strongly convex functions class
                    self.list_of_class_constraints.append(fj - fi + 
                                        gj * (xi - xj)
                                        + self.mu / 2 * (xi - xj) ** 2)
                    


class SmoothConvexFunction(SmoothStronglyConvexFunction):
    def __init__(self,
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        # Inherit from SmoothStronglyConvexFunction as a special case of it with mu=0.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         mu=0,
                         L=L)

        assert self.L < np.inf, print("use the class ConvexFunction instead")
