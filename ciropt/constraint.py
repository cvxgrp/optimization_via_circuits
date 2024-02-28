"""
The code is borrowed from the PEPit https://github.com/PerformanceEstimation/PEPit
"""

class Constraint(object):
    """
    A :class:`Constraint` encodes either an equality or an inequality between two :class:`Expression` objects.

    A :class:`Constraint` must be understood either as
    `self.expression` = 0 or `self.expression`  <= 0
    depending on the value of `self.equality_or_inequality`.
    """
    counter = 0

    def __init__(self, expression, equality_or_inequality):
        """
        :class:`Constraint` objects can also be instantiated via the following arguments.
        Args:
            expression (Expression): an object of class Expression
            equality_or_inequality (str): either 'equality' or 'inequality'.
        """
        # Update the counter
        self.counter = Constraint.counter
        Constraint.counter += 1
        # Store the underlying expression
        self.expression = expression
        # Verify that 'equality_or_inequality' is well defined and store its value
        assert equality_or_inequality in {'equality', 'inequality'}
        self.equality_or_inequality = equality_or_inequality
        # The value of the underlying expression must be stored in self._value.
        self._value = None
        # Moreover, the associated dual variable value must be stored in self._dual_variable_value.
        self._dual_variable_value = None

    def eval(self):
        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self._value is None:
            try:
                self._value = self.expression.eval()
            except ValueError("The PEP must be solved to evaluate Expressions!"):
                raise ValueError("The PEP must be solved to evaluate Constraints!")

        return self._value

    def eval_dual(self):
        # If the attribute _dual_variable_value is not None, then simply return it.
        # Otherwise, raise a ValueError.
        if self._dual_variable_value is None:
            # The PEP would have filled the attribute at the end of the solve.
            raise ValueError("The PEP must be solved to evaluate Constraints dual variables!")

        return self._dual_variable_value
    