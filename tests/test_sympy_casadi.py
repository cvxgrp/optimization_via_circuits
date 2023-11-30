import ciropt as co
import casadi as ca
import sympy as sp
import numpy as np




def test_sympy_expression_to_casadi():
    opti = ca.Opti()
    ca_vars = { 'b': opti.variable(),
                'd': opti.variable(),
                'h': opti.variable(),}
    b, d, h = sp.symbols('b d h')
    # strongly convex function -- unique minimizer
    expr = (b - d - 5)**2 + (h - d - 10)**2 + 0.01 * (b**2 + d**2 + h**2)
    ca_expr2 = co.sympy_expression_to_casadi(expr, ca_vars, opti)
    opti.minimize(ca_expr2)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
    sol = opti.solve()
    vals2 = sol.value(ca_expr2), sol.value(ca_vars["b"]), sol.value(ca_vars["d"]), sol.value(ca_vars["h"])

    opti = ca.Opti()
    ca_vars = { 'b': opti.variable(),
                'd': opti.variable(),
                'h': opti.variable(),}
    ca_expr = (ca_vars["b"] - ca_vars["d"] - 5)**2 + (ca_vars["h"] - ca_vars["d"]- 10)**2 + 0.01 * (ca_vars["h"]**2 + ca_vars["d"]**2 + ca_vars["b"]**2)
    opti.minimize(ca_expr)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
    sol = opti.solve()
    vals = sol.value(ca_expr), sol.value(ca_vars["b"]), sol.value(ca_vars["d"]), sol.value(ca_vars["h"])

    for v1, v2 in zip(vals, vals2):
        assert np.allclose(v1, v2)
    print(vals, vals2)
    print("PASSED")



def test_linearize():
    a, b, c, d = sp.symbols('a b c d')

    expr1 = a**2 + 3 * b*c*d - 22*d**3 + 100
    expr2 = co.linearize_expression(expr1)
    co.equal_expression(expr1, expr2)

    expr1 = -a*d*b**2 + 44*d - 6
    expr2 = co.linearize_expression(expr1)
    co.equal_expression(expr1, expr2)

    expr1 = -50*a*d*b**2 + 44*c**3 - 6*a**4
    expr2 = co.linearize_expression(expr1)
    co.equal_expression(expr1, expr2)

    print("PASSED")


if __name__ == "__main__":
    test_sympy_expression_to_casadi()
    test_linearize()