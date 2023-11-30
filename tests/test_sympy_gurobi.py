import ciropt as co
import gurobipy as gp
import sympy as sp
import numpy as np




def test_sympy_expression_to_gurobi():
    model = gp.Model() 
    model.Params.LogToConsole = 0
    gp_vars = { 'b': model.addVar(name='b', lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY),
                    'd': model.addVar(name='d', lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY),
                    'h': model.addVar(name='h', lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY),}
    model.update()
    gp_expr1 = (gp_vars["b"] - gp_vars["d"] - 5)**2 + (gp_vars["h"] - gp_vars["d"]- 10)**2 + 0.01 * (gp_vars["h"]**2 + gp_vars["d"]**2 + gp_vars["b"]**2)
    model.setObjective( gp_expr1, gp.GRB.MINIMIZE)
    model.optimize()
    vals1 = gp_expr1.getValue(), gp_vars["b"].X, gp_vars["d"].X, gp_vars["h"].X
    print(f"{vals1=}")

    model = gp.Model()
    # model.Params.LogToConsole = 0 
    gp_vars = { 'b': model.addVar(name='b', lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY),
                'd': model.addVar(name='d', lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY),
                'h': model.addVar(name='h', lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY),}
    model.update()
    b, d, h = sp.symbols('b d h')
    # strongly convex function -- unique minimizer
    expr = (b - d - 5)**2 + (h - d - 10)**2 + 0.01 * (b**2 + d**2 + h**2)
    gp_expr2 = co.sympy_expression_to_gurobi(expr, gp_vars, model)
    model.params.NonConvex = 2
    model.params.FeasibilityTol = 1e-6
    model.params.TimeLimit = 50
    model.setObjective( gp_expr2, gp.GRB.MINIMIZE)
    # model.addConstr( gp_expr2 <= 10 )
    # model.addConstr( gp_expr2 >= -10 )
    model.optimize()
    vals2 = gp_expr2.getValue(), gp_vars["b"].X, gp_vars["d"].X, gp_vars["h"].X
    print(f"{vals2=}")
    for v1, v2 in zip(vals1, vals2):
        assert np.allclose(v1, v2)

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
    test_sympy_expression_to_gurobi()
    test_linearize()