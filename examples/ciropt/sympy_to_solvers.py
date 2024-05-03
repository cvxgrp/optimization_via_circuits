import sympy as sp
import numpy as np
import gurobipy as gp
import cvxpy as cp


from ciropt.utils import *



def gp_linearize_monomial(monomial, gp_vars, model):
    """
    Introduce quadratic constraints to make current monomial linear w.r.t. new variables
    """
    if monomial == []:
        return 1
    monomial = sorted(monomial)
    for i in range(len(monomial) - 1):
        new_variable = "_".join(monomial[:i+2])
        if new_variable not in gp_vars:
            # store a quadratic constraint for a new variable concatenation
            gp_vars[new_variable] = model.addVar(name=new_variable, lb=-1.*gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            model.update()
            model.addConstr( gp_vars[new_variable] == gp_vars["_".join(monomial[:i+1])] * gp_vars[monomial[i+1]] )
        if i == len(monomial)-2:
            assert "_".join(monomial) == new_variable
    return gp_vars["_".join(monomial)]


def find_lambda_nu(monomial):
    count_lamb, count_nu = 0, 0
    idx_lamb, idx_nu = [], []
    for idx, var in enumerate(monomial):
        if var[:4] == "lamb":
            count_lamb += 1
            idx_lamb = idx
        elif var[:2] == "nu":
            count_nu += 1
            idx_nu = idx
    return count_lamb, count_nu, idx_lamb, idx_nu


def gurobi_to_numpy(mat):
    M = np.zeros(mat.shape, dtype=object)
    for i in range(M.shape[0]):
        if len(M.shape) == 2:
            for j in range(M.shape[1]):
                M[i, j] = mat[i,j] if type(mat[i,j])==float else mat[i, j].getValue()
        else:
            M[i] = mat[i] if type(mat[i])==float else mat[i].getValue()
    return M


def sympy_expression_to_gurobi(sp_expression, gp_vars, model):
    if not isinstance(sp_expression, sp.Basic):
        return float(sp_expression)
    elif sp.simplify(sp_expression).free_symbols == set() or sp_expression.is_number:
        return float(sp.simplify(sp_expression))
    polynomial = sp.Poly(sp.simplify(sp_expression))
    coeffs = polynomial.coeffs()
    gp_expr = 0
    for i, monomial in enumerate(polynomial.monoms()):
        variables_in_monomial = []
        for var, exp in zip(polynomial.gens, monomial):
            variables_in_monomial += var.name.split("_") * exp
        if variables_in_monomial != list():
            count_lamb, count_nu, lamb_idx, nu_idx = find_lambda_nu(variables_in_monomial)
            assert count_nu + count_lamb <= 1 # one dual variable per expression/primal constraint
            if count_lamb > 0:
                name = variables_in_monomial[lamb_idx]
                variables_in_monomial.pop(lamb_idx)
                prefix, ab, _ = name.split("|")
                a, b = map(int, ab.split("."))
                lamb_ab = gp_vars[prefix][a, b].item()
                gp_monomial = lamb_ab * gp_linearize_monomial(variables_in_monomial, gp_vars, model)
            elif count_nu > 0:
                name = variables_in_monomial[nu_idx]
                variables_in_monomial.pop(nu_idx)
                prefix, ab, _ = name.split("|")
                a, b = map(int, ab.split("."))
                nu_ab = gp_vars[prefix][a, b].item()
                gp_monomial = nu_ab * gp_linearize_monomial(variables_in_monomial, gp_vars, model)
            else:
                gp_monomial = gp_linearize_monomial(variables_in_monomial, gp_vars, model)
        else:
            gp_monomial = 1
        gp_expr += float(coeffs[i]) * gp_monomial
    return gp_expr


def sympy_matrix_to_gurobi(sp_matrix, gp_vars, model):
    M = np.zeros(sp_matrix.shape, dtype=object)
    for i in range(M.shape[0]):
        if len(M.shape) == 2:
            for j in range(M.shape[1]):
                M[i, j] = sympy_expression_to_gurobi(sp_matrix[i, j], gp_vars, model)
        else:
            M[i] = sympy_expression_to_gurobi(sp_matrix[i], gp_vars, model)
    return M


def sympy_expression_to_casadi_lamb(sp_expression, ca_vars, model):
    """
    Convert sympy expression to expression using casadi variables
    """
    if not isinstance(sp_expression, sp.Basic):
        return float(sp_expression)
    elif sp.simplify(sp_expression).free_symbols == set() or sp_expression.is_number:
        return float(sp.simplify(sp_expression))
    polynomial = sp.Poly(sp.simplify(sp_expression))
    coeffs = polynomial.coeffs()
    gp_expr = 0
    for i, monomial in enumerate(polynomial.monoms()):
        variables_in_monomial = []
        for var, exp in zip(polynomial.gens, monomial):
            variables_in_monomial += var.name.split("_") * exp
        if variables_in_monomial != list():
            count_lamb, count_nu, lamb_idx, nu_idx = find_lambda_nu(variables_in_monomial)
            assert count_nu + count_lamb <= 1 # one dual variable per expression/primal constraint
            if count_lamb > 0:
                name = variables_in_monomial[lamb_idx]
                variables_in_monomial.pop(lamb_idx)
                prefix, ab, _ = name.split("|")
                a, b = map(int, ab.split("."))
                try:
                    lamb_ab = ca_vars[prefix][a, b]
                except:
                    print(name, prefix, a, b, sp_expression)
                    lamb_ab = ca_vars[prefix][a, b]
                gp_monomial = lamb_ab * ca_linearize_monomial(variables_in_monomial, ca_vars, model)
            elif count_nu > 0:
                name = variables_in_monomial[nu_idx]
                variables_in_monomial.pop(nu_idx)
                prefix, ab, _ = name.split("|")
                a, b = map(int, ab.split("."))
                nu_ab = ca_vars[prefix][a, b]
                gp_monomial = nu_ab * ca_linearize_monomial(variables_in_monomial, ca_vars, model)
            else:
                gp_monomial = ca_linearize_monomial(variables_in_monomial, ca_vars, model)
        else:
            gp_monomial = 1
        gp_expr += float(coeffs[i]) * gp_monomial
    return gp_expr


def sympy_matrix_to_casadi(sp_matrix, ca_vars, model):
    if not len(sp_matrix.shape) == 2:
        sp_matrix = sp_matrix.reshape(-1, 1)
    M = np.zeros(sp_matrix.shape)
    for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = sympy_expression_to_casadi_lamb(sp_matrix[i, j], ca_vars, model)
                M += val * prod_one_hot(M.shape, i, j)
    return M


def ca_linearize_monomial(monomial, ca_vars, opti):
    if monomial == []:
        return 1
    monomial = sorted(monomial)
    for i in range(len(monomial) - 1):
        new_variable = "_".join(monomial[:i+2])
        if new_variable not in ca_vars:
            # store a quadratic constraint for a new variable concatenation
            ca_vars[new_variable] = opti.variable()
            opti.subject_to( ca_vars[new_variable] == ca_vars["_".join(monomial[:i+1])] * ca_vars[monomial[i+1]] )
        if i == len(monomial)-2:
            assert "_".join(monomial) == new_variable
    return ca_vars["_".join(monomial)]


def sympy_expression_to_casadi(sp_expression, ca_vars, opti):
    if not isinstance(sp_expression, sp.Basic):
        return float(sp_expression)
    elif sp.simplify(sp_expression).free_symbols == set() or sp_expression.is_number:
        return float(sp.simplify(sp_expression))
    polynomial = sp.Poly(sp.simplify(sp_expression))
    coeffs = polynomial.coeffs()
    ca_expr = 0
    for i, monomial in enumerate(polynomial.monoms()):
        variables_in_monomial = []
        for var, exp in zip(polynomial.gens, monomial):
            variables_in_monomial += var.name.split("_") * exp
        if variables_in_monomial != list():
            ca_monomial = ca_linearize_monomial(variables_in_monomial, ca_vars, opti)
        else:
            ca_monomial = 1
        ca_expr += float(coeffs[i]) * ca_monomial
    return ca_expr


def cvx_linearize_monomial(monomial, var_x, name2idx):
    if monomial == []:
        return 1
    monomial = sorted(monomial)
    for i in range(len(monomial) - 1):
        new_variable = "_".join(monomial[:i+2])
        if i == len(monomial)-2:
            assert "_".join(monomial) == new_variable
    return var_x[name2idx["_".join(monomial)]]


def sympy_expression_to_cvx(sp_expression, var_x, name2idx):
    # expression in terms of parameters: b, d, h, alpha, beta,
    if not isinstance(sp_expression, sp.Basic):
        return float(sp_expression)
    elif sp.simplify(sp_expression).free_symbols == set() or sp_expression.is_number:
        return float(sp.simplify(sp_expression))
    polynomial = sp.Poly(sp.simplify(sp_expression))
    coeffs = polynomial.coeffs()
    cvx_expr = 0
    for i, monomial in enumerate(polynomial.monoms()):
        variables_in_monomial = []
        for var, exp in zip(polynomial.gens, monomial):
            variables_in_monomial += var.name.split("_") * exp
        if variables_in_monomial != list():
            cvx_monomial = cvx_linearize_monomial(variables_in_monomial, var_x, name2idx)
        else:
            cvx_monomial = 1
        cvx_expr += float(coeffs[i]) * cvx_monomial
    return cvx_expr


def init_ca_ipopt(init_values, opti, vec_indices, var_x):
    if init_values is not None:
        for name in init_values.keys():
            if name in vec_indices:
                opti.set_initial(get_vec_var(var_x, name, vec_indices), init_values[name])
            elif name == "x":
                opti.set_initial(var_x[:init_values[name].size], init_values[name])
                

def ca_dict_total_variable_size(ca_vars):
    size = 0
    for k, var in ca_vars.items():
        size += np.prod(var.size())
    return size