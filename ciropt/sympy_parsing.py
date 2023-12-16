import sympy as sp
import numpy as np
import gurobipy as gp
import cvxpy as cp




def const_coeff_expr(expr, name2idx, x_size):
    row = np.zeros(len(name2idx))
    if not isinstance(expr, sp.Basic):
        row[0] = float(expr)
    elif sp.simplify(expr).free_symbols == set() or expr.is_number:
        row[0] = float(sp.simplify(expr))
    else:
        polynomial = sp.Poly(sp.simplify(expr))
        expr_coeffs = polynomial.coeffs()
        for i, monomial in enumerate(polynomial.monoms()):
            variables_in_monomial = []
            assert sum(monomial) <= 1, print("not linear expression in variables")
            for var, exp in zip(polynomial.gens, monomial):
                variables_in_monomial += [var.name] * exp
            variables_in_monomial = sorted(variables_in_monomial)
            if variables_in_monomial != list():
                var_name = "_".join(variables_in_monomial)
            else:
                var_name = 1
            if var_name in name2idx:
                var_idx = name2idx[var_name]
            row[var_idx] = expr_coeffs[i]
    return np.concatenate([row, np.zeros(x_size - len(name2idx))], axis=0).reshape(-1, 1)


def sp_v_coeff_matrix(sp_exp, core_vars):
    # get constant coefficient matrices from linear expressions at the 
    # coordinates of F, G stored in sp_exp 
    # all matrices in sp_exp are linear in the linearized variables of core_vars
    v_coeffs = {"F":[], "G":[]}
    v_names = [1] + core_vars
    name2idx = {var : idx for idx, var in enumerate(v_names)}
    v_k_list = list(sp_exp.keys())
    for k in v_k_list:
        for mtype, mat_expr in sp_exp[k].items():
            # elements of mat_expr must be linear in v_names
            # coeff_matrix is of size |flatten(F or G)| x |v|
            coeff_matrix = np.zeros((mat_expr.size, len(v_names)))
            for expr_idx, expr in enumerate(mat_expr.flatten(order='C')):
                if not isinstance(expr, sp.Basic):
                    coeff_matrix[expr_idx, 0] = float(expr)
                    continue
                elif sp.simplify(expr).free_symbols == set() or expr.is_number:
                    coeff_matrix[expr_idx, 0] = float(sp.simplify(expr))
                    continue 
                polynomial = sp.Poly(sp.simplify(expr))
                # print(polynomial)
                expr_coeffs = polynomial.coeffs()
                for i, monomial in enumerate(polynomial.monoms()):
                    variables_in_monomial = []
                    # assert sum(monomial) <= 1, print("not linear expression in variables")
                    for var, exp in zip(polynomial.gens, monomial):
                        assert not "_" in var.name
                        variables_in_monomial += [var.name] * exp
                    variables_in_monomial = sorted(variables_in_monomial)
                    if variables_in_monomial != list():
                        var_name = "_".join(variables_in_monomial)
                    else:
                        var_name = 1
                    if var_name in name2idx:
                        var_idx = name2idx[var_name]
                    else:
                        variables_in_monomial = sorted(("_".join(variables_in_monomial)).split("_"))
                        # add new variable and all the prefix variables
                        assert len(variables_in_monomial) >= 2, print(variables_in_monomial)
                        prefix = variables_in_monomial[0]
                        for el in variables_in_monomial[1:]:
                            prefix += "_" + el
                            if prefix not in name2idx:
                                v_names += [prefix]
                                name2idx[prefix] = len(name2idx)
                        assert prefix == var_name
                        var_idx = name2idx[var_name]
                        coeff_matrix = np.hstack(( coeff_matrix, np.zeros((coeff_matrix.shape[0], len(name2idx) - coeff_matrix.shape[1])) ))
                    coeff_matrix[expr_idx, var_idx] = expr_coeffs[i]
                    assert coeff_matrix.shape[1] == len(name2idx), print(coeff_matrix.shape, len(name2idx))
            v_coeffs[mtype] += [coeff_matrix]
    return v_coeffs, v_names, name2idx, v_k_list


def evaluate_monomial(v_name, params):
    res = 1
    if type(v_name) == int: return v_name
    for var in v_name.split("_"):
        res *= params[var]
    return res


def linear_matrix_expr_to_coeff_matrix(mat_expr, name2idx):
    # elements of mat_expr must be linear in sp_v = name2idx.keys()
    coeff_matrix = np.zeros((mat_expr.size, len(name2idx)))

    for expr_idx, expr in enumerate(mat_expr.flatten(order='C')):
        if not isinstance(expr, sp.Basic):
            coeff_matrix[expr_idx, 0] = float(expr)
            continue
        elif sp.simplify(expr).free_symbols == set() or expr.is_number:
            coeff_matrix[expr_idx, 0] = float(sp.simplify(expr))
            continue 
        polynomial = sp.Poly(sp.simplify(expr))
        expr_coeffs = polynomial.coeffs()
        for i, monomial in enumerate(polynomial.monoms()):
            variables_in_monomial = []
            assert sum(monomial) <= 1, print("not linear expression in variables")
            for var, exp in zip(polynomial.gens, monomial):
                variables_in_monomial += [var.name] * exp
            if variables_in_monomial != list():
                var_name = "_".join(sorted(variables_in_monomial))
            else:
                var_name = 1
            var_idx = name2idx[var_name]
            coeff_matrix[expr_idx, var_idx] = expr_coeffs[i]
    return coeff_matrix


def coeff_matrix_to_linear_matrix_expr(coeff_matrix, sp_v, shape):
    return ( coeff_matrix @ sp_v[ : coeff_matrix.shape[1]]).reshape(shape, order='C')


def multiply_dicts(dict1, dict2):
    product_dict = dict()
    for key1 in dict1.keys():
        for key2 in dict2.keys():
            product_key = (key1, key2)
            # product_value = linearize_expression(dict1[key1] * dict2[key2])
            product_value = sp.simplify(dict1[key1] * dict2[key2])
            if product_key in product_dict.keys():
                product_dict[product_key] += product_value
            else:
                product_dict[product_key] = product_value
    return product_dict


def merge_dict(dict1, dict2):
    # the union of the 2 inputs with added values
    merged_dict = dict1.copy()
    for key in dict2.keys():
        if key in dict1.keys():
            # merged_dict[key] = linearize_expression(merged_dict[key] + dict2[key])
            merged_dict[key] = sp.simplify(merged_dict[key] + dict2[key])
        else:
            merged_dict[key] = dict2[key]
    return merged_dict


def prune_dict(my_dict):
    pruned_dict = dict()
    for key in my_dict.keys():
        if my_dict[key] != 0:
            pruned_dict[key] = my_dict[key]
    return pruned_dict


def check_degree_expression(expr, degree):
    if not isinstance(expr, sp.Basic):
        return expr
    polynomial = sp.Poly(sp.simplify(expr))
    for monomial in polynomial.monoms():
        if np.array(monomial).sum() >= degree + 1:
            return False
    return True


def linearize_expression(expr):
    if not isinstance(expr, sp.Basic) or sp.simplify(expr).free_symbols == set() or expr.is_number:
        return expr
    polynomial = sp.Poly(sp.simplify(expr))
    lin_expr = 0
    coeffs = polynomial.coeffs()
    for i, monomial in enumerate(polynomial.monoms()):
        variables_in_monomial = []
        # make monomial linear with long variable
        for var, exp in zip(polynomial.gens, monomial):
            variables_in_monomial += [var.name] * exp
        if variables_in_monomial != list():
            name = "_".join(sorted(variables_in_monomial))
            new_var = sp.symbols("_".join(sorted(name.split("_"))))
        else:
            new_var = 1
        lin_expr += coeffs[i] * new_var
    return sp.simplify(lin_expr)


def collapse_expression(expr):
    # collapse long symbols into a monomial with simple variables and exponents
    if not isinstance(expr, sp.Basic):
        return float(expr)
    elif sp.simplify(expr).free_symbols == set() or expr.is_number:
        return float(sp.simplify(expr))
    polynomial = sp.Poly(sp.simplify(expr))
    collapse_expr = 0
    coeffs = polynomial.coeffs()
    for i, monomial in enumerate(polynomial.monoms()):
        variables_in_monomial = []
        for var, exp in zip(polynomial.gens, monomial):
            variables_in_monomial += var.name.split("_") * exp
        if variables_in_monomial != list():
            new_monomial = sp.Mul(*[sp.symbols(name) for name in variables_in_monomial])
        else:
            new_monomial = 1
        collapse_expr += coeffs[i] * new_monomial
    return collapse_expr


def is_zero_expr(expr):
    # expression is 0 if all the coefficients are 0, numerical 
    if isinstance(expr, (int, float, complex)) or expr.is_number:
        return np.allclose(float(expr), 0)
    polynomial = sp.Poly(sp.simplify(expr))
    coeffs = polynomial.coeffs()
    for i, monomial in enumerate(polynomial.monoms()): 
        if not np.allclose(float(coeffs[i]), 0):
            return False
    else:
        return True

def equal_expression(expr1, expr2):
    ex1, ex2 = collapse_expression(expr1), collapse_expression(expr2)
    if is_zero_expr(ex1 - ex2):
        return True
    if isinstance(ex1, sp.Basic) and not isinstance(ex2, sp.Basic):
        if not ex1.is_number:
            return False
        else:
            return np.allclose(float(ex1), float(ex2))
    if isinstance(ex2, sp.Basic) and not isinstance(ex1, sp.Basic):
        if not ex2.is_number:
            return False
        else:
            return np.allclose(float(ex1), float(ex2))
    res = sp.simplify(ex1 - ex2)
    return res == 0 or res.is_number and np.allclose(float(res), 0) 

    
def check_linear_expression(expr):
    polynomial = sp.Poly(sp.simplify(expr))
    for monomial in polynomial.monoms():
        if np.array(monomial).sum() >= 2:
            return False
    return True


def simplify_matrix(expr):
    res = np.zeros(expr.shape, dtype=object)
    for i in range(res.shape[0]):
        if len(expr.shape) == 2:
            for j in range(res.shape[1]):
                res[i, j] = linearize_expression(expr[i, j])
        else:
            res[i] = linearize_expression(expr[i])
    return res


def equal_sp_arrays(a, b):
    if a.shape != b.shape:
        if a.squeeze().shape == b.squeeze().shape:
            a = a.squeeze()
            b = b.squeeze()
        else:
            return False
    for i in range(a.shape[0]):
        if len(a.shape) == 2:
            for j in range(a.shape[1]):
                if not equal_expression(a[i, j], b[i, j]):
                    return False
        else:
            if not equal_expression(a[i], b[i]):
                return False
    return True
    

def vars_to_vector(v_names, params):
    vec = np.zeros((len(v_names), 1))
    for i, var in enumerate(v_names):
        vec[i] = evaluate_monomial(var, params)
    return vec
