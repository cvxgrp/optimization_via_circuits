import sympy as sp
import numpy as np
import gurobipy as gp




def ca_add_bounds(opti, bounds, ca_vars):
    if bounds is not None:
            for name in bounds.keys():
                if "ub" in bounds[name]:
                    opti.subject_to( ca_vars[name] <= bounds[name]["ub"] )
                if "lb" in bounds[name]:
                    opti.subject_to( ca_vars[name] >= bounds[name]["lb"] )


def cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lambs):
    if bounds is not None:
        for name in bounds.keys():
            if name in cvx_vars: this_var = cvx_vars[name]
            elif name in name2idx: this_var = var_x[name2idx[name]]
            elif name == "lamb": this_var = I_lambs @ var_x
            else: continue
            print("set bounds for %s"%name)
            if "ub" in bounds[name]:
                constraints += [ this_var <= bounds[name]["ub"] ]
            if "lb" in bounds[name]:
                constraints += [ this_var >= bounds[name]["lb"] ]
    return constraints


def get_PPt_element(vec_P, k1, k2):
    def get_start_idx(k):
        return k * (k+1) // 2 
    s1 = get_start_idx(k1)
    s2 = get_start_idx(k2)
    row1 = vec_P[s1 : s1 + min(k1, k2) + 1]
    row2 = vec_P[s2 : s2 + min(k1, k2) + 1]
    return row1.T @ row2


def get_PPt_matrix(x, vec_indices, k1, k2):
    # selection matrices to get (PP^T)_{k1, k2} from x
    # PP^T_{k1, k2} = P_{k1,:}(P_{k2,:})^T = (S1 @ x).T @ (S2 @ x) = S1.T @ S2 @ xx^T
    def get_start_idx(k):
        return k * (k+1) // 2 
    def selection_matrix(num_rows, c1, c3):
        return np.concatenate([np.zeros((num_rows, c1)), \
                                           np.eye(num_rows), \
                                           np.zeros((num_rows, c3))], axis=1)
    s1 = vec_indices["P"][0] + get_start_idx(k1) 
    s2 = vec_indices["P"][0] + get_start_idx(k2) 
    num_rows = min(k1, k2) + 1 
    S1 = selection_matrix(num_rows, s1, x.shape[0] - num_rows - s1 )
    S2 = selection_matrix(num_rows, s2, x.shape[0] - num_rows - s2 )
    return S1, S2


def evaluate_monomial(v_name, params):
    res = 1
    if type(v_name) == int: return v_name
    for var in v_name.split("_"):
        res *= params[var]
    return res


def vars_to_vector(v_names, params):
    vec = np.zeros((len(v_names), 1))
    for i, var in enumerate(v_names):
        vec[i] = evaluate_monomial(var, params)
    return vec


def get_vec_var(x, var_name, vec_indices, matrix=False):
    if matrix:
        num_rows = vec_indices[var_name][1] + 1 - vec_indices[var_name][0] 
        selection_matrix = np.concatenate([np.zeros((num_rows, vec_indices[var_name][0])), \
                                           np.eye(num_rows), \
                                           np.zeros((num_rows, x.shape[0] - vec_indices[var_name][1] - 1))], axis=1)
        return selection_matrix
    else:
        return x[vec_indices[var_name][0] : vec_indices[var_name][1] + 1]


def reshape_lamb_2d(lamb):
    size = int(np.sqrt(lamb.size)) + 1
    res = np.zeros((size, size))
    count = 0
    for i in range(size):
        for j in range(size):
            if i == j: continue
            res[i, j] = lamb[count]
            count += 1
    return res


def flatten_lamb(lamb):
    # remove diagonal entries
    res = []
    for i in range(lamb.shape[0]):
        for j in range(lamb.shape[0]):
            if i == j: continue
            res += [lamb[i, j]]
    return np.array(res).reshape(-1, 1)


def cholseky_matrix(Z, eps=1e-9):
    Lamb, V = np.linalg.eigh(Z)
    # print(Lamb)
    assert np.allclose(V @ np.diag(Lamb) @ V.T, Z)
    Z_plus = V @ np.diag(np.maximum(Lamb, eps)) @ V.T
    P = np.linalg.cholesky(Z_plus)
    return P


def stack_vectors(vectors, max_len):
    stacked_matrix = np.zeros((len(vectors), vectors[0].shape[0], max_len))
    for i, vector in enumerate(vectors):
        stacked_matrix[i, :, : vector.shape[1]] = vector
    return stacked_matrix


def sp_v_coeff_matrix(sp_exp, core_vars):
    # get constant coefficient matrices from linear expressions at the coordinates of F, G stored in sp_exp 
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


def gp_linearize_monomial(monomial, gp_vars, model):
    # introduce quadratic constraints to make current monomial linear w.r.t. new variables
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


def find_lambda(monomial):
    count = 0
    indices = []
    for idx, var in enumerate(monomial):
        if var[:4] == "lamb":
            count += 1
            indices = idx
    return count, indices


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
            count, lamb_idx = find_lambda(variables_in_monomial)
            if count > 0:
                assert count == 1
                name = variables_in_monomial[lamb_idx]
                variables_in_monomial.pop(lamb_idx)
                # if name not in gp_vars:
                #     gp_vars[name] = model.addVar(name=name, lb=0, ub=gp.GRB.INFINITY)
                #     model.update()
                # lamb_ij = gp_vars[name]
                prefix, ab, _ = name.split("|")
                a, b = map(int, ab.split("."))
                lamb_ab = gp_vars[prefix][a, b].item()
                gp_monomial = lamb_ab * gp_linearize_monomial(variables_in_monomial, gp_vars, model)
            else:
                gp_monomial = gp_linearize_monomial(variables_in_monomial, gp_vars, model)
        else:
            gp_monomial = 1
        gp_expr += float(coeffs[i]) * gp_monomial
    return gp_expr


def gp_trace(M):
    tr = 0
    for i in range(M.shape[0]):
        tr += M[i, i].item()
    return tr


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
            count, lamb_idx = find_lambda(variables_in_monomial)
            if count > 0:
                assert count == 1
                name = variables_in_monomial[lamb_idx]
                variables_in_monomial.pop(lamb_idx)
                prefix, ab, _ = name.split("|")
                a, b = map(int, ab.split("."))
                lamb_ab = ca_vars[prefix][a, b]
                gp_monomial = lamb_ab * ca_linearize_monomial(variables_in_monomial, ca_vars, model)
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


def multiply_dicts(dict1, dict2):
    product_dict = dict()
    for key1 in dict1.keys():
        for key2 in dict2.keys():
            product_key = (key1, key2)
            product_value = linearize_expression(dict1[key1] * dict2[key2])
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
            merged_dict[key] = linearize_expression(merged_dict[key] + dict2[key])
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


def one_hot(n, i, flatten=False):
    if flatten:
        return np.eye(n)[:, i]
    else:
        return np.eye(n)[:, i:i+1]


def symm_prod_one_hot(n, i, j):
    a = one_hot(n, i)
    b = one_hot(n, j)
    return symm_prod(a, b)


def prod_one_hot(shape, i, j):
    a = np.zeros(shape)
    a[i, j] = 1
    return a


def symm_prod(a, b=None):
    # assert (a @ b.T).shape == (b @ a.T).shape
    if b is None:
        b = a
    return 0.5 * (a @ b.T + b @ a.T)


def dict_parameters_ciropt(sol, ca_vars, all=False):
    res = {} 
    if all:
        keys_list = ca_vars.keys()
    else:
        keys_list = ['b', 'h', 'd', 'alpha', 'beta']
    for key in keys_list:
        try: res[key] = sol.value(ca_vars[key])
        except: pass
    return res


def dict_parameters_ciropt_gp(model, gp_vars, all=False, Xn=False):
    res = {} 
    # all_vars = model.getVars()
    # values = model.getAttr("X", all_vars)
    if all:
        keys_list = gp_vars.keys()
    else:
        keys_list = ['b', 'h', 'd', 'alpha', 'beta']
    for key in keys_list:
        if Xn:
            res[key] = gp_vars[key].Xn
        else:
            try: res[key] = gp_vars[key].X
            except: pass
    return res


def gp_print_solutions(model, gp_vars, all=False):
    for i in range(model.SolCount):
        model.Params.SolutionNumber = i
        print(f"{i+1}: obj = {model.PoolObjVal}")
        # print(model.printQuality())
        print(dict_parameters_ciropt_gp(model, gp_vars, all=all, Xn=True))
    model.Params.SolutionNumber = 0


def matrix_to_diff_psd(A):
    # split matrix A = A_plus - A_minus
    # difference of two PSD matrices
    symm = np.allclose(A, A.T)
    if symm:
        evals, V = np.linalg.eigh(A)
        inv_V = V.T
    else:
        evals, V = np.linalg.eig(A)
        inv_V = np.linalg.inv(V)
    # assert np.allclose(A, (evals * V) @ inv_V)
    idx = np.argsort(evals)
    evals = evals[np.argsort(evals)]
    V = V[:, idx]
    inv_V = inv_V[idx, :]
    # assert np.allclose(A, (evals * V) @ inv_V)
    idx_positive_evals = np.where(evals>0)[0]
    if idx_positive_evals.size > 0:
        pos_idx = idx_positive_evals[0]
        if pos_idx >= 1:
            A_minus = -(evals[:pos_idx] * V[:, :pos_idx]) @ inv_V[:pos_idx, :]
            A_plus = (evals[pos_idx:] * V[:, pos_idx:]) @ inv_V[pos_idx:, :]
        else:
            A_plus = A # A is PSD 
            A_minus = 0
    else:
        A_plus = 0 
        A_minus = - A # A is NSD
    return A_plus, A_minus
