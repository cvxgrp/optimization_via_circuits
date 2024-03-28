import sympy as sp
import numpy as np
import gurobipy as gp
import cvxpy as cp



def ca_add_bounds(opti, bounds, ca_vars, name2idx):
    if bounds is not None:
        for name in bounds.keys():
            if name in ca_vars: this_var = ca_vars[name]
            elif name in name2idx: this_var = ca_vars["x"][name2idx[name]]
            if "ub" in bounds[name]:
                opti.subject_to( this_var <= bounds[name]["ub"] )
            if "lb" in bounds[name]:
                opti.subject_to( this_var >= bounds[name]["lb"] )


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


def flatten_lower_tri(P):
    res = np.zeros(P.shape[0] * (P.shape[0]+1)//2)
    for i in range(P.shape[0]):
        res[i*(i+1)//2 : i*(i+1)//2 + i + 1] = P[i, :i+1]
    return res


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


def gp_trace(M):
    tr = 0
    for i in range(M.shape[0]):
        tr += M[i, i].item()
    return tr


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
        keys_list = ['b', 'h', 'd', 'alpha', 'beta', 'gamma', 'delta']
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
    assert np.allclose(A, A_plus - A_minus)
    return A_plus, A_minus
