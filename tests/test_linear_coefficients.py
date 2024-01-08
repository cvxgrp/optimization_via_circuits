import casadi as ca
import numpy as np
import ciropt as co
import sympy as sp



def get_start_idx(k):
    if k == 0:
        return 0
    return k * (k-1) // 2 + 1

def main():
    L_smooth = 1.
    mu = 0.0001
    Capacitance = 2.
    Inductance = 2.
    R = 1.

    # solver = "ca_canonical"
    solver = "ca_canonical_X"
    problem = co.accelerated_gradient_circuit(mu, L_smooth, R, Capacitance, Inductance)
    problem.obj = problem.b + problem.d
    res, sol, sp_exp = problem.solve(solver=solver, verbose=False, debug=True)[:3]

    sp_v = np.array([1] + [sp.symbols(name) for name in problem.v_names[1:]])
    name2idx = problem.name2idx

    for k in sp_exp.keys():
        for mat_expr in sp_exp[k].values():
            coeff_matrix = co.linear_matrix_expr_to_coeff_matrix(mat_expr, name2idx)
            assert co.equal_sp_arrays(co.coeff_matrix_to_linear_matrix_expr(coeff_matrix, \
                                                                    sp_v, mat_expr.shape), mat_expr)
        
    print("PASSED")

    core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
    # check coefficient matrices, ie, C @ v == \nabla l(F, G)
    v_coeffs, v_names, name2idx2, v_k_list = co.sp_v_coeff_matrix(sp_exp, core_vars)
    sp_v2 = np.array([1] + [sp.symbols(name) for name in v_names[1:]])

    for i, k in enumerate(v_k_list):
        for mtype, mat_expr in sp_exp[k].items():
            coeff_matrix = v_coeffs[mtype][i]
            assert co.equal_sp_arrays(co.coeff_matrix_to_linear_matrix_expr(coeff_matrix, sp_v2, mat_expr.shape), mat_expr)
    print("PASSED")

    
    a = np.zeros((10, 10))
    count = 0
    for i in range(10):
        a[i, :i+1] = np.arange(count, count + i + 1)
        count += i + 1
    print(a)
    vec_diag_P_idx = np.cumsum(np.arange(10 + 1))[1:] - 1
    Q = np.zeros((10, 10 * (10 + 1) // 2))
    np.put_along_axis(Q, vec_diag_P_idx.reshape(-1, 1), 1, axis=1)
    vec_a = np.arange(10 * (10 + 1) // 2)
    diag_a = Q @ vec_a
    assert np.allclose(diag_a, np.diag(a))
    for k1 in range(10):
        for k2 in range(10):
            assert np.allclose( (a @ a.T)[k1, k2], co.get_PPt_element(vec_a, k1, k2))
    print("PASSED")

    for _ in range(10):
        dim_G = 6
        x = np.random.randn(100, 1)
        vec_indices = { "v"   : [0, 9],\
                                "lamb": [10, 70], \
                                "P"  : [71, 71 + dim_G * (dim_G + 1)//2]} 
        var_name = "P"
        l = co.get_vec_var(x, var_name, vec_indices, matrix=False)
        S = co.get_vec_var(x, var_name, vec_indices, matrix=True)
        vec_P = co.get_vec_var(x, "P", vec_indices)
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                val = co.get_PPt_element(vec_P, k1, k2)
                S1, S2 = co.get_PPt_matrix(x, vec_indices, k1, k2)
                assert np.allclose(val, x.T @ S1.T @ S2 @ x)
        assert np.allclose(S @ x, l)
    print("PASSED")


if __name__ == "__main__":
    main()