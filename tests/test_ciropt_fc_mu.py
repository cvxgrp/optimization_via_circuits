import casadi as ca
import numpy as np
import ciropt as co
import sympy as sp


def eval_ca_function(vars, eval_vars, func):
    y_func = ca.Function('y_func', vars, [func])
    return y_func(*eval_vars)


def main():
    mu = 0.000001
    Capacitance = 1.

    solver = "ipopt"

    # Ciropt formulation
    problem = co.gradient_flow_circuit(mu, np.inf, Capacitance)
    problem.obj = problem.b
    res, sol, sp_exp = problem.solve(solver=solver, debug=True, verbose=False)[:3]
    ca_vars = problem.vars
    print(res)

    co_vars = co.dict_parameters_ciropt(sol, ca_vars, all=True)

    # Manual problem formulation
    size_I = 4          # I = { star, 1, 1.5, 2 }
    dim_G = 5           # { x_star, x_1, y_1, y_1p5, y_2 }
    dim_F = 4           # { f_star, f_1, f_1p5, f_2 }
    opti = ca.Opti()

    b = opti.variable()
    h = opti.variable()
    alpha = opti.variable()
    beta = opti.variable()
    alpha_h = opti.variable()
    beta_h = opti.variable()
    P_full = opti.variable(dim_G, dim_G)
    P = ca.tril(P_full)
    lamb0 = opti.variable(size_I, size_I)

    y_star = np.zeros(dim_G).reshape(-1, 1)
    y_1 = co.one_hot(dim_G, 2)
    y_1p5 = co.one_hot(dim_G, 3) 
    y_2 = co.one_hot(dim_G, 4) 
    Gs = [y_star, y_1, y_1p5, y_2]

    f_star = co.one_hot(dim_F, 0) # np.zeros((dim_F, 1))
    f_1   = co.one_hot(dim_F, 1) 
    f_1p5 = co.one_hot(dim_F, 2)
    f_2   = co.one_hot(dim_F, 3)
    F = [f_star, f_1, f_1p5, f_2]

    x_star = co.one_hot(dim_G, 0)
    x_1    = co.one_hot(dim_G, 1)
    x_1p5 = x_1 - (alpha_h / Capacitance) * y_1
    x_2   = x_1 - (beta_h / Capacitance) * y_1 - ((h - beta_h) / Capacitance) * y_1p5 
    Xs = [x_star, x_1, x_1p5, x_2]

    sp_vars = {name : sp.symbols(name) for name in ca_vars.keys()}
    sp_x_1p5 = x_1 - (sp_vars["alpha_h"] / Capacitance) * y_1
    sp_x_2   = x_1 - (sp_vars["beta_h"] / Capacitance) * y_1 - ((sp_vars["h"] - sp_vars["beta_h"]) / Capacitance) * y_1p5 
    sp_Xs = [x_star, x_1, sp_x_1p5, sp_x_2]

    star_idx = 4
    A = lambda i,j: co.symm_prod(Gs[j], Xs[i] - Xs[j])
    B = lambda i,j: co.symm_prod(Xs[i] - Xs[j], Xs[i] - Xs[j])
    C = lambda i,j: co.symm_prod(Gs[i] - Gs[j], Gs[i] - Gs[j])
    a = lambda i,j: F[j] - F[i]

    sp_A = lambda i,j: co.symm_prod(Gs[j], sp_Xs[i] - sp_Xs[j])
    sp_B = lambda i,j: co.symm_prod(sp_Xs[i] - sp_Xs[j], sp_Xs[i] - sp_Xs[j])
    sp_C = lambda i,j: co.symm_prod(Gs[i] - Gs[j], Gs[i] - Gs[j])

    assert (dim_G, dim_G) == A(1, 2).shape == B(1, 2).shape == C(1, 2).shape
    assert a(1, 2).shape == (dim_F, 1)

    opti.minimize( -b )
    opti.subject_to( alpha_h == alpha * h )
    opti.subject_to( beta_h == beta * h )
    opti.subject_to( ca.reshape(lamb0, (-1, 1)) >= np.zeros((size_I * size_I, 1)) )
    opti.subject_to( ca.diag(P) >= np.zeros((dim_G, 1)) )
    opti.subject_to( ca.diag(lamb0) == np.zeros((size_I, 1)) )
    # star:0, 1:1, 1p5:2, 2:3
    sum_ij_La = 0
    sum_ij_AC = 0
    for i in range(size_I):
        for j in range(size_I):
            if i == j: continue
            sum_ij_La += lamb0[i, j] * a(i, j)
            sum_ij_AC += lamb0[i, j] * (A(i, j) + (mu/2) * B(i, j))

    Fweights_d = - b * a(1, 0)
    Gweights_d = (Capacitance / 2) * (B(3, 0) - B(1, 0))
    opti.subject_to( sum_ij_La - Fweights_d  == np.zeros((dim_F, 1))) 
    opti.subject_to( sum_ij_AC  - P @ P.T - Gweights_d == np.zeros((dim_G, dim_G)))

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)
    sol = opti.solve()
    assert sol.stats()['success'], print(sol.stats())
    print(sol.value(b), sol.value(h), sol.value(alpha), sol.value(beta))

    # compare symbolic expressions 
    size_I_function = size_I
    for i in range(size_I_function):
        for j in range(size_I_function):
            if i == j: continue
            F1, G1 = sp_exp[(0, 0, i, j)]["F"], sp_exp[(0, 0, i, j)]["G"]
            F2 = a(i, j)
            G2 = sp_A(i, j) + (mu/2) * sp_B(i, j)
            assert co.equal_sp_arrays(G1, G2), print(f"{i=}, {j=} \n{G1=} \n{G2=}")
            assert co.equal_sp_arrays(F1, F2), print(f"{i=}, {j=} \n{F1=} \n{F2=}\n")

    F1, G1 = sp_exp["FG_d"]["F"], sp_exp["FG_d"]["G"]
    F2 = sp_vars["b"] * a(1, 0)
    G2 = (Capacitance / 2) * (sp_B(3, 0) - sp_B(1, 0))
    assert co.equal_sp_arrays(G1, G2), print(f"{i=}, {j=} \n{G1=} \n{G2=}")
    assert co.equal_sp_arrays(-F1, F2), print(f"{i=}, {j=} \n{F1=} \n{F2=}\n")
    print("PASSED equal symbolic expression")

    # compare expression evaluations 
    h_init = co_vars["h"]
    b_init = co_vars["b"]
    alpha_init = co_vars["alpha"]
    beta_init = co_vars["beta"]
    lamb_init = co_vars["lamb0"]
    co_vars["P_full"] = co_vars["P"]
    P_full_init = co_vars["P_full"]
    alpha_h_init = alpha_init * h_init
    beta_h_init = beta_init * h_init

    vars = [b, h, alpha, beta, alpha_h, beta_h, P_full, lamb0]
    eval_vars = [b_init, h_init, alpha_init, beta_init, alpha_h_init, beta_h_init, P_full_init, lamb_init]

    for name in ['h', 'b', 'alpha', 'beta', 'P_full', 'lamb0']:
        ca_y_eval = eval_ca_function(vars, eval_vars, locals()[name])
        assert np.allclose(ca_y_eval, co_vars[name]), print(name)

    for name in ['sum_ij_AC', 'sum_ij_La', 'Fweights_d', 'Gweights_d']:
        ca_y_eval = eval_ca_function(vars, eval_vars, locals()[name])
        assert np.allclose(ca_y_eval, problem.ca_expressions[name]) or \
            np.allclose(np.array(ca_y_eval.full()).flatten(), problem.ca_expressions[name]), print(name)

    val_ca_P = eval_ca_function(vars, eval_vars, locals()['P_full'])
    sum1 = np.array((eval_ca_function(vars, eval_vars, locals()['sum_ij_AC']) \
        - eval_ca_function(vars, eval_vars, locals()['Gweights_d']) \
        - val_ca_P @ val_ca_P.T).full())
    sum2 = np.array((eval_ca_function(vars, eval_vars, locals()['sum_ij_La']) \
        - eval_ca_function(vars, eval_vars, locals()['Fweights_d'])).full())

    assert np.allclose(0, np.linalg.norm(sum1.flatten())) and np.allclose(0, np.linalg.norm(sum2.flatten()))

    print("PASSED equal expression evaluations")



if __name__ == "__main__":
    main()