import casadi as ca
import numpy as np
import ciropt as co
import sympy as sp
import ciropt.function as co_func


def eval_ca_function(vars, eval_vars, func):
        g_func = ca.Function('g_func', vars, [func])
        return g_func(*eval_vars)


def accelerated_gradient_circuit(mu, L_smooth, R, Capacitance, Inductance, params=None): 
    problem = co.CircuitOpt()
    package = co_func 
    h, alpha, beta, eta, rho = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho

    func = co.define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    v_C_1 = problem.set_initial_point()
    i_L_1 = problem.set_initial_point()
    x_1 = R * i_L_1 + v_C_1
    y_1, f_1 = func.oracle(x_1)

    i_L_1p5 = i_L_1 + (alpha * h / Inductance) * (v_C_1 - (x_1 - R * y_1)) 
    v_C_1p5 = v_C_1 - (alpha * h / Capacitance) * y_1 
    x_1p5 = R * i_L_1p5 + v_C_1p5
    y_1p5, f_1p5 = func.oracle(x_1p5)

    i_L_2 = i_L_1 + (beta * h / Inductance) * (v_C_1 - (x_1 - R * y_1)) + \
                    ((1 - beta) * h / Inductance) * (v_C_1p5 - (x_1p5 - R * y_1p5))
    v_C_2 = v_C_1 - (beta * h / Capacitance) * y_1 - ((1 - beta) * h / Capacitance) * y_1p5  
    x_2 = R * i_L_2 + v_C_2
    y_2, f_2 = func.oracle(x_2)

    E_1 = (Capacitance/2) * (v_C_1 - x_star)**2 + (Inductance/2) * (i_L_1 - y_star) ** 2
    E_2 = (Capacitance/2) * (v_C_2 - x_star)**2 + (Inductance/2) * (i_L_2 - y_star) ** 2
    Delta_1 = rho * R * (y_1 - i_L_1)**2 + eta * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem



def main():
    L_smooth = 1.
    mu = 0.0001
    Capacitance = 2.
    Inductance = 2.
    R = 1.

    solver = "ipopt"
    # ciropt definitions
    problem = accelerated_gradient_circuit(mu, L_smooth, R, Capacitance, Inductance)
    problem.obj = problem.eta + problem.rho 
    res, sol, sp_exp = problem.solve(verbose=False, debug=True, solver=solver)[:3]
    ca_vars = problem.vars
    print(f"{res=}")


    co_vars = co.dict_parameters_ciropt(sol, ca_vars, all=True)

    # Grammian definitions
    size_I = 4          # I = { star, 1, 1.5, 2 }
    dim_G = 6           # { x_star, v_C_1, i_L_1, g_1, g_1p5, g_2 }
    dim_F = 4           # { f_star, f_1, f_1p5, f_2}

    opti = ca.Opti()

    eta = opti.variable()
    rho = opti.variable()
    h = opti.variable()
    alpha = opti.variable()
    beta = opti.variable()
    alpha_h = opti.variable()
    beta_h = opti.variable()
    alpha_h_h = opti.variable()
    alpha_beta_h_h = opti.variable()
    P_full = opti.variable(dim_G, dim_G)
    P = ca.tril(P_full)
    lamb0 = opti.variable(size_I, size_I)

    g_star = np.zeros(dim_G).reshape(-1, 1)
    g_1 = co.one_hot(dim_G, 3)
    g_1p5 = co.one_hot(dim_G, 4) 
    g_2 = co.one_hot(dim_G, 5) 
    Gs = [g_star, g_1, g_1p5, g_2]

    f_star = co.one_hot(dim_F, 0) # np.zeros((dim_F, 1))
    f_1   = co.one_hot(dim_F, 1) 
    f_1p5 = co.one_hot(dim_F, 2)
    f_2   = co.one_hot(dim_F, 3)
    F = [f_star, f_1, f_1p5, f_2]

    x_star = co.one_hot(dim_G, 0)
    # x_1    = co.one_hot(dim_G, 1)
    # i_L_1    = co.one_hot(dim_G, 2)
    # v_C_1 = x_1 + R * g_1 - R * i_L_1
    v_C_1    = co.one_hot(dim_G, 1)
    i_L_1    = co.one_hot(dim_G, 2)
    x_1 = v_C_1 + R * i_L_1
    i_L_1p5 = i_L_1 + (alpha_h / Inductance) * (v_C_1 - (x_1 - R * g_1)) 
    v_C_1p5 = v_C_1 - (alpha_h / Capacitance) * g_1
    x_1p5 = R * i_L_1p5 + v_C_1p5

    i_L_2 = i_L_1 + (beta_h / Inductance) * (v_C_1 - (x_1 - R * g_1)) + \
                    ((h - beta_h) / Inductance) * (v_C_1p5 - (x_1p5 - R * g_1p5)) 
    v_C_2 = v_C_1 - (beta_h / Capacitance) * g_1 - ((h - beta_h) / Capacitance) * g_1p5
    x_2 = R * i_L_2 + v_C_2
    Xs = [x_star, x_1, x_1p5, x_2]

    sp_vars = {name : sp.symbols(name) for name in ca_vars.keys()}
    sp_i_L_1p5 = i_L_1 + (sp_vars["alpha_h"] / Inductance) * (v_C_1 - (x_1 - R * g_1) ) 
    sp_v_C_1p5 = v_C_1 - (sp_vars["alpha_h"] / Capacitance) * g_1
    sp_x_1p5 = R * sp_i_L_1p5 + sp_v_C_1p5

    sp_i_L_2 = i_L_1 + (sp_vars["beta_h"] / Inductance) * (v_C_1 - (x_1 - R * g_1)) + \
                    ((sp_vars["h"] - sp_vars["beta_h"]) / Inductance) * (sp_v_C_1p5 - (sp_x_1p5 - R * g_1p5))
    sp_v_C_2 = v_C_1 - (sp_vars["beta_h"] / Capacitance) * g_1 - \
                    ((sp_vars["h"] - sp_vars["beta_h"]) / Capacitance) * g_1p5 
    sp_x_2 = R * sp_i_L_2 + sp_v_C_2

    sp_Xs = [x_star, x_1, sp_x_1p5, sp_x_2]

    star_idx = 4
    A = lambda i,j: co.symm_prod(Gs[j], Xs[i] - Xs[j])
    B = lambda i,j: co.symm_prod(Xs[i] - Xs[j])
    C = lambda i,j: co.symm_prod(Gs[i] - Gs[j])
    D = lambda i,j: co.symm_prod(Xs[i] - Xs[j] - (1/L_smooth) * (Gs[i] - Gs[j]))
    W_11 = co.symm_prod(Gs[1] - i_L_1, Gs[1] - i_L_1) 
    E_1 = (Capacitance / 2) * (co.symm_prod(v_C_1 - x_star, v_C_1 - x_star)) \
            +(Inductance / 2) * (co.symm_prod(i_L_1 - g_star, i_L_1 - g_star))
    E_2 = (Capacitance / 2) * (co.symm_prod(v_C_2 - x_star, v_C_2 - x_star)) \
            +(Inductance / 2) * (co.symm_prod(i_L_2 - g_star, i_L_2 - g_star))                      
    a = lambda i,j: F[j] - F[i]

    sp_A = lambda i,j: co.symm_prod(Gs[j], sp_Xs[i] - sp_Xs[j])
    sp_B = lambda i,j: co.symm_prod(sp_Xs[i] - sp_Xs[j])
    sp_C = lambda i,j: co.symm_prod(Gs[i] - Gs[j])
    sp_D = lambda i,j: co.symm_prod(sp_Xs[i] - sp_Xs[j] - (1/L_smooth) * (Gs[i] - Gs[j]))
    sp_E_1 = (Capacitance / 2) * (co.symm_prod(v_C_1 - x_star, v_C_1 - x_star)) \
        +(Inductance / 2) * (co.symm_prod(i_L_1 - g_star, i_L_1 - g_star))
    sp_E_2 = (Capacitance / 2) * (co.symm_prod(sp_v_C_2 - x_star, sp_v_C_2 - x_star)) \
        +(Inductance / 2) * (co.symm_prod(sp_i_L_2 - g_star, sp_i_L_2 - g_star)) 

    assert (dim_G, dim_G) == A(1, 2).shape == B(1, 2).shape == C(1, 2).shape
    assert a(1, 2).shape == (dim_F, 1)

    opti.minimize( - eta - rho )
    opti.subject_to( alpha_h == alpha * h )
    opti.subject_to( beta_h == beta * h )
    opti.subject_to( alpha_h_h == alpha_h * h )
    opti.subject_to( alpha_beta_h_h == alpha_h * beta_h )
    opti.subject_to( ca.reshape(lamb0, (-1, 1)) >= np.zeros((size_I * size_I, 1)) )
    opti.subject_to( ca.diag(P) >= np.zeros((dim_G, 1)) )
    opti.subject_to( ca.diag(lamb0) == np.zeros((size_I, 1)) )
    # star:0, 1:1, 1p5:2, 2:3
    sum_ij_La = 0
    sum_ij_AC = 0
    # fj - fi + gj * (xi - xj) + 1/(2*L) * (gi - gj) ** 2 + mu / (2 * (1 - mu / L)) * (xi - xj - 1/L * (gi - gj))**2
    for i in range(size_I):
        for j in range(size_I):
            if i == j: continue
            sum_ij_La += lamb0[i, j] * a(i, j)
            sum_ij_AC += lamb0[i, j] * (A(i, j) + (1./(2 * L_smooth)) * C(i, j) + mu / (2 * (1 - mu / L_smooth)) * D(i, j))

    Fweights_obj = - eta * a(1, 0)
    Gweights_obj = E_2 - E_1 + rho * R * W_11
    opti.subject_to( sum_ij_La - Fweights_obj  == np.zeros((dim_F, 1))) 
    opti.subject_to( sum_ij_AC  - P @ P.T - Gweights_obj == np.zeros((dim_G, dim_G)))

    opts = {'ipopt.max_iter':50000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)
    sol = opti.solve()
    assert sol.stats()['success'], print(sol.stats())
    print(sol.value(eta), sol.value(h), sol.value(rho), sol.value(alpha), sol.value(beta))

    # compare matrices
    size_I_function = size_I
    for i in range(size_I_function):
        for j in range(size_I_function):
            if i == j: continue
            F1, G1 = sp_exp[(0, 0, i, j)]["F"], sp_exp[(0, 0, i, j)]["G"]
            F2 = a(i, j)
            G2 = co.simplify_matrix(sp_A(i, j) + (1./(2*L_smooth)) * sp_C(i, j) + mu / (2 * (1 - mu / L_smooth)) * sp_D(i, j))
            assert co.equal_sp_arrays(G1, G2), print(f"{i=}, {j=} \n{G1=} \n{G2=}, \n{G1-G2=}")
            assert co.equal_sp_arrays(F1, F2), print(f"{i=}, {j=} \n{F1=} \n{F2=}\n")

    F1, G1 = sp_exp["FG_obj"]["F"], sp_exp["FG_obj"]["G"]
    F2 = sp_vars["eta"] * a(1, 0)
    G2 =  sp_E_2 - sp_E_1 + sp_vars["rho"] * R * W_11
    assert co.equal_sp_arrays(G1, G2), print(f"{G1=} \n{G2=}")
    assert co.equal_sp_arrays(-F1, F2), print(f"{F1=} \n{F2=}\n")
    print("PASSED equal symbolic expression")

    # compare expression evaluations 
    h_init = co_vars["h"]
    b_init = co_vars["eta"]
    d_init = co_vars["rho"]
    alpha_init = co_vars["alpha"]
    beta_init = co_vars["beta"]
    lamb_init = co_vars["lamb0"]
    co_vars["P_full"] = co_vars["P"]
    P_full_init = co_vars["P_full"]
    alpha_h_init = alpha_init * h_init
    beta_h_init = beta_init * h_init

    vars = [eta, rho, h, alpha, beta, alpha_h, beta_h, P_full, lamb0]
    eval_vars = [b_init, d_init, h_init, alpha_init, beta_init, alpha_h_init, beta_h_init, P_full_init, lamb_init]

    for name in ['h', 'eta', 'rho', 'alpha', 'beta', 'P_full', 'lamb0']:
        ca_g_eval = eval_ca_function(vars, eval_vars, locals()[name])
        assert np.allclose(ca_g_eval, co_vars[name]), print(name)

    for name in ['sum_ij_AC', 'sum_ij_La', 'Fweights_obj', 'Gweights_obj']:
        ca_g_eval = eval_ca_function(vars, eval_vars, locals()[name])
        assert np.allclose(ca_g_eval, problem.ca_expressions[name]) or \
            np.allclose(np.array(ca_g_eval.full()).flatten(), problem.ca_expressions[name]), print(name)

    val_ca_P = eval_ca_function(vars, eval_vars, locals()['P_full'])
    sum1 = np.array((eval_ca_function(vars, eval_vars, locals()['sum_ij_AC']) \
        - eval_ca_function(vars, eval_vars, locals()['Gweights_obj']) \
        - val_ca_P @ val_ca_P.T).full())
    sum2 = np.array((eval_ca_function(vars, eval_vars, locals()['sum_ij_La']) \
        - eval_ca_function(vars, eval_vars, locals()['Fweights_obj'])).full())

    assert np.allclose(0, np.linalg.norm(sum1.flatten())) and np.allclose(0, np.linalg.norm(sum2.flatten()))

    print("PASSED equal expression evaluations")


if __name__ == "__main__":
    main()

