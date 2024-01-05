import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt

 


def define_function(problem, mu, L_smooth, package):
    if mu != 0 and L_smooth < np.inf:
        func = problem.declare_function( package.SmoothStronglyConvexFunction, L=L_smooth, mu=mu) 
    elif mu != 0:
        func = problem.declare_function( package.StronglyConvexFunction, mu=mu) 
    elif L_smooth < np.inf:
        func = problem.declare_function( package.SmoothConvexFunction, L=L_smooth) 
    else:
        func = problem.declare_function( package.ConvexFunction) 
    return func


def gradient_flow_circuit(mu, L_smooth, Capacitance, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        h, alpha, beta, b = params["h"], params["alpha"], params["beta"], params["b"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, b = problem.h, problem.alpha, problem.beta, problem.b
    func = define_function(problem, mu, L_smooth, package )
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    x_1 = problem.set_initial_point()
    y_1, f_1 = func.oracle(x_1)

    x_1p5 = x_1 - (alpha * h / Capacitance) * y_1 
    y_1p5, f_1p5 = func.oracle(x_1p5)

    x_2 = x_1  - (beta * h / Capacitance) * y_1 - ((1 - beta) * h / Capacitance) * y_1p5
    y_2, f_2 = func.oracle(x_2)

    E_1 = (Capacitance/2) * (x_1 - x_star)**2
    E_2 = (Capacitance/2) * (x_2 - x_star)**2
    Delta_1 = b * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def accelerated_gradient_circuit(mu, L_smooth, R, Capacitance, Inductance, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    func = define_function(problem, mu, L_smooth, package )
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    # x_1 = problem.set_initial_point()
    # i_L_1 = problem.set_initial_point()
    # y_1, f_1 = func.oracle(x_1)
    # v_C_1 = x_1 + R * y_1  - R * i_L_1
    v_C_1 = problem.set_initial_point()
    i_L_1 = problem.set_initial_point()
    x_1, y_1, f_1 = proximal_step((R * i_L_1 + v_C_1), func, R)

    i_L_1p5 = i_L_1 + (alpha * h / Inductance) * (v_C_1 - x_1) 
    v_C_1p5 = v_C_1 - (alpha * h / Capacitance) * y_1 
    x_1p5, y_1p5, f_1p5 = proximal_step((R * i_L_1p5 + v_C_1p5), func, R)

    i_L_2 = i_L_1 + (beta * h / Inductance) * (v_C_1 - x_1) + \
                    ((1 - beta) * h / Inductance) * (v_C_1p5 - x_1p5)
    v_C_2 = v_C_1 - (beta * h / Capacitance) * y_1 - ((1 - beta) * h / Capacitance) * y_1p5  
    x_2, y_2, f_2 = proximal_step((R * i_L_2 + v_C_2), func, R)

    E_1 = (Capacitance/2) * (v_C_1 - x_star)**2 + (Inductance/2) * (i_L_1 - y_star) ** 2
    E_2 = (Capacitance/2) * (v_C_2 - x_star)**2 + (Inductance/2) * (i_L_2 - y_star) ** 2
    Delta_1 = d * R * (y_1 - i_L_1)**2 + b * (f_1 - f_star)
    # Delta_1 = b * (f_1 - f_star) 
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def dual_decomposition(n_func, mu, L_smooth, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        h, alpha, beta, b = params["h"], params["alpha"], params["beta"], params["b"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, b = problem.h, problem.alpha, problem.beta, problem.b

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    ys_star = [0] * n_func
    for i in range(n_func):
        yi, _ = fs[i].oracle(x_star)
        ys_star[i] = yi

    i_L_1 = [0] * n_func
    xs_1 = [0] * n_func
    for i in range(n_func):
        if i < n_func-1:
            yi_1 = problem.set_initial_point()
            i_L_1[i] = yi_1
            if i == 0: i_L_1[n_func-1] = -yi_1
            else: i_L_1[n_func-1] -= yi_1
        else:
            yi_1 = i_L_1[n_func-1]
        xi_1 = co_func.subgrad_conjugate(yi_1, fs[i])
        xs_1[i] = xi_1
        if i == 0: z_1 = xi_1
        else: z_1 += xi_1
    z_1 = z_1 / n_func

    i_L_1p5 = [0] * n_func
    for i in range(n_func):
        i_L_1p5[i] = i_L_1[i] + (alpha * h / Inductance) * (z_1 - xs_1[i])
    xs_1p5 = [0] * n_func
    for i in range(n_func):
        xi_1p5 = co_func.subgrad_conjugate(i_L_1p5[i], fs[i])
        xs_1p5[i] = xi_1p5
        if i == 0: z_1p5 = xi_1p5
        else: z_1p5 += xi_1p5
    z_1p5 = z_1p5 / n_func

    i_L_2 = [0] * n_func
    for i in range(n_func):
        i_L_2[i] = i_L_1[i] + (beta * h / Inductance) * (z_1 - xs_1[i]) \
                              + ((1 - beta) * h / Inductance) * (z_1p5 - xs_1p5[i])

    E_1, E_2, Delta_1 = 0, 0, 0
    for i in range(n_func):
        E_1 += (Inductance/2) * (i_L_1[i] - ys_star[i])**2
        E_2 += (Inductance/2) * (i_L_2[i] - ys_star[i])**2
        Delta_1 += b * ((xs_1[i] - x_star) * (i_L_1[i] - ys_star[i]))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def primal_decomposition2(mu, L_smooth, Capacitance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    y2_star, _ = f2.oracle(x_star)
    # y2_star = y_star - y1_star

    z_1 = problem.set_initial_point()
    y1_1, _ = f1.oracle(z_1)
    y2_1, _ = f2.oracle(z_1)

    z_1p5 = z_1 - (alpha * h / Capacitance) * (y1_1 + y2_1)
    y1_1p5, _ = f1.oracle(z_1p5)
    y2_1p5, _ = f2.oracle(z_1p5)

    z_2 = z_1  - (beta * h / Capacitance) * (y1_1 + y2_1) \
               - ((1 - beta) * h / Capacitance) * (y1_1p5 + y2_1p5)
    # y1_2, _ = f1.oracle(z_2)
    # y2_2, _ = f2.oracle(z_2)

    E_1 = (Capacitance/2) * (z_1 - x_star)**2
    E_2 = (Capacitance/2) * (z_2 - x_star)**2
    Delta_1 = b * ((z_1 - x_star) * (y1_1 - y1_star) + (z_1 - x_star) * (y2_1 - y2_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def primal_decomposition(n_func, mu, L_smooth, Capacitance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        h, alpha, beta, b = params["h"], params["alpha"], params["beta"], params["b"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, b = problem.h, problem.alpha, problem.beta, problem.b

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)

    z_1 = problem.set_initial_point()
    for i in range(n_func):
        yi_1, _ = fs[i].oracle(z_1)
        if i == 0: sum_y_1 = yi_1
        else: sum_y_1 += yi_1

    z_1p5 = z_1 - (alpha * h / Capacitance) * sum_y_1
    for i in range(n_func):
        yi_1p5, _ = fs[i].oracle(z_1p5)
        if i == 0: sum_y_1p5 = yi_1p5
        else: sum_y_1p5 += yi_1p5

    z_2 = z_1  - (beta * h / Capacitance) * sum_y_1 \
               - ((1 - beta) * h / Capacitance) * sum_y_1p5

    E_1 = (Capacitance/2) * (z_1 - x_star)**2
    E_2 = (Capacitance/2) * (z_2 - x_star)**2
    Delta_1 = b * ((z_1 - x_star) * (sum_y_1 - y_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def douglas_rachford_splitting(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    y2_star = y_star - y1_star

    # i_L_0 = problem.set_initial_point()
    x2_0 = problem.set_initial_point()

    # x1_1, y1_1, f1_1 = proximal_step(x2_0 + R * i_L_0, f1, R)
    # x2_1, y2_1, f2_1 = proximal_step(x1_1 - R * i_L_0, f2, R)
    x1_1, y1_1, f1_1 = proximal_step(x2_0, f1, R)
    x2_1, y2_1, f2_1 = proximal_step(x1_1, f2, R)
    i_L_1 = (h / Inductance)  * (x2_1 - x1_1)
    # problem.add_constraint(Constraint(f_star - f1_1 - f2_1, "inequality"))

    x1_1p5, y1_1p5, f1_1p5 = proximal_step(x2_1 + R * i_L_1, f1, R)
    x2_1p5, y2_1p5, f2_1p5 = proximal_step(x1_1p5 - R * i_L_1, f2, R)
    i_L_1p5 = i_L_1 + (alpha * h / Inductance) * (x2_1 - x1_1)
    # problem.add_constraint(Constraint(f_star - f1_1p5 - f2_1p5, "inequality"))

    x1_2, y1_2, f1_2 = proximal_step(x2_1p5 + R * i_L_1p5, f1, R)
    x2_2, y2_2, f2_2 = proximal_step(x1_2 - R * i_L_1p5, f2, R)
    i_L_2 = i_L_1 + (beta * h / Inductance) *  (x2_1 - x1_1) \
                  + ((1 - beta) * h / Inductance) * (x2_1p5 - x1_1p5)
    # problem.add_constraint(Constraint(f_star - f1_2 - f2_2, "inequality"))

    E_1 = (Inductance/2) * (i_L_1 - y1_star) ** 2 
    E_2 = (Inductance/2) * (i_L_2 - y1_star) ** 2 

    Delta_1 = d * R * (y1_1 - i_L_1)**2 + b * (f1_1 + f2_1 \
                            - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - f_star)
    # Delta_1 = b * (f1_1 + f2_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def admm_consensus(n_func, mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        Constraint = pep_constr
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        Constraint = co_constr
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    ys_star = [0] * n_func
    # ys_star[-1] = y_star
    for i in range(n_func):#(n_func-1):
        gi, _ = fs[i].oracle(x_star)
        ys_star[i] = gi
        # ys_star[-1] -= gi

    i_Ls_1 = [0] * n_func
    # initialize currents on inductors with \sum_l i_(L_l)(0)=0
    for i in range(n_func-1):
        i_Ls_1[i] = problem.set_initial_point()
        if i == 0: i_Ls_1[-1] = -i_Ls_1[i]
        else: i_Ls_1[-1] -= i_Ls_1[i]
    # initial potential at the bottom of circuit is e_0 = 0

    triplets_1 = [0] * n_func
    sum_fi_1 = 0
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1[i]), fs[i], R)
        if i == 0: e_1 = xi
        else: e_1 = e_1 + xi
        triplets_1[i] = (xi, gi, fi)
        sum_fi_1 += fi
    # potential at the bottom of circuit is average of potentials at fi's
    e_1 = e_1 / n_func
    L_x_e_ystar1 = 0
    for i in range(n_func):
        L_x_e_ystar1 += - ys_star[i] * (triplets_1[i][0] - e_1)
    problem.add_constraint(Constraint(f_star - sum_fi_1 - L_x_e_ystar1, "inequality"))

    i_Ls_1p5 = [0] * n_func
    sum_fi_1p5 = 0
    for i in range(n_func):
        i_Ls_1p5[i] = i_Ls_1[i] + (alpha * h / Inductance) * (e_1 - triplets_1[i][0]) 
    triplets_1p5 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1p5[i] + e_1), fs[i], R)
        if i == 0: e_1p5 = xi
        else: e_1p5 = e_1p5 + xi
        triplets_1p5[i] = (xi, gi, fi)
        sum_fi_1p5 += fi
    e_1p5 = e_1p5 / n_func
    L_x_e_ystar1p5 = 0
    for i in range(n_func):
        L_x_e_ystar1p5 += - ys_star[i] * (triplets_1p5[i][0] - e_1p5)
    # problem.add_constraint(Constraint(f_star - sum_fi_1p5 - L_x_e_ystar1p5, "inequality"))

    i_Ls_2 = [0] * n_func
    for i in range(n_func):
        i_Ls_2[i] = i_Ls_1[i] + (beta * h / Inductance) * (e_1 - triplets_1[i][0]) \
                              + ((1 - beta) * h / Inductance) * (e_1p5 - triplets_1p5[i][0])
    triplets_2 = [0] * n_func
    sum_fi_2 = 0
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_2[i] + e_1p5), fs[i], R)
        if i == 0: e_2 = xi
        else: e_2 = e_2 + xi
        triplets_2[i] = (xi, gi, fi)
        sum_fi_2 += fi
    e_2 = e_2 / n_func
    L_x_e_ystar2 = 0
    for i in range(n_func):
        L_x_e_ystar2 += - ys_star[i] * (triplets_2[i][0] - e_2)
    problem.add_constraint(Constraint(f_star - sum_fi_2 - L_x_e_ystar2, "inequality"))

    E_1 = 0; E_2 = 0; f_1 = 0
    Delta_1 = - b * f_star
    for i in range(n_func):
        E_1 += (Inductance/2) * (i_Ls_1[i] - ys_star[i]) ** 2
        E_2 += (Inductance/2) * (i_Ls_2[i] - ys_star[i]) ** 2
        # f_1 += triplets_1[i][2]
        Delta_1 += d * R * (triplets_1[i][1] - i_Ls_1[i])**2 + b *( triplets_1[i][2] - ys_star[i] * (triplets_1[i][0] - x_star))
    # Delta_1 += b * (sum_fi_1 - f_star)
    # Delta_1 = b * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def admm_consensus_proof(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        Constraint = pep_constr
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        Constraint = co_constr
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    rho = 1 / R
    f = define_function(problem, mu, L_smooth, package)
    g = define_function(problem, mu, L_smooth, package)
    func = f + g
    x_star, y_star, func_star = func.stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f.oracle(x_star)
    y2_star = y_star - y1_star
    z1 = problem.set_initial_point()
    y1_1 = problem.set_initial_point()
    y2_1 = - y1_1

    # x1_1, grad_f_1, f_1 = proximal_step(z0, f, 1/rho)
    # x2_1, grad_g_1, g_1 = proximal_step(z0, g, 1/rho)
    # x1_1, grad_f_1, f_1 = proximal_step(- (1/rho) * y1_0, f, 1/rho)
    # x2_1, grad_g_1, g_1 = proximal_step(- (1/rho) * y2_0, g, 1/rho)
    # z1 = (x1_1 + x2_1) / 2
    # y1_1 = y1_0 - h * rho * (z1 - x1_1)
    # y2_1 = y2_0 - h * rho * (z1 - x2_1)
    # problem.add_constraint(Constraint(func_star - (f_1 + g_1 - y1_star * (x1_1 - z1) - y2_star * (x2_1 - z1)), "inequality"))

    x1_2, grad_f_2, f_2 = proximal_step(z1 - (1/rho) * y1_1, f, 1/rho)
    x2_2, grad_g_2, g_2 = proximal_step(z1 - (1/rho) * y2_1, g, 1/rho)
    z2 = (x1_2 + x2_2) / 2
    y1_2 = y1_1 - h * rho * (z2 - x1_2)
    y2_2 = y2_1 - h * rho * (z2 - x2_2)
    # problem.add_constraint(Constraint(func_star - (f_2 + g_2 - y1_star * (x1_2 - z2) - y2_star * (x2_2 - z2)), "inequality"))

    E_1 = (1/rho) * (y1_1 + y1_star)**2 + (1/rho) * (y2_1 + y2_star)**2 + 2 * rho * (z1 - x_star)**2 
    E_2 = (1/rho) * (y1_2 + y1_star)**2 + (1/rho) * (y2_2 + y2_star)**2 + 2 * rho * (z2 - x_star)**2 
    Delta_1 = b * (rho * (x1_2 - z2)**2 + rho * (x2_2 - z2)**2) + d * rho * (z1 - z2)**2 

    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem



def decomp_euler_consensus2(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        Constraint = pep_constr
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        Constraint = co_constr
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    y2_star, _ = f2.oracle(x_star)
    # y2_star = y_star - y1_star

    i_L1_1 = problem.set_initial_point()
    e_1 = problem.set_initial_point()
    i_L2_1 = - i_L1_1
    # initial potential at the bottom of circuit is e_1 = 0
    # _ = f1.oracle(e_0); _ = f2.oracle(e_0)

    x1_1, y1_1, f1_1 = proximal_step(e_1 + (R * i_L1_1), f1, R)
    x2_1 = 2 * e_1 - x1_1
    y2_1, f2_1 = f2.oracle(x2_1)

    # e_1 = (x1_1 + x2_1) / 2
    # _ = f1.oracle(e_1); _ = f2.oracle(e_1)
    (f1 + f2).oracle(e_1)
    # problem.add_constraint(Constraint(f_star - (f1_2 + f2_2 - y1_star * (x1_2 - e_2) - y2_star * (x2_2 - e_2)), "inequality"))
    i_L1_2 = i_L1_1 + (h / Inductance) * (e_1 - x1_1) 
    i_L2_2 = i_L2_1 + (h / Inductance) * (e_1 - x2_1) 
    
    # L_x_e_ystar2 = - y1_star * (x1_2 - e_2) - y2_star * (x2_2 - e_2)
    # problem.add_constraint(Constraint(f_star - f1_2 - f2_2 - L_x_e_ystar2, "inequality"))

    E_1 = (Inductance/2) * (i_L1_1 - y1_star) ** 2 + (Inductance/2) * (i_L2_1 - y2_star) ** 2
    E_2 = (Inductance/2) * (i_L1_2 - y1_star) ** 2 + (Inductance/2) * (i_L2_2 - y2_star) ** 2
    Delta_1 = d * R * (y1_1 - i_L1_1)**2 + d * R * (y2_1 - i_L2_1)**2  + b * (f1_1 + f2_1 \
                            - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - f_star)
    # Delta_1 = d * R * (y1_1 - i_L1_1)**2 + d * R * (y2_1 - i_L2_1)**2  \
    #         + b * ((x1_1 - x_star) * (y1_1 - y1_star) + (x2_1 - x_star) * (y2_1 - y2_star))
    # Delta_1 = d * R * (y1_2 - i_L1_1)**2 + d * R * (y2_2 - i_L2_1)**2  \
    #         + b * ((x1_2 - x_star) * (y1_2 - y1_star) + (x2_2 - x_star) * (y2_2 - y2_star))
    # Delta_1 = d * R * (y1_2 - i_L1_2)**2 + d * R * (y2_2 - i_L2_2)**2  \
    #         + b * ((x1_2 - x_star) * (y1_2 - y1_star) + (x2_2 - x_star) * (y2_2 - y2_star))
    problem.set_performance_metric((E_2 - E_1) + Delta_1)
    return problem


def admm_consensus2(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        Constraint = pep_constr
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        Constraint = co_constr
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    y2_star, _ = f2.oracle(x_star)
    # y2_star = y_star - y1_star

    i_L1_1 = problem.set_initial_point()
    e_1 = problem.set_initial_point()
    i_L2_1 = - i_L1_1
    # initial potential at the bottom of circuit is e_1 = 0
    # _ = f1.oracle(e_1); _ = f2.oracle(e_1)

    # x1_1, y1_1, f1_1 = proximal_step(e_0 + (R * i_L1_0), f1, R)
    # x2_1, y2_1, f2_1 = proximal_step(e_0 + (R * i_L2_0), f2, R)
    # e_1 = (x1_1 + x2_1) / 2
    # _ = f1.oracle(e_1); _ = f2.oracle(e_1)
    # i_L1_1 = i_L1_0 + (h / Inductance) * (e_1 - x1_1) 
    # i_L2_1 = i_L2_0 + (h / Inductance) * (e_1 - x2_1) 

    x1_2, y1_2, f1_2 = proximal_step(e_1 + (R * i_L1_1), f1, R)
    x2_2, y2_2, f2_2 = proximal_step(e_1 + (R * i_L2_1), f2, R)
    e_2 = (x1_2 + x2_2) / 2
    _ = f1.oracle(e_2); _ = f2.oracle(e_2)
    # problem.add_constraint(Constraint(f_star - (f1_2 + f2_2 - y1_star * (x1_2 - e_2) - y2_star * (x2_2 - e_2)), "inequality"))
    i_L1_2 = i_L1_1 + (h / Inductance) * (e_2 - x1_2) 
    i_L2_2 = i_L2_1 + (h / Inductance) * (e_2 - x2_2) 

    # problem.add_constraint(Constraint((e_1 - x1_1)**2 + (e_1 - x2_1)**2 - (e_2 - x1_1)**2 - (e_2 - x2_1)**2, "inequality"))
    # problem.add_constraint(Constraint((e_2 - x1_2)**2 + (e_2 - x2_2)**2 - (e_1 - x1_2)**2 - (e_1 - x2_2)**2, "inequality"))
    
    # L_x_e_ystar2 = - y1_star * (x1_2 - e_2) - y2_star * (x2_2 - e_2)
    # problem.add_constraint(Constraint(f_star - f1_2 - f2_2 - L_x_e_ystar2, "inequality"))

    E_1 = (Inductance/2) * (i_L1_1 - y1_star) ** 2 + (Inductance/2) * (i_L2_1 - y2_star) ** 2
    E_2 = (Inductance/2) * (i_L1_2 - y1_star) ** 2 + (Inductance/2) * (i_L2_2 - y2_star) ** 2
    # Delta_1 = d * R * (y1_1 - i_L1_1)**2 + d * R * (y2_1 - i_L2_1)**2  + b * (f1_1 + f2_1 \
    #                         - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - f_star)
    # Delta_1 = d * R * (y1_1 - i_L1_1)**2 + d * R * (y2_1 - i_L2_1)**2  \
    #         + b * ((x1_1 - x_star) * (y1_1 - y1_star) + (x2_1 - x_star) * (y2_1 - y2_star))
    Delta_1 = d * R * (y1_2 - i_L1_1)**2 + d * R * (y2_2 - i_L2_1)**2  \
            + b * ((x1_2 - x_star) * (y1_2 - y1_star) + (x2_2 - x_star) * (y2_2 - y2_star))
    # Delta_1 = d * R * (y1_2 - i_L1_2)**2 + d * R * (y2_2 - i_L2_2)**2  \
    #         + b * ((x1_2 - x_star) * (y1_2 - y1_star) + (x2_2 - x_star) * (y2_2 - y2_star))
    problem.set_performance_metric((E_2 - E_1) + Delta_1)
    return problem


def admm_consensus2_i1(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        Constraint = pep_constr
        h, alpha, beta, b, d = params["h"], params["alpha"], params["beta"], params["b"], params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        Constraint = co_constr
        h, alpha, beta, b, d = problem.h, problem.alpha, problem.beta, problem.b, problem.d

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    y2_star, _ = f2.oracle(x_star)

    i_L1_1 = problem.set_initial_point()
    i_L2_1 = - i_L1_1
    # initial potential at the bottom of circuit is e_0 = 0

    x1_1, y1_1, f1_1 = proximal_step((R * i_L1_1), f1, R)
    x2_1, y2_1, f2_1 = proximal_step((R * i_L2_1), f2, R)

    # potential at the bottom of circuit is average of potentials at fi's
    e_1 = (x1_1 + x2_1) / 2
    L_x_e_ystar1 = - y1_star * (x1_1 - e_1) - y2_star * (x2_1 - e_1)
    problem.add_constraint(Constraint(f_star - f1_1 - f2_1 - L_x_e_ystar1, "inequality"))

    i_L1_1p5 = i_L1_1 + (alpha * h / Inductance) * (e_1 - x1_1) 
    i_L2_1p5 = i_L2_1 + (alpha * h / Inductance) * (e_1 - x2_1) 
    x1_1p5, y1_1p5, f1_1p5 = proximal_step((R * i_L1_1p5 + e_1), f1, R)
    x2_1p5, y2_1p5, f2_1p5 = proximal_step((R * i_L2_1p5 + e_1), f2, R)
    e_1p5 = (x1_1p5 + x2_1p5) / 2

    i_L1_2 = i_L1_1 + (beta * h / Inductance) * (e_1 - x1_1) \
                              + ((1 - beta) * h / Inductance) * (e_1p5 - x1_1p5)
    i_L2_2 = i_L2_1 + (beta * h / Inductance) * (e_1 - x2_1) \
                              + ((1 - beta) * h / Inductance) * (e_1p5 - x2_1p5)
    x1_2, y1_2, f1_2 = proximal_step((R * i_L1_2 + e_1p5), f1, R)
    x2_2, y2_2, f2_2 = proximal_step((R * i_L2_2 + e_1p5), f2, R)
    e_2 = (x1_2 + x2_2) / 2
    L_x_e_ystar2 = - y1_star * (x1_2 - e_2) - y2_star * (x2_2 - e_2)
    problem.add_constraint(Constraint(f_star - f1_2 - f2_2 - L_x_e_ystar2, "inequality"))

    E_1 = (Inductance/2) * (i_L1_1 - y1_star) ** 2 + (Inductance/2) * (i_L2_1 - y2_star) ** 2
    E_2 = (Inductance/2) * (i_L1_2 - y1_star) ** 2 + (Inductance/2) * (i_L2_2 - y2_star) ** 2
    Delta_1 = d * R * (y1_1 - i_L1_1)**2 + d * R * (y2_1 - i_L2_1)**2  + b * (f1_1 + f2_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem
