import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt



def admm_euler_consensus2(n_func, mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, alpha, beta, b, d, gamma = params["h"], params["alpha"], params["beta"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, alpha, beta, b, d, gamma = problem.h, problem.alpha, problem.beta, problem.b, problem.d, problem.gamma

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    ys_star = [0] * n_func
    ys_star[-1] = y_star
    for i in range(n_func-1):
        gi, _ = fs[i].oracle(x_star)
        ys_star[i] = gi
        ys_star[-1] -= gi

    i_Ls_1 = [0] * n_func
    # initialize currents on inductors with \sum_l i_(L_l)(0)=0
    for i in range(n_func-1):
        i_Ls_1[i] = problem.set_initial_point()
        if i == 0: i_Ls_1[-1] = -i_Ls_1[i]
        else: i_Ls_1[-1] -= i_Ls_1[i]
    e_1 = problem.set_initial_point()
    triplets_1 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1[i] + e_1), fs[i], R)
        if i == 0: sum_xi_1 = xi
        else: sum_xi_1 = sum_xi_1 + xi
        triplets_1[i] = (xi, gi, fi)
    i_Ls_1p5 = [0] * n_func
    for i in range(n_func):
        i_Ls_1p5[i] = i_Ls_1[i] + (alpha * h / Inductance) * (e_1 - triplets_1[i][0])
    sum_xi_1 = sum_xi_1 / n_func
    problem.add_constraint(Constraint( (e_1 - sum_xi_1) ** 2, "inequality"))
    problem.add_constraint(Constraint(-(e_1 - sum_xi_1) ** 2, "inequality"))

    triplets_1p5 = [0] * n_func
    e_1p5 = problem.set_initial_point()
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1p5[i] + e_1p5), fs[i], R)
        if i == 0: sum_xi_1p5 = xi
        else: sum_xi_1p5 = sum_xi_1p5 + xi
        triplets_1p5[i] = (xi, gi, fi)
    i_Ls_2 = [0] * n_func
    for i in range(n_func):
        i_Ls_2[i] = i_Ls_1[i] + (beta * h / Inductance) * (e_1 - triplets_1[i][0]) \
                              + ((1 - beta) * h / Inductance) * (e_1p5 - triplets_1p5[i][0])
    sum_xi_1p5 = sum_xi_1p5 / n_func
    problem.add_constraint(Constraint( (e_1p5 - sum_xi_1p5) ** 2, "inequality"))
    problem.add_constraint(Constraint(-(e_1p5 - sum_xi_1p5) ** 2, "inequality"))
    
    E_1 = 0; E_2 = 0
    Delta_1 = - b * f_star
    for i in range(n_func):
        E_1 += (Inductance/2) * (i_Ls_1[i] - ys_star[i]) ** 2
        E_2 += (Inductance/2) * (i_Ls_2[i] - ys_star[i]) ** 2
        Delta_1 += d * R * (triplets_1[i][1] - i_Ls_1[i]) ** 2 \
                 + b *( triplets_1[i][2] - ys_star[i] * (triplets_1[i][0] - x_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def ppm_circuit_original(mu, L_smooth, Capacitance, R, params=None): 
    # fixed stepsize h = CR
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        b = params["b"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        b = problem.b
    func = define_function(problem, mu, L_smooth, package )
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    x_1 = problem.set_initial_point()
    x_2, y_2, f_2 = proximal_step(x_1, func, R)

    E_1 = (Capacitance/2) * (x_1 - x_star)**2
    E_2 = (Capacitance/2) * (x_2 - x_star)**2
    Delta_1 = b * (f_2 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def primal_decomposition2(mu, L_smooth, Capacitance, params=None):
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


def admm_consensus2(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    # y2_star, _ = f2.oracle(x_star)
    y2_star = y_star - y1_star

    i_L1_1 = problem.set_initial_point()
    e_1 = problem.set_initial_point()
    i_L2_1 = - i_L1_1

    x1_2, y1_2, f1_2 = proximal_step(e_1 + (R * i_L1_1), f1, R)
    x2_2, y2_2, f2_2 = proximal_step(e_1 + (R * i_L2_1), f2, R)
    e_2 = (x1_2 + x2_2) / 2
    # _ = f1.oracle(e_2); _ = f2.oracle(e_2)
    i_L1_2 = i_L1_1 + (h / Inductance) * (e_2 - x1_2) 
    i_L2_2 = i_L2_1 + (h / Inductance) * (e_2 - x2_2) 

    E_1 = (Inductance/2) * (i_L1_1 - y1_star) ** 2 + (Inductance/2) * (i_L2_1 - y2_star) ** 2 \
            + gamma * (e_1 - x_star)**2
    E_2 = (Inductance/2) * (i_L1_2 - y1_star) ** 2 + (Inductance/2) * (i_L2_2 - y2_star) ** 2 \
            + gamma * (e_2 - x_star)**2
    Delta_1 = d * R * (y1_2 - i_L1_1)**2 + d * R * (y2_2 - i_L2_1)**2  \
            + b * ((x1_2 - x_star) * (y1_2 - y1_star) + (x2_2 - x_star) * (y2_2 - y2_star))
    problem.set_performance_metric((E_2 - E_1) + Delta_1)
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