import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.circuit_opt import CircuitOpt


from ciropt.function import SmoothStronglyConvexFunction, StronglyConvexFunction, \
                            SmoothConvexFunction, ConvexFunction

 


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


def admm_consensus(n_func, mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        h, alpha, beta, b = params["h"], params["alpha"], params["beta"], params["b"] #, params["d"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        h, alpha, beta, b = problem.h, problem.alpha, problem.beta, problem.b #, problem.d

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    gs_star = [0] * n_func
    for i in range(n_func):
        gi, fi = fs[i].oracle(x_star)
        gs_star[i] = gi

    i_Ls_1 = [0] * n_func
    # initialize currents on inductors with \sum_l i_(L_l)(0)=0
    for i in range(n_func-1):
        i_Ls_1[i] = problem.set_initial_point()
        if i == 0: i_Ls_1[-1] = -i_Ls_1[i]
        else: i_Ls_1[-1] = i_Ls_1[-1] - i_Ls_1[i]
    # initial potential at the bottom of circuit is e_0 = 0

    triplets_1 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1[i]), fs[i], R)
        if i == 0: e_1 = xi
        else: e_1 = e_1 + xi
        triplets_1[i] = (xi, gi, fi)
    # potential at the bottom of circuit is average of potentials at fi's
    e_1 = e_1 / n_func

    i_Ls_1p5 = [0] * n_func
    for i in range(n_func):
        i_Ls_1p5[i] = i_Ls_1[i] + (alpha * h / Inductance) * (e_1 - triplets_1[i][0]) 
    triplets_1p5 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1p5[i] + e_1), fs[i], R)
        if i == 0: e_1p5 = xi
        else: e_1p5 = e_1p5 + xi
        triplets_1p5[i] = (xi, gi, fi)
    e_1p5 = e_1p5 / n_func

    i_Ls_2 = [0] * n_func
    for i in range(n_func):
        i_Ls_2[i] = i_Ls_1[i] + (beta * h / Inductance) * (e_1 - triplets_1[i][0]) \
                              + ((1 - beta) * h / Inductance) * (e_1p5 - triplets_1p5[i][0])
    triplets_2 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1p5[i] + e_1p5), fs[i], R)
        if i == 0: e_2 = xi
        else: e_2 = e_2 + xi
        triplets_2[i] = (xi, gi, fi)
    e_2 = e_2 / n_func

    E_1 = 0; E_2 = 0; f_1 = 0
    for i in range(n_func):
        E_1 += (Inductance/2) * (i_Ls_1[i] - gs_star[i]) ** 2
        E_2 += (Inductance/2) * (i_Ls_2[i] - gs_star[i]) ** 2
        f_1 += triplets_1[i][2]
    #     if i == 0: Delta_1 = d * R * (triplets_1[i][1] - i_Ls_1[i])**2 
    #     else: Delta_1 = Delta_1 + d * R * (triplets_1[i][1] - i_Ls_1[i])**2 
    # Delta_1 += b * (f_1 - f_star)
    Delta_1 = b * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem