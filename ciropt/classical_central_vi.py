import numpy as np
import sympy as sp

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt
from ciropt.utils import define_function




def gradient_flow_circuit(mu, L_smooth, Capacitance, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        h, alpha, beta, eta = params["h"], params["alpha"], params["beta"], params["eta"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, eta = problem.h, problem.alpha, problem.beta, problem.eta
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
    Delta_1 = eta * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def gradient_flow_circuit_tuneC(mu, L_smooth, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        h, alpha, beta, eta, C, invC = params["h"], params["alpha"], params["beta"], params["eta"], params["C"], params["invC"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        C = sp.symbols('C')
        invC = sp.symbols("invC")
        problem.discretization_params = sorted(problem.discretization_params + ["C", "invC"])
        h, alpha, beta, eta = problem.h, problem.alpha, problem.beta, problem.eta
    func = define_function(problem, mu, L_smooth, package )
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    x_1 = problem.set_initial_point()
    y_1, f_1 = func.oracle(x_1)

    x_1p5 = x_1 - (alpha * h * invC) * y_1 
    y_1p5, f_1p5 = func.oracle(x_1p5)

    x_2 = x_1  - (beta * h * invC) * y_1 - ((1 - beta) * h * invC) * y_1p5
    y_2, f_2 = func.oracle(x_2)

    E_1 = (C/2) * (x_1 - x_star)**2
    E_2 = (C/2) * (x_2 - x_star)**2
    Delta_1 = eta * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def accelerated_gradient_circuit(mu, L_smooth, R, Capacitance, Inductance, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        h, alpha, beta, eta, rho = params["h"], params["alpha"], params["beta"], params["eta"], params["rho"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, eta, rho = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho

    func = define_function(problem, mu, L_smooth, package)
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


def accelerated_gradient_circuit_tuneRLC(mu, L_smooth, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        h, alpha, beta, eta, rho, C, invC, L, invL, R = params["h"], params["alpha"], params["beta"], params["eta"], \
                                    params["rho"], params["C"], params["invC"], params["L"], params["invL"], params["R"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        C, R, L = sp.symbols('C'), sp.symbols('R'), sp.symbols('L')
        invC, invL = sp.symbols("invC"), sp.symbols("invL")
        problem.discretization_params = sorted(problem.discretization_params + 
                                               ["C", "invC", "L", "invL", "R"])
        h, alpha, beta, eta, rho = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho

    func = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    v_C_1 = problem.set_initial_point()
    i_L_1 = problem.set_initial_point()
    x_1 = R * i_L_1 + v_C_1
    y_1, f_1 = func.oracle(x_1)

    i_L_1p5 = i_L_1 + (alpha * h * invL) * (v_C_1 - (x_1 - R * y_1)) 
    v_C_1p5 = v_C_1 - (alpha * h * invC) * y_1 
    x_1p5 = R * i_L_1p5 + v_C_1p5
    y_1p5, f_1p5 = func.oracle(x_1p5)

    i_L_2 = i_L_1 + (beta * h * invL) * (v_C_1 - (x_1 - R * y_1)) + \
                    ((1 - beta) * h * invL) * (v_C_1p5 - (x_1p5 - R * y_1p5))
    v_C_2 = v_C_1 - (beta * h * invC) * y_1 - ((1 - beta) * h * invC) * y_1p5  
    x_2 = R * i_L_2 + v_C_2
    y_2, f_2 = func.oracle(x_2)

    E_1 = (C/2) * (v_C_1 - x_star)**2 + (L/2) * (i_L_1 - y_star) ** 2
    E_2 = (C/2) * (v_C_2 - x_star)**2 + (L/2) * (i_L_2 - y_star) ** 2
    Delta_1 = rho * R * (y_1 - i_L_1)**2 + eta * (f_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def ppm_circuit(mu, L_smooth, Capacitance, R, params=None): 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        h, alpha, beta, eta, rho = params["h"], params["alpha"], params["beta"], params["eta"], params["rho"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        h, alpha, beta, eta, rho = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho
    func = define_function(problem, mu, L_smooth, package )
    x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)

    x_1 = problem.set_initial_point()
    x1_prox, _, _ = proximal_step(x_1, func, R)

    x_1p5 = x_1 - (alpha * h / Capacitance) * (x_1 - x1_prox) / R 
    x1p5_prox, _, _ = proximal_step(x_1p5, func, R)

    i_R_1 = (x_1 - x1_prox) / R
    x_2 = x_1  - (beta * h / Capacitance) * (x_1 - x1_prox) / R  \
               - ((1 - beta) * h / Capacitance) * (x_1p5 - x1p5_prox) / R
    y_2, f_2 = func.oracle(x_2)

    E_1 = (Capacitance/2) * (x_1 - x_star)**2
    E_2 = (Capacitance/2) * (x_2 - x_star)**2
    Delta_1 = eta * (f_2 - f_star) + rho * R * (i_R_1 - y_star)**2
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def prox_gradient_circuit(mu, L_smooth, R, Capacitance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        h, eta, rho = params["h"], params["eta"], params["rho"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        h, eta, rho = problem.h, problem.eta, problem.rho

    f = define_function(problem, mu, L_smooth, package)
    g = define_function(problem, 0, np.inf, package)
    func = f + g
    x_star, y_star, func_star = func.stationary_point(return_gradient_and_function_value=True)
    y2_star, _ = f.oracle(x_star)
    y1_star = y_star - y2_star
    e_star = x_star - R * y2_star

    x2_1 = problem.set_initial_point()
    y2_1, _ = f.oracle(x2_1)
    e_1 = x2_1 - R * y2_1 
    # compute gradient of moreau envelope: \nabla ^R g(e_1)
    x1_1, y1_1, _ = proximal_step(e_1, g, R)
    nabla_R_g_e1 = (e_1 - x1_1) / R

    x2_2 = x2_1 - (h / Capacitance) * (nabla_R_g_e1 + y2_1)
    y2_2, _ = f.oracle(x2_2)
    e_2 = x2_2 - R * y2_2
    x1_2, y1_2, _ = proximal_step(e_2, g, R)

    E_1 = (Capacitance / 2) * (e_1 - e_star)**2 
    E_2 = (Capacitance / 2) * (e_2 - e_star)**2 

    Delta_1 = eta * (1/L_smooth - R) * (y2_1 - y2_star)**2 + rho * R * (nabla_R_g_e1 - y1_star)**2 

    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def dual_decomposition(n_func, mu, L_smooth, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        h, alpha, beta, eta = params["h"], params["alpha"], params["beta"], params["eta"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, eta = problem.h, problem.alpha, problem.beta, problem.eta

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
    # initialize currents on inductors to sum to 0
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
        Delta_1 += eta * ((xs_1[i] - x_star) * (i_L_1[i] - ys_star[i]))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def primal_decomposition(n_func, mu, L_smooth, Capacitance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        h, alpha, beta, eta = params["h"], params["alpha"], params["beta"], params["eta"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        h, alpha, beta, eta = problem.h, problem.alpha, problem.beta, problem.eta

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
    Delta_1 = eta * ((z_1 - x_star) * (sum_y_1 - y_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def douglas_rachford_splitting(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, alpha, beta, eta, rho, gamma = params["h"], params["alpha"], params["beta"], params["eta"], params["rho"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step
        h, alpha, beta, eta, rho, gamma = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    # y2_star = y_star - y1_star
    y2_star, _ = f2.oracle(x_star)
    problem.add_constraint(Constraint((y1_star + y2_star - y_star) ** 2, "equality"))

    i_L_0 = problem.set_initial_point()
    x2_0 = problem.set_initial_point()

    x1_1, y1_1, f1_1 = proximal_step(x2_0 + R * i_L_0, f1, R)
    x2_1, y2_1, f2_1 = proximal_step(x1_1 - R * i_L_0, f2, R)
    i_L_1 = i_L_0 + (h / Inductance)  * (x2_1 - x1_1)

    x1_1p5, y1_1p5, f1_1p5 = proximal_step(x2_1 + R * i_L_1, f1, R)
    x2_1p5, y2_1p5, f2_1p5 = proximal_step(x1_1p5 - R * i_L_1, f2, R)
    i_L_1p5 = i_L_1 + (alpha * h / Inductance) * (x2_1 - x1_1)

    x1_2, y1_2, f1_2 = proximal_step(x2_1p5 + R * i_L_1p5, f1, R)
    x2_2, y2_2, f2_2 = proximal_step(x1_2 - R * i_L_1p5, f2, R)
    i_L_2 = i_L_1 + (beta * h / Inductance) *  (x2_1 - x1_1) \
                  + ((1 - beta) * h / Inductance) * (x2_1p5 - x1_1p5)

    E_1 = (Inductance/2) * (i_L_1 - y1_star) ** 2 + gamma * (x2_1 - x_star)**2
    E_2 = (Inductance/2) * (i_L_2 - y1_star) ** 2 + gamma * (x2_2 - x_star)**2

    Delta_1 = rho * R * (y1_1 - i_L_1)**2 \
              + eta * (f1_1 + f2_1 - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def davis_yin_splitting(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, alpha, beta, eta, rho, gamma = params["h"], params["alpha"], params["beta"], params["eta"], params["rho"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step
        h, alpha, beta, eta, rho, gamma = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2 + f3).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    y2_star, _ = f2.oracle(x_star)
    y3_star, _ = f3.oracle(x_star)
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star - y_star) ** 2, "equality"))

    i_L_0 = problem.set_initial_point()
    x2_0 = problem.set_initial_point()
    y3_0, _ = f3.oracle(x2_0)
    # s = R
    e_0 = x2_0 - R * i_L_0 - R * y3_0

    x1_1, y1_1, f1_1 = proximal_step(2 * x2_0 - e_0 - R * y3_0, f1, R)
    x2_1, y2_1, f2_1 = proximal_step(x1_1 + e_0 - x2_0, f2, R)
    i_L_1 = i_L_0 + (h / Inductance)  * (x2_1 - x1_1)
    y3_1, f3_1 = f3.oracle(x2_1)
    e_1 = x2_1 - R * y3_1 - R * i_L_1

    x1_1p5, y1_1p5, f1_1p5 = proximal_step(2 * x2_1 - e_1 - R * y3_1, f1, R)
    x2_1p5, y2_1p5, f2_1p5 = proximal_step(x1_1p5 + e_1 - x2_1, f2, R)
    i_L_1p5 = i_L_1 + (alpha * h / Inductance) * (x2_1p5 - x1_1p5)
    y3_1p5, _ = f3.oracle(x2_1p5)
    e_1p5 = x2_1p5 - R * y3_1p5 - R * i_L_1p5

    x1_2, y1_2, f1_2 = proximal_step(2 * x2_1p5 - e_1p5 - R * y3_1p5, f1, R)
    x2_2, y2_2, f2_2 = proximal_step(x1_2 + e_1p5 - x2_1p5, f2, R)
    i_L_2 = i_L_1 + (beta * h / Inductance) *  (x2_1 - x1_1) \
                  + ((1 - beta) * h / Inductance) * (x2_1p5 - x1_1p5)
    y3_2, _ = f3.oracle(x2_2)
    e_2 = x2_2 - R * y3_2 - R * i_L_2

    e_star = x_star - R * y1_star - R * y3_star
    
    E_1 = (Inductance/2) * (i_L_1 - y1_star) ** 2 + gamma * (e_1 - e_star)**2  
    E_2 = (Inductance/2) * (i_L_2 - y1_star) ** 2 + gamma * (e_2 - e_star)**2  

    Delta_1 = rho * (1 / R) * (x1_1 - x2_1)**2 \
              + eta * (f1_1 + f2_1 + f3_1 - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - y3_star * (x2_1 - x_star) - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def admm_consensus(n_func, mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, alpha, beta, eta, rho, gamma = params["h"], params["alpha"], params["beta"], params["eta"], params["rho"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, alpha, beta, eta, rho, gamma = problem.h, problem.alpha, problem.beta, problem.eta, problem.rho, problem.gamma

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    ys_star = [0] * n_func
    for i in range(n_func):
        gi, _ = fs[i].oracle(x_star)
        ys_star[i] = gi
        if i == 0:     
            sum_ys_star = gi
        else:
            sum_ys_star += gi 
    problem.add_constraint(Constraint((sum_ys_star - y_star) ** 2, "equality"))

    i_Ls_1 = [0] * n_func
    e_1 = problem.set_initial_point()
    # initialize currents on inductors with \sum_l i_(L_l)(0)=0
    for i in range(n_func-1):
        i_Ls_1[i] = problem.set_initial_point()
        if i == 0: i_Ls_1[-1] = -i_Ls_1[i]
        else: i_Ls_1[-1] -= i_Ls_1[i]
 
    triplets_1p5 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1[i] + e_1), fs[i], R)
        if i == 0: e_1p5 = xi
        else: e_1p5 = e_1p5 + xi
        triplets_1p5[i] = (xi, gi, fi)
    i_Ls_1p5 = [0] * n_func
    for i in range(n_func):
        i_Ls_1p5[i] = i_Ls_1[i] + (alpha * h / Inductance) * (e_1p5 - triplets_1p5[i][0])
    e_1p5 = e_1p5 / n_func

    triplets_2 = [0] * n_func
    for i in range(n_func):
        xi, gi, fi = proximal_step((R * i_Ls_1p5[i] + e_1p5), fs[i], R)
        if i == 0: e_2 = xi
        else: e_2 = e_2 + xi
        triplets_2[i] = (xi, gi, fi)
    e_2 = e_2 / n_func
    i_Ls_2 = [0] * n_func
    for i in range(n_func):
        i_Ls_2[i] = i_Ls_1[i] + (beta * h / Inductance) * (e_1p5 - triplets_1p5[i][0]) \
                              + ((1 - beta) * h / Inductance) * (e_2 - triplets_2[i][0])

    E_1 = gamma * (e_1 - x_star)**2
    E_2 = gamma * (e_2 - x_star)**2
    Delta_1 = - eta * f_star
    for i in range(n_func):
        E_1 += (Inductance/2) * (i_Ls_1[i] - ys_star[i]) ** 2
        E_2 += (Inductance/2) * (i_Ls_2[i] - ys_star[i]) ** 2
        Delta_1 += rho * R * (triplets_2[i][1] - i_Ls_2[i])**2 \
                    + eta * (triplets_2[i][2] - ys_star[i] * (triplets_2[i][0] - x_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def admm_euler_consensus(n_func, mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, eta, rho = params["h"], params["eta"], params["rho"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, eta, rho = problem.h, problem.eta, problem.rho

    fs = [0] * n_func
    for i in range(n_func):
        fs[i] = define_function(problem, mu, L_smooth, package)
        if i == 0: f = fs[i]
        else: f += fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    ys_star = [0] * n_func
    for i in range(n_func):
        gi, _ = fs[i].oracle(x_star)
        ys_star[i] = gi
        if i == 0:     
            sum_ys_star = gi
        else:
            sum_ys_star += gi 
    problem.add_constraint(Constraint((sum_ys_star - y_star) ** 2, "equality"))

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
    i_Ls_2 = [0] * n_func
    for i in range(n_func): 
        i_Ls_2[i] = i_Ls_1[i] + (h / Inductance) * (e_1 - triplets_1[i][0])
    sum_xi_1 = sum_xi_1 / n_func
    problem.add_constraint(Constraint( (e_1 - sum_xi_1) ** 2, "equality"))

    E_1 = 0; E_2 = 0
    Delta_1 = - eta * f_star
    for i in range(n_func):
        E_1 += (Inductance/2) * (i_Ls_1[i] - ys_star[i]) ** 2
        E_2 += (Inductance/2) * (i_Ls_2[i] - ys_star[i]) ** 2
        Delta_1 += rho * R * (triplets_1[i][1] - i_Ls_1[i]) ** 2 \
                 + eta *( triplets_1[i][2] - ys_star[i] * (triplets_1[i][0] - x_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def admm_consensus_proof(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func
        proximal_step = pep_proximal_step
        h, eta, rho = params["h"], params["eta"], params["rho"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        proximal_step = co_func.proximal_step
        h, eta, rho = problem.h, problem.eta, problem.rho

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

    x1_2, grad_f_2, f_2 = proximal_step(z1 - (1/rho) * y1_1, f, 1/rho)
    x2_2, grad_g_2, g_2 = proximal_step(z1 - (1/rho) * y2_1, g, 1/rho)
    z2 = (x1_2 + x2_2) / 2
    y1_2 = y1_1 - h * rho * (z2 - x1_2)
    y2_2 = y2_1 - h * rho * (z2 - x2_2)

    E_1 = (1/rho) * (y1_1 + y1_star)**2 + (1/rho) * (y2_1 + y2_star)**2 + 2 * rho * (z1 - x_star)**2 
    E_2 = (1/rho) * (y1_2 + y1_star)**2 + (1/rho) * (y2_2 + y2_star)**2 + 2 * rho * (z2 - x_star)**2 
    Delta_1 = eta * (rho * (x1_2 - z2)**2 + rho * (x2_2 - z2)**2) + rho * rho * (z1 - z2)**2 

    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem

