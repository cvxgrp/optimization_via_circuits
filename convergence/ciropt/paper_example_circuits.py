import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt
from ciropt.utils import define_function




def example_fC3RC_circuit(mu, L_smooth, Capacitance, R, params=None): 
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

    z_1 = problem.set_initial_point()
    e2_1 = problem.set_initial_point()
    x_1, _, _ = proximal_step(z_1, func, R/2)
    y_1 = (2 / R) * (z_1 - x_1)


    e2_1p5 = e2_1 - (alpha * h / (2 * R * Capacitance)) * (R * y_1 + 3 * e2_1)
    z_1p5 = z_1 - (alpha * h / (4 * R * Capacitance)) * (5 * R * y_1 + 3 * e2_1)
    x_1p5, _, _ = proximal_step(z_1p5, func, R/2)
    y_1p5 = (2 / R) * (z_1p5 - x_1p5)

    e2_2 = e2_1  - (beta * h / (2 * R * Capacitance)) *  (R * y_1 + 3 * e2_1)  \
            - ((1 - beta) * h / (2 * R * Capacitance)) * (R * y_1p5 + 3 * e2_1p5)
    z_2 = z_1  - (beta * h / (4 * R * Capacitance)) *  (5 * R * y_1 + 3 * e2_1)  \
            - ((1 - beta) * h / (4 * R * Capacitance)) * (5 * R * y_1p5 + 3 *  e2_1p5)
    x_2, _, _ = proximal_step(z_2, func, R/2)
    y_2 = (2 / R) * (z_2 - x_2)

    v_C1_1 = e2_1 / 2 - z_1
    v_C1_2 = e2_2 / 2 - z_2
    v_C2_1 = e2_1 
    v_C2_2 = e2_2 
    e1_1 = (e2_1 - R * y_1) / 2
    E_1 = (Capacitance/2) * (v_C1_1 + x_star)**2 + (Capacitance/2) * (v_C2_1)**2
    E_2 = (Capacitance/2) * (v_C1_2 + x_star)**2 + (Capacitance/2) * (v_C2_2)**2
    Delta_1 = b * (x_1 - x_star) * (y_1 - y_star) \
            + d * (1/R) * ((e1_1)**2 + (e1_1 - e2_1)**2 + (e2_1)**2)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def example_fCR_circuit(mu, L_smooth, Capacitance, R, params=None): 
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

    v_C_1 = problem.set_initial_point()
    x_1, y_1, f_1 = proximal_step(v_C_1, func, R)

    v_C_1p5 = v_C_1 - (alpha * h / Capacitance) * y_1
    x_1p5, y_1p5, f_1p5 = proximal_step(v_C_1p5, func, R)

    v_C_2 = v_C_1  - (beta * h / Capacitance) * y_1  \
               - ((1 - beta) * h / Capacitance) * y_1p5
    x_2, y_2, f_2 = proximal_step(v_C_2, func, R)

    E_1 = (Capacitance/2) * (v_C_1 - x_star)**2
    E_2 = (Capacitance/2) * (v_C_2 - x_star)**2
    Delta_1 = b * (x_1 - x_star) * (y_1 - y_star) + d * R * (y_1)**2
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem
