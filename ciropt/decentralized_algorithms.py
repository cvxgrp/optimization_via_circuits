import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt
from ciropt.utils import define_function



def decentralized_admm_consensus_l3(mu, L_smooth, R, Inductance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
        # h, alpha, b, d, gamma = params["h"], params["alpha"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, alpha, b, d, gamma = problem.h, problem.alpha, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + f3).stationary_point(return_gradient_and_function_value=True)
    y1_star, f1_star = f1.oracle(x_star)
    y2_star, f2_star = f2.oracle(x_star)
    y3_star, f3_star = f3.oracle(x_star)

    y2_21_star = problem.set_initial_point()
    y2_23_star = y2_star - y2_21_star
    # when f is not differentiable
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star - y_star) ** 2, "equality"))
    # currents on each new at equilibrium sum to 0
    problem.add_constraint(Constraint((y2_21_star + y1_star) ** 2, "equality"))
    problem.add_constraint(Constraint((y2_23_star + y3_star) ** 2, "equality"))

    e_12_1 = problem.set_initial_point()
    e_23_1 = problem.set_initial_point()
    # initialize currents on inductors to sum to 0 on every edge
    i_L_12n1_1 = problem.set_initial_point()
    i_L_23n2_1 = problem.set_initial_point()

    x1_2, y1_2, f1_2 = proximal_step((R * i_L_12n1_1 + e_12_1), f1, R)
    x2_2, y2_2, f2_2 = proximal_step(((-R * i_L_12n1_1 + e_12_1) + (R * i_L_23n2_1 + e_23_1))/2, f2, R/2)
    x3_2, y3_2, f3_2 = proximal_step((-R * i_L_23n2_1 + e_23_1), f3, R)

    e_12_2 = (x1_2 + x2_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    i_L_12n1_2 = i_L_12n1_1 + ( h / Inductance) * (e_12_2 - x1_2)
    i_L_23n2_2 = i_L_23n2_1 + ( h / Inductance) * (e_23_2 - x2_2)

    E_1 = gamma * (e_12_1 - x_star)**2 + gamma * (e_23_1 - x_star)**2 \
            + (Inductance/2) * (i_L_12n1_1 - y1_star) ** 2 \
            + (Inductance/2) * (-i_L_12n1_1 - y2_21_star) ** 2 \
            + (Inductance/2) * (i_L_23n2_1 - y2_23_star) ** 2 \
            + (Inductance/2) * (-i_L_23n2_1 - y3_star) ** 2 
    E_2 = gamma * (e_12_2 - x_star)**2 + gamma * (e_23_2 - x_star)**2 \
        + (Inductance/2) * (i_L_12n1_2 - y1_star) ** 2 \
        + (Inductance/2) * (-i_L_12n1_2 - y2_21_star) ** 2 \
        + (Inductance/2) * (i_L_23n2_2 - y2_23_star) ** 2 \
        + (Inductance/2) * (-i_L_23n2_2 - y3_star) ** 2 
    
    Delta_2 = d * (1/R) * ((e_12_2 - x1_2)**2 + (e_12_2 - x2_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_23_2 - x3_2)**2 ) \
              + b * ( f1_2 - f1_star - y1_star * (x1_2 - x_star)\
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def decentralized_gradient_descent_line3(mu, L_smooth, R, Capacitance, params=None):
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
    f3 = define_function(problem, mu, L_smooth, package)
    
    # stepsize = h / Capacitance
    x2_star = problem.set_initial_point()
    y2_star, f2_star = f2.oracle(x2_star)
    x1_star, y1_star, f1_star = proximal_step(x2_star, f1, R)
    x3_star, y3_star, f3_star = proximal_step(x2_star, f3, R)
    x2_star_v2, y2_star_v2, f2_star_v2 = proximal_step((x1_star + x3_star)/2, f2, R/2)
    problem.add_constraint(Constraint( (x2_star - x2_star_v2) ** 2, "equality"))

    x1_1 = problem.set_initial_point()
    y1_1, _ = f1.oracle(x1_1) 
    x2_1 = problem.set_initial_point()
    y2_1, _ = f2.oracle(x2_1)
    x3_1 = problem.set_initial_point()
    y3_1, _ = f3.oracle(x3_1)

    x1_2 = x1_1 - (h / Capacitance) * (y1_1 + (x1_1 - x2_1) / R ) 
    y1_2, f1_2 = f1.oracle(x1_2)
    x2_2 = x2_1 - ( h / Capacitance) * (y2_1 + (x2_1 - x1_1 + x2_1 - x3_1) / R ) 
    y2_2, f2_2 = f2.oracle(x2_2)
    x3_2 = x3_1 - ( h / Capacitance) * (y3_1 + (x3_1 - x2_1) / R )
    y3_2, f3_2 = f3.oracle(x3_2)

    E_1 = (Capacitance/2) * ((x1_1 - x1_star)**2 + (x2_1 - x2_star)**2 + (x3_1 - x3_star)**2)
    E_2 = (Capacitance/2) * ((x1_2 - x1_star)**2 + (x2_2 - x2_star)**2 + (x3_2 - x3_star)**2) 

    Delta_2 = d * (1 / R) * ((x1_2 - x2_2 - (x1_star - x2_star))**2 + (x3_2 - x2_2 - (x3_star - x2_star))**2) + \
              b * ((x1_2 - x1_star) * (y1_2 - y1_star) + (x2_2 - x2_star) * (y2_2 - y2_star) 
                                                        + (x3_2 - x3_star) * (y3_2 - y3_star))
    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem

