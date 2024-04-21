import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt
from ciropt.utils import define_function




def dadmm_L_graph6(mu, L_smooth, R, L_extra, Inductance, params=None):
    # graph with 6 nodes and 7 edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    # 1 -- 2 -- 4 -- 5
    #   \  |  /  \
    #      3       6 
    # D-ADMM with inductor between nodes x4 and x5
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)
    f4 = define_function(problem, mu, L_smooth, package)
    f5 = define_function(problem, mu, L_smooth, package)
    f6 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + f4 + f5 + f6).stationary_point(return_gradient_and_function_value=True)
    y1_star, f1_star = f1.oracle(x_star)
    y2_star, f2_star = f2.oracle(x_star)
    y3_star, f3_star = f3.oracle(x_star)
    y4_star, f4_star = f4.oracle(x_star)
    y5_star, f5_star = f5.oracle(x_star)
    y6_star, f6_star = f6.oracle(x_star)
    # when f is not differentiable
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star + y4_star + y5_star + y6_star - y_star) ** 2, "equality"))

    y1_12_star = problem.set_initial_point()
    y1_13_star = y1_star - y1_12_star
    y2_23_star = problem.set_initial_point()
    y2_24_star = y2_star + y1_12_star - y2_23_star
    y3_34_star = y3_star + y1_13_star + y2_23_star
    i_L_star = problem.set_initial_point()
    y5_54_star = y5_star - i_L_star
    # y5_54_star + i_L = y5_star;     y6_64_star = y6_star
    # currents on each new at equilibrium sum to 0
    problem.add_constraint(Constraint((y2_24_star + y3_34_star + y5_star + y6_star + y4_star) ** 2, "equality"))

    # edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    e_12_1 = problem.set_initial_point()
    e_13_1 = problem.set_initial_point()
    e_23_1 = problem.set_initial_point()
    e_24_1 = problem.set_initial_point()
    e_34_1 = problem.set_initial_point()
    e_45_1 = problem.set_initial_point()
    e_46_1 = problem.set_initial_point()
    # initialize currents on inductors to sum to 0 on every edge
    i_L_12_1 = problem.set_initial_point()
    i_L_13_1 = problem.set_initial_point()
    i_L_23_1 = problem.set_initial_point()
    i_L_24_1 = problem.set_initial_point()
    i_L_34_1 = problem.set_initial_point()
    i_L_45_1 = problem.set_initial_point()
    i_L_46_1 = problem.set_initial_point()
    i_L_1 = problem.set_initial_point() # extra current x4 -> x5
    # x5_22 = problem.set_initial_point()
    x1_2, y1_2, f1_2 = proximal_step((1/2)*((R * i_L_12_1 + e_12_1) + (R * i_L_13_1 + e_13_1)), f1, R/2)
    x2_2, y2_2, f2_2 = proximal_step((1/3)*((-R * i_L_12_1 + e_12_1) + (R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_24_1 + e_24_1)), f2, R/3)
    x3_2, y3_2, f3_2 = proximal_step((1/3)*((-R * i_L_13_1 + e_13_1) + (-R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_34_1 + e_34_1)), f3, R/3)
    x4_2, y4_2, f4_2 = proximal_step((1/4)*((-R * i_L_24_1 + e_24_1) + (-R * i_L_34_1 + e_34_1) \
                                            + (R * i_L_45_1 + e_45_1) + (R * i_L_46_1 + e_46_1) - R * i_L_1), f4, R/4)
    x5_2, y5_2, f5_2 = proximal_step((-R * i_L_45_1 + e_45_1 + R * i_L_1), f5, R)
    x6_2, y6_2, f6_2 = proximal_step(-R * i_L_46_1 + e_46_1, f6, R)
    # problem.add_constraint(Constraint((x5_2 - x5_22) ** 2, "equality"))

    e_12_2 = (x1_2 + x2_2) / 2
    e_13_2 = (x1_2 + x3_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    e_24_2 = (x2_2 + x4_2) / 2
    e_34_2 = (x3_2 + x4_2) / 2
    e_45_2 = (x4_2 + x5_2) / 2
    e_46_2 = (x4_2 + x6_2) / 2
    i_L_12_2 = i_L_12_1 + ( h / Inductance) * (e_12_2 - x1_2)      
    i_L_13_2 = i_L_13_1 + ( h / Inductance) * (e_13_2 - x1_2)
    i_L_23_2 = i_L_23_1 + ( h / Inductance) * (e_23_2 - x2_2)
    i_L_24_2 = i_L_24_1 + ( h / Inductance) * (e_24_2 - x2_2)
    i_L_34_2 = i_L_34_1 + ( h / Inductance) * (e_34_2 - x3_2)
    i_L_45_2 = i_L_45_1 + ( h / Inductance) * (e_45_2 - x4_2)
    i_L_46_2 = i_L_46_1 + ( h / Inductance) * (e_46_2 - x4_2)
    i_L_2 = i_L_1 + ( h / L_extra) * (x4_2 - x5_2)
    
    # energy of two inductors on each net is twice the energy on one inductor
    E_1 = gamma * ((e_12_1 - x_star)**2 + (e_13_1 - x_star)**2 + (e_23_1 - x_star)**2 + (e_24_1 - x_star)**2\
                   + (e_34_1 - x_star)**2 + (e_45_1 - x_star)**2 + (e_46_1 - x_star)**2) +  \
            + Inductance * (i_L_12_1 - y1_12_star) ** 2 + Inductance * (i_L_13_1 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_1 - y2_23_star) ** 2 + Inductance * (i_L_24_1 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_1 - y3_34_star) ** 2 + Inductance * (i_L_45_1 + y5_54_star) ** 2 \
            + Inductance * (i_L_46_1 + y6_star) ** 2 + (L_extra/2) * (i_L_1 - i_L_star) ** 2
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_13_2 - x_star)**2 + (e_23_2 - x_star)**2 + (e_24_2 - x_star)**2\
                   + (e_34_2 - x_star)**2 + (e_45_2 - x_star)**2 + (e_46_2 - x_star)**2) +  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_13_2 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_2 - y2_23_star) ** 2 + Inductance * (i_L_24_2 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 + Inductance * (i_L_45_2 + y5_54_star) ** 2 \
            + Inductance * (i_L_46_2 + y6_star) ** 2  + (L_extra/2) * (i_L_2 - i_L_star) ** 2
    # currents on resistors on each net sum to 0
    Delta_2 = d * ((2/R) * ((e_12_2 - x1_2)**2 + (e_13_2 - x3_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_24_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2 + (e_45_2 - x5_2)**2 + (e_46_2 - x6_2)**2 ))  \
              + b * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star) \
                    + f5_2 - f5_star - y5_star * (x5_2 - x_star) \
                    + f6_2 - f6_star - y6_star * (x6_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def dadmm_R_graph6(mu, L_smooth, R, R_extra, Inductance, params=None):
    # graph with 6 nodes and 7 edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    # 1 -- 2 -- 4 -- 5
    #   \  |  /  \
    #      3       6 
    # D-ADMM with resistor between nodes x4 and x5
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)
    f4 = define_function(problem, mu, L_smooth, package)
    f5 = define_function(problem, mu, L_smooth, package)
    f6 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + f4 + f5 + f6).stationary_point(return_gradient_and_function_value=True)
    y1_star, f1_star = f1.oracle(x_star)
    y2_star, f2_star = f2.oracle(x_star)
    y3_star, f3_star = f3.oracle(x_star)
    y4_star, f4_star = f4.oracle(x_star)
    y5_star, f5_star = f5.oracle(x_star)
    y6_star, f6_star = f6.oracle(x_star)
    # when f is not differentiable
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star + y4_star + y5_star + y6_star - y_star) ** 2, "equality"))

    y1_12_star = problem.set_initial_point()
    y1_13_star = y1_star - y1_12_star
    y2_23_star = problem.set_initial_point()
    y2_24_star = y2_star + y1_12_star - y2_23_star
    y3_34_star = y3_star + y1_13_star + y2_23_star
    # y5_54_star = y5_star;     y6_64_star = y6_star
    # currents on each new at equilibrium sum to 0
    problem.add_constraint(Constraint((y2_24_star + y3_34_star + y5_star + y6_star + y4_star) ** 2, "equality"))

    # edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    e_12_1 = problem.set_initial_point()
    e_13_1 = problem.set_initial_point()
    e_23_1 = problem.set_initial_point()
    e_24_1 = problem.set_initial_point()
    e_34_1 = problem.set_initial_point()
    e_45_1 = problem.set_initial_point()
    e_46_1 = problem.set_initial_point()
    # initialize currents on inductors to sum to 0 on every edge
    i_L_12_1 = problem.set_initial_point()
    i_L_13_1 = problem.set_initial_point()
    i_L_23_1 = problem.set_initial_point()
    i_L_24_1 = problem.set_initial_point()
    i_L_34_1 = problem.set_initial_point()
    i_L_45_1 = problem.set_initial_point()
    i_L_46_1 = problem.set_initial_point()
    x5_22 = problem.set_initial_point()
    x1_2, y1_2, f1_2 = proximal_step((1/2)*((R * i_L_12_1 + e_12_1) + (R * i_L_13_1 + e_13_1)), f1, R/2)
    x2_2, y2_2, f2_2 = proximal_step((1/3)*((-R * i_L_12_1 + e_12_1) + (R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_24_1 + e_24_1)), f2, R/3)
    x3_2, y3_2, f3_2 = proximal_step((1/3)*((-R * i_L_13_1 + e_13_1) + (-R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_34_1 + e_34_1)), f3, R/3)
    c = 4/R + 1./R_extra
    x4_2, y4_2, f4_2 = proximal_step((1/c)*((-R * i_L_24_1 + e_24_1)/R + (-R * i_L_34_1 + e_34_1)/R \
                                            + (R * i_L_45_1 + e_45_1)/R + (R * i_L_46_1 + e_46_1)/R + x5_22/R_extra), f4, 1/c)
    c = 1/R + 1./R_extra
    x5_2, y5_2, f5_2 = proximal_step((1/c) * ((-R * i_L_45_1 + e_45_1) / R + x4_2 / R_extra), f5, 1./c)
    x6_2, y6_2, f6_2 = proximal_step(-R * i_L_46_1 + e_46_1, f6, R)
    problem.add_constraint(Constraint((x5_2 - x5_22) ** 2, "equality"))

    e_12_2 = (x1_2 + x2_2) / 2
    e_13_2 = (x1_2 + x3_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    e_24_2 = (x2_2 + x4_2) / 2
    e_34_2 = (x3_2 + x4_2) / 2
    e_45_2 = (x4_2 + x5_2) / 2
    e_46_2 = (x4_2 + x6_2) / 2
    i_L_12_2 = i_L_12_1 + ( h / Inductance) * (e_12_2 - x1_2)      
    i_L_13_2 = i_L_13_1 + ( h / Inductance) * (e_13_2 - x1_2)
    i_L_23_2 = i_L_23_1 + ( h / Inductance) * (e_23_2 - x2_2)
    i_L_24_2 = i_L_24_1 + ( h / Inductance) * (e_24_2 - x2_2)
    i_L_34_2 = i_L_34_1 + ( h / Inductance) * (e_34_2 - x3_2)
    i_L_45_2 = i_L_45_1 + ( h / Inductance) * (e_45_2 - x4_2)
    i_L_46_2 = i_L_46_1 + ( h / Inductance) * (e_46_2 - x4_2)
    
    # energy of two inductors on each net is twice the energy on one inductor
    E_1 = gamma * ((e_12_1 - x_star)**2 + (e_13_1 - x_star)**2 + (e_23_1 - x_star)**2 + (e_24_1 - x_star)**2\
                   + (e_34_1 - x_star)**2 + (e_45_1 - x_star)**2 + (e_46_1 - x_star)**2) +  \
            + Inductance * (i_L_12_1 - y1_12_star) ** 2 + Inductance * (i_L_13_1 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_1 - y2_23_star) ** 2 + Inductance * (i_L_24_1 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_1 - y3_34_star) ** 2 + Inductance * (i_L_45_1 + y5_star) ** 2 \
            + Inductance * (i_L_46_1 + y6_star) ** 2
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_13_2 - x_star)**2 + (e_23_2 - x_star)**2 + (e_24_2 - x_star)**2\
                   + (e_34_2 - x_star)**2 + (e_45_2 - x_star)**2 + (e_46_2 - x_star)**2) +  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_13_2 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_2 - y2_23_star) ** 2 + Inductance * (i_L_24_2 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 + Inductance * (i_L_45_2 + y5_star) ** 2 \
            + Inductance * (i_L_46_2 + y6_star) ** 2  
    # currents on resistors on each net sum to 0
    Delta_2 = d * ((2/R) * ((e_12_2 - x1_2)**2 + (e_13_2 - x3_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_24_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2 + (e_45_2 - x5_2)**2 + (e_46_2 - x6_2)**2 ) \
                    + (1/R) * (x4_2 - x5_2)**2)  \
              + b * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star) \
                    + f5_2 - f5_star - y5_star * (x5_2 - x_star) \
                    + f6_2 - f6_star - y6_star * (x6_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def dadmm_C_graph6(mu, L_smooth, R, Capacitance, Inductance, params=None):
    # graph with 6 nodes and 7 edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    # 1 -- 2 -- 4 -- 5
    #   \  |  /  \
    #      3       6 
    # D-ADMM with Capacitor at e_45
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)
    f4 = define_function(problem, mu, L_smooth, package)
    f5 = define_function(problem, mu, L_smooth, package)
    f6 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + f4 + f5 + f6).stationary_point(return_gradient_and_function_value=True)
    y1_star, f1_star = f1.oracle(x_star)
    y2_star, f2_star = f2.oracle(x_star)
    y3_star, f3_star = f3.oracle(x_star)
    y4_star, f4_star = f4.oracle(x_star)
    y5_star, f5_star = f5.oracle(x_star)
    y6_star, f6_star = f6.oracle(x_star)
    # when f is not differentiable
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star + y4_star + y5_star + y6_star - y_star) ** 2, "equality"))

    y1_12_star = problem.set_initial_point()
    y1_13_star = y1_star - y1_12_star
    y2_23_star = problem.set_initial_point()
    y2_24_star = y2_star + y1_12_star - y2_23_star
    y3_34_star = y3_star + y1_13_star + y2_23_star
    # y5_54_star = y5_star;     y6_64_star = y6_star
    # currents on each new at equilibrium sum to 0
    problem.add_constraint(Constraint((y2_24_star + y3_34_star + y5_star + y6_star + y4_star) ** 2, "equality"))

    # edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    e_12_1 = problem.set_initial_point()
    e_13_1 = problem.set_initial_point()
    e_23_1 = problem.set_initial_point()
    e_24_1 = problem.set_initial_point()
    e_34_1 = problem.set_initial_point()
    e_45_1 = problem.set_initial_point()
    e_46_1 = problem.set_initial_point()
    # initialize currents on inductors to sum to 0 on every edge
    i_L_12_1 = problem.set_initial_point()
    i_L_13_1 = problem.set_initial_point()
    i_L_23_1 = problem.set_initial_point()
    i_L_24_1 = problem.set_initial_point()
    i_L_34_1 = problem.set_initial_point()
    i_L_45_1 = problem.set_initial_point()
    i_L_54_1 = problem.set_initial_point()
    i_L_46_1 = problem.set_initial_point()


    x1_2, y1_2, f1_2 = proximal_step((1/2)*((R * i_L_12_1 + e_12_1) + (R * i_L_13_1 + e_13_1)), f1, R/2)
    x2_2, y2_2, f2_2 = proximal_step((1/3)*((-R * i_L_12_1 + e_12_1) + (R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_24_1 + e_24_1)), f2, R/3)
    x3_2, y3_2, f3_2 = proximal_step((1/3)*((-R * i_L_13_1 + e_13_1) + (-R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_34_1 + e_34_1)), f3, R/3)
    x4_2, y4_2, f4_2 = proximal_step((1/4)*((-R * i_L_24_1 + e_24_1) + (-R * i_L_34_1 + e_34_1) \
                                            + (R * i_L_45_1 + e_45_1) + (R * i_L_46_1 + e_46_1)), f4, R/4)
    x5_2, y5_2, f5_2 = proximal_step(R * i_L_54_1 + e_45_1, f5, R)
    x6_2, y6_2, f6_2 = proximal_step(-R * i_L_46_1 + e_46_1, f6, R)

    e_12_2 = (x1_2 + x2_2) / 2
    e_13_2 = (x1_2 + x3_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    e_24_2 = (x2_2 + x4_2) / 2
    e_34_2 = (x3_2 + x4_2) / 2
    e_45_2 = e_45_1 - ( h / Capacitance) * (i_L_45_1 + i_L_54_1 + (2 * e_45_1 - x4_2 - x5_2) / R)
    e_46_2 = (x4_2 + x6_2) / 2
    i_L_12_2 = i_L_12_1 + ( h / Inductance) * (e_12_2 - x1_2)      
    i_L_13_2 = i_L_13_1 + ( h / Inductance) * (e_13_2 - x1_2)
    i_L_23_2 = i_L_23_1 + ( h / Inductance) * (e_23_2 - x2_2)
    i_L_24_2 = i_L_24_1 + ( h / Inductance) * (e_24_2 - x2_2)
    i_L_34_2 = i_L_34_1 + ( h / Inductance) * (e_34_2 - x3_2)
    i_L_45_2 = i_L_45_1 + ( h / Inductance) * (e_45_2 - x4_2)
    i_L_54_2 = i_L_54_1 + ( h / Inductance) * (e_45_2 - x5_2)
    i_L_46_2 = i_L_46_1 + ( h / Inductance) * (e_46_2 - x4_2)
    
    # energy of two inductors on each net is twice the energy on one inductor
    E_1 = gamma * ((e_12_1 - x_star)**2 + (e_13_1 - x_star)**2 + (e_23_1 - x_star)**2 + (e_24_1 - x_star)**2\
                   + (e_34_1 - x_star)**2 + (e_46_1 - x_star)**2) +  \
            + Inductance * (i_L_12_1 - y1_12_star) ** 2 + Inductance * (i_L_13_1 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_1 - y2_23_star) ** 2 + Inductance * (i_L_24_1 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_1 - y3_34_star) ** 2 \
            + (Inductance/2) * (i_L_45_1 - y4_star) ** 2 + (Inductance/2) * (i_L_54_1 - y5_star) ** 2 \
            + Inductance * (i_L_46_1 + y6_star) ** 2 \
            + (Capacitance / 2) * (e_45_1 - x_star) ** 2
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_13_2 - x_star)**2 + (e_23_2 - x_star)**2 + (e_24_2 - x_star)**2\
                   + (e_34_2 - x_star)**2 + (e_46_2 - x_star)**2) +  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_13_2 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_2 - y2_23_star) ** 2 + Inductance * (i_L_24_2 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 \
            + (Inductance/2) * (i_L_45_2 - y4_star) ** 2 + (Inductance/2) * (i_L_54_2 - y5_star) ** 2 \
            + Inductance * (i_L_46_2 + y6_star) ** 2 \
            + (Capacitance / 2) * (e_45_2 - x_star) ** 2
    # currents on resistors on each net sum to 0
    Delta_2 = d * ((2/R) * ((e_12_2 - x1_2)**2 + (e_13_2 - x3_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_24_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2  + (e_46_2 - x6_2)**2 )  \
                    + (1/R) * ((e_45_2 - x5_2)**2 + (e_45_2 - x4_2)**2)) \
              + b * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star) \
                    + f5_2 - f5_star - y5_star * (x5_2 - x_star) \
                    + f6_2 - f6_star - y6_star * (x6_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def decentralized_admm_cycle4(mu, L_smooth, R, Inductance, params=None):
    # cycle graph with 4 nodes and 4 edges (1,2), (2,3), (3,4), (4, 1)
    # 1 -- 2 -- 3 
    #   \     /  
    #      4        
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)
    f4 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + f4).stationary_point(return_gradient_and_function_value=True)
    y1_star, f1_star = f1.oracle(x_star)
    y2_star, f2_star = f2.oracle(x_star)
    y3_star, f3_star = f3.oracle(x_star)
    y4_star, f4_star = f4.oracle(x_star)
    # when f is not differentiable
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star + y4_star - y_star) ** 2, "equality"))

    y1_12_star = problem.set_initial_point()
    y1_14_star = y1_star - y1_12_star
    y2_23_star = y2_star + y1_12_star
    y3_34_star = y3_star + y2_23_star
    # currents on each new at equilibrium sum to 0
    problem.add_constraint(Constraint((y1_14_star + y3_34_star + y4_star) ** 2, "equality"))

    # edges (1,2), (2,3), (3, 4), (4, 1)
    e_12_1 = problem.set_initial_point()
    e_23_1 = problem.set_initial_point()
    e_34_1 = problem.set_initial_point()
    e_41_1 = problem.set_initial_point()
    # initialize currents on inductors to sum to 0 on every edge
    i_L_12_1 = problem.set_initial_point()
    i_L_23_1 = problem.set_initial_point()
    i_L_34_1 = problem.set_initial_point()
    i_L_41_1 = problem.set_initial_point()

    x1_2, y1_2, f1_2 = proximal_step((1/2)*((R * i_L_12_1 + e_12_1) + (-R * i_L_41_1 + e_41_1)), f1, R/2)
    x2_2, y2_2, f2_2 = proximal_step((1/2)*((-R * i_L_12_1 + e_12_1) + (R * i_L_23_1 + e_23_1)), f2, R/2)
    x3_2, y3_2, f3_2 = proximal_step((1/2)*((-R * i_L_23_1 + e_23_1) + (R * i_L_34_1 + e_34_1)), f3, R/2)
    x4_2, y4_2, f4_2 = proximal_step((1/2)*((R * i_L_41_1 + e_41_1) + (-R * i_L_34_1 + e_34_1)), f4, R/2)


    e_12_2 = (x1_2 + x2_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    e_34_2 = (x3_2 + x4_2) / 2
    e_41_2 = (x4_2 + x1_2) / 2
    i_L_12_2 = i_L_12_1 + ( h / Inductance) * (e_12_2 - x1_2)   
    i_L_23_2 = i_L_23_1 + ( h / Inductance) * (e_23_2 - x2_2)
    i_L_34_2 = i_L_34_1 + ( h / Inductance) * (e_34_2 - x3_2)
    i_L_41_2 = i_L_41_1 + ( h / Inductance) * (e_41_2 - x4_2)
    
    # energy of two inductors on each net is twice the energy on one inductor
    E_1 = gamma * ((e_12_1 - x_star)**2 + (e_23_1 - x_star)**2 \
                 + (e_34_1 - x_star)**2 + (e_41_1 - x_star)**2)  \
            + Inductance * (i_L_12_1 - y1_12_star) ** 2 + Inductance * (i_L_23_1 - y2_23_star) ** 2 \
            + Inductance * (i_L_34_1 - y3_34_star) ** 2 + Inductance * (i_L_41_1 + y1_14_star) ** 2 
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_23_2 - x_star)**2 \
                 + (e_34_2 - x_star)**2 + (e_41_2 - x_star)**2)  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_23_2 - y2_23_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 + Inductance * (i_L_41_2 + y1_14_star) ** 2
    # currents on resistors on each net sum to 0
    Delta_2 = d * (2/R) * ((e_12_2 - x1_2)**2 + (e_23_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2 + (e_41_2 - x1_2)**2 )  \
              + b * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def decentralized_admm_graph6(mu, L_smooth, R, Inductance, params=None):
    # graph with 6 nodes and 7 edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    # 1 -- 2 -- 4 -- 5
    #   \  |  /  \
    #      3       6 
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    f3 = define_function(problem, mu, L_smooth, package)
    f4 = define_function(problem, mu, L_smooth, package)
    f5 = define_function(problem, mu, L_smooth, package)
    f6 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + f4 + f5 + f6).stationary_point(return_gradient_and_function_value=True)
    y1_star, f1_star = f1.oracle(x_star)
    y2_star, f2_star = f2.oracle(x_star)
    y3_star, f3_star = f3.oracle(x_star)
    y4_star, f4_star = f4.oracle(x_star)
    y5_star, f5_star = f5.oracle(x_star)
    y6_star, f6_star = f6.oracle(x_star)
    # when f is not differentiable
    problem.add_constraint(Constraint((y1_star + y2_star + y3_star + y4_star + y5_star + y6_star - y_star) ** 2, "equality"))

    y1_12_star = problem.set_initial_point()
    y1_13_star = y1_star - y1_12_star
    y2_23_star = problem.set_initial_point()
    y2_24_star = y2_star + y1_12_star - y2_23_star
    y3_34_star = y3_star + y1_13_star + y2_23_star
    # y5_54_star = y5_star, y6_64_star = y6_star
    # currents on each new at equilibrium sum to 0
    problem.add_constraint(Constraint((y2_24_star + y3_34_star + y5_star + y6_star + y4_star) ** 2, "equality"))

    # edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    e_12_1 = problem.set_initial_point()
    e_13_1 = problem.set_initial_point()
    e_23_1 = problem.set_initial_point()
    e_24_1 = problem.set_initial_point()
    e_34_1 = problem.set_initial_point()
    e_45_1 = problem.set_initial_point()
    e_46_1 = problem.set_initial_point()
    # initialize currents on inductors to sum to 0 on every edge
    i_L_12_1 = problem.set_initial_point()
    i_L_13_1 = problem.set_initial_point()
    i_L_23_1 = problem.set_initial_point()
    i_L_24_1 = problem.set_initial_point()
    i_L_34_1 = problem.set_initial_point()
    i_L_45_1 = problem.set_initial_point()
    i_L_46_1 = problem.set_initial_point()

    x1_2, y1_2, f1_2 = proximal_step((1/2)*((R * i_L_12_1 + e_12_1) + (R * i_L_13_1 + e_13_1)), f1, R/2)
    x2_2, y2_2, f2_2 = proximal_step((1/3)*((-R * i_L_12_1 + e_12_1) + (R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_24_1 + e_24_1)), f2, R/3)
    x3_2, y3_2, f3_2 = proximal_step((1/3)*((-R * i_L_13_1 + e_13_1) + (-R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_34_1 + e_34_1)), f3, R/3)
    x4_2, y4_2, f4_2 = proximal_step((1/4)*((-R * i_L_24_1 + e_24_1) + (-R * i_L_34_1 + e_34_1) \
                                            + (R * i_L_45_1 + e_45_1) + (R * i_L_46_1 + e_46_1)), f4, R/4)
    x5_2, y5_2, f5_2 = proximal_step(-R * i_L_45_1 + e_45_1, f5, R)
    x6_2, y6_2, f6_2 = proximal_step(-R * i_L_46_1 + e_46_1, f6, R)

    e_12_2 = (x1_2 + x2_2) / 2
    e_13_2 = (x1_2 + x3_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    e_24_2 = (x2_2 + x4_2) / 2
    e_34_2 = (x3_2 + x4_2) / 2
    e_45_2 = (x4_2 + x5_2) / 2
    e_46_2 = (x4_2 + x6_2) / 2
    i_L_12_2 = i_L_12_1 + ( h / Inductance) * (e_12_2 - x1_2)      
    i_L_13_2 = i_L_13_1 + ( h / Inductance) * (e_13_2 - x1_2)
    i_L_23_2 = i_L_23_1 + ( h / Inductance) * (e_23_2 - x2_2)
    i_L_24_2 = i_L_24_1 + ( h / Inductance) * (e_24_2 - x2_2)
    i_L_34_2 = i_L_34_1 + ( h / Inductance) * (e_34_2 - x3_2)
    i_L_45_2 = i_L_45_1 + ( h / Inductance) * (e_45_2 - x4_2)
    i_L_46_2 = i_L_46_1 + ( h / Inductance) * (e_46_2 - x4_2)
    
    # energy of two inductors on each net is twice the energy on one inductor
    E_1 = gamma * ((e_12_1 - x_star)**2 + (e_13_1 - x_star)**2 + (e_23_1 - x_star)**2 + (e_24_1 - x_star)**2\
                   + (e_34_1 - x_star)**2 + (e_45_1 - x_star)**2 + (e_46_1 - x_star)**2) +  \
            + Inductance * (i_L_12_1 - y1_12_star) ** 2 + Inductance * (i_L_13_1 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_1 - y2_23_star) ** 2 + Inductance * (i_L_24_1 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_1 - y3_34_star) ** 2 + Inductance * (i_L_45_1 + y5_star) ** 2 \
            + Inductance * (i_L_46_1 + y6_star) ** 2
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_13_2 - x_star)**2 + (e_23_2 - x_star)**2 + (e_24_2 - x_star)**2\
                   + (e_34_2 - x_star)**2 + (e_45_2 - x_star)**2 + (e_46_2 - x_star)**2) +  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_13_2 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_2 - y2_23_star) ** 2 + Inductance * (i_L_24_2 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 + Inductance * (i_L_45_2 + y5_star) ** 2 \
            + Inductance * (i_L_46_2 + y6_star) ** 2  
    # currents on resistors on each net sum to 0
    Delta_2 = d * (2/R) * ((e_12_2 - x1_2)**2 + (e_13_2 - x3_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_24_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2 + (e_45_2 - x5_2)**2 + (e_46_2 - x6_2)**2 )  \
              + b * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star) \
                    + f5_2 - f5_star - y5_star * (x5_2 - x_star) \
                    + f6_2 - f6_star - y6_star * (x6_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def decentralized_admm_consensus_l3(mu, L_smooth, R, Inductance, params=None):
    # line graph with 3 nodes
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP()
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        # h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
        h, alpha, beta, b, d, gamma = params["h"], params["alpha"], params["beta"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, alpha, beta, b, d, gamma = problem.h, problem.alpha, problem.beta, problem.b, problem.d, problem.gamma

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

    x1_1p5, y1_1p5, f1_1p5 = proximal_step((R * i_L_12n1_1 + e_12_1), f1, R)
    x2_1p5, y2_1p5, f2_1p5 = proximal_step(((-R * i_L_12n1_1 + e_12_1) + (R * i_L_23n2_1 + e_23_1))/2, f2, R/2)
    x3_1p5, y3_1p5, f3_1p5 = proximal_step((-R * i_L_23n2_1 + e_23_1), f3, R)

    e_12_1p5 = (x1_1p5 + x2_1p5) / 2
    e_23_1p5 = (x2_1p5 + x3_1p5) / 2
    i_L_12n1_1p5 = i_L_12n1_1 + ( alpha * h / Inductance) * (e_12_1p5 - x1_1p5)
    i_L_23n2_1p5 = i_L_23n2_1 + ( alpha * h / Inductance) * (e_23_1p5 - x2_1p5)

    x1_2, y1_2, f1_2 = proximal_step((R * i_L_12n1_1p5 + e_12_1p5), f1, R)
    x2_2, y2_2, f2_2 = proximal_step(((-R * i_L_12n1_1p5 + e_12_1p5) + (R * i_L_23n2_1p5 + e_23_1p5))/2, f2, R/2)
    x3_2, y3_2, f3_2 = proximal_step((-R * i_L_23n2_1p5 + e_23_1p5), f3, R)

    e_12_2 = (x1_2 + x2_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    i_L_12n1_2 = i_L_12n1_1 + ( beta * h / Inductance) * (e_12_1p5 - x1_1p5) \
                            + ( (1-beta) * h / Inductance) * (e_12_2 - x1_2)      
    i_L_23n2_2 = i_L_23n2_1 + ( beta * h / Inductance) * (e_23_1p5 - x2_1p5) \
                            + ( (1-beta) * h / Inductance) * (e_23_2 - x2_2)
    
    # y1_12_star = y1_star; y3_32_star = y3_star
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


def diffusion_line3(mu, L_smooth, R, Capacitance, params=None):
    """
    Mixing matrix for a line graph 
    W = [[1/3, 2/3, 0], 
         [1/3, 1/3, 1/3],
         [0, 1/3, 2/3]]
    """
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
    
    x1_star = problem.set_initial_point()
    x2_star = problem.set_initial_point()
    x3_star = problem.set_initial_point()
    y1_star, f1_star = f1.oracle(x1_star)
    y2_star, f2_star = f2.oracle(x2_star)
    y3_star, f3_star = f3.oracle(x3_star)
    e1_star = x1_star - R * y1_star
    e2_star = x2_star - R * y2_star
    e3_star = x3_star - R * y3_star
    problem.add_constraint(Constraint( ((e1_star - 2 * e2_star + e3_star) / (3 * R) - y2_star) ** 2, "equality"))
    problem.add_constraint(Constraint(((e2_star - e1_star) / (3 * R) - y1_star) ** 2, "equality"))
    problem.add_constraint(Constraint(((e2_star - e3_star) / (3 * R) - y3_star) ** 2, "equality"))

    x1_1 = problem.set_initial_point()
    y1_1, _ = f1.oracle(x1_1) 
    x2_1 = problem.set_initial_point()
    y2_1, _ = f2.oracle(x2_1)
    x3_1 = problem.set_initial_point()
    y3_1, _ = f3.oracle(x3_1)
    e1_1 = x1_1 - R * y1_1
    e2_1 = x2_1 - R * y2_1
    e3_1 = x3_1 - R * y3_1

    x1_2 = x1_1 - (h / Capacitance) * (y1_1 + (1/(3*R)) * ((x1_1 - R * y1_1) - (x2_1 - R * y2_1)))
    y1_2, f1_2 = f1.oracle(x1_2)
    x2_2 = x2_1 - (h / Capacitance) * (y2_1 + (1/(3*R)) * (2 * (x2_1 - R * y2_1) - (x1_1 - R * y1_1) - (x3_1 - R * y3_1)))
    y2_2, f2_2 = f2.oracle(x2_2)
    x3_2 = x3_1 - (h / Capacitance) * (y3_1 + (1/(3*R)) * ((x3_1 - R * y3_1) - (x2_1 - R * y2_1)))
    y3_2, f3_2 = f3.oracle(x3_2)
    e1_2 = x1_2 - R * y1_2
    e2_2 = x2_2 - R * y2_2
    e3_2 = x3_2 - R * y3_2

    E_1 = (Capacitance/2) * ((e1_1 - e1_star)**2 + (e2_1 - e2_star)**2 + (e3_1 - e3_star)**2)
    E_2 = (Capacitance/2) * ((e1_2 - e1_star)**2 + (e2_2 - e2_star)**2 + (e3_2 - e3_star)**2) 

    Delta_1 = d * (1 / (3 * R)) * ((e1_1 - e2_1 - (e1_star - e2_star))**2 + (e3_1 - e2_1 - (e3_star - e2_star))**2) + \
              b * (1/L_smooth - R) * ((y1_1 - y1_star)**2 + (y2_1 - y2_star)**2 + (y3_1 - y3_star)**2)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
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
