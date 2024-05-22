import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt
from ciropt.utils import define_function



def dadmm_C_graph6_rk(mus, L_smooths, R, Capacitance, Inductance, params=None):
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
        h, eta, rho, gamma, alpha, beta = params["h"], params["eta"], params["rho"], params["gamma"], params["alpha"], params["beta"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, eta, rho, gamma, alpha, beta = problem.h, problem.eta, problem.rho, problem.gamma, problem.alpha, problem.beta

    f1 = define_function(problem, mus[0], L_smooths[0], package)
    f2 = define_function(problem, mus[0], L_smooths[0], package)
    f3 = define_function(problem, mus[0], L_smooths[0], package)
    f4 = define_function(problem, mus[1], L_smooths[1], package)
    f5 = define_function(problem, mus[1], L_smooths[1], package)
    f6 = define_function(problem, mus[0], L_smooths[0], package)

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


    x1_1p5, y1__1p5, f1_1p5 = proximal_step((1/2)*((R * i_L_12_1 + e_12_1) + (R * i_L_13_1 + e_13_1)), f1, R/2)
    x2_1p5, y2_1p5, f2_1p5 = proximal_step((1/3)*((-R * i_L_12_1 + e_12_1) + (R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_24_1 + e_24_1)), f2, R/3)
    x3_1p5, y3_2, f3_1p5 = proximal_step((1/3)*((-R * i_L_13_1 + e_13_1) + (-R * i_L_23_1 + e_23_1) \
                                            + (R * i_L_34_1 + e_34_1)), f3, R/3)
    x4_1p5, y4_1p5, f4_1p5 = proximal_step((1/4)*((-R * i_L_24_1 + e_24_1) + (-R * i_L_34_1 + e_34_1) \
                                            + (R * i_L_45_1 + e_45_1) + (R * i_L_46_1 + e_46_1)), f4, R/4)
    x5_1p5, y5_1p5, f5_1p5 = proximal_step(R * i_L_54_1 + e_45_1, f5, R)
    x6_1p5, y6_1p5, f6_1p5 = proximal_step(-R * i_L_46_1 + e_46_1, f6, R)

    e_12_1p5 = (x1_1p5 + x2_1p5) / 2
    e_13_1p5 = (x1_1p5 + x3_1p5) / 2
    e_23_1p5 = (x2_1p5 + x3_1p5) / 2
    e_24_1p5 = (x2_1p5 + x4_1p5) / 2
    e_34_1p5 = (x3_1p5 + x4_1p5) / 2
    e_45_1p5 = e_45_1 - ( alpha * h / Capacitance) * (i_L_45_1 + i_L_54_1 + (2 * e_45_1 - x4_1p5 - x5_1p5) / R)
    e_46_1p5 = (x4_1p5 + x6_1p5) / 2
    i_L_12_1p5 = i_L_12_1 + ( alpha * h / Inductance) * (e_12_1p5 - x1_1p5)      
    i_L_13_1p5 = i_L_13_1 + ( alpha * h / Inductance) * (e_13_1p5 - x1_1p5)
    i_L_23_1p5 = i_L_23_1 + ( alpha * h / Inductance) * (e_23_1p5 - x2_1p5)
    i_L_24_1p5 = i_L_24_1 + ( alpha * h / Inductance) * (e_24_1p5 - x2_1p5)
    i_L_34_1p5 = i_L_34_1 + ( alpha * h / Inductance) * (e_34_1p5 - x3_1p5)
    i_L_45_1p5 = i_L_45_1 + ( alpha * h / Inductance) * (e_45_1p5 - x4_1p5)
    i_L_54_1p5 = i_L_54_1 + ( alpha * h / Inductance) * (e_45_1p5 - x5_1p5)
    i_L_46_1p5 = i_L_46_1 + ( alpha * h / Inductance) * (e_46_1p5 - x4_1p5)


    x1_2, y1_2, f1_2 = proximal_step((1/2)*((R * i_L_12_1p5 + e_12_1p5) + (R * i_L_13_1p5 + e_13_1)), f1, R/2)
    x2_2, y2_2, f2_2 = proximal_step((1/3)*((-R * i_L_12_1p5 + e_12_1) + (R * i_L_23_1p5 + e_23_1) \
                                            + (R * i_L_24_1p5 + e_24_1)), f2, R/3)
    x3_2, y3_2, f3_2 = proximal_step((1/3)*((-R * i_L_13_1p5 + e_13_1) + (-R * i_L_23_1p5 + e_23_1) \
                                            + (R * i_L_34_1p5 + e_34_1)), f3, R/3)
    x4_2, y4_2, f4_2 = proximal_step((1/4)*((-R * i_L_24_1p5 + e_24_1) + (-R * i_L_34_1p5 + e_34_1) \
                                            + (R * i_L_45_1p5 + e_45_1) + (R * i_L_46_1p5 + e_46_1)), f4, R/4)
    x5_2, y5_2, f5_2 = proximal_step(R * i_L_54_1p5 + e_45_1p5, f5, R)
    x6_2, y6_2, f6_2 = proximal_step(-R * i_L_46_1p5 + e_46_1p5, f6, R)

    e_12_2 = (x1_2 + x2_2) / 2
    e_13_2 = (x1_2 + x3_2) / 2
    e_23_2 = (x2_2 + x3_2) / 2
    e_24_2 = (x2_2 + x4_2) / 2
    e_34_2 = (x3_2 + x4_2) / 2
    e_45_2 = e_45_1 - ( beta * h / Capacitance) * (i_L_45_1 + i_L_54_1 + (2 * e_45_1 - x4_1p5 - x5_1p5) / R) \
                    - ( (1 - beta) * h / Capacitance) * (i_L_45_1p5 + i_L_54_1p5 + (2 * e_45_1p5 - x4_2 - x5_2) / R)
    e_46_2 = (x4_2 + x6_2) / 2
    i_L_12_2 = i_L_12_1 + ( beta * h / Inductance) * (e_12_1p5 - x1_1p5) + ( (1 - beta) * h / Inductance) * (e_12_2 - x1_2)      
    i_L_13_2 = i_L_13_1 + ( beta * h / Inductance) * (e_13_1p5 - x1_1p5) + ( (1 - beta) * h / Inductance) * (e_13_2 - x1_2)
    i_L_23_2 = i_L_23_1 + ( beta * h / Inductance) * (e_23_1p5 - x2_1p5) + ( (1 - beta) * h / Inductance) * (e_23_2 - x2_2)
    i_L_24_2 = i_L_24_1 + ( beta * h / Inductance) * (e_24_1p5 - x2_1p5) + ( (1 - beta) * h / Inductance) * (e_24_2 - x2_2)
    i_L_34_2 = i_L_34_1 + ( beta * h / Inductance) * (e_34_1p5 - x3_1p5) + ( (1 - beta) * h / Inductance) * (e_34_2 - x3_2)
    i_L_45_2 = i_L_45_1 + ( beta * h / Inductance) * (e_45_1p5 - x4_1p5) + ( (1 - beta) * h / Inductance) * (e_45_2 - x4_2)
    i_L_54_2 = i_L_54_1 + ( beta * h / Inductance) * (e_45_1p5 - x5_1p5) + ( (1 - beta) * h / Inductance) * (e_45_2 - x5_2)
    i_L_46_2 = i_L_46_1 + ( beta * h / Inductance) * (e_46_1p5 - x4_1p5) + ( (1 - beta) * h / Inductance) * (e_46_2 - x4_2)
    
    # energy of two inductors on each net is twice the energy on one inductor
    E_1 = gamma * ((e_12_1 - x_star)**2 + (e_13_1 - x_star)**2 + (e_23_1 - x_star)**2 + (e_24_1 - x_star)**2\
                   + (e_34_1 - x_star)**2 + (e_46_1 - x_star)**2) +  \
            + Inductance * (i_L_12_1 - y1_12_star) ** 2 + Inductance * (i_L_13_1 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_1 - y2_23_star) ** 2 + Inductance * (i_L_24_1 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_1 - y3_34_star) ** 2 \
            + (Inductance/2) * (i_L_45_1 + y5_star) ** 2 + (Inductance/2) * (i_L_54_1 - y5_star) ** 2 \
            + Inductance * (i_L_46_1 + y6_star) ** 2 \
            + (Capacitance / 2) * (e_45_1 - x_star) ** 2
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_13_2 - x_star)**2 + (e_23_2 - x_star)**2 + (e_24_2 - x_star)**2\
                   + (e_34_2 - x_star)**2 + (e_46_2 - x_star)**2) +  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_13_2 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_2 - y2_23_star) ** 2 + Inductance * (i_L_24_2 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 \
            + (Inductance/2) * (i_L_45_2 + y5_star) ** 2 + (Inductance/2) * (i_L_54_2 - y5_star) ** 2 \
            + Inductance * (i_L_46_2 + y6_star) ** 2 \
            + (Capacitance / 2) * (e_45_2 - x_star) ** 2
    # currents on resistors on each net sum to 0
    Delta_2 = rho * ((2/R) * ((e_12_2 - x1_2)**2 + (e_13_2 - x3_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_24_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2  + (e_46_2 - x6_2)**2 )  \
                    + (1/R) * ((e_45_2 - x5_2)**2 + (e_45_2 - x4_2)**2)) \
              + eta * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star) \
                    + f5_2 - f5_star - y5_star * (x5_2 - x_star) \
                    + f6_2 - f6_star - y6_star * (x6_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def dadmm_C_graph6(mus, L_smooths, R, Capacitance, Inductance, params=None):
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
        h, eta, rho, gamma = params["h"], params["eta"], params["rho"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step 
        h, eta, rho, gamma = problem.h, problem.eta, problem.rho, problem.gamma

    f1 = define_function(problem, mus[0], L_smooths[0], package)
    f2 = define_function(problem, mus[0], L_smooths[0], package)
    f3 = define_function(problem, mus[0], L_smooths[0], package)
    f4 = define_function(problem, mus[1], L_smooths[1], package)
    f5 = define_function(problem, mus[1], L_smooths[1], package)
    f6 = define_function(problem, mus[0], L_smooths[0], package)

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
            + (Inductance/2) * (i_L_45_1 + y5_star) ** 2 + (Inductance/2) * (i_L_54_1 - y5_star) ** 2 \
            + Inductance * (i_L_46_1 + y6_star) ** 2 \
            + (Capacitance / 2) * (e_45_1 - x_star) ** 2
    E_2 = gamma * ((e_12_2 - x_star)**2 + (e_13_2 - x_star)**2 + (e_23_2 - x_star)**2 + (e_24_2 - x_star)**2\
                   + (e_34_2 - x_star)**2 + (e_46_2 - x_star)**2) +  \
            + Inductance * (i_L_12_2 - y1_12_star) ** 2 + Inductance * (i_L_13_2 - y1_13_star) ** 2 \
            + Inductance * (i_L_23_2 - y2_23_star) ** 2 + Inductance * (i_L_24_2 - y2_24_star) ** 2 \
            + Inductance * (i_L_34_2 - y3_34_star) ** 2 \
            + (Inductance/2) * (i_L_45_2 + y5_star) ** 2 + (Inductance/2) * (i_L_54_2 - y5_star) ** 2 \
            + Inductance * (i_L_46_2 + y6_star) ** 2 \
            + (Capacitance / 2) * (e_45_2 - x_star) ** 2
    # currents on resistors on each net sum to 0
    Delta_2 = rho * ((2/R) * ((e_12_2 - x1_2)**2 + (e_13_2 - x3_2)**2 \
                         + (e_23_2 - x2_2)**2 + (e_24_2 - x2_2)**2 \
                         + (e_34_2 - x4_2)**2  + (e_46_2 - x6_2)**2 )  \
                    + (1/R) * ((e_45_2 - x5_2)**2 + (e_45_2 - x4_2)**2)) \
              + eta * ( f1_2 - f1_star - y1_star * (x1_2 - x_star) \
                    + f2_2 - f2_star - y2_star * (x2_2 - x_star) \
                    + f3_2 - f3_star - y3_star * (x3_2 - x_star) \
                    + f4_2 - f4_star - y4_star * (x4_2 - x_star) \
                    + f5_2 - f5_star - y5_star * (x5_2 - x_star) \
                    + f6_2 - f6_star - y6_star * (x6_2 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


######### Paper simple example #########


def example_fC3RC_circuit(mu, L_smooth, Capacitance, R, params=None): 
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
    Delta_1 = eta * (x_1 - x_star) * (y_1 - y_star) \
            + rho * (1/R) * ((e1_1)**2 + (e1_1 - e2_1)**2 + (e2_1)**2)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem
