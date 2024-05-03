import numpy as np

import PEPit
import PEPit.functions as pep_func
from PEPit.constraint import Constraint as pep_constr
from PEPit.primitive_steps import proximal_step as pep_proximal_step

import ciropt.function as co_func
from ciropt.constraint import Constraint as co_constr
from ciropt.circuit_opt import CircuitOpt
from ciropt.utils import define_function


def pg_extra_line2(mu, L_smooth, R, R_12, params=None):
    # line graph with 3 nodes
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

    # f1 = define_function(problem, mu, L_smooth, package)
    # f2 = define_function(problem, mu, L_smooth, package)
    f1 = define_function(problem, mu, 10000, package)
    f2 = define_function(problem, mu, 10000, package)
    # f1 = define_function(problem, mu, np.inf, package)
    # f2 = define_function(problem, mu, np.inf, package)
    h1 = define_function(problem, mu, L_smooth, package)
    h2 = define_function(problem, mu, L_smooth, package)

    x_star, y_star, f_star = (f1 + f2 + h1 + h2).stationary_point(return_gradient_and_function_value=True)
    y_f1_star, f1_star = f1.oracle(x_star)
    y_f2_star, f2_star = f2.oracle(x_star)
    y_h1_star, h1_star = h1.oracle(x_star)
    y_h2_star, h2_star = h2.oracle(x_star)
    # y_h2_star = y_star - y_f1_star - y_f2_star - y_h1_star
    # h2_star = f_star - f1_star - f2_star - h1_star

    i_L_12_star = - y_f1_star - y_h1_star
    i_L_21_star = - y_f2_star - y_h2_star

    # when f is not differentiable
    problem.add_constraint(Constraint((y_f1_star + y_f2_star + y_h1_star + y_h2_star - y_star) ** 2, "equality"))

    x1_1 = problem.set_initial_point()
    x2_1 = problem.set_initial_point()
    y_f1_1, f1_1 = f1.oracle(x1_1)
    y_f2_1, f2_1 = f2.oracle(x2_1)
    y_h1_1, h1_1 = h1.oracle(x1_1)
    y_h2_1, h2_1 = h2.oracle(x2_1)
    # problem.add_constraint(Constraint((y_f1_1 + y_f2_1 + y_h1_1 + y_h2_1) ** 2, "equality"))
    # y_h2_1 = - y_f1_1 - y_f2_1 - y_h1_1
    # _, h2_1 = h2.oracle(x2_1)
    
    # i_L_12_1 = - y_f1_1 - y_h1_1 - 1/R_12 * (x1_1 - x2_1) hold only in continuous setup
    i_L_12_1 =  problem.set_initial_point() 
    i_L_21_1 = - i_L_12_1
    
    # update
    e1_1 = R/R_12 * x2_1 + ( 1 - R/R_12 ) * x1_1 - R * y_h1_1 - R * i_L_12_1
    e2_1 = R/R_12 * x1_1 + ( 1 - R/R_12 ) * x2_1 - R * y_h2_1 - R * i_L_21_1
    x1_2, y_f1_2, f1_2 = proximal_step(e1_1, f1, R)
    x2_2, y_f2_2, f2_2 = proximal_step(e2_1, f2, R)
    i_L_12_2 = i_L_12_1 + h/R_12 * (x1_1 - x2_1)
    i_L_21_2 = i_L_21_1 + h/R_12 * (x2_1 - x1_1)    
    
    L_12 = R_12


    i_L_12_3 = i_L_12_2 + h/R_12 * (x1_2 - x2_2)
    i_L_21_3 = i_L_21_2 + h/R_12 * (x2_2 - x1_2)    


    # Energy
    # E_1 = gamma * (x1_1 - x_star)**2 + gamma * (x2_1 - x_star)**2 \
    #         + (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 
    # E_2 = gamma * (x1_2 - x_star)**2 + gamma * (x2_2 - x_star)**2 \
    #         + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2   

    # E_1 = (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 
    # E_2 = (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2    

    # E_1 = gamma * ( (x1_1 - x2_1)**2 ) \
    #     + (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_1 - i_L_21_star) ** 2 
    # E_2 = gamma * ( (x1_2 - x2_2)**2 ) \
    #     + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_2 - i_L_21_star) ** 2  

    E_1 = gamma * ( (x1_1 - x_star)**2 + (x2_1 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_2 - i_L_21_star) ** 2 
    E_2 = gamma * ( (x1_2 - x_star)**2 + (x2_2 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_3 - i_L_12_star ) ** 2 + (L_12/2) * (i_L_21_3 - i_L_21_star ) ** 2  

    # E_1 = gamma * ( (x1_1 - x_star)**2 + (x2_1 - x_star)**2 ) \
    #     + (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_1 - i_L_21_star) ** 2 
    # E_2 = gamma * ( (x1_2 - x_star)**2 + (x2_2 - x_star)**2 ) \
    #     + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_2 - i_L_21_star) ** 2  

    # E_1 = gamma * ( (x1_1 + x2_1 - 2 * x_star)**2 ) \
    #     + (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_1 - i_L_21_star) ** 2 
    # E_2 = gamma * ( (x1_2 + x2_2 - 2 * x_star)**2 ) \
    #     + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_2 - i_L_21_star) ** 2 

    # E_1 = gamma * ( (y_f1_1 - y_f1_star)**2 + (y_f2_1 - y_f2_star)**2 ) \
    #     + (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_1 - i_L_21_star) ** 2 
    # E_2 = gamma * ( (y_f1_2 - y_f1_star)**2 + (y_f2_2 - y_f2_star)**2 ) \
    #     + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_2 - i_L_21_star) ** 2  

    # E_1 = (L_12/2) * (i_L_12_1 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_1 - i_L_21_star) ** 2 
    # E_2 = (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2 + (L_12/2) * (i_L_21_2 - i_L_21_star) ** 2  

    # Delta_2 = d * 1/R_12 * ((x2_2 - x1_2)**2) \
    #           + b * ( f1_2 - f1_star - y_f1_star * (x1_2 - x_star) \
    #                 + f2_2 - f2_star - y_f2_star * (x2_2 - x_star)\
    #                 + h1_1 - h1_star - y_h1_star * (x1_1 - x_star)\
    #                 + h2_1 - h2_star - y_h2_star * (x2_1 - x_star))

    # Delta_2 = d * 1/R_12 * ((x2_1 - x1_1)**2) \
    #           + b * ( f1_2 - f1_star - y_f1_star * (x1_2 - x_star) \
    #                 + f2_2 - f2_star - y_f2_star * (x2_2 - x_star)\
    #                 + h1_1 - h1_star - y_h1_star * (x1_1 - x_star)\
    #                 + h2_1 - h2_star - y_h2_star * (x,2_1 - x_star))
    
    # Delta_2 = d * 1/R_12 * ((x2_1 - x1_1)**2) \
    #           + b * ( ( y_f1_2 - y_f1_star ) * (x1_2 - x_star) \
    #                 + ( y_f2_2 - y_f2_star ) * (x2_2 - x_star) \
    #                 + ( y_h1_1 - y_h1_star ) * (x1_1 - x_star) \
    #                 + ( y_h2_1 - y_h2_star ) * (x2_1 - x_star) )

    # Delta_2 = d * ( (x1_2 - x1_1) - (x2_2 - x2_1) )**2 \
    #           + b * ( ( y_f1_2 - y_f2_2 ) * (x1_2 - x2_2) \
    #                 + ( y_h1_1 - y_h2_1 ) * (x1_1 - x2_1)  )

    # Delta_2 = d * ( (x1_2 - x1_1) - (x2_2 - x2_1) )**2 \
    #           + b * ( ( y_f1_2 - y_f2_2 ) * (x1_2 - x2_2) \
    #                 + ( y_h1_1 - y_h2_1 ) * (x1_1 - x2_1)  )

    # Delta_2 = b * ( f1_2 - f1_star - y_f1_star * (x1_2 - x_star) \
    #             + f2_2 - f2_star - y_f2_star * (x2_2 - x_star) ) \
    #         + d *( h1_1 - h1_star - y_h1_star * (x1_1 - x_star)\
    #             + h2_1 - h2_star - y_h2_star * (x2_1 - x_star))
    
    Delta_2 = b * ( (y_f1_2 - y_f1_star) * (x1_2 - x_star) \
                + (y_f2_2 - y_f2_star) * (x2_2 - x_star) ) \
            + d *( (y_h1_1 - y_h1_star) * (x1_1 - x_star)\
                + (y_h2_1 - y_h2_star) * (x2_1 - x_star))

    # Delta_2 = b * ( f1_1 - f1_star - y_f1_star * (x1_1 - x_star) \
    #             + f2_1 - f2_star - y_f2_star * (x2_1 - x_star) ) \
    #         + d *( h1_1 - h1_star - y_h1_star * (x1_1 - x_star)\
    #             + h2_1 - h2_star - y_h2_star * (x2_1 - x_star))

    # Delta_2 = d * 1/R_12 * ((x2_1 - x1_1)**2) \
    #           + b * ( f1_2 - f1_star \
    #                 + f2_2 - f2_star \
    #                 + h1_1 - h1_star \
    #                 + h2_1 - h2_star )

    # Delta_2 = b * ( f1_2 - f1_star + f2_2 - f2_star ) \
    #             + d *( h1_1 - h1_star + h2_1 - h2_star )

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem




















def pg_extra_line3(mu, L_smooth, R, Capacitance, params=None):
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
