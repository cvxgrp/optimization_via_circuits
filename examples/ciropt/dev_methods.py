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

    i_L_12_star = - y_f1_star - y_h1_star
    i_L_21_star = - y_f2_star - y_h2_star

    problem.add_constraint(Constraint((y_f1_star + y_f2_star + y_h1_star + y_h2_star - y_star) ** 2, "equality"))

    x1_1 = problem.set_initial_point()
    x2_1 = problem.set_initial_point()
    y_f1_1, f1_1 = f1.oracle(x1_1)
    y_f2_1, f2_1 = f2.oracle(x2_1)
    y_h1_1, h1_1 = h1.oracle(x1_1)
    y_h2_1, h2_1 = h2.oracle(x2_1)    

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
    E_1 = gamma * ( (x1_1 - x_star)**2 + (x2_1 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2  
    E_2 = gamma * ( (x1_2 - x_star)**2 + (x2_2 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_3 - i_L_12_star ) ** 2

 
    # Delta
    Delta_2 = b * ( (y_f1_2 - y_f1_star) * (x1_2 - x_star) \
                + (y_f2_2 - y_f2_star) * (x2_2 - x_star) ) \
            + d *( (y_h1_1 - y_h1_star) * (x1_1 - x_star)  \
                + (y_h2_1 - y_h2_star) * (x2_1 - x_star))

    # Delta_2 = b * ( f1_2 - f1_star - y_f1_star * (x1_2 - x_star) \
    #             + f2_2 - f2_star - y_f2_star * (x2_2 - x_star) ) \
    #         + d *( h1_1 - h1_star - y_h1_star * (x1_1 - x_star)  \
    #             + h2_1 - h2_star - y_h2_star * (x2_1 - x_star))

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def pg_extra_line3(mu, L_smooth_h, L_smooth_f, R, W, params=None):
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

    f1 = define_function(problem, mu, L_smooth_f, package)
    f2 = define_function(problem, mu, L_smooth_f, package)
    f3 = define_function(problem, mu, L_smooth_f, package)
    h1 = define_function(problem, mu, L_smooth_h, package)
    h2 = define_function(problem, mu, L_smooth_h, package)
    h3 = define_function(problem, mu, L_smooth_h, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + h1 + h2 + h3).stationary_point(return_gradient_and_function_value=True)
    y_f1_star, f1_star = f1.oracle(x_star)
    y_f2_star, f2_star = f2.oracle(x_star)
    y_f3_star, f3_star = f3.oracle(x_star)
    y_h1_star, h1_star = h1.oracle(x_star)
    y_h2_star, h2_star = h2.oracle(x_star)
    y_h3_star, h3_star = h3.oracle(x_star)

    i_L_12_star = - y_f1_star - y_h1_star
    i_L_23_star = y_f3_star + y_h3_star

    problem.add_constraint(Constraint((y_f1_star + y_f2_star + y_f3_star + y_h1_star + y_h2_star + y_h3_star - y_star) ** 2, "equality"))

    x1_1 = problem.set_initial_point()
    x2_1 = problem.set_initial_point()
    x3_1 = problem.set_initial_point()
    y_f1_1, f1_1 = f1.oracle(x1_1)
    y_f2_1, f2_1 = f2.oracle(x2_1)
    y_f3_1, f3_1 = f3.oracle(x3_1)
    y_h1_1, h1_1 = h1.oracle(x1_1)
    y_h2_1, h2_1 = h2.oracle(x2_1)    
    y_h3_1, h3_1 = h3.oracle(x3_1)    

    i_L_12_1 =  problem.set_initial_point() 
    i_L_23_1 =  problem.set_initial_point() 

    # update
    e1_1 = W[0][0] * x1_1 + W[0][1] * x2_1 + W[0][2] * x3_1 - R * y_h1_1 - R * i_L_12_1
    e2_1 = W[1][0] * x1_1 + W[1][1] * x2_1 + W[1][2] * x3_1 - R * y_h2_1 + R * i_L_12_1 - R * i_L_23_1
    e3_1 = W[2][0] * x1_1 + W[2][1] * x2_1 + W[2][2] * x3_1 - R * y_h3_1 + R * i_L_23_1
    x1_2, y_f1_2, f1_2 = proximal_step(e1_1, f1, R)
    x2_2, y_f2_2, f2_2 = proximal_step(e2_1, f2, R)
    x3_2, y_f3_2, f3_2 = proximal_step(e3_1, f3, R)

    R_12, R_23 = R / W[0][1], R / W[1][2]
    L_12, L_23 = R_12, R_23

    i_L_12_2 = i_L_12_1 + h/R_12 * (x1_1 - x2_1)
    i_L_23_2 = i_L_23_1 + h/R_23 * (x2_1 - x3_1)    
    
    i_L_12_3 = i_L_12_2 + h/R_12 * (x1_2 - x2_2)
    i_L_23_3 = i_L_23_2 + h/R_23 * (x2_2 - x3_2)    



    # Energy
    E_1 = gamma * ( (x1_1 - x_star)**2 + (x2_1 - x_star)**2 + (x3_1 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2  + (L_23/2) * (i_L_23_2 - i_L_23_star) ** 2  
    E_2 = gamma * ( (x1_2 - x_star)**2  + (x2_2 - x_star)**2 + (x3_2 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_3 - i_L_12_star) ** 2  + (L_23/2) * (i_L_23_3 - i_L_23_star) ** 2  

 
    # Delta
    Delta_2 = b * ( (y_f1_2 - y_f1_star) * (x1_2 - x_star) \
                + (y_f2_2 - y_f2_star) * (x2_2 - x_star) \
                + (y_f3_2 - y_f3_star) * (x3_2 - x_star) ) \
            + d *( (y_h1_1 - y_h1_star) * (x1_1 - x_star)  \
                + (y_h2_1 - y_h2_star) * (x2_1 - x_star) \
                + (y_h3_1 - y_h3_star) * (x3_1 - x_star) )

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def pg_extra_line3_C(mu, L_smooth_h, L_smooth_f, R, C, W, params=None):
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

    f1 = define_function(problem, mu, L_smooth_f, package)
    f2 = define_function(problem, mu, L_smooth_f, package)
    f3 = define_function(problem, mu, L_smooth_f, package)
    h1 = define_function(problem, mu, L_smooth_h, package)
    h2 = define_function(problem, mu, L_smooth_h, package)
    h3 = define_function(problem, mu, L_smooth_h, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + h1 + h2 + h3).stationary_point(return_gradient_and_function_value=True)
    y_f1_star, f1_star = f1.oracle(x_star)
    y_f2_star, f2_star = f2.oracle(x_star)
    y_f3_star, f3_star = f3.oracle(x_star)
    y_h1_star, h1_star = h1.oracle(x_star)
    y_h2_star, h2_star = h2.oracle(x_star)
    y_h3_star, h3_star = h3.oracle(x_star)

    i_L_12_star = - y_f1_star - y_h1_star
    i_L_32_star = - y_f3_star - y_h3_star
    i_L_23_star = - i_L_32_star

    problem.add_constraint(Constraint((y_f1_star + y_f2_star + y_f3_star + y_h1_star + y_h2_star + y_h3_star - y_star) ** 2, "equality"))

    x1_1 = problem.set_initial_point()
    x2_1 = problem.set_initial_point()
    x3_1 = problem.set_initial_point()
    e23_1 = problem.set_initial_point()
    y_f1_1, f1_1 = f1.oracle(x1_1)
    y_f2_1, f2_1 = f2.oracle(x2_1)
    y_f3_1, f3_1 = f3.oracle(x3_1)
    y_h1_1, h1_1 = h1.oracle(x1_1)
    y_h2_1, h2_1 = h2.oracle(x2_1)    
    y_h3_1, h3_1 = h3.oracle(x3_1)    

    i_L_12_1 =  problem.set_initial_point() 
    i_L_21_1 =  -i_L_12_1
    i_L_23_1 =  problem.set_initial_point() 
    i_L_32_1 =  problem.set_initial_point() 

    # update
    e1_1 = W[0][0] * x1_1 + W[0][1] * x2_1 + W[0][2] * x3_1 - R * y_h1_1 - R * i_L_12_1
    e2_1 = W[1][0] * x1_1 + W[1][1] * x2_1 + W[1][2] * e23_1 - R * y_h2_1 - R * i_L_21_1 - R * i_L_23_1
    e3_1 = W[2][0] * x1_1 + W[2][1] * e23_1 + W[2][2] * x3_1 - R * y_h3_1 - R * i_L_32_1
    x1_2, y_f1_2, f1_2 = proximal_step(e1_1, f1, R)
    x2_2, y_f2_2, f2_2 = proximal_step(e2_1, f2, R)
    x3_2, y_f3_2, f3_2 = proximal_step(e3_1, f3, R)

    R_12, R_23 = R / W[0][1], R / W[1][2]
    L_12, L_23 = R_12, R_23

    e23_2 = e23_1 + h/C  * ( i_L_23_1 + 1/R_23 * ( x2_1 - e23_1 ) + i_L_32_1 + 1/R_23 * ( x3_1 - e23_1 ) )

    i_L_12_2 = i_L_12_1 + h/R_12 * (x1_1 - x2_1)
    i_L_23_2 = i_L_23_1 + h/R_23 * (x2_1 - e23_1)    
    i_L_32_2 = i_L_32_1 + h/R_23 * (x3_1 - e23_1)
    
    i_L_12_3 = i_L_12_2 + h/R_12 * (x1_2 - x2_2)
    i_L_23_3 = i_L_23_2 + h/R_23 * (x2_2 - e23_2)    
    i_L_32_3 = i_L_32_2 + h/R_23 * (x3_2 - e23_2)   



    # Energy
    E_1 = gamma * ( (x1_1 - x_star)**2 + (x2_1 - x_star)**2 + (x3_1 - x_star)**2 ) + (C/2) * (e23_1 - x_star) ** 2 \
        + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2  + (L_23/2) * (i_L_23_2 - i_L_23_star) ** 2 + (L_23/2) * (i_L_32_2 - i_L_32_star) ** 2 
    E_2 = gamma * ( (x1_2 - x_star)**2  + (x2_2 - x_star)**2 + (x3_2 - x_star)**2 ) + (C/2) * (e23_2 - x_star) ** 2 \
        + (L_12/2) * (i_L_12_3 - i_L_12_star) ** 2  + (L_23/2) * (i_L_23_3 - i_L_23_star) ** 2 + (L_23/2) * (i_L_32_3 - i_L_32_star) ** 2 

 
    # Delta
    Delta_2 = b * ( (y_f1_2 - y_f1_star) * (x1_2 - x_star) \
                + (y_f2_2 - y_f2_star) * (x2_2 - x_star) \
                + (y_f3_2 - y_f3_star) * (x3_2 - x_star) ) \
            + d *( (y_h1_1 - y_h1_star) * (x1_1 - x_star)  \
                + (y_h2_1 - y_h2_star) * (x2_1 - x_star) \
                + (y_h3_1 - y_h3_star) * (x3_1 - x_star) )

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem


def pg_extra_line5(mu, L_smooth_h, L_smooth_f, R, W, params=None):
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

    f1 = define_function(problem, mu, L_smooth_f, package)
    f2 = define_function(problem, mu, L_smooth_f, package)
    f3 = define_function(problem, mu, L_smooth_f, package)
    f4 = define_function(problem, mu, L_smooth_f, package)
    f5 = define_function(problem, mu, L_smooth_f, package)
    h1 = define_function(problem, mu, L_smooth_h, package)
    h2 = define_function(problem, mu, L_smooth_h, package)
    h3 = define_function(problem, mu, L_smooth_h, package)
    h4 = define_function(problem, mu, L_smooth_h, package)
    h5 = define_function(problem, mu, L_smooth_h, package)

    x_star, y_star, f_star = (f1 + f2 + f3 + f4 + f5 + h1 + h2 + h3 + h4 + h5).stationary_point(return_gradient_and_function_value=True)
    y_f1_star, f1_star = f1.oracle(x_star)
    y_f2_star, f2_star = f2.oracle(x_star)
    y_f3_star, f3_star = f3.oracle(x_star)
    y_f4_star, f4_star = f4.oracle(x_star)
    y_f5_star, f5_star = f5.oracle(x_star)
    y_h1_star, h1_star = h1.oracle(x_star)
    y_h2_star, h2_star = h2.oracle(x_star)
    y_h3_star, h3_star = h3.oracle(x_star)
    y_h4_star, h4_star = h4.oracle(x_star)
    y_h5_star, h5_star = h5.oracle(x_star)

    problem.add_constraint(Constraint((y_f1_star + y_f2_star + y_f3_star + y_f4_star + y_f5_star \
                                       + y_h1_star + y_h2_star + y_h3_star + y_h4_star + y_h5_star - y_star) ** 2, "equality"))

    # should modify from here
    i_L_12_star = - y_f1_star - y_h1_star
    i_L_23_star = y_f3_star + y_h3_star

    x1_1 = problem.set_initial_point()
    x2_1 = problem.set_initial_point()
    x3_1 = problem.set_initial_point()
    y_f1_1, f1_1 = f1.oracle(x1_1)
    y_f2_1, f2_1 = f2.oracle(x2_1)
    y_f3_1, f3_1 = f3.oracle(x3_1)
    y_h1_1, h1_1 = h1.oracle(x1_1)
    y_h2_1, h2_1 = h2.oracle(x2_1)    
    y_h3_1, h3_1 = h3.oracle(x3_1)    

    i_L_12_1 =  problem.set_initial_point() 
    i_L_23_1 =  problem.set_initial_point() 

    # update
    e1_1 = W[0][0] * x1_1 + W[0][1] * x2_1 + W[0][2] * x3_1 - R * y_h1_1 - R * i_L_12_1
    e2_1 = W[1][0] * x1_1 + W[1][1] * x2_1 + W[1][2] * x3_1 - R * y_h2_1 + R * i_L_12_1 - R * i_L_23_1
    e3_1 = W[2][0] * x1_1 + W[2][1] * x2_1 + W[2][2] * x3_1 - R * y_h3_1 + R * i_L_23_1
    x1_2, y_f1_2, f1_2 = proximal_step(e1_1, f1, R)
    x2_2, y_f2_2, f2_2 = proximal_step(e2_1, f2, R)
    x3_2, y_f3_2, f3_2 = proximal_step(e3_1, f3, R)

    R_12, R_23 = R / W[0][1], R / W[1][2]
    L_12, L_23 = R_12, R_23

    i_L_12_2 = i_L_12_1 + h/R_12 * (x1_1 - x2_1)
    i_L_23_2 = i_L_23_1 + h/R_23 * (x2_1 - x3_1)    
    
    i_L_12_3 = i_L_12_2 + h/R_12 * (x1_2 - x2_2)
    i_L_23_3 = i_L_23_2 + h/R_23 * (x2_2 - x3_2)    



    # Energy
    E_1 = gamma * ( (x1_1 - x_star)**2 + (x2_1 - x_star)**2 + (x3_1 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_2 - i_L_12_star) ** 2  + (L_23/2) * (i_L_23_2 - i_L_23_star) ** 2  
    E_2 = gamma * ( (x1_2 - x_star)**2  + (x2_2 - x_star)**2 + (x3_2 - x_star)**2 ) \
        + (L_12/2) * (i_L_12_3 - i_L_12_star) ** 2  + (L_23/2) * (i_L_23_3 - i_L_23_star) ** 2  

 
    # Delta
    Delta_2 = b * ( (y_f1_2 - y_f1_star) * (x1_2 - x_star) \
                + (y_f2_2 - y_f2_star) * (x2_2 - x_star) \
                + (y_f3_2 - y_f3_star) * (x3_2 - x_star) ) \
            + d *( (y_h1_1 - y_h1_star) * (x1_1 - x_star)  \
                + (y_h2_1 - y_h2_star) * (x2_1 - x_star) \
                + (y_h3_1 - y_h3_star) * (x3_1 - x_star) )

    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem




def drs_variant(mu, L_smooth, R, Inductance, Capacitance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        # h, alpha, beta, b, d, gamma = params["h"], params["alpha"], params["beta"], params["b"], params["d"], params["gamma"]
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step
        # h, alpha, beta, b, d, gamma = problem.h, problem.alpha, problem.beta, problem.b, problem.d, problem.gamma
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    # y2_star = y_star - y1_star
    y2_star, _ = f2.oracle(x_star)
    problem.add_constraint(Constraint((y1_star + y2_star - y_star) ** 2, "equality"))

    i_L_0 = problem.set_initial_point()
    x2_0 = problem.set_initial_point()
    x1_0 = problem.set_initial_point()
    i_C_0 = problem.set_initial_point()

    h=0.1

    x1_1, y1_1, f1_1 = proximal_step(x2_0 + R * i_L_0 + R * i_C_0, f1, R)
    x2_1, y2_1, f2_1 = proximal_step(x1_1 - R * i_L_0 - R * i_C_0, f2, R)
    # i_L_1 = i_L_0 + (h / Inductance)  * (x2_1 - x1_1)
    # i_C_1 = (Capacitance / h)  * (x2_1 + x1_1 - x1_0 - x2_0)
    i_L_1 = i_L_0 + h / Inductance  * (x2_1 - x1_1)
    i_C_1 = Capacitance/h  * (x2_1 + x1_1 - x1_0 - x2_0)

    x1_2, y1_2, f1_2 = proximal_step(x2_1 + R * i_L_1 + R * i_C_1, f1, R)
    x2_2, y2_2, f2_2 = proximal_step(x1_2 - R * i_L_1 - R * i_C_1, f2, R)
    # i_L_2 = i_L_1 + h / Inductance *  (x2_2 - x1_2) 
    # i_C_2 = (Capacitance / h)  * (x2_2 + x1_2 - x1_1 - x2_1) 
    i_L_2 = i_L_1 + h/Inductance *  (x2_2 - x1_2) 
    i_C_2 = Capacitance/h  * (x2_2 + x1_2 - x1_1 - x2_1) 

    E_1 = (Inductance/2) * (i_L_1 - y1_star) ** 2 + gamma * (x2_1 - x_star)**2 + (Capacitance/2) * (x1_0 - x2_0) ** 2 
    E_2 = (Inductance/2) * (i_L_2 - y1_star) ** 2 + gamma * (x2_2 - x_star)**2 + (Capacitance/2) * (x1_0 - x2_0) ** 2 

    Delta_1 = d * R * (y1_1 - i_L_1)**2 \
              + b * (f1_1 + f2_1 - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - f_star)
    # Delta_1 = b * (f1_1 + f2_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem


def drs_variant2(mu, L_smooth, R, Inductance, Capacitance, params=None):
    if params is not None:
        # verification mode: PEP
        problem = PEPit.PEP() 
        package = pep_func 
        Constraint = pep_constr
        proximal_step = pep_proximal_step
        # h, alpha, beta, b, d, gamma = params["h"], params["alpha"], params["beta"], params["b"], params["d"], params["gamma"]
        h, b, d, gamma = params["h"], params["b"], params["d"], params["gamma"]
    else:
        # Ciropt mode
        problem = CircuitOpt()
        package = co_func
        Constraint = co_constr
        proximal_step = co_func.proximal_step
        # h, alpha, beta, b, d, gamma = problem.h, problem.alpha, problem.beta, problem.b, problem.d, problem.gamma
        h, b, d, gamma = problem.h, problem.b, problem.d, problem.gamma

    f1 = define_function(problem, mu, L_smooth, package)
    f2 = define_function(problem, mu, L_smooth, package)
    x_star, y_star, f_star = (f1 + f2).stationary_point(return_gradient_and_function_value=True)
    y1_star, _ = f1.oracle(x_star)
    # y2_star = y_star - y1_star
    y2_star, _ = f2.oracle(x_star)
    problem.add_constraint(Constraint((y1_star + y2_star - y_star) ** 2, "equality"))

    i_L_0 = problem.set_initial_point()
    x2_0 = problem.set_initial_point()
    x1_0 = problem.set_initial_point()
    v_C_0 = problem.set_initial_point()
    y1_0, f1_0 = f1.oracle(x1_0) # f1=g
    i_C_0 = - i_L_0 - y1_0 - v_C_0/R
    
    # x1_1, y1_1, f1_1 = proximal_step(x2_0 + R * i_L_0 + R * i_C_1, f1, R)
    x2_1, y2_1, f2_1 = proximal_step(x1_0 - R * i_L_0 - R * i_C_0, f2, R)
    v_C_1 = v_C_0 + (h / Capacitance)  * i_C_0
    x1_1 = x2_1 - v_C_1
    i_L_1 = i_L_0 + (h / Inductance)  * (x2_1 - x1_1)    
    y1_1, f1_1 = f1.oracle(x1_1) # f1=g
    i_C_1 = - i_L_1 - y1_1 - v_C_1/R

    x2_2, y2_2, f2_2 = proximal_step(x1_1 - R * i_L_1 - R * i_C_1, f2, R)
    v_C_2 = v_C_1 + (h / Capacitance)  * i_C_1
    x1_2 = x2_2 - v_C_2
    i_L_2 = i_L_1 + (h / Inductance)  * (x2_2 - x1_2)

    E_1 = (Inductance/2) * (i_L_1 - y1_star) ** 2 + gamma * (x2_1 - x_star)**2 + (Capacitance/2) * (x1_1 - x2_1) ** 2 
    E_2 = (Inductance/2) * (i_L_2 - y1_star) ** 2 + gamma * (x2_2 - x_star)**2 + (Capacitance/2) * (x1_2 - x2_2) ** 2 

    Delta_1 = d * R * (y1_1 - i_L_1)**2 \
            + b * (f1_0 + f2_1 - y1_star * (x1_0 - x_star) - y2_star * (x2_1 - x_star) - f_star)
            #   + b * (f1_1 + f2_1 - y1_star * (x1_1 - x_star) - y2_star * (x2_1 - x_star) - f_star)
    # Delta_1 = b * (f1_1 + f2_1 - f_star)
    problem.set_performance_metric(E_2 - (E_1 - Delta_1))
    return problem