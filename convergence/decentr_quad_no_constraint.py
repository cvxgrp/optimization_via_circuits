"""
    Some functions are from  https://github.com/jaewookjsuh/continuous_anchor_acceleration_experiment/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
from scipy.stats import ortho_group



# closd form
def prox_fj_quad_no_constraint(z, rho, Qj, bj, vector_size):
    """
    return minimizer to 1/2 * x^T * Q_i * x + b_i^T * x + (1/(2 * rho)) * \|x - z\|_2^2
    """
    x = np.squeeze( np.dot( np.linalg.inv(Qj + 1/rho * np.eye(vector_size)) , ( 1/rho * z - bj[0] ).T ) )
    return x

def cvx_prox_fj_quad_no_constraint(z, rho, problem_spec, problem_data, j, eps=1e-4):
    """
    return minimizer to 1/2 * x^T * Q_i * x + b_i^T * x + (1/(2 * rho)) * \|x - z\|_2^2
    """

    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    Q = problem_data['Q']
    b = problem_data['b']

    x = cp.Variable(vector_size)

    f = 1/2 * cp.quad_form(x, Q[j]) 
    f += cp.sum(cp.multiply(b[j][0], x))
    f += (1/(2*rho)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal", print(prob.status)
    return x.value


def min_cvx_quad_no_constraint(problem_spec, problem_data):
    """
    return minimizer to sum_{i} 1/2 * x^T * Q_i * x + sum_{i} b_i^T * x 
    """

    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    Q = problem_data['Q']
    b = problem_data['b']

    x = cp.Variable(vector_size)
    f = 0
    for j in range(n_node):
        f += 1/2 * cp.quad_form(x, Q[j]) + cp.sum(cp.multiply(b[j][0], x))

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal"
    # x_cvx = np.repeat(x.value, n_node).reshape(vector_size, n_node).T
    return f.value, x.value

def min_quad_no_constraint(problem_spec, problem_data):
    """
    return minimizer to sum_{i} 1/2 * x^T * Q_i * x + sum_{i} b_i^T * x 
    """

    n_node = problem_spec['n_node']
    Q = problem_data['Q']
    b = problem_data['b']

    Q_sum = np.zeros((Q[0].shape))
    b_sum = np.zeros((b[0].shape))
    for j in range(n_node):
        Q_sum += Q[j]
        b_sum += b[j]
    x_star = np.dot( np.linalg.inv( Q_sum ) , ( - b_sum[0] ).T )

    f_star = 0 
    f_star = 1/2 * x_star @ Q_sum @ x_star.T + np.dot(b_sum[0], x_star)
    return f_star, x_star


def data_generation(problem_spec) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    theta = problem_spec['sc_theta'] 
    sc_perturb = problem_spec['sc_perturb'] 

    Q, b = [], []
    for j in range(n_node):
        b_normal = np.random.normal(loc=0, scale=1, size=(1, vector_size))
        sq_Q = np.random.normal(loc=0, scale=1, size=(vector_size, vector_size))
        # b.append( 1/(np.sqrt(vector_size)) * np.random.uniform(low=0, high=1, size=(1, vector_size)) )
        # sq_Q = 1/(n_node * vector_size) * np.random.uniform(low=-1, high=1, size=(vector_size, vector_size))
        Q.append( 1/(n_node * vector_size) * np.dot( sq_Q.T, sq_Q ) )
        b.append( 1/(n_node * vector_size) * b_normal )
        # Q.append( 1/(n_node * vector_size) * np.dot( sq_Q.T, sq_Q ) )
        # b.append( 1/(n_node * vector_size) * b_normal )

        ### check the eigenvalues
        # _, s, _ = np.linalg.svd(np.dot( sq_Q.T, sq_Q ))
        # print([np.max(s), np.min(s)])
    
    if sc_perturb:
        for j in range(n_node):
            if j == 3 or j == 4:
                # Q[j] += theta * np.eye(vector_size)
                # Q[j] = 50 * Q[j]
                # sq_Q = np.random.normal(loc=2, scale=1, size=(vector_size, vector_size))
                # D = np.diag(np.random.uniform(low=0, high=2, size=vector_size))
                D = np.diag(np.random.uniform(low=2, high=4, size=vector_size))
                # D = np.diag(np.random.normal(1, 1, vector_size))
                # D = np.diag(np.random.normal(10, 1, vector_size))
                # D = np.diag(np.random.normal(3, 1, vector_size))
                # D = np.diag(np.random.normal(0.5, 1, vector_size))
                U = ortho_group.rvs(dim=vector_size)
                # sq_Q = np.dot(D,U)
                # Q[j] = np.dot( sq_Q.T, sq_Q )
                # D = np.diag(np.abs(np.diag(D)))                 
                # D = np.diag( np.where(np.diag(D) < 0, 0, np.diag(D)))
                # print(D)
                Q[j] = np.dot( U.T, np.dot(D,U) )

    problem_data = {'Q' : Q, 'b' : b}
    return problem_data


def graph_generation_nodes6():
    # graph with 6 nodes and 7 edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    # 1 -- 2 -- 4 -- 5
    #   \  |  /  \
    #      3       6 
    n_node = 6

    # (1) network
    G = nx.Graph()
    G.add_nodes_from([1, n_node])
    G.add_edges_from([(1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)])

    # (2) mixing matrices : metropolis-hastings weights
    W = np.zeros((n_node, n_node))
    for edge in list(G.edges) :
        (i,j) = edge
        W[i-1,j-1] = 1/(np.maximum(G.degree(i), G.degree(j))+1)
        W[j-1,i-1] = 1/(np.maximum(G.degree(j), G.degree(i))+1)
    for i in range(n_node) :
        W[i,i] = 1 - np.sum(W[i])
    [_, S, V] = np.linalg.svd((1/2)*(np.eye(W.shape[0])-W))
    Vred = V[0:n_node-1]
    Sred = S[0:n_node-1]

    network_data = {'G' : G, 'W' : W, 'Vred' : Vred, 'Sred' : Sred}
    return network_data


def p_extra_quad_no_constraint(problem_spec, problem_data, network_data, x_opt_star, f_star, printing=False, freq=200) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    rho = problem_data['rho']
    # eps = problem_spec['sc_eps']

    Q = problem_data['Q']
    b = problem_data['b']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    x_0 = np.zeros((n_node,vector_size))
    for j in range(n_node):
        x_0[j] = b[j]

    x_k = np.array(x_0)
    w_0 = np.zeros((n_node,vector_size))
    w_k = np.array(w_0)
    # b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        f_val = 0
        for jj in range(n_node) :
            Q_temp = Q[jj]
            b_temp = b[jj]
            e_k_jj = (W[jj]@x_k_prev - w_k_prev[jj])
            # x_k[jj] = cvx_prox_fj_quad_no_constraint(e_k_jj, rho, problem_spec, problem_data, jj, eps=1e-4)
            # x_k[jj] = prox_fj_quad_no_constraint(e_k_jj, rho, problem_spec, problem_data, jj)
            x_k[jj] = prox_fj_quad_no_constraint(e_k_jj, rho, Q_temp, b_temp, vector_size)
            # f_val += (1/2 * x_k[jj] @ Q[jj] @ x_k[jj].T + b[jj] @ x_k[jj])[0]
            f_val += 1/2 * x_k[jj] @ Q[jj] @ x_k[jj].T + np.dot(b[jj][0], x_k[jj])
        w_k = w_k_prev + 1/2*(np.eye(n_node) - W) @ x_k_prev
        
        op_norm.append(Mnormsq(x_k-x_k_prev, w_k-w_k_prev, rho, network_data, n_node))        

        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        # const_vio.append(np.sum((A@x_k.T - b_stack)**2))
        f_reldiff.append(np.abs((f_val - f_star)/f_star))
        if printing and (ii % freq == 0 or ii == itr_num-1):
            print(f"{ii=}, {f_reldiff[-1]=}, {err_opt_reldiff[-1]=}")

    return op_norm, err_opt_star, err_opt_reldiff, const_vio, f_reldiff

"""
    Helper functions: calculates inner product and norm induced by metric matrix M (which is dependent on rho).

    (1) Minner  : calculates M-inner product between (x_1, w_1) and (x_2, w_2)
    (2) Mnormsq : calculates squared M-norm of (x, w)
    (3) Mnorm   : calculates M-norm of (x, w)
"""

def Minner(x1, w1, x2, w2, rho, network_data, n_node) :
    Vred = network_data['Vred']
    Sred = network_data['Sred']

    norm_squared = (1/rho)*np.sum(x1*x2) + (1/rho)*np.sum(x1*w2) + (1/rho)*np.sum(w1*x2)
    u1 = (1/rho)*Vred.T@((Vred@w1)*np.reshape(1/np.sqrt(Sred), (n_node-1,1)))
    u2 = (1/rho)*Vred.T@((Vred@w2)*np.reshape(1/np.sqrt(Sred), (n_node-1,1)))
    norm_squared = norm_squared + rho*np.sum(u1*u2)

    return norm_squared

def Mnormsq(x, w, rho, network_data, n_node) :
    return 1/4*Minner(x, w, x, w, rho, network_data, n_node)

def Mnorm(x, w, rho, network_data, n_node) :
    return 1/2*np.sqrt(Minner(x, w, x, w, rho, network_data, n_node))