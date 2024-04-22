"""
    Some functions are from  https://github.com/jaewookjsuh/continuous_anchor_acceleration_experiment/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp




def cvx_prox_fj_geom_mean(z, alpha, b, vector_size):
    """
    return minimizer to (1/2)\|x - b\|_2^2 + (1/(2 * alpha)) * \|x - z\|_2^2
    """
    x = cp.Variable(vector_size)
    f = (1/2) * cp.norm(x - b, 2) + (1/(2*alpha)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal", print(prob.status)
    return x.value


def cvx_geom_median(problem_spec, problem_data):
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    b = problem_data['b']

    x = cp.Variable(vector_size)
    f = 0

    for jj in range(n_node) :
        b_temp = b[jj*vector_size : (jj+1)*vector_size]
        f += 1/2 * cp.norm(x - b_temp, 2) 

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal"
    x_cvx = np.repeat(x.value, n_node).reshape(vector_size, n_node).T
    return f.value, x_cvx


def data_generation(problem_spec) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']

    # vectors which geom. median we want to find
    # b = np.random.randn(n_node * vector_size)
    b = np.random.rand(n_node * vector_size) * 200 - 100

    problem_data = {'b' : b}
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


def p_extra_dgeom_median(method_ver, problem_spec, problem_data, network_data, x_opt_star, f_star, params=None) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    alpha = problem_data['alpha']

    b = problem_data['b']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    x_0 = np.zeros((n_node,vector_size))
    x_k = np.array(x_0)
    w_0 = np.zeros((n_node,vector_size))
    w_k = np.array(w_0)
    # b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    e_k = np.array(x_0)

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        e_k_prev = np.array(e_k)
        f_val = 0
        for jj in range(n_node) :
            b_temp = b[jj*vector_size:(jj+1)*vector_size]
            e_k[jj] = b_temp - (W[jj]@x_k_prev - w_k_prev[jj])
            x_k[jj] = b_temp - (e_k[jj] / np.linalg.norm(e_k[jj])) * np.maximum(np.linalg.norm(e_k[jj])-alpha, 0.)

            f_val += 1/2*np.linalg.norm((x_k[jj] - b_temp), ord=2)

        w_k = w_k_prev + 1/2*(np.eye(n_node)-W)@x_k_prev
        
        op_norm.append(Mnormsq(x_k-x_k_prev, w_k-w_k_prev, alpha, network_data, n_node))        

        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        # const_vio.append(np.sum((A@x_k.T - b_stack)**2))
        f_reldiff.append(np.abs(f_star - f_val)/f_star)

    return op_norm, err_opt_star, err_opt_reldiff, const_vio, f_reldiff


"""
    Helper functions: calculates inner product and norm induced by metric matrix M (which is dependent on alpha).

    (1) Minner  : calculates M-inner product between (x_1, w_1) and (x_2, w_2)
    (2) Mnormsq : calculates squared M-norm of (x, w)
    (3) Mnorm   : calculates M-norm of (x, w)
"""

def Minner(x1, w1, x2, w2, alpha, network_data, n_node) :
    Vred = network_data['Vred']
    Sred = network_data['Sred']

    norm_squared = (1/alpha)*np.sum(x1*x2) + (1/alpha)*np.sum(x1*w2) + (1/alpha)*np.sum(w1*x2)
    u1 = (1/alpha)*Vred.T@((Vred@w1)*np.reshape(1/np.sqrt(Sred), (n_node-1,1)))
    u2 = (1/alpha)*Vred.T@((Vred@w2)*np.reshape(1/np.sqrt(Sred), (n_node-1,1)))
    norm_squared = norm_squared + alpha*np.sum(u1*u2)

    return norm_squared

def Mnormsq(x, w, alpha, network_data, n_node) :
    return 1/4*Minner(x, w, x, w, alpha, network_data, n_node)

def Mnorm(x, w, alpha, network_data, n_node) :
    return 1/2*np.sqrt(Minner(x, w, x, w, alpha, network_data, n_node))