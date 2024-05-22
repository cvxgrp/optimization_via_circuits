"""
    Some functions are from  https://github.com/jaewookjsuh/continuous_anchor_acceleration_experiment/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp

from scipy.stats import ortho_group



def prox_fj_geom_mean(z, rho, b, rescale):
    """
    return minimizer to \|x - b\|_2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    rho = rescale * rho
    e = b - z
    x = b - (e / np.linalg.norm(e)) * np.maximum(np.linalg.norm(e) - rho, 0.)
    return x

def prox_fj_geom_mean_sc(z, rho, b, c, Qj, theta=1e-4):
    """
    return minimizer to theta * \|Q x - c\|_2^2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    I = np.eye(Qj.shape[1])
    A = 2 * theta * np.dot(Qj.T, Qj) + (1 / rho) * I
    b = 2 * theta * np.dot(Qj.T, c) + (1 / rho) * z
    x_star = np.linalg.solve(A, b)
    return x_star

def cvx_prox_fj_geom_mean(z, rho, b, rescale, vector_size):
    """
    return minimizer to rescale * \|x - b\|_2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    x = cp.Variable(vector_size)
    f = rescale * cp.norm(x - b, 2) + (1/(2*rho)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal", print(prob.status)
    return x.value


def cvx_prox_fj_geom_mean_sc(z, rho, b, c, Qj, vector_size, theta=1e-4):
    """
    return minimizer to  theta * \|x - c\|_2^2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    x = cp.Variable(vector_size)
    f = theta * cp.sum_squares( Qj @ x - c ) + (1/(2*rho)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal", print(prob.status)
    return x.value


def cvx_geom_median(problem_spec, problem_data):
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    theta = problem_spec['sc_theta']
    rescale = problem_spec['rescale']
    b = problem_data['b']
    c = problem_data['c']
    Q = problem_data['Q']

    x = cp.Variable(vector_size)
    f = 0

    for jj in range(n_node) :
        b_temp = b[jj*vector_size : (jj+1)*vector_size]
        c_temp = c[jj*vector_size : (jj+1)*vector_size]
        if jj in [3, 4]:
            f += theta * cp.sum_squares( Q[jj] @ x -c_temp) 
        else:
            f += rescale * cp.norm(x - b_temp, 2)  

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal"
    x_cvx = np.repeat(x.value, n_node).reshape(vector_size, n_node).T
    return f.value, x_cvx


def data_generation(problem_spec) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    translation = problem_spec['translation']
    sc_index_set = problem_spec['sc_index_set'] 
    Q = []

    # vectors which geom. median we want to find
    # b = np.random.randn(n_node * vector_size)
    b = np.random.rand(n_node * vector_size) * 200 - 100
    # b = np.random.rand(n_node * vector_size) * 10000 - 100
    
    if translation=="b":
        c = b
    elif translation=="rand":
        # c = np.random.rand(n_node * vector_size) * 100000 - 100
        c = np.random.rand(n_node * vector_size) * 200 - 100
    else:
        c = np.zeros(n_node * vector_size)

    for j in range(n_node):
        Q.append(np.zeros((vector_size, vector_size)))

    for j in sc_index_set:
        # D = np.diag(np.arange(1, vector_size + 1))
        # D = np.diag(np.arange(1, vector_size + 1))
        # D = np.diag(np.linspace(1, 300, vector_size))
        D = np.diag(np.linspace(1, 1000, vector_size))
        U = ortho_group.rvs(dim=vector_size)
        sq_Q = np.dot(D,U)
        Q[j] = np.dot( sq_Q.T, sq_Q )

    problem_data = {'b' : b, 'c' : c, 'Q' : Q}
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


def p_extra_dgeom_median(method_ver, problem_spec, problem_data, network_data, x_opt_star, f_star, printing=False, freq=200, params=None) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    rho = problem_data['rho']
    theta = problem_spec['sc_theta']
    rescale = problem_spec['rescale']

    b = problem_data['b']
    c = problem_data['c']
    Q = problem_data['Q']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    # x_0 = np.reshape(b, (n_node, vector_size))
    x_0 = np.zeros((n_node, vector_size))
    x_k = np.array(x_0)
    w_0 = np.zeros((n_node,vector_size))
    w_k = np.array(w_0)

    e_k = np.array(x_0)

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        f_val = 0
        for jj in range(n_node) :
            b_temp = b[jj*vector_size : (jj+1)*vector_size]
            c_temp = c[jj*vector_size : (jj+1)*vector_size]
            z_temp = (W[jj]@x_k_prev - w_k_prev[jj])
            Q_temp = Q[jj]
            if jj in [3, 4]:
                x_k[jj] = prox_fj_geom_mean_sc(z_temp, rho, b_temp, c_temp, Q_temp, theta=theta)
                f_val += theta * np.linalg.norm( Q_temp @ x_k[jj] - c_temp, ord=2)**2
            else:
                e_k[jj] = b_temp - z_temp
                x_k[jj] = prox_fj_geom_mean(z_temp, rho, b_temp, rescale)
                f_val += rescale * np.linalg.norm((x_k[jj] - b_temp), ord=2)

        w_k = w_k_prev + 1/2*(np.eye(n_node) - W) @ x_k_prev
        
        # op_norm.append(Mnormsq(x_k-x_k_prev, w_k-w_k_prev, rho, network_data, n_node))        

        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        f_reldiff.append(np.abs(f_star - f_val)/f_star)
        if printing and (ii % freq == 0 or ii == itr_num-1):
            print(f"{ii=}, {f_reldiff[-1]=}, {err_opt_reldiff[-1]=}")

    return op_norm, err_opt_star, err_opt_reldiff, const_vio, f_reldiff