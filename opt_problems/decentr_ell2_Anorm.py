"""
    Some functions are from  https://github.com/jaewookjsuh/continuous_anchor_acceleration_experiment/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp




def prox_fj_geom_mean(z, rho, b):
    """
    return minimizer to \|x - b\|_2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    e = b - z
    x = b - (e / np.linalg.norm(e)) * np.maximum(np.linalg.norm(e) - rho, 0.)
    return x

def cvx_prox_fj_geom_mean(z, rho, b, vector_size):
    """
    return minimizer to \|x - b\|_2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    x = cp.Variable(vector_size)
    f = cp.norm(x - b, 2) + (1/(2*rho)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal", print(prob.status)
    return x.value


def prox_fj_geom_mean_sc(z, rho, b, P):
    """
    return minimizer to  (1/2)\|x - b\|_P^2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    tilde_P = P + np.eye(P.shape[0]) / rho
    new_z = b + np.linalg.solve(tilde_P, (z - b) / rho )
    return new_z

def cvx_prox_fj_geom_mean_sc(z, rho, b, P, vector_size):
    """
    return minimizer to  (1/2)\|x - b\|_P^2 + (1/(2 * rho)) * \|x - z\|_2^2
    """
    x = cp.Variable(vector_size)
    f = 0.5 * cp.quad_form(x - b, P) + (1/(2*rho)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal", print(prob.status)
    return x.value


def cvx_geom_median(problem_spec, problem_data):
    n_nodes = problem_spec['n_nodes']
    vector_size = problem_spec['vector_size']
    b = problem_data['b']
    P = problem_data['P']

    x = cp.Variable(vector_size)
    f = 0

    for jj in range(n_nodes) :
        bj = b[jj*vector_size : (jj+1)*vector_size]
        Pj = P[jj]  
        if jj in [3, 4]:
            f += 0.5 * cp.quad_form(x - bj, Pj) 
        else:
            f += cp.norm(x - bj, 2)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal"
    x_cvx = np.repeat(x.value, n_nodes).reshape(vector_size, n_nodes).T
    return f.value, x_cvx


def data_generation(problem_spec) :
    n_nodes = problem_spec['n_nodes']
    vector_size = problem_spec['vector_size']

    # vectors which geom. median we want to find
    # b = np.random.randn(n_nodes * vector_size)
    b = np.random.rand(n_nodes * vector_size) * 200 - 100
    
    Ps = []
    for _ in range(n_nodes):
        Q = np.linalg.qr(np.random.randn(vector_size, vector_size))[0]
        eigs = 1 + 10**(np.random.rand(vector_size) * 3)
        P = (Q * eigs) @ Q.T
        assert np.allclose(np.sort(eigs), np.sort(np.linalg.eigvalsh(P))), \
                print(np.linalg.norm(np.sort(eigs) - np.sort(np.linalg.eigvalsh(P))))
        Ps += [P]

    problem_data = {'b' : b, 'P' : Ps}
    return problem_data


def graph_generation_nodes6():
    # graph with 6 nodes and 7 edges (1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)
    # 1 -- 2 -- 4 -- 5
    #   \  |  /  \
    #      3       6 
    n_nodes = 6

    # (1) network
    G = nx.Graph()
    G.add_nodes_from([1, n_nodes])
    G.add_edges_from([(1,2), (1,3), (2,3), (2,4), (3,4), (4,5), (4,6)])

    # (2) mixing matrices : metropolis-hastings weights
    W = np.zeros((n_nodes, n_nodes))
    for edge in list(G.edges) :
        (i,j) = edge
        W[i-1,j-1] = 1/(np.maximum(G.degree(i), G.degree(j))+1)
        W[j-1,i-1] = 1/(np.maximum(G.degree(j), G.degree(i))+1)
    for i in range(n_nodes) :
        W[i,i] = 1 - np.sum(W[i])
    [_, S, V] = np.linalg.svd((1/2)*(np.eye(W.shape[0])-W))
    Vred = V[0:n_nodes-1]
    Sred = S[0:n_nodes-1]

    network_data = {'G' : G, 'W' : W, 'Vred' : Vred, 'Sred' : Sred}
    return network_data


def p_extra_dgeom_median(method_ver, problem_spec, problem_data, network_data, x_opt_star, f_star, printing=False, freq=200) :
    n_nodes = problem_spec['n_nodes']
    vector_size = problem_spec['vector_size']
    rho = problem_data['rho']

    b = problem_data['b']
    P = problem_data['P']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    err_opt_star, err_opt_reldiff, const_vio, f_reldiff = [], [], [], []

    x_0 = np.reshape(b, (n_nodes, vector_size))
    x_k = np.array(x_0)
    w_0 = np.zeros((n_nodes,vector_size))
    w_k = np.array(w_0)

    e_k = np.array(x_0)

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        f_val = 0
        for jj in range(n_nodes) :
            bj = b[jj*vector_size : (jj+1)*vector_size]
            Pj = P[jj]
            z_temp = (W[jj]@x_k_prev - w_k_prev[jj])
            if jj in [3, 4]:
                x_k[jj] = prox_fj_geom_mean_sc(z_temp, rho, bj, Pj) 
                f_val += 0.5 * (x_k[jj] - bj).T @ Pj @ (x_k[jj] - bj)
            else:
                e_k[jj] = bj - z_temp
                x_k[jj] = prox_fj_geom_mean(z_temp, rho, bj)
                f_val += np.linalg.norm((x_k[jj] - bj), ord=2)

        w_k = w_k_prev + 1/2*(np.eye(n_nodes) - W) @ x_k_prev
              
        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        f_reldiff.append(np.abs(f_star - f_val)/f_star)
        if printing and (ii % freq == 0 or ii == itr_num-1):
            print(f"{ii=}, {f_reldiff[-1]=}, {err_opt_reldiff[-1]=}")

    return err_opt_star, err_opt_reldiff, const_vio, f_reldiff

