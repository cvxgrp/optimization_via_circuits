"""
    Some functions are from  https://github.com/jaewookjsuh/continuous_anchor_acceleration_experiment/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
from scipy.stats import ortho_group


def prox_fj_quad_constraint(z, rho, aj, bj):
    """
    return minimizer to \delta_{ x | a_j^T x - b <=0 } + 1/2*rho || x - z ||^2
    """
    if aj.T @ z - bj <= 0: 
        x = z
    else:
        x = z + (bj - aj.T @ z) / np.sum(np.square(aj))  * aj
    return x

def grad_hj_quad_constraint(z, Qj, pj):
    """
    return gradient of 1/2 * x^T * Q_j * x + p_j^T * x
    """
    return Qj @ z + pj


def min_cvx_quad_constraint(problem_spec, problem_data, contrained):
    """
    return minimizer to sum_{j} 1/2 * x^T * Q_j * x + sum_{j} p_j^T * x 
             subject to a_i^T * x <= b_i
    """

    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    Q = problem_data['Q']
    p = problem_data['p']
    a = problem_data['a']
    b = problem_data['b']

    x = cp.Variable(vector_size)
    f = 0
    for j in range(n_node):
        f += 1/2 * cp.quad_form(x, Q[j]) + cp.sum(cp.multiply(p[j], x))

    constraints = []
    if contrained:        
        for j in range(n_node):
            constraints.append( a[j].T @ x <= b[j] )

    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve()
    assert prob.status=="optimal"
    return f.value, x.value


def data_generation(problem_spec) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    theta = problem_spec['sc_theta'] 
    sc_perturb = problem_spec['sc_perturb'] 
    sc_index_set = problem_spec['sc_index_set'] 

    Q, p = [], []
    for j in range(n_node):
        p_normal = np.random.normal(loc=0, scale=1, size=vector_size)
        sq_Q = np.random.normal(loc=0, scale=1, size=(vector_size, vector_size))
        Q.append( 1/n_node * np.dot( sq_Q.T, sq_Q ) )
        p.append( 1/n_node * p_normal )
    
    a, b = [], []
    for j in range(n_node):
        a_normal = np.random.normal(loc=1, scale=1, size=vector_size)
        b_normal = np.random.normal(0, 1)
        a.append( a_normal )
        b.append( b_normal )
    
        if sc_perturb and (j in sc_index_set):
            ### Uniform distributed diagonal
            # D = np.diag(np.random.uniform(low=0, high=2, size=vector_size))
            # D = np.diag(np.random.uniform(low=2, high=4, size=vector_size))

            ### Gaussian diagonals
            D = np.diag(np.random.normal(np.sqrt(theta), 1, vector_size))                
            U = ortho_group.rvs(dim=vector_size)
            sq_Q = np.dot(D,U)
            Q[j] += np.dot( sq_Q.T, sq_Q )
    problem_data = {'Q' : Q, 'p' : p, 'a' : a, 'b' : b}
    return problem_data


def graph_generation(n_node):
    # (1) network
    G = nx.Graph()
    G.add_nodes_from([1, n_node])

    G.add_edges_from([ (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), \
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), \
    (18, 19), (19, 20), (1, 4), (1, 5), (2, 5), (3, 6), (5, 8), (7, 10), (9, 12), \
    (11, 14), (13, 16), (15, 18), (17, 20), (2, 7), (4, 9), (6, 11), (8, 13), (10, 15), (12, 17), (14, 19) ])

    # G = nx.complete_graph(n_node)

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