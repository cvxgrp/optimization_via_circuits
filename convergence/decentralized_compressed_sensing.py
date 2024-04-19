"""
    Some functions are from  https://github.com/jaewookjsuh/continuous_anchor_acceleration_experiment/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp




def cvx_prox_fj_hj(z, alpha, rho, A_temp, b_temp, vector_size):
    """
    return minimizer to (1/2)\|A_jx - b_j\|_2^2 + \rho\|x\|_1 + (1/(2 * alpha)) * \|x - z\|_2^2
    """
    x = cp.Variable(vector_size)
    f = (1/2) * cp.sum_squares(A_temp @ x - b_temp) + rho * cp.sum(cp.abs(x)) + (1/(2*alpha)) * cp.sum_squares(x - z)

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal"
    return x.value


def dadmm(alg_type, problem_spec, problem_data, network_data, x_opt_star, f_star, 
          params=None, printing=False, freq=50):
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    alpha = problem_data['alpha']
    rho = problem_data['rho']
    A = problem_data['A']
    b = problem_data['b']
    x_0_data = problem_data['x_0_data']
    x_star = problem_data['x_star']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    G = network_data["G"]
    node_degrees = [degree for node, degree in sorted(G.degree(), key=lambda x: x[0])]
    adjacency = [[] for _ in range(G.number_of_nodes())]
    for node, adjacencies in G.adjacency():
        adjacency[node-1] = [e-1 for e in list(adjacencies.keys())]
    if alg_type == "dadmm": 
        R = params["R"]
        step_size = 1./R
    elif alg_type == "circuit":
        R, h, Inductance = params["R"], params["h"], params["Inductance"]
        step_size = h / Inductance

    As, bs = [], []
    for jj in range(n_node):
        As += [A[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]]
        bs += [b[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]]
    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    x_0 = np.array(x_0_data) # (n_node, vector_size)
    x_k = np.array(x_0)
    i_L_k = [np.zeros((deg, vector_size)) for deg in node_degrees]
    e_k = [np.zeros((deg, vector_size)) for deg in node_degrees]
    b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))


    for ii in range(itr_num):
        if ii >= 1: x_k_prev = np.array(x_k)
        i_L_k_prev = [np.array(el) for el in i_L_k]
        e_k_prev = [np.array(el) for el in e_k]
        f_val = 0

        for jj in range(n_node):
            zj = (1./node_degrees[jj]) * (R * i_L_k_prev[jj] + e_k_prev[jj]).sum(axis=0)
            x_k[jj] = cvx_prox_fj_hj(zj, R/node_degrees[jj], rho, As[jj], bs[jj], vector_size)
            f_val += 1/2*np.sum((As[jj] @ x_k[jj] - bs[jj])**2) + rho * np.sum(np.abs(x_k[jj]))

        for j in range(n_node):
            for l_idx, l in enumerate(adjacency[j]):
                e_k[j][l_idx] = (1/2) * (x_k[j] + x_k[l])
        for j in range(n_node):
            i_L_k[j] = i_L_k_prev[j] + step_size * ( e_k[j] - x_k[j])
        
        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        const_vio.append(np.sum((A@x_k.T - b_stack)**2))
        f_reldiff.append(np.abs(f_star - f_val)/f_star)
        if printing and ii % freq == 0:
            print(f"{ii=}, {f_reldiff[-1]=}, {err_opt_reldiff[-1]=}")

    return err_opt_star, err_opt_reldiff, const_vio, f_reldiff


def cvx_compressed_sensing(problem_spec, problem_data):
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    rho = problem_data['rho']
    A = problem_data['A']
    b = problem_data['b']

    x = cp.Variable(vector_size)
    f = n_node * rho * cp.sum(cp.abs(x))

    for jj in range(n_node) :
        A_temp = A[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]
        b_temp = b[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]

        f += 1/2 * cp.sum_squares(A_temp @ x - b_temp) 

    prob = cp.Problem(cp.Minimize(f), [])
    prob.solve()
    assert prob.status=="optimal"
    x_cvx = np.repeat(x.value, n_node).reshape(vector_size, n_node).T
    return f.value, x_cvx


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



"""
    These functions generate the decentralized problem and solves it with PG-EXTRA and its combination with anchor acceleration.

    (1) data_generation : randomly generates problem data including
        - the measurement matrix A
        - (noisy) measurement b
        - true solution x_true, and its stacked version x_star
        - initial value x_0_data
    
    (2) graph_generation : generate the graph representing the network the decentralized optimization problem is to be solved.

    (3) obtain_opt_prob_solution : obtain true solution to the L1-regularized problem, which can be different from x_true due to noisy measurement.
"""

def data_generation(problem_spec) :
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    n_nonzero_entry = problem_spec['n_nonzero_entry']
    
    # (1) sparse signal x_true
    x_true = np.zeros(vector_size)
    x_true[:n_nonzero_entry] = np.random.randn(n_nonzero_entry)
    np.random.shuffle(x_true)   # sparse signal x_true
    x_star = np.array(n_node*[x_true])

    # (2) stack of initial iterates x_0_data
    x_0_data = np.zeros((n_node, vector_size))
    # x_0_data = np.random.randn(n_node, vector_size)

    # (3) measurement matrices
    A = np.random.randn(n_sensor, vector_size)
    for ii in range(n_node) :   # normalize each measurement matrices
        A[n_sensor_per_node*ii:n_sensor_per_node*(ii+1)] = A[n_sensor_per_node*ii:n_sensor_per_node*(ii+1)] / np.linalg.norm(A[n_sensor_per_node*ii:n_sensor_per_node*(ii+1)])
    b = A@x_true + 0.01 * np.random.randn(n_sensor)    # noisy measurements
    # b = A@x_true # exact measurement

    problem_data = {'x_true' : x_true, 'x_star' : x_star, 'x_0_data' : x_0_data, 'A' : A, 'b' : b}
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


def obtain_opt_prob_solution(problem_spec, problem_data, network_data) :
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    alpha = problem_data['alpha']
    rho = problem_data['rho']
    A = problem_data['A']
    b = problem_data['b']
    x_0_data = problem_data['x_0_data'] 
    itr_num = problem_data['itr_num']
    W = network_data['W']

    x_0 = np.array(x_0_data) # (n_node, vector_size)
    x_k = np.array(x_0)
    w_k = np.zeros((n_node,vector_size))
    b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    for _ in range(2*itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        for jj in range(n_node):
            A_temp = A[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]
            b_temp = b[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]
            x_k[jj] = W[jj]@x_k_prev - alpha * A_temp.T@(A_temp@x_k_prev[jj] - b_temp) - w_k_prev[jj]
            x_k[jj] = np.sign(x_k[jj]) * np.maximum(np.abs(x_k[jj])- alpha * rho, 0.)
        w_k = w_k_prev + 1/2*(np.eye(n_node) - W) @ x_k_prev
    x_opt_star = x_k
    f_star = 0
    for jj in range(n_node):
        A_temp = A[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]
        b_temp = b[jj*n_sensor_per_node : (jj+1)*n_sensor_per_node]
        f_star += 1/2*np.sum((A_temp @ x_k[jj] - b_temp)**2) + rho * np.sum(np.abs(x_k[jj]))
  
    return x_opt_star, f_star


def pg_extra(method_ver, problem_spec, problem_data, network_data, x_opt_star, f_star, params=None) :
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    alpha = problem_data['alpha']
    rho = problem_data['rho']
    A = problem_data['A']
    b = problem_data['b']
    x_0_data = problem_data['x_0_data']
    x_star = problem_data['x_star']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    x_0 = np.array(x_0_data)
    x_k = np.array(x_0)
    w_0 = np.zeros((n_node,vector_size))
    w_k = np.array(w_0)
    b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    e_k = np.array(x_0)

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        e_k_prev = np.array(e_k)
        f_val = 0
        for jj in range(n_node) :
            A_temp = A[jj*n_sensor_per_node:(jj+1)*n_sensor_per_node]
            b_temp = b[jj*n_sensor_per_node:(jj+1)*n_sensor_per_node]
            e_k[jj] = W[jj]@x_k_prev - alpha*A_temp.T@(A_temp@x_k_prev[jj]-b_temp) - w_k_prev[jj]
            x_k[jj] = np.sign(e_k[jj])*np.maximum(np.abs(e_k[jj])-alpha*rho, 0.)
            f_val += 1/2*np.sum((A_temp @ x_k[jj] - b_temp)**2) + rho * np.sum(np.abs(x_k[jj]))
        w_k = w_k_prev + 1/2*(np.eye(n_node)-W)@x_k_prev
        
        op_norm.append(Mnormsq(x_k-x_k_prev, w_k-w_k_prev, alpha, network_data, n_node))        

        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        const_vio.append(np.sum((A@x_k.T - b_stack)**2))
        f_reldiff.append(np.abs(f_star - f_val)/f_star)

    return op_norm, err_opt_star, err_opt_reldiff, const_vio, f_reldiff

