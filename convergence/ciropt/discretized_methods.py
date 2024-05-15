import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp





def dadmm(alg_type, problem_spec, problem_data, network_data, x_opt_star, f_star, 
          prox_operators, fi_operators, params=None, printing=False, sc_index_set = None, freq=50):
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    
    # x_0_data = problem_data['x_0_data']
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
    elif alg_type == "cir_dadmm":
        R, h, Inductance = params["R"], params["h"], params["Inductance"]
        step_size = h / Inductance
    elif alg_type == "cir_dadmm_l":
        R, h, Inductance, L_extra = params["R"], params["h"], params["Inductance"], params["L_extra"]
        step_size = h / Inductance
    elif alg_type == "cir_dadmm_c":
        R, h, Inductance, Capacitance = params["R"], params["h"], params["Inductance"], params["Capacitance"]
        step_size = h / Inductance

    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    x_0 = np.reshape(problem_data['b'], (n_node, vector_size))
    # x_0 = np.zeros((n_node, vector_size))    
    x_k = np.array(x_0)
    i_L_k = [np.zeros((deg, vector_size)) for deg in node_degrees]
    e_k = [np.zeros((deg, vector_size)) for deg in node_degrees]
    
    # if alg_type == "cir_dadmm_l":
    #     i_Lextra_k = np.zeros((1, vector_size))

    for ii in range(itr_num):
        if ii >= 1: x_k_prev = np.array(x_k)
        i_L_k_prev = [np.array(el) for el in i_L_k]
        e_k_prev = [np.array(el) for el in e_k]
        # if alg_type == "cir_dadmm_l":
        #     i_Lextra_k_prev = np.array(i_Lextra_k)
        f_val = 0

        for jj in range(n_node):
            zj = (1./node_degrees[jj]) * (R * i_L_k_prev[jj] + e_k_prev[jj]).sum(axis=0)
            # if alg_type == "cir_dadmm_l" and jj == 3:
            #     zj = (1./node_degrees[jj]) * (R * i_L_k_prev[jj] + e_k_prev[jj] - R * i_Lextra_k_prev).sum(axis=0)
            # elif alg_type == "cir_dadmm_l" and jj == 4:
            #     zj = (1./node_degrees[jj]) * (R * i_L_k_prev[jj] + e_k_prev[jj] + R * i_Lextra_k_prev).sum(axis=0)
            # else:
            #     zj = (1./node_degrees[jj]) * (R * i_L_k_prev[jj] + e_k_prev[jj]).sum(axis=0)
            x_k[jj] = prox_operators[jj](zj, R/node_degrees[jj])
            f_val += fi_operators[jj](x_k[jj])

        for j in range(n_node):
            for l_idx, l in enumerate(adjacency[j]):
                # if alg_type == "cir_dadmm_c" and ((j == 3 and l==4) or (j==4 and l==3)):
                if alg_type == "cir_dadmm_c" and {j,l}.issubset(sc_index_set):
                    j_idx = adjacency[l].index(j)
                    e_k[j][l_idx] = e_k_prev[j][l_idx] - (h / Capacitance) * (i_L_k_prev[j][l_idx] + i_L_k_prev[l][j_idx] + (2 * e_k_prev[j][l_idx] - x_k[j] - x_k[l])/R)
                else:
                    e_k[j][l_idx] = (1/2) * (x_k[j] + x_k[l])
        for j in range(n_node):
            i_L_k[j] = i_L_k_prev[j] + step_size * ( e_k[j] - x_k[j])
            
        # if alg_type == "cir_dadmm_l":
        #     i_Lextra_k = i_Lextra_k_prev + (h / L_extra) * (x_k[3] - x_k[4])
        
        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        # const_vio.append(np.sum((A@x_k.T - b_stack)**2))
        f_reldiff.append(np.abs((f_val - f_star)/f_star))
        if printing and (ii % freq == 0 or ii == itr_num-1):
            # print(f"{ii=}, {f_reldiff[-1]=}, {err_opt_reldiff[-1]=}")
            print(f"{ii=}, {f_reldiff[-1]=}")

    return err_opt_star, err_opt_reldiff, const_vio, f_reldiff



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

def pg_extra(alg_type, problem_spec, problem_data, network_data, x_opt_star, f_star, 
          prox_operators, fi_operators, params=None, printing=False, sc_index_set = None, freq=50) :
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    rho = problem_data['rho']
    # eps = problem_spec['sc_eps']

    Q = problem_data['Q']
    b = problem_data['b']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    if alg_type == "pg_extra": 
        step_size_L = 1/2
    elif alg_type == "pg_extra_c":
        R, h, Capacitance = params["R"], params["h"], params["Capacitance"]
        rho = R
        step_size_L = h
        step_size_C = h / Capacitance
        R_matrix = 1/R * W

    err_opt_star, err_opt_reldiff, op_norm, const_vio, f_reldiff = [], [], [], [], []

    x_0 = np.zeros((n_node,vector_size))
    for j in range(n_node):
        x_0[j] = b[j]

    x_k = np.array(x_0)
    w_0 = np.zeros((n_node,vector_size))
    w_k = np.array(w_0)
    # b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    if alg_type=="pg_extra_c":
        i_L_0 = np.zeros((n_node,n_node,vector_size))
        i_L_k = np.array(i_L_0)
        tile_0 = np.zeros((n_node,n_node,vector_size))
        tile_k = np.array(tile_0)
        bridge_0 = np.zeros((n_node,n_node,vector_size))
        bridge_k = np.array(bridge_0)

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        f_val = 0

        if alg_type=="pg_extra":
            w_k = w_k_prev + step_size_L*(np.eye(n_node) - W) @ x_k_prev            
        elif alg_type=="pg_extra_c":
            tile_k_prev = np.array(tile_k)
            i_L_k_prev = np.array(i_L_k)
            for jj in range(n_node) :
                for ll in range(n_node) : # change to adjacency[ll]
                    if jj != ll :
                        if {jj,ll}.issubset(sc_index_set):
                            bridge_k[jj][ll] = i_L_k_prev[jj][ll] + R_matrix[jj][ll] * ( x_k_prev[jj] - tile_k_prev[jj][ll] )
                            i_L_k[jj][ll] = i_L_k_prev[jj][ll] + step_size_L * R_matrix[jj][ll] * ( x_k_prev[jj] - tile_k_prev[jj][ll] )
                        else :
                            bridge_k[jj][ll] = i_L_k_prev[jj][ll] + R_matrix[jj][ll] * ( x_k_prev[jj] - x_k_prev[ll] )
                            i_L_k[jj][ll] = i_L_k_prev[jj][ll] + step_size_L * R_matrix[jj][ll] * ( x_k_prev[jj] - x_k_prev[ll] )
            for jj in range(n_node) :
                for ll in range(n_node) :
                    if jj != ll :
                        tile_k[jj][ll] = tile_k_prev[jj][ll] + step_size_C * ( bridge_k[jj][ll] + bridge_k[ll][jj] )

        for jj in range(n_node) :
            Q_temp = Q[jj]
            b_temp = b[jj]
            # x_k_prev_temp = x_k_prev[jj]
            if alg_type=="pg_extra":
                e_k_jj = (W[jj]@x_k_prev - w_k_prev[jj])
            elif alg_type=="pg_extra_c":                
                sum_bridge = np.array(np.zeros(vector_size))
                for ll in range(n_node) : # change to adjacency[ll]
                    sum_bridge += bridge_k[jj][ll]
                e_k_jj = x_k_prev[jj] - R * sum_bridge
            x_k[jj] = prox_operators[jj](e_k_jj, rho, Q_temp, b_temp, vector_size)
            # f_val += (1/2 * x_k[jj] @ Q[jj] @ x_k[jj].T + b[jj] @ x_k[jj])[0]
            f_val += 1/2 * x_k[jj] @ Q[jj] @ x_k[jj].T + np.dot(b[jj][0], x_k[jj])
        
        op_norm.append(Mnormsq(x_k-x_k_prev, w_k-w_k_prev, rho, network_data, n_node))        

        err_opt_star.append(np.sqrt(np.sum((x_k - x_opt_star)**2)))
        err_opt_reldiff.append(np.sqrt(np.sum((x_k - x_opt_star)**2)) / np.sqrt(np.sum((x_0 - x_opt_star)**2)))
        # const_vio.append(np.sum((A@x_k.T - b_stack)**2))
        f_reldiff.append(np.abs((f_val - f_star)/f_star))
        if printing and (ii % freq == 0 or ii == itr_num-1):
            print(f"{ii=}, {f_reldiff[-1]=}, {err_opt_reldiff[-1]=}")

    return op_norm, err_opt_star, err_opt_reldiff, const_vio, f_reldiff



