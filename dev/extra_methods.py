import casadi as ca
import numpy as np
import sympy as sp

from ciropt.point import Point
from ciropt.expression import Expression
from ciropt.function import Function
from ciropt.utils import *



  

def gp_solve_qcqp_sdp_relax(verbose=True, debug=False, time_limit=1000, psm=1, ps=10, heur=0.001, method=0, bounds=None): 
    # formulate problem explicitly as QCQP using x and matrix X
    # and relax it to convex SDP to get bounds on the variables
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    sp_exp = {}
    core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
    bounds_names = sorted(['alpha', 'beta', 'h', 'b', 'd', 'P', "lamb"])
    sp_v = core_vars[:]
    for f_idx, function in enumerate(list_of_leaf_functions):
        function.add_class_constraints()
        size_I_function = len(function.list_of_points)
        counter = 0
        for i in range(size_I_function):
            for j in range(size_I_function):
                if i == j: continue
                f_interpolation_ineq_ij = function.list_of_class_constraints[counter]
                Fweights_ij, Gweights_ij = _expression_to_matrix(f_interpolation_ineq_ij, dim_G, dim_F)
                sp_exp[(i, j)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                counter += 1
        assert len(function.list_of_class_constraints) == size_I_function * (size_I_function - 1), \
            print(len(function.list_of_class_constraints), size_I_function * (size_I_function - 1), size_I_function)

    descent = E_2 - E_1 + Delta_1
    Fweights_d, Gweights_d = _expression_to_matrix(descent, dim_G, dim_F) 
    sp_exp["FG_d"] = {"F" : Fweights_d, "G" : Gweights_d}

    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)

    num_fi = len(list_of_leaf_functions)
    v_size = len(v_names)
    x_size = len(v_names) + num_fi * size_I_function * (size_I_function-1) +  dim_G * (dim_G + 1) // 2
    
    model = gp.Model() 
    # bounds
    bounds_vars = model.addMVar((len(bounds_names)), name="bounds")
    # SDP variables
    var_x = model.addMVar((x_size, 1), name="x", lb=-100000, ub=100000)
    sqrt_Z = model.addMVar((x_size + 1, x_size + 1))
    var_X = model.addMVar((x_size, x_size))
    Z = sqrt_Z @ sqrt_Z.T
    model.update()
    model.addConstr( var_x[name2idx["b"]].item() >= 0 )
    model.addConstr( var_x[name2idx["h"]].item() >= 0 )
    model.addConstr( var_x[name2idx["d"]].item() >= 0 )
    # constraints to encode X \succeq x x^T
    model.addConstr( sqrt_Z.diagonal() >= np.zeros(x_size + 1) )
    # implied linear constraints
    model.addConstrs( Z[i, i].item() >= 0 for i in range(x_size + 1))
    model.addConstrs( Z[i, j].item() >= -(Z[i, i] + Z[j, j])/2 for i in range(x_size + 1) for j in range(x_size + 1) )
    model.addConstrs( Z[i, j].item() <=  (Z[i, i] + Z[j, j])/2 for i in range(x_size + 1) for j in range(x_size + 1) )
    model.addConstrs( Z[i, j].item() ==  Z[j, i].item() for i in range(x_size + 1) for j in range(x_size + 1) )
    for i in range(x_size):  # sqrt_Z is lower triangular
        for j in range(i + 1, x_size):
            model.addConstr( sqrt_Z[i, j] == 0 )
    model.addConstr(Z[ : x_size, : x_size] == var_X)
    model.addConstr(Z[x_size : x_size+1, : x_size] == var_x.T)
    model.addConstr(Z[ : x_size, x_size : x_size+1] == var_x)
    model.addConstr(Z[x_size, x_size].item() == 1)

    model.addConstr(var_X == var_X.T)
    diag_X = var_X.diagonal().reshape(-1, 1)
    model.addConstr( diag_X >= np.zeros((x_size, 1)) )
    # is true when equality holds X = x x^T
    model.addConstr( var_X >= -(np.ones((x_size, 1)) @ diag_X.T + diag_X @ np.ones((x_size, 1)).T ) / 2 )
    model.addConstr( var_X <=  (np.ones((x_size, 1)) @ diag_X.T + diag_X @ np.ones((x_size, 1)).T ) / 2 )
    
    # model.addConstr( var_x[name2idx["alpha"]].item() >= -1)
    # model.addConstr( var_x[name2idx["alpha"]].item() <= 1 )
    # model.addConstr( var_x[name2idx["beta"]].item() >= -1)
    # model.addConstr( var_x[name2idx["beta"]].item() <= 1 )
    model.addConstr( var_x[0].item() == 1 )

    # additional RLT cuts constraints
    # ...

    assert v_k_list[-1] == "FG_d", print(v_k_list)

    vec_indices = { "v"   : [0, v_size - 1],\
                    "lamb": [v_size, v_size + num_fi * size_I_function * (size_I_function-1) - 1 ], \
                    "P"  : [v_size + num_fi * size_I_function * (size_I_function-1), x_size - 1]} 

    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)

    # matrix coefficient for F is 0 constraints
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
    assert sum_ij_F.shape == (I_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs.shape[0], dim_F, v_size))
    res = np.zeros((dim_F, 1))
    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs.shape[0], I_v.shape[0]), \
        print(obj_F[k : k+1, :].shape, I_v.shape, sum_ij_F[:, k, :].shape, I_lambs.shape) 
        model.addConstr(  (obj_F[k : k+1, :] @ I_v @ var_x).item()  - gp_trace(I_lambs.T @ sum_ij_F[:, k, :] @ I_v @ var_X) == 0)

    # matrix coefficient for G is 0 constraints
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
            model.addConstr(  (obj_G[k_idx : k_idx+1, :] @ I_v @ var_x).item()  \
                            + gp_trace(S1.T @ S2 @ var_X) \
                            - gp_trace(I_lambs.T @ sum_ij_G[:, k_idx, :] @ I_v @ var_X) == 0)

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    # v_quad_constraints = []
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        # print(f"{name=}, {pref_v=}, {v=}")
        ai = - one_hot(v_size, name2idx[name])
        # include both permutations for the product 
        # Qi1 = prod_one_hot((v_size, v_size), name2idx[pref_v], name2idx[v]) 
        Qi2 = prod_one_hot((v_size, v_size), name2idx[v], name2idx[pref_v])
        # v_quad_constraints += [(ai, Qi1), (ai, Qi2)]
        # opti.subject_to( vec_v.T @ Qi1 @ vec_v + ai.T @ vec_v == 0)
        model.addConstr( gp_trace(I_v.T @ Qi2 @ I_v @ var_X) + (ai.T @ I_v @ var_x).item() == 0)

    # lambda >= 0 constraints
    model.addConstr( I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1)))
    # diag(P) >= 0 constraints
    vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
    Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
    np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
    # P_diag_constraints = [Q]
    # print( Q.shape, vec_P.shape, dim_G)
    model.addConstr( Q @ I_P @ var_x >= np.zeros((dim_G, 1)) )

    # bounds constraints
    for b_idx, name in enumerate(bounds_names):
        if name in vec_indices:
            var_name = get_vec_var(var_x, name, vec_indices)
            model.addConstrs(var_name[i, 0].item() >= -bounds_vars[b_idx].item() for i in range(var_name.size))
            model.addConstrs(var_name[i, 0].item() <= bounds_vars[b_idx].item() for i in range(var_name.size))
        else:
            model.addConstr(var_x[name2idx[name], 0].item() <= bounds_vars[b_idx].item())
            model.addConstr(var_x[name2idx[name], 0].item() >= -bounds_vars[b_idx].item())

    if not verbose:
        model.Params.LogToConsole = 0
    model.update()
    gp_vars = {name : var_x[name2idx[name]].item() for name in core_vars}
    for i in range(bounds_vars.size):
        model.setObjectiveN(-bounds_vars[i].item(), index=i, priority=1)
    # model.setObjective(-bounds_vars[0].item())
    
    model.Params.TimeLimit = time_limit
    model.Params.FeasibilityTol = 1e-9
    model.Params.ModelSense = 1
    model.update()
    print(f"{model.Params.NonConvex=} \n{model.Params.TimeLimit=} \n{model.Params.FeasibilityTol=} \n{model.Params.Method=} \n{model.Params.PoolSearchMode=} \n{model.Params.Heuristics=}")
    model.optimize()

    model.Params.DualReductions = 0
    model.reset()
    model.optimize()

    model = model
    name2idx = name2idx
    v_names = v_names
    bounds_vars = {name:bounds_vars.X[b_idx] for b_idx, name in enumerate(bounds_names) }
    return bounds_vars, model, sp_exp


def _expression_to_casadi(expression, ca_vars, opti, dim_G, dim_F, debug=False):
    Fweights = np.zeros((dim_F, 1))
    Gweights = np.zeros((dim_G, dim_G))
    if debug:
        sp_FG = {"F" : np.zeros((dim_F, 1), dtype=object), "G" : np.zeros((dim_G, dim_G), dtype=object)}
    for key, sp_weight in expression.decomposition_dict.items():
        weight = sympy_expression_to_casadi(sp_weight, ca_vars, opti)
        if not isinstance(sp_weight, sp.Basic) or sp.simplify(sp_weight).free_symbols == set():
            sp_weight = float(sp_weight)
        if type(key) == Expression: # Function values are stored in F
            assert key.get_is_leaf()
            Fweights += weight * one_hot(dim_F, key.counter)
            if debug:
                sp_FG["F"][key.counter] += sp_weight
        elif type(key) == tuple: # Inner products are stored in G
            point1, point2 = key
            assert point1.get_is_leaf() and point2.get_is_leaf()
            Gweights += weight * symm_prod_one_hot(dim_G, point1.counter, point2.counter)
            if debug:
                sp_FG["G"][point1.counter, point2.counter] += sp_weight/2
                sp_FG["G"][point2.counter, point1.counter] += sp_weight/2
    if debug:
        return Fweights, Gweights, sp_FG
    else:
        return Fweights, Gweights


def ca_solve_long(verbose=True, debug=False):
    dim_G = Point.counter
    dim_F = Expression.counter
    print(f"{dim_G=}, {dim_F=}")
    opti = ca.Opti()
    P_full = opti.variable(dim_G, dim_G)
    P = ca.tril(P_full)
    opti.subject_to( ca.diag(P) >= np.zeros((dim_G, 1)) )
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    sum_ij_La = 0; sum_ij_AC = 0
    ca_vars = { 'b': opti.variable(),
                'd': opti.variable(),
                'h': opti.variable(),
                'alpha': opti.variable(),
                'beta': opti.variable(), 
                'P_full': P_full }
    sp_exp = {}
        
    for function in list_of_leaf_functions:
        function.add_class_constraints()
        size_I_function = len(function.list_of_points)
        lamb = opti.variable(size_I_function, size_I_function)
        ca_vars["lamb"] = lamb
        # assert dim_G - size_I_function <= 2, print(dim_G - size_I_function)
        opti.subject_to( ca.reshape(lamb, (-1, 1)) >= np.zeros((size_I_function * size_I_function, 1)) )
        # opti.subject_to( ca.diag(lamb) == np.zeros((size_I_function, 1)) )
        counter = 0
        for i in range(size_I_function):
            for j in range(size_I_function):
                if i == j: continue
                f_interpolation_ineq_ij = function.list_of_class_constraints[counter]
                if debug:
                    Fweights_ij, Gweights_ij, sp_FG = _expression_to_casadi(f_interpolation_ineq_ij, ca_vars, opti, dim_G, dim_F, debug=debug)
                    sp_exp[(i, j)] = sp_FG
                else:
                    Fweights_ij, Gweights_ij = _expression_to_casadi(f_interpolation_ineq_ij, ca_vars, opti, dim_G, dim_F, debug=debug)
                counter += 1
                sum_ij_La += lamb[i, j] * Fweights_ij
                sum_ij_AC += lamb[i, j] * Gweights_ij
        assert len(function.list_of_class_constraints) == size_I_function * (size_I_function - 1), \
            print(len(function.list_of_class_constraints), size_I_function * (size_I_function - 1), size_I_function)

    descent = E_2 - E_1 + Delta_1
    if debug:
        Fweights_d, Gweights_d, sp_FG = _expression_to_casadi(descent, ca_vars, opti, dim_G, dim_F, debug=debug)
        sp_exp["FG_d"] = sp_FG
    else:
        Fweights_d, Gweights_d = _expression_to_casadi(descent, ca_vars, opti, dim_G, dim_F)   
    opti.subject_to( sum_ij_La - Fweights_d == np.zeros((dim_F, 1))) 
    opti.subject_to( sum_ij_AC - P @ P.T - Gweights_d == np.zeros((dim_G, dim_G)))
    if verbose:
        opts = {}
    else:
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    # opti.minimize( - ca_vars["b"] - ca_vars["d"] )
    opti.minimize( - sympy_expression_to_casadi(obj, ca_vars, opti))
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve() # QCQP for solving CircuitOpt
    except:
        if debug:
            ca_vars = ca_vars
        return None, ca_vars, None, sp_exp
    assert sol.stats()['success'], print(sol.stats())
    # print(dim_F, dim_G)
    if debug:
        ca_vars = ca_vars
        ca_expressions = {'sum_ij_La':sol.value(sum_ij_La),\
                            'sum_ij_AC':sol.value(sum_ij_AC),\
                            'Fweights_d':sol.value(Fweights_d),\
                            'Gweights_d':sol.value(Gweights_d),\
                            'P_full':sol.value(P_full),}
    opti = opti
    return dict_parameters_ciropt(sol, ca_vars), ca_vars, sol, sp_exp
   

