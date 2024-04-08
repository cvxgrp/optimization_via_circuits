import casadi as ca
import numpy as np
import sympy as sp
import scipy

from ciropt.point import Point
from ciropt.expression import Expression
from ciropt.function import Function
from ciropt.constraint import Constraint
from ciropt.utils import *
from ciropt.sympy_parsing import *
from ciropt.sympy_to_solvers import *


def solve_ipopt_qcqp_matrix(self, verbose=True, init_values=None, bounds=None, debug=False, **kwargs):
    """
    Use ipopt to find proofs and formulate problem explicitly as QCQP using matrix X:
        with variables x = [vec(v), vec(lamb), vec(nu), vec(P)], and X = xx^T 
    """
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]

    discretization_params = self.discretization_params
    sp_exp, total_I_size, total_eq_size  = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]

    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)

    v_size = len(v_names)
    x_size = len(v_names) + total_I_size + total_eq_size +  dim_G * (dim_G + 1) // 2
    opti = ca.Opti()
    var_x1 = opti.variable(x_size + 100, 1)
    var_x = var_x1[:x_size]
    var_X = var_x @ var_x.T
    opti.subject_to(var_x[0] == 1)
    ca_vars = {"x": var_x}
    assert v_k_list[-1] == "FG_d", print(v_k_list)

    print(f"Ipopt total # of variables = {np.prod(var_x1.size())}")
    print(f"Actual # of variables = {x_size}")

    vec_indices = { "v"   : [0, v_size - 1],
                    "lamb": [v_size, v_size + total_I_size - 1 ], 
                    "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ], \
                    "P"  : [v_size + total_I_size + total_eq_size, x_size - 1]} 

    init_ca_ipopt(init_values, opti, vec_indices, var_x)

    # selection matrices for variables v, lamb, nu, P
    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)
    vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
    I_lambs_nus = get_vec_var(var_x, "lamb_nu", vec_indices, matrix=True)

    # matrix coefficient for variable F is 0
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
    assert sum_ij_F.shape == (I_lambs_nus.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs_nus.shape[0], dim_F, v_size))
    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs_nus.shape[0], I_v.shape[0])
        opti.subject_to(  obj_F[k : k+1, :] @ I_v @ var_x - ca.trace(I_lambs_nus.T @ sum_ij_F[:, k, :] @ I_v @ var_X) == 0)

    # matrix coefficient for variable G is 0
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lambs_nus.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs_nus.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            # PP^T_{k1, k2} = P_{k1,:}(P_{k2,:})^T = (S1 @ x).T @ (S2 @ x) = S1.T @ S2 @ xx^T
            S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
            opti.subject_to(  obj_G[k_idx : k_idx+1, :] @ I_v @ var_x \
                            + ca.trace(S1.T @ S2 @ var_X)\
                            - ca.trace(((I_lambs_nus.T @ sum_ij_G[:, k_idx, :]) @ I_v) @ var_X) == 0)

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        # print(f"{name=}, {pref_v=}, {v=}")
        ai = - one_hot(v_size, name2idx[name])
        # include both permutations for the product 
        Qi = symm_prod_one_hot(v_size, name2idx[v], name2idx[pref_v])
        opti.subject_to( ca.trace(((I_v.T @ Qi) @ I_v) @ var_X)+ ai.T @ I_v @ var_x == 0)

    # bounds
    ca_add_bounds(opti, bounds, ca_vars, name2idx)

    # lambda >= 0 constraints
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    opti.subject_to( I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1)))
    # diag(P) >= 0 constraints
    vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
    Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
    np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
    # P_diag_constraints = [Q]
    # print( Q.shape, vec_P.shape, dim_G)
    opti.subject_to( Q @ I_P @ var_x >= np.zeros((dim_G, 1)) )

    opts = {'ipopt.max_iter':50000}
    if not verbose:
        opts.update({'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
    ca_vars = {name : var_x[idx] for name, idx in name2idx.items()}
    opti.minimize( -sympy_expression_to_casadi(self.obj, ca_vars, opti))
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve() # QCQP for solving CircuitOpt
    except:
        if debug:
            self.ca_vars = ca_vars
        print("Could not find a solution using Ipopt")
        return None, ca_vars, None, sp_exp
    assert sol.stats()['success'], print(sol.stats())
    self.opti = opti
    self.name2idx = name2idx
    self.v_names = v_names
    self.vars = {"x": sol.value(var_x), "v_names":v_names}
    return dict_parameters_ciropt(sol, ca_vars), sol, sp_exp


def solve_gp(self, verbose=True, debug=False, time_limit=1000, ftol=1e-9, heur=0.001, method=0, bounds=None, **kwargs):
    """
    Use gurobipy branch-and-bound to solve a QCQP
        keep all the variables separately, without forming a unified (vectorized) variable
    """
    try:
        import gurobipy as gp
    except ImportError:
        raise Exception("Gurobi package is not installed.")
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    model = gp.Model() 

    gp_vars = { 'b': model.addVar(name='b', lb=0., ub=1000.),
                'd': model.addVar(name='d', lb=0., ub=1000.),
                'h': model.addVar(name='h', lb=0., ub=1000.),
                'gamma': model.addVar(name='gamma', lb=0., ub=1000.),
                'alpha': model.addVar(name='alpha', lb=-1., ub=1.),
                'beta': model.addVar(name='beta', lb=-1., ub=1.),  
                'P': model.addMVar((dim_G, dim_G), name='P', lb=-1000, ub=1000)  }
    model.update()
    if bounds is not None:
        for name in bounds.keys():
            if not name in gp_vars: continue
            if "ub" in bounds[name]:
                model.addConstr( gp_vars[name] <= bounds[name]["ub"] )
            if "lb" in bounds[name]:
                model.addConstr( gp_vars[name] >= bounds[name]["lb"] )
    # P is lower triangular
    model.addConstr( gp_vars["P"].diagonal() >= np.zeros(dim_G) )
    for i in range(dim_G):
        for j in range(i + 1, dim_G):
            model.addConstr( gp_vars["P"][i, j] == 0 )

    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    sp_exp, total_I_size, total_eq_size, sum_ij_La, sum_ij_AC = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)
    shift_f_idx = 0; ineq_size = 0; eq_size = 0
    if self.list_of_constraints != list():
        eq_size = total_eq_size
        ineq_size = len(self.list_of_constraints) - total_eq_size
        if ineq_size >= 1:
            shift_f_idx = 1
            name = "lamb0"
            gp_vars[name] = model.addMVar((ineq_size, 1), name=name, lb=0, ub=1000)
            model.update()
        if eq_size >= 1:
            name = "nu0"
            gp_vars[name] = model.addMVar((eq_size, 1), name=name, lb=-1000, ub=1000)
            model.update()

    for f_idx, function in enumerate(list_of_leaf_functions):
        name = "lamb%d"%(f_idx + shift_f_idx)
        size_I_function = len(function.list_of_points)
        gp_vars[name] = model.addMVar((size_I_function, size_I_function), name=name, lb=0, ub=1000)
        model.update()
        if bounds is not None and name in bounds:
            if "ub" in bounds[name]:
                model.addConstr( gp_vars[name] <= bounds[name]["ub"] )
            if "lb" in bounds[name]:
                model.addConstr( gp_vars[name] >= bounds[name]["lb"] )
        # model.addConstr( gp_vars[name].diagonal() == np.zeros(size_I_function) )

    assert sum_ij_La.shape == sp_exp["FG_d"]["F"].shape and sum_ij_AC.shape == sp_exp["FG_d"]["G"].shape
    sp_z1 = simplify_matrix(sum_ij_La - sp_exp["FG_d"]["F"])
    sp_z2 = simplify_matrix(sum_ij_AC - sp_exp["FG_d"]["G"]) 
    z1 = sympy_matrix_to_gurobi(sp_z1, gp_vars, model)
    z2 = sympy_matrix_to_gurobi(sp_z2, gp_vars, model)
    PPt = gp_vars["P"] @ gp_vars["P"].T
    model.addConstrs(z1[i] == 0 for i in range(dim_F))
    model.addConstrs(z2[i, j] - PPt[i, j].item() == 0 for i in range(dim_G) for j in range(dim_G))
    if not verbose:
        model.Params.LogToConsole = 0
    # model.setObjective( - gp_vars["b"] - gp_vars["d"] )
    model.update()
    model.setObjective( -sympy_expression_to_gurobi(self.obj, gp_vars, model), gp.GRB.MINIMIZE)
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    model.Params.FeasibilityTol = ftol
    model.Params.Method = method
    model.Params.PoolSearchMode = 1
    model.Params.Heuristics = heur
    # model.tune()
    # for i in range(model.tuneResultCount):
    #     model.getTuneResult(i)
    #     model.write('tune'+str(i)+'.prm')
    model.update()  
    model.optimize()
    if debug:
        self.gp_expressions = {'z1': z1,\
                            'z2': z2,\
                            'P': gp_vars["P"], \
                            'sum_ij_La':sympy_matrix_to_gurobi(simplify_matrix(sum_ij_La), gp_vars, model), \
                            'sum_ij_AC':sympy_matrix_to_gurobi(simplify_matrix(sum_ij_AC), gp_vars, model), \
                            'Fweights_d':sympy_matrix_to_gurobi(simplify_matrix(sp_exp["FG_d"]["F"]), gp_vars, model), \
                            'Gweights_d':sympy_matrix_to_gurobi(simplify_matrix(sp_exp["FG_d"]["G"]), gp_vars, model) }
    self.model = model
    self.vars = gp_vars
    return dict_parameters_ciropt_gp(model, gp_vars), model, sp_exp


def solve_gp_qcqp_matrix(self, verbose=True, debug=False, time_limit=1000, ftol=1e-9, heur=0.001, method=0, bounds=None, **kwargs):
    """
    Use gurobipy branch-and-bound to solve a QCQP in matrix form
        with variables x = [vec(v), vec(lamb), vec(nu), vec(P)], and X = xx^T 
    """
    try:
        import gurobipy as gp
    except ImportError:
        raise Exception("Gurobi package is not installed.")
    # formulate problem explicitly as QCQP using x and matrix X
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]

    discretization_params = self.discretization_params
    sp_exp, total_I_size, total_eq_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]

    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)

    v_size = len(v_names)
    x_size = len(v_names) + total_I_size + total_eq_size + dim_G * (dim_G + 1) // 2
    model = gp.Model() 
    var_x = model.addMVar((x_size, 1), name="x", lb=-1000, ub=1000)
    model.update()
    model.addConstr( var_x[name2idx["b"]].item() >= 0 )
    model.addConstr( var_x[name2idx["h"]].item() >= 0 )
    model.addConstr( var_x[name2idx["d"]].item() >= 0 )
    model.addConstr( var_x[name2idx["alpha"]].item() >= -1)
    model.addConstr( var_x[name2idx["alpha"]].item() <= 1 )
    model.addConstr( var_x[name2idx["beta"]].item() >= -1)
    model.addConstr(var_x[name2idx["beta"]].item() <= 1 )
    var_X = var_x @ var_x.T
    model.addConstr( var_x[0].item() == 1 )

    # additional RLT cuts constraints
    # constraints += [ var_X - lb_x @ var_x.T - var_x @ lb_x.T >= - lb_x @ lb_x.T, \
    #                  var_X - ub_x @ var_x.T - var_x @ ub_x.T >= - ub_x @ ub_x.T, \
    #                  var_X - lb_x @ var_x.T - var_x @ ub_x.T <= - lb_x @ ub_x.T ]

    assert v_k_list[-1] == "FG_d", print(v_k_list)

    vec_indices = { "v"   : [0, v_size - 1],\
                    "lamb": [v_size, v_size + total_I_size - 1 ], \
                    "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ], \
                    "P"  : [v_size + total_I_size + total_eq_size, x_size - 1]} 

    # x = [v, vec(lambda), vec(P)]
    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
    I_lamb_nu = get_vec_var(var_x, "lamb_nu", vec_indices, matrix=True)
    I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)

    # matrix coefficient for variable F is 0
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
    assert sum_ij_F.shape == (I_lamb_nu.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lamb_nu.shape[0], dim_F, v_size))

    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lamb_nu.shape[0], I_v.shape[0])
        M1 = obj_F[k : k+1, :] @ I_v
        M2 = I_lamb_nu.T @ sum_ij_F[:, k, :] @ I_v
        print(type(M1), type(var_X), type(M2))
        model.addConstr(  (M1 @ var_x).item()  - gp_trace(M2 @ var_X) == 0)

    # matrix coefficient for variable G is 0
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lamb_nu.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lamb_nu.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            # PP^T_{k1, k2} = P_{k1,:}(P_{k2,:})^T = (S1 @ x).T @ (S2 @ x) = S1.T @ S2 @ xx^T
            S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
            M1 = obj_G[k_idx : k_idx+1, :] @ I_v
            M2 = S1.T @ S2
            M3 = I_lamb_nu.T @ sum_ij_G[:, k_idx, :] @ I_v
            model.addConstr(  (M1 @ var_x).item()  \
                            + gp_trace(M2 @ var_X) \
                            - gp_trace(M3 @ var_X) == 0)

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        ai = - one_hot(v_size, name2idx[name])
        # include both permutations for the product 
        Qi = symm_prod_one_hot(v_size, name2idx[pref_v], name2idx[v])
        M1 = (I_v.T @ Qi @ I_v)
        M2 = ai.T @ I_v
        model.addConstr( gp_trace(M1 @ var_X) + (M2 @ var_x).item() == 0)
        # model.addConstr( gp_trace((I_v.T @ Qi2 @ I_v) @ var_X) + ((ai.T @ I_v) @ var_x).item() == 0)

    # lambda >= 0 constraints
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    model.addConstr( I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1)))
    # diag(P) >= 0 constraints
    vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
    Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
    np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
    # P_diag_constraints = [Q]
    M1 = Q @ I_P
    model.addConstr( M1 @ var_x >= np.zeros((dim_G, 1)) )

    if not verbose:
        model.Params.LogToConsole = 0
    model.update()
    gp_vars = {name : var_x[name2idx[name]].item() for name in discretization_params}
    model.setObjective( -sympy_expression_to_gurobi(self.obj, gp_vars, model), gp.GRB.MINIMIZE)
    
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    model.Params.FeasibilityTol = ftol
    model.Params.Method = method
    model.Params.PoolSearchMode = 1
    model.Params.Heuristics = heur
    model.update()
    model.optimize()

    self.model = model
    self.name2idx = name2idx
    self.v_names = v_names
    self.vars = {"x": var_x.X, "v_names":v_names}
    return dict_parameters_ciropt_gp(model, gp_vars), model, sp_exp
   

def solve_qcqp_sni(self, verbose=True, max_iter=1000, debug=False, bounds=None, **kwargs):
    """
    Use Suggest-and-Improve QCQP framework to find proof
        with variables x = [vec(v), vec(lamb), vec(nu), vec(P)]
    """
    try:
        import qcqp as sni
    except ImportError:
        raise Exception("QCQP package is not installed.")
    # solve QCQP problem using suggest and improve procedure
    # using x=[vec(v), vec(lamb)] and Z
    # every matrix can be decomposed onto a difference of two PSD matrices
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    
    discretization_params = self.discretization_params

    sp_exp, total_I_size, total_eq_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]

    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)
    v_size = len(v_names)
    x_size = len(v_names) + total_I_size + total_eq_size + dim_G * (dim_G + 1) // 2
    P_size = dim_G * (dim_G + 1) // 2
    
    # sni variables
    sni_f0 = None
    sni_fs = [] # Pc, qc, rc, op_name
    # variables
    var_x = cp.Variable((x_size, 1))
    
    sni_fs += [(scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(one_hot(x_size, 0)), -1, "=="),
            #    (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(-one_hot(x_size, name2idx["b"])), 0, "<="),
            #    (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(-one_hot(x_size, name2idx["h"])), 0, "<="),
            #    (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(-one_hot(x_size, name2idx["d"])), 0, "<="),
                (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(-one_hot(x_size, name2idx["alpha"])), -1, "<="),
                (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(one_hot(x_size, name2idx["alpha"])), -1, "<="),
                (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(-one_hot(x_size, name2idx["beta"])), -1, "<="),
                (scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(one_hot(x_size, name2idx["beta"])), -1, "<=")
                ]

    assert v_k_list[-1] == "FG_d", print(v_k_list)

    vec_indices = { "v"   : [0, v_size - 1],\
                    "lamb": [v_size, v_size + total_I_size - 1 ],
                    "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ], \
                    "P"  : [v_size + total_I_size + total_eq_size, x_size - 1]} 

    # x = [v, vec(lambda), vec(P)]
    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
    I_lamb_nu = get_vec_var(var_x, "lamb_nu", vec_indices, matrix=True)
    I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)
    # bounds
    # constraints = cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lambs)

    # matrix coefficient for variable F is 0
    # |F| x |v|
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    # |lambs| x |F| x |v|
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size) 
    assert sum_ij_F.shape == (I_lamb_nu.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lamb_nu.shape[0], dim_F, v_size))
    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lamb_nu.shape[0], I_v.shape[0]) 
        block_Q_F = 0.5 * sum_ij_F[:, k, :]
        Q_F = scipy.sparse.bmat([[None, block_Q_F.T, None], 
                                 [block_Q_F, None, None], 
                                 [None, None, scipy.sparse.csr_matrix((P_size, P_size))]]).tocsr()
        q_F = scipy.sparse.csc_matrix((obj_F[k : k+1, :] @ I_v).T)
        sni_fs += [(-Q_F, q_F, 0, "==")]
        

    # matrix coefficient for variable G is 0
    # |G| x |v|
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    # |lambs| x |G| x |v|
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lamb_nu.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lamb_nu.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            block_Q_G = 0.5 * sum_ij_G[:, k_idx, :]
            # PP^T_{k1, k2} = P_{k1,:}(P_{k2,:})^T = (S1 @ x).T @ (S2 @ x) = S1.T @ S2 @ xx^T
            S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
            Q_G = (scipy.sparse.bmat([[None, block_Q_G.T, None], 
                                      [block_Q_G, None, None], 
                                      [None, None, scipy.sparse.csr_matrix((P_size, P_size))]]) \
                    - scipy.sparse.csr_matrix(S1.T @ S2)).tocsr()
            q_G = scipy.sparse.csc_matrix((obj_G[k_idx : k_idx+1, :] @ I_v).T)
            sni_fs += [(-Q_G, q_G, 0, "==")]

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue # monomial with degree 1
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        ai = scipy.sparse.csc_matrix(-one_hot(x_size, name2idx[name]))
        Qi = scipy.sparse.csr_matrix(symm_prod_one_hot(x_size, name2idx[pref_v], name2idx[v]))
        sni_fs += [(Qi, ai, 0, "==")] 

    # lambda >= 0 constraints
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    for i in range(I_lambs.shape[0]):
        sni_fs += [(scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(- I_lambs[i:i+1].T), 0, "<=")]
    # diag(P) >= 0 constraints
    vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
    Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
    np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
    M1 = Q @ I_P
    # model.addConstr( M1 @ var_x >= np.zeros((dim_G, 1)) )
    for i in range(M1.shape[0]):
        sni_fs += [(scipy.sparse.csr_matrix((x_size, x_size)), scipy.sparse.csc_matrix(- M1[i:i+1].T), 0, "<=")]

    q0 = scipy.sparse.csc_matrix(const_coeff_expr(-self.obj, name2idx, x_size))
    sni_f0 = (scipy.sparse.csr_matrix((x_size, x_size)), q0, 0)
    prob = sni.QCQPProblem(sni_f0, sni_fs, var_x)
    qcqp = sni.QCQP(prob)

    # SDR solution is cached and not solved again
    qcqp.suggest(sni.SDR, solver=cp.CLARABEL)
    print("SDR lower bound: %.4f" % qcqp.sdr_bound)
    f_dccp, v_dccp = qcqp.improve(sni.DCCP)
    print("Penalty CCP: objective %.4f, violation %.4f, v==0: %d" % (f_dccp, v_dccp, int(np.allclose(v_dccp, 0))))
    # sample from the semidefinite relaxation
    qcqp.suggest(sni.SDR, solver=cp.CLARABEL)
    print("SDR lower bound: %.4f" % qcqp.sdr_bound)
    f_dccp, v_dccp = qcqp.improve(sni.DCCP)
    f_dccp, v_dccp = qcqp.improve(sni.COORD_DESCENT, phase1=False)
    print("Penalty CCP + coordinate descent: objective %.4f, violation %.4f" % (f_dccp, v_dccp))

    # qcqp.suggest(sni.SDR, solver=cp.CLARABEL)
    # print("SDR lower bound: %.4f" % qcqp.sdr_bound)
    # f_ipopt, v_ipopt = qcqp.improve(sni.COORD_DESCENT)
    # f_ipopt, v_ipopt = qcqp.improve(sni.IPOPT, verbose=True, init=False)
    # print("Coordinate descent + ipopt: objective %.4f, violation %.4f" % (f_ipopt, v_ipopt))

    # qcqp.suggest(sni.RANDOM, solver=cp.CLARABEL)
    # # print("SDR lower bound: %.4f" % qcqp.sdr_bound)
    # f_ipopt, v_ipopt = qcqp.improve(sni.IPOPT, max_iter=500000, verbose=True, init=False)
    # print("Ipopt: objective %.4f, violation %.4f" % (f_ipopt, v_ipopt))

    qcqp.suggest(sni.SDR, solver=cp.CLARABEL)
    f_admm, v_admm = qcqp.improve(sni.COORD_DESCENT)
    f_admm, v_admm = qcqp.improve(sni.ADMM, phase1=False)
    print("Coordinate descent + nonconvex ADMM: objective %.4f, violation %.4f" % (f_admm, v_admm))

    self.prob = prob

    self.name2idx = name2idx
    self.v_names = v_names
    self.var_x = var_x
    self.vars = {"x": var_x.value, "v_names":v_names}
    vars_vals = {name:var_x[name2idx[name], 0].value for name in discretization_params}
    return vars_vals, prob, sp_exp


def solve_dccp(self, verbose=True, max_iter=1000, debug=False, bounds=None, **kwargs):
    """
    Find proof by solving QCQP problem using convex-concave procedure (CCP)
        using x=[vec(v), vec(lamb)] and Z
        CCP: every matrix can be decomposed onto a difference of two PSD matrices
    """
    try:
        import dccp
    except ImportError:
        raise Exception("DCCP package is not installed.")
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    
    discretization_params = self.discretization_params

    sp_exp, total_I_size, total_eq_size  = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]

    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)
    v_size = len(v_names)
    x_size = len(v_names) + total_I_size + total_eq_size 
    
    # variables
    var_x = cp.Variable((x_size, 1))
    var_Z = cp.Variable((dim_G, dim_G), PSD=True)
    constraints = [ var_x[0] == 1 ]
    cvx_vars = { "Z": var_Z,  "x": var_x}

    constraints += [ #cp.abs(var_Z) <= 1000,
                    cp.abs(var_x) <= 10000]

    constraints += [ var_x[name2idx["b"]] >= 0., \
                        var_x[name2idx["h"]] >= 0., \
                        var_x[name2idx["d"]] >= 0, 
                        -1 <= var_x[name2idx["alpha"]], 
                            var_x[name2idx["alpha"]] <= 1 , 
                        -1 <= var_x[name2idx["beta"]], 
                            var_x[name2idx["beta"]] <= 1
                    ]

    assert v_k_list[-1] == "FG_d", print(v_k_list)

    vec_indices = { "v"   : [0, v_size - 1],\
                    "lamb": [v_size, v_size + total_I_size - 1 ],
                    "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ]} 

    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
    I_lamb_nu = get_vec_var(var_x, "lamb_nu", vec_indices, matrix=True)
    
    # bounds
    constraints = cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lamb_nu)
    var_v = get_vec_var(var_x, "v", vec_indices)

    # matrix coefficient for variable F is 0
    # |F| x |v|
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    # |lambs| x |F| x |v|
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size) 
    assert sum_ij_F.shape == (I_lamb_nu.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lamb_nu.shape[0], dim_F, v_size))
    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lamb_nu.shape[0], I_v.shape[0]) 
        block_Q_F = 0.5 * sum_ij_F[:, k, :]
        Q_F = np.block([[np.zeros((v_size, v_size)), block_Q_F.T], 
                        [block_Q_F, np.zeros((total_I_size + total_eq_size, total_I_size + total_eq_size))]])
        q_F = obj_F[k : k+1, :]
        Q_F_plus, Q_F_minus = matrix_to_diff_psd(Q_F)
        if type(Q_F_plus) == int:
            constraints += [ q_F @ var_v + cp.quad_form(var_x, Q_F_minus) == 0]
        elif type(Q_F_minus) == int:
            constraints += [ q_F @ var_v == cp.quad_form(var_x, Q_F_plus)]
        else:
            constraints += [ q_F @ var_v + cp.quad_form(var_x, Q_F_minus) == cp.quad_form(var_x, Q_F_plus)]
        

    # matrix coefficient for variable G is 0
    # |G| x |v|
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    # |lambs| x |G| x |v|
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lamb_nu.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lamb_nu.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            block_Q_G = 0.5 * sum_ij_G[:, k_idx, :]
            Q_G = np.block([[np.zeros((v_size, v_size)), block_Q_G.T], 
                            [block_Q_G, np.zeros((total_I_size + total_eq_size, total_I_size + total_eq_size))]])
            q_G = obj_G[k_idx : k_idx+1, :]
            Q_G_plus, Q_G_minus = matrix_to_diff_psd(Q_G)
            if type(Q_G_plus) == int:
                constraints += [ q_G @ var_v + var_Z[k1, k2] + cp.quad_form(var_x, Q_G_minus) == 0]
            elif type(Q_G_minus) == int:
                constraints += [ q_G @ var_v + var_Z[k1, k2] == cp.quad_form(var_x, Q_G_plus)]
            else:
                constraints += [ q_G @ var_v + var_Z[k1, k2] + cp.quad_form(var_x, Q_G_minus) == cp.quad_form(var_x, Q_G_plus)]

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue # monomial with degree 1
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        # ai = - one_hot(v_size, name2idx[name])
        # include both permutations for the product 
        i_idx, j_idx = min(name2idx[pref_v], name2idx[v]), max(name2idx[pref_v], name2idx[v])
        if i_idx == j_idx:
            constraints += [ cp.square(var_x[i_idx]) == var_x[name2idx[name]] ]
        else:
            # Qi = symm_prod_one_hot(v_size, name2idx[pref_v], name2idx[v]) 
            Qi_plus = 0.25 * np.ones((2,2))
            Qi_minus = 0.25 * np.array(([1, -1], [-1, 1]))
            assert np.allclose(Qi_plus - Qi_minus, np.array([[0, 0.5],[0.5, 0] ]))
            var_x_ij = cp.vstack([var_x[i_idx], var_x[j_idx]])
            constraints += [ cp.quad_form(var_x_ij, Qi_plus) == cp.quad_form(var_x_ij, Qi_minus) + var_x[name2idx[name]] ]

    # lambda >= 0 constraints
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    constraints += [ I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1))]

    obj = -sympy_expression_to_cvx(self.obj, var_x, name2idx)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    print("problem is DCP:", prob.is_dcp())   # false
    print("problem is DCCP:", dccp.is_dccp(prob))  # true
    prob.solve(method='dccp', verbose=verbose, max_iter=max_iter)
    print(f"{prob.status=}")
    self.prob = prob
    self.name2idx = name2idx
    self.v_names = v_names
    self.var_x = var_x
    self.vars = {"x": var_x.value, "Z": var_Z.value, "v_names":v_names}
    vars_vals = {name:var_x[name2idx[name], 0].value for name in discretization_params}
    return vars_vals, prob, sp_exp


def solve_sdp_relax(self, verbose=True, var_bound=None, debug=False, bounds=None, cvx_solver=cp.CLARABEL, **kwargs):
    """
    Formulate problem explicitly as QCQP using x and matrices X and Z
        keep Z=PP^T as it is, do not aggregate it into x = vec(p, lambda)
    and relax it to convex SDP relaxation or to get bounds on the variables
    """
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    
    discretization_params = self.discretization_params
    bounds_names = sorted(['alpha', 'beta', 'Z', "lamb"])

    sp_exp, total_I_size, total_eq_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]

    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)
    v_size = len(v_names)
    x_size = len(v_names) + total_I_size + total_eq_size
    
    # bounds
    if var_bound is not None:
        bounds_vars = cp.Variable((len(bounds_names)), nonneg=True)
        bnames2idx = {bname:idx for idx, bname in enumerate(bounds_names)}
    # SDP variables
    var_x = cp.Variable((x_size, 1))
    W = cp.Variable((x_size + 1, x_size + 1), PSD=True)
    var_X = cp.Variable((x_size, x_size), PSD=True)
    var_Z = cp.Variable((dim_G, dim_G), PSD=True)
    constraints = [ var_X[:, :1] == var_x, \
                    var_X[:1, :] == var_x.T ]
    cvx_vars = { "Z": var_Z,  "x": var_x, "X": var_X}

    constraints += [ var_x[name2idx["b"]] >= 0.00001, \
                        var_x[name2idx["h"]] >= 0.00001, \
                        var_x[name2idx["d"]] >= 0]
    # constraints to encode X >> xx^T
    constraints += [W[ : x_size, : x_size] == var_X, \
                    W[x_size : x_size+1, : x_size] == var_x.T, \
                    W[ : x_size, x_size : x_size+1] == var_x, \
                    W[x_size, x_size] == 1 ]
    # implied linear constraints for PSD matrix W, X
    diag_W = cp.diag(W)
    diag_X = cp.diag(var_X)
    constraints += [ diag_W >= np.zeros((x_size + 1)), \
                    W >= -(cp.outer(np.ones((x_size + 1)), diag_W) + cp.outer(diag_W, np.ones((x_size + 1))) ) / 2 , \
                    W <=  (cp.outer(np.ones((x_size + 1)), diag_W) + cp.outer(diag_W, np.ones((x_size + 1))) ) / 2, \
                    # is true when equality holds X = xx^T
                    diag_X >= np.zeros((x_size)), \
                    var_X >= -(cp.outer(np.ones((x_size)), diag_X) + cp.outer(diag_X, np.ones((x_size))) ) / 2 , \
                    var_X <=  (cp.outer(np.ones((x_size)), diag_X) + cp.outer(diag_X, np.ones((x_size))) ) / 2 ]

    constraints += [    -1 <= var_x[name2idx["alpha"]], 
                            var_x[name2idx["alpha"]] <= 1 , 
                        -1 <= var_x[name2idx["beta"]], 
                            var_x[name2idx["beta"]] <= 1, 
                        var_x[0] == 1 
                    ]

    assert v_k_list[-1] == "FG_d", print(v_k_list)

    vec_indices = { "v"   : [0, v_size - 1],\
                    "lamb": [v_size, v_size + total_I_size - 1 ],
                    "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ]} 

    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
    I_lamb_nu = get_vec_var(var_x, "lamb_nu", vec_indices, matrix=True)

    constraints = cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lamb_nu)

    # matrix coefficient for variable F is 0
    # |F| x |v|
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    # |lambs| x |F| x |v|
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
    assert sum_ij_F.shape == (I_lamb_nu.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lamb_nu.shape[0], dim_F, v_size))
    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lamb_nu.shape[0], I_v.shape[0])
        block_Q_F = 0.5 * sum_ij_F[:, k, :]
        Q_F = scipy.sparse.bmat([[None, block_Q_F.T], 
                                 [block_Q_F, None,]]).tocsr()
        q_F = scipy.sparse.csr_matrix(obj_F[k : k+1, :] @ I_v)
        constraints += [ cp.sum(q_F @ var_x) == cp.trace(Q_F @ var_X)]

    # matrix coefficient for variable G is 0
    # |G| x |v|
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    # |lambs| x |G| x |v|
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lamb_nu.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lamb_nu.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            block_Q_G = 0.5 * sum_ij_G[:, k_idx, :]
            Q_G = (scipy.sparse.bmat([[None, block_Q_G.T], 
                                      [block_Q_G, None]])).tocsr()
            q_G = scipy.sparse.csr_matrix(obj_G[k_idx : k_idx+1, :] @ I_v)
            constraints += [ cp.sum(q_G @ var_x) + var_Z[k1, k2] == cp.trace(Q_G @ var_X) ]

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        ai = - one_hot(x_size, name2idx[name])
        # include both permutations for the product 
        Qi = scipy.sparse.csr_matrix(symm_prod_one_hot(x_size, name2idx[pref_v], name2idx[v]))
        constraints += [ cp.trace( Qi  @ var_X) + cp.sum(ai.T @ var_x) == 0 ]

    # lambda >= 0 constraints
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    constraints += [ I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1))]

    if var_bound is not None:
        # bounds constraints
        for name in bounds_names:
            b_idx = bnames2idx[name]
            if name in vec_indices: # v, lamb
                var_name = get_vec_var(var_x, name, vec_indices)
                constraints += [cp.abs(var_name[:, 0]) <= bounds_vars[b_idx] * np.ones(var_name.size)]
            elif name == "Z":
                constraints += [cp.diag(var_Z) <= bounds_vars[b_idx]]
            else:
                constraints += [cp.abs(var_x[name2idx[name], 0]) <= bounds_vars[b_idx]]

        constraints += [ bounds_vars[bnames2idx["Z"]] <= cp.trace(var_Z), \
                         bounds_vars[bnames2idx["lamb_nu"]] <= cp.pnorm(I_lamb_nu @ var_x, 0.999),
                         bounds_vars[bnames2idx["alpha"]] == var_x[name2idx["alpha"]],
                         bounds_vars[bnames2idx["beta"]] == var_x[name2idx["beta"]],   ]

        obj = -cp.sum(bounds_vars[bnames2idx[var_bound]])
    else:
        obj = -sympy_expression_to_cvx(self.obj, var_x, name2idx)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    # 'SCS', 'CVXOPT','SDPA', cp.CLARABEL, cp.MOSEK
    prob.solve(solver=cvx_solver, verbose=verbose)
    print(f"{prob.status=}")
    self.prob = prob
    self.name2idx = name2idx
    self.v_names = v_names
    self.var_x = var_x
    self.vars = {"x": var_x.value, "Z": var_Z.value, "v_names":v_names}
    if var_bound is not None:
        res = self.bounds_vars = {name:bounds_vars.value[b_idx] for b_idx, name in enumerate(bounds_names) }
        self.bounds_vars["P"] = np.sqrt(self.bounds_vars["Z"])
    else:
        res = {name:var_x[name2idx[name], 0].value for name in discretization_params}
    return res, prob, sp_exp
    

def bounds_sdp_relax_all(self, verbose=True, cvx_solver=cp.CLARABEL, debug=False, bounds=None, **kwargs):
    """
    Formulate problem explicitly as QCQP using variables x and matrix X
        variable x = vec(v, lambda, P), aggregating Z=PP^T
    and relax it to convex SDP to get bounds on the variables
    """
    dim_G = Point.counter
    dim_F = Expression.counter 
    print(f"{dim_G=}, {dim_F=}")
    list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
    
    discretization_params = self.discretization_params
    # bounds_names = sorted(['alpha', 'beta', 'h', 'b', 'd', 'P', "lamb"])
    bounds_names = sorted(['P', "lamb", "nu"])

    sp_exp, total_I_size, total_eq_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]


    v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)
    v_size = len(v_names)
    x_size = len(v_names) + total_I_size + total_eq_size + dim_G * (dim_G + 1) // 2
    P_size = dim_G * (dim_G + 1) // 2
    
    # bounds
    bounds_vars = cp.Variable((len(bounds_names)))
    bnames2idx = {bname:idx for idx, bname in enumerate(bounds_names)}
    # SDP variables
    var_x = cp.Variable((x_size, 1))
    W = cp.Variable((x_size + 1, x_size + 1), PSD=True)
    var_X = cp.Variable((x_size, x_size), PSD=True)
    constraints = [ var_X[:, :1] == var_x, \
                    var_X[:1, :] == var_x.T ]
    cvx_vars = {"x": var_x, "X": var_X}

    constraints += [ var_x[name2idx["b"]] >= 0.05, \
                        var_x[name2idx["h"]] >= 0.01, \
                        var_x[name2idx["d"]] >= 0]
    # constraints to encode X \succeq xx^T
    constraints += [W[ : x_size, : x_size] == var_X, \
                    W[x_size : x_size+1, : x_size] == var_x.T, \
                    W[ : x_size, x_size : x_size+1] == var_x, \
                    W[x_size, x_size] == 1]

    constraints += [ 
                    var_x[name2idx["alpha"]] >= -1, \
                    var_x[name2idx["alpha"]] <= 1 , \
                    var_x[name2idx["beta"]] >= -1, \
                    var_x[name2idx["beta"]] <= 1, \
                    var_x[0] == 1 ]

    assert v_k_list[-1] == "FG_d", print(v_k_list)

    vec_indices = { "v"   : [0, v_size - 1],\
                    "lamb": [v_size, v_size + total_I_size - 1 ], \
                    "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ], \
                    "P"  : [v_size + total_I_size + total_eq_size, x_size - 1]} 

    I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
    # I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
    I_lamb_nu = get_vec_var(var_x, "lamb_nu", vec_indices, matrix=True)
    I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)

    constraints = cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lamb_nu)

    # matrix coefficient for variable F is 0
    # |F| x |v|
    obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
    # |lambs| x |F| x |v|
    sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
    assert sum_ij_F.shape == (I_lamb_nu.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lamb_nu.shape[0], dim_F, v_size))
    for k in range(dim_F):
        assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lamb_nu.shape[0], I_v.shape[0])
        block_Q_F = 0.5 * sum_ij_F[:, k, :]
        Q_F = scipy.sparse.bmat([[None, block_Q_F.T, None], 
                                    [block_Q_F, None, None], 
                                    [None, None, scipy.sparse.csr_matrix((P_size, P_size))]]).tocsr()
        q_F = scipy.sparse.csr_matrix(obj_F[k : k+1, :] @ I_v)
        constraints += [ cp.sum(q_F @ var_x) == cp.trace( Q_F @ var_X) ]

    # matrix coefficient for variable G is 0
    obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
    sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
    assert sum_ij_G.shape == (I_lamb_nu.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lamb_nu.shape[0], dim_G*dim_G, v_size))
    for k1 in range(dim_G):
        for k2 in range(dim_G):
            k_idx = k1 * dim_G + k2
            block_Q_G = 0.5 * sum_ij_G[:, k_idx, :]
            # PP^T_{k1, k2} = P_{k1,:}(P_{k2,:})^T = (S1 @ x).T @ (S2 @ x) = S1.T @ S2 @ xx^T
            S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
            Q_G = (scipy.sparse.bmat([[None, block_Q_G.T, None], 
                                    [block_Q_G, None, None], 
                                    [None, None, scipy.sparse.csr_matrix((P_size, P_size))]]) \
                    - scipy.sparse.csr_matrix(S1.T @ S2)).tocsr()
            q_G = scipy.sparse.csr_matrix(obj_G[k_idx : k_idx+1, :] @ I_v)
            constraints += [ cp.sum(q_G @ var_x) == cp.trace( Q_G @ var_X) ]

    # v variables quadratic constraints
    # v^T Qi v + ai^T v = 0 
    for name in v_names[1:]:
        vars = name.split("_")
        if len(vars) == 1: continue
        pref_v, v = "_".join(vars[:-1]), vars[-1]
        ai = -one_hot(x_size, name2idx[name])
        # include both permutations for the product 
        Qi = symm_prod_one_hot(x_size, name2idx[v], name2idx[pref_v])
        constraints += [ cp.trace( Qi @ var_X) + cp.sum(ai.T @ var_x) == 0 ]

    # lambda >= 0 constraints
    I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
    constraints += [ I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1))]
    # diag(P) >= 0 constraints
    vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
    Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
    np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
    constraints += [ Q @ I_P @ var_x >= np.zeros((dim_G, 1)) ]

    # bounds constraints
    for b_idx, name in enumerate(bounds_names):
        if name in vec_indices:
            var_name = get_vec_var(var_x, name, vec_indices)
            constraints += [cp.abs(var_name[:, 0]) <= bounds_vars[b_idx] * np.ones(var_name.size)]
        else:
            constraints += [cp.abs(var_x[name2idx[name], 0]) <= bounds_vars[b_idx]]

    constraints += [cp.square(bounds_vars[bnames2idx["P"]]) <= cp.trace(I_P @ var_X @ I_P.T), \
                    bounds_vars[bnames2idx["lamb"]] <= cp.pnorm(I_lamb_nu @ var_x, 0.999)]
    assert (I_P @ var_X @ I_P.T).shape[0] == dim_G * (dim_G + 1) // 2

    obj = - cp.sum(bounds_vars)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    # 'SCS', 'CVXOPT','SDPA', cp.CLARABEL, cp.MOSEK
    prob.solve(solver=cvx_solver, verbose=verbose)
    print(f"{prob.status=}")
    self.prob = prob
    self.name2idx = name2idx
    self.v_names = v_names
    self.var_x = var_x
    self.bounds_vars = {name:bounds_vars.value[b_idx] for b_idx, name in enumerate(bounds_names) }
    # self.bounds_vars["P"] = np.sqrt(self.bounds_vars["Z"])
    return self.bounds_vars, prob, sp_exp

