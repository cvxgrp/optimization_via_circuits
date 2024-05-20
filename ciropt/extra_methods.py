import numpy as np
import scipy
import cvxpy as cp

from ciropt.point import Point
from ciropt.expression import Expression
from ciropt.function import Function
from ciropt.utils import *
from ciropt.sympy_parsing import *
from ciropt.sympy_to_solvers import *






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

    constraints += [ var_x[name2idx["eta"]] >= 0.00001, \
                        var_x[name2idx["h"]] >= 0.00001, \
                        var_x[name2idx["rho"]] >= 0]
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

    assert v_k_list[-1] == "FG_obj", print(v_k_list)

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
    

def solve_bisection_eta(self, freq=20, params=None, max_iters=30, cvx_solver=cp.CLARABEL, **kwargs):
        # given discretization parameters in params
        # do bisection on eta by checking if the resulting SDP subject to params and eta
        # is feasible
        dim_G = Point.counter
        dim_F = Expression.counter 
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                if function.get_is_leaf()]
        discretization_params = self.discretization_params
        assert sorted(list(params.keys())) == discretization_params
        sp_exp, total_I_size, total_eq_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]
        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)
        inputs = (v_coeffs, v_names, v_k_list, name2idx, total_I_size, total_eq_size, sp_exp)

        eta = params["eta"] if "eta" in params else 10.
        for t in range(max_iters):
            params["eta"] = eta
            try:
                prob = self.solve_fix_discr_sdp(params=params, cvx_solver=cvx_solver, inputs=inputs, verbose=False)[1]
                if t % freq == 0: print(f"{t=}, {eta=}, {prob.status=}")
                if prob.status == 'optimal':
                    return params, eta
            except: pass
            eta /= 2
        return params, eta


def solve_fix_discr_sdp(self, params=None, cvx_solver=cp.CLARABEL, verbose=True, debug=False, inputs=None, **kwargs):
        # for fixed alpha, beta, h, eta, rho, gamma 
        # solve the corresponding SDP
        dim_G = Point.counter
        dim_F = Expression.counter 
        if inputs is None:
            list_of_leaf_functions = [function for function in Function.list_of_functions
                                    if function.get_is_leaf()]
            discretization_params = self.discretization_params
            assert sorted(list(params.keys())) == discretization_params
            sp_exp, total_I_size, total_eq_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:3]

            v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, discretization_params)
            if debug:   assert set(list(name2idx.keys())) == set(v_names)
            lambda_size = total_I_size
            nu_size = total_eq_size
        else:
            v_coeffs, v_names, v_k_list, name2idx, lambda_size, nu_size, sp_exp = inputs
        
        v_value = vars_to_vector(v_names, params)
        v_size = len(v_names)
        
        # SDP variables
        var_lambda_nu = cp.Variable((lambda_size + nu_size, 1))
        Z = cp.Variable((dim_G, dim_G), PSD=True)
        constraints = [ var_lambda_nu[:lambda_size] >= 0]
        assert v_k_list[-1] == "FG_obj", print(v_k_list)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (lambda_size + nu_size, dim_F, v_size)
        constraints += [ cp.vec(obj_F @ v_value, order="C") == cp.vec(var_lambda_nu.T @ (sum_ij_F @ v_value).squeeze(), order="C")]

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (lambda_size + nu_size, dim_G*dim_G, v_size)
        vec_Z = cp.vec(Z, order="C")
        constraints += [cp.vec(obj_G @ v_value, order="C") + vec_Z == cp.vec(var_lambda_nu.T @ (sum_ij_G @ v_value).squeeze(), order="C") ]

        prob = cp.Problem(cp.Minimize(0), constraints)
        # 'SCS', 'CVXOPT','SDPA', cp.CLARABEL, cp.MOSEK
        prob.solve(solver=cvx_solver, verbose=verbose)
        if verbose:
            print(f"{prob.status=}")
        # if prob.status != 'optimal':
        #     raise Exception(f"Not a feasible solution, {prob.status=}")
        self.prob = prob
        self.name2idx = name2idx
        self.v_names = v_names
        self.vars = {"Z" : Z.value, "lambdas_nus" : var_lambda_nu.value}
        self.vars.update(params)
        return self.vars, prob, sp_exp


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

    gp_vars = { 'eta': model.addVar(name='eta', lb=0., ub=1000.),
                'rho': model.addVar(name='rho', lb=0., ub=1000.),
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
        name = "lamb%rho"%(f_idx + shift_f_idx)
        size_I_function = len(function.list_of_points)
        gp_vars[name] = model.addMVar((size_I_function, size_I_function), name=name, lb=0, ub=1000)
        model.update()
        if bounds is not None and name in bounds:
            if "ub" in bounds[name]:
                model.addConstr( gp_vars[name] <= bounds[name]["ub"] )
            if "lb" in bounds[name]:
                model.addConstr( gp_vars[name] >= bounds[name]["lb"] )
        # model.addConstr( gp_vars[name].diagonal() == np.zeros(size_I_function) )

    assert sum_ij_La.shape == sp_exp["FG_obj"]["F"].shape and sum_ij_AC.shape == sp_exp["FG_obj"]["G"].shape
    sp_z1 = simplify_matrix(sum_ij_La - sp_exp["FG_obj"]["F"])
    sp_z2 = simplify_matrix(sum_ij_AC - sp_exp["FG_obj"]["G"]) 
    z1 = sympy_matrix_to_gurobi(sp_z1, gp_vars, model)
    z2 = sympy_matrix_to_gurobi(sp_z2, gp_vars, model)
    PPt = gp_vars["P"] @ gp_vars["P"].T
    model.addConstrs(z1[i] == 0 for i in range(dim_F))
    model.addConstrs(z2[i, j] - PPt[i, j].item() == 0 for i in range(dim_G) for j in range(dim_G))
    if not verbose:
        model.Params.LogToConsole = 0
    # model.setObjective( - gp_vars["eta"] - gp_vars["rho"] )
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
                            'Fweights_obj':sympy_matrix_to_gurobi(simplify_matrix(sp_exp["FG_obj"]["F"]), gp_vars, model), \
                            'Gweights_obj':sympy_matrix_to_gurobi(simplify_matrix(sp_exp["FG_obj"]["G"]), gp_vars, model) }
    self.model = model
    self.vars = gp_vars
    return dict_parameters_ciropt_gp(model, gp_vars), model, sp_exp