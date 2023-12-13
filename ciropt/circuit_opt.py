import casadi as ca
import numpy as np
import sympy as sp
import cvxpy as cp
import dccp


from ciropt.point import Point
from ciropt.expression import Expression
from ciropt.function import Function
from ciropt.utils import *

"""
Some of the code is borrowed from the PEPit https://github.com/PerformanceEstimation/PEPit
"""


class CircuitOpt(object):
    counter = 0

    def __init__(self):
        self._reset_classes()
        self._init_discretization_parameters()
        self.counter = CircuitOpt.counter
        CircuitOpt.counter += 1

        self.list_of_functions = list()
        self.list_of_points = list()
        self.list_of_performance_metrics = list()


    def _init_discretization_parameters(self):
        self.b = sp.symbols('b')
        self.d = sp.symbols('d')
        self.h = sp.symbols('h')
        self.alpha = sp.symbols('alpha')
        self.beta = sp.symbols('beta')


    @staticmethod
    def _reset_classes():
        Expression.counter = 0
        Expression.list_of_leaf_expressions = list()
        Function.counter = 0
        Function.list_of_functions = list()
        CircuitOpt.counter = 0
        Point.counter = 0
        Point.list_of_leaf_points = list()


    def declare_function(self, function_class, **kwargs):
        f = function_class(is_leaf=True, decomposition_dict=None, **kwargs)
        self.list_of_functions.append(f)
        return f
    

    def set_initial_point(self):
        x = Point(is_leaf=True, decomposition_dict=None)
        self.list_of_points.append(x)
        return x
    
    
    def set_performance_metric( self,  perf_metric):
        """
        perf_metric = E_2 - (E_1 - Delta_1)
        Delta_1 = b * (tilde_f_1 - f_star) + d * (R * \|i_R\|^2_2)
        """
        assert isinstance(perf_metric, Expression)
        self.perf_metric = perf_metric
        assert check_degree_expression(self.perf_metric, 2)


    def _expression_to_matrix(self, expression, dim_G, dim_F, sp_v=None):
        # convert expression into matrix representation: function of F and G
        Fweights = np.zeros((dim_F), dtype=object)
        Gweights = np.zeros((dim_G, dim_G), dtype=object)
        for key, sp_weight in expression.decomposition_dict.items():
            if not isinstance(sp_weight, sp.Basic) or sp.simplify(sp_weight).free_symbols == set():
                sp_weight = float(sp_weight)
            if type(key) == Expression: # Function values are stored in F
                assert key.get_is_leaf()
                Fweights[key.counter] += sp_weight
            elif type(key) == tuple: # Inner products are stored in G
                point1, point2 = key
                assert point1.get_is_leaf() and point2.get_is_leaf()
                Gweights[point1.counter, point2.counter] += sp_weight/2
                Gweights[point2.counter, point1.counter] += sp_weight/2
        return Fweights, Gweights
    

    def circuit_symbolic_matrices(self, list_of_leaf_functions, dim_G, dim_F):
        sum_ij_La = np.zeros(dim_F, dtype=object)
        sum_ij_AC  = np.zeros((dim_G, dim_G), dtype=object)
        sp_exp = {}
        total_I_size = 0
        for f_idx, function in enumerate(list_of_leaf_functions):
            function.add_class_constraints()
            size_I_function = len(function.list_of_points)
            counter = 0
            prefix_name = "lamb%d"%f_idx
            for i in range(size_I_function):
                for j in range(size_I_function):
                    if i == j: continue 
                    lamb_ij = sp.symbols(prefix_name + "|%d.%d|"%(i,j))
                    f_interpolation_ineq_ij = function.list_of_class_constraints[counter]
                    Fweights_ij, Gweights_ij = self._expression_to_matrix(f_interpolation_ineq_ij, dim_G, dim_F)
                    sp_exp[(f_idx, i, j)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                    counter += 1
                    assert sum_ij_La.shape == Fweights_ij.shape and sum_ij_AC.shape == Gweights_ij.shape
                    sum_ij_La += lamb_ij * Fweights_ij
                    sum_ij_AC += lamb_ij * Gweights_ij
            total_I_size += counter
            assert len(function.list_of_class_constraints) == size_I_function * (size_I_function - 1)

        Fweights_d, Gweights_d = self._expression_to_matrix(self.perf_metric, dim_G, dim_F) 
        sp_exp["FG_d"] = {"F" : Fweights_d, "G" : Gweights_d}
        assert sum_ij_La.shape == Fweights_d.shape and sum_ij_AC.shape == Gweights_d.shape
        return sp_exp, total_I_size, sum_ij_La, sum_ij_AC
        

    def solve_gp(self, verbose=True, debug=False, time_limit=1000, ftol=1e-9, heur=0.001, method=0, bounds=None, **kwargs):
        # lazy form of optimization problem
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        model = gp.Model() 
 
        gp_vars = { 'b': model.addVar(name='b', lb=0., ub=1000.),
                    'd': model.addVar(name='d', lb=0., ub=1000.),
                    'h': model.addVar(name='h', lb=0., ub=1000.),
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
        sp_exp, total_I_size, sum_ij_La, sum_ij_AC = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)
        for f_idx in range(len(list_of_leaf_functions)):
            name = "lamb%d"%(f_idx)
            gp_vars[name] = model.addMVar((dim_F, dim_F), name=name, lb=0, ub=1000)
            model.update()
            if bounds is not None and name in bounds:
                if "ub" in bounds[name]:
                    model.addConstr( gp_vars[name] <= bounds[name]["ub"] )
                if "lb" in bounds[name]:
                    model.addConstr( gp_vars[name] >= bounds[name]["lb"] )
            model.addConstr( gp_vars[name].diagonal() == np.zeros(dim_F) )

        assert sum_ij_La.shape == sp_exp["FG_d"]["F"].shape and sum_ij_AC.shape == sp_exp["FG_d"]["G"].shape
        sp_z1 = simplify_matrix(sum_ij_La - sp_exp["FG_d"]["F"])
        sp_z2 = simplify_matrix(sum_ij_AC - sp_exp["FG_d"]["G"]) # sum_ij_AC - P @ P.T - Gweights_d
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

    
    def solve_ca(self, verbose=True, debug=False, init_vals=None, bounds=None, **kwargs):
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        opti = ca.Opti()        
        ca_vars = { 'b': opti.variable(),
                    'd': opti.variable(),
                    'h': opti.variable(),
                    'alpha': opti.variable(),
                    'beta': opti.variable(), 
                    'P': opti.variable(dim_G, dim_G) }
        P = ca.tril(ca_vars["P"])
        opti.subject_to( ca.diag(P) >= np.zeros((dim_G, 1)) )
        ca_add_bounds(opti, bounds, ca_vars)

        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        
        sp_exp, total_I_size, sum_ij_La, sum_ij_AC = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)
        size_I_function = int(np.ceil(np.sqrt(total_I_size)))
        for f_idx in range(len(list_of_leaf_functions)):
            lamb = opti.variable(size_I_function, size_I_function)
            ca_vars["lamb%d"%f_idx] = lamb
            opti.subject_to( ca.reshape(lamb, (-1, 1)) >= np.zeros((size_I_function * size_I_function, 1)) )

        sp_z1 = simplify_matrix(sum_ij_La - sp_exp["FG_d"]["F"])
        sp_z2 = simplify_matrix(sum_ij_AC - sp_exp["FG_d"]["G"]) # sum_ij_AC - P @ P.T - Gweights_d
        z1 = sympy_matrix_to_casadi(sp_z1, ca_vars, opti)
        z2 = sympy_matrix_to_casadi(sp_z2, ca_vars, opti)
        opti.subject_to( z1 == np.zeros((dim_F, 1))) 
        opti.subject_to( z2 - P @ P.T == np.zeros((dim_G, dim_G)))
        opts = {'ipopt.max_iter':50000}
        if not verbose:
            opts.update({'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
        print(f"{opts=}")
        if init_vals is not None:
            for name, var_value in init_vals.items():
                opti.set_initial(ca_vars[name], var_value)

        opti.minimize( - sympy_expression_to_casadi(self.obj, ca_vars, opti))
        opti.solver('ipopt', opts)
        try:
            sol = opti.solve() # QCQP for solving CircuitOpt
        except:
            if debug:
                self.ca_vars = ca_vars
            return None, ca_vars, None, sp_exp
        assert sol.stats()['success'], print(sol.stats())
        if debug:
            self.ca_expressions = {'sum_ij_La':sol.value(sympy_matrix_to_casadi(sum_ij_La, ca_vars, opti)),\
                                'sum_ij_AC':sol.value(sympy_matrix_to_casadi(sum_ij_AC, ca_vars, opti)),\
                                'Fweights_d':sol.value(sympy_matrix_to_casadi(sp_exp["FG_d"]["F"], ca_vars, opti)),\
                                'Gweights_d':sol.value(sympy_matrix_to_casadi(sp_exp["FG_d"]["G"], ca_vars, opti)),\
                                'z1': z1,\
                                'z2': z2,\
                                'P': P}
        self.opti = opti
        self.vars = ca_vars
        return dict_parameters_ciropt(sol, ca_vars), sol, sp_exp
    
  
    def solve_ca_canonical(self, verbose=True, bounds=None, debug=False, **kwargs):
        # formulate problem explicitly as QCQP with matrices
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]

        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)

        v_size = len(v_names)
        x_size = len(v_names) + total_I_size +  dim_G * (dim_G + 1) // 2
        opti = ca.Opti()
        var_x = opti.variable(x_size + 20, 1)
        opti.subject_to(var_x[0] == 1)

        # ca_add_bounds(opti, bounds, ca_vars)

        assert v_k_list[-1] == "FG_d", print(v_k_list)

        vec_indices = { "v"   : [0, v_size - 1],\
                        "lamb": [v_size, v_size + total_I_size - 1 ], \
                        "P"  : [v_size + total_I_size, x_size - 1]} 

        vec_v = get_vec_var(var_x, "v", vec_indices)
        vec_lambs = get_vec_var(var_x, "lamb", vec_indices)
        vec_P = get_vec_var(var_x, "P", vec_indices)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (vec_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (vec_lambs.shape[0], dim_F, v_size))
        res = np.zeros((dim_F, 1))
        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == vec_v.shape[0] and sum_ij_F[:, k, :].shape == (vec_lambs.shape[0], vec_v.shape[0]) 
            opti.subject_to(  obj_F[k : k+1, :] @ vec_v - vec_lambs.T @ sum_ij_F[:, k, :] @ vec_v == 0)

        # opti.subject_to( res == np.zeros((dim_F, 1)) )

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (vec_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (vec_lambs.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                P_k1P_k2 = get_PPt_element(vec_P, k1, k2)
                opti.subject_to(  obj_G[k_idx : k_idx+1, :] @ vec_v + P_k1P_k2 - vec_lambs.T @ sum_ij_G[:, k_idx, :] @ vec_v == 0)

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
            Qi = symm_prod_one_hot((v_size, v_size), name2idx[v], name2idx[pref_v])
            opti.subject_to( vec_v.T @ Qi @ vec_v + ai.T @ vec_v == 0)

        # lambda >= 0 constraints
        opti.subject_to( vec_lambs >= np.zeros(vec_lambs.shape))
        # diag(P) >= 0 constraints
        vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
        Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
        np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
        # P_diag_constraints = [Q]
        # print( Q.shape, vec_P.shape, dim_G)
        opti.subject_to( Q @ vec_P >= np.zeros((dim_G, 1)) )

        opts = {}
        if not verbose:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        ca_vars = {name : var_x[idx] for name, idx in name2idx.items()}
        opti.minimize( - sympy_expression_to_casadi(self.obj, ca_vars, opti))
        opti.solver('ipopt', opts)
        sol = opti.solve() # QCQP for solving CircuitOpt
        assert sol.stats()['success'], print(sol.stats())
        self.opti = opti
        self.name2idx = name2idx
        self.v_names = v_names
        self.vars = {"x": sol.value(var_x)}
        return dict_parameters_ciropt(sol, ca_vars), sol, sp_exp
    

    def solve_ca_canonical_X(self, verbose=True, debug=False, **kwargs):
        # formulate problem explicitly as QCQP using x and matrix X
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]

        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)

        v_size = len(v_names)
        x_size = len(v_names) + total_I_size +  dim_G * (dim_G + 1) // 2
        opti = ca.Opti()
        var_x1 = opti.variable(x_size + 20, 1)
        var_x = var_x1[:x_size]
        var_X = var_x @ var_x.T
        opti.subject_to(var_x[0] == 1)
        assert v_k_list[-1] == "FG_d", print(v_k_list)

        vec_indices = { "v"   : [0, v_size - 1],\
                        "lamb": [v_size, v_size + total_I_size - 1 ], \
                        "P"  : [v_size + total_I_size, x_size - 1]} 

        I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
        I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
        I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (I_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs.shape[0], dim_F, v_size))
        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs.shape[0], I_v.shape[0])
            opti.subject_to(  obj_F[k : k+1, :] @ I_v @ var_x - ca.trace(I_lambs.T @ sum_ij_F[:, k, :] @ I_v @ var_X) == 0)

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (I_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
                opti.subject_to(  obj_G[k_idx : k_idx+1, :] @ I_v @ var_x \
                                + ca.trace(S1.T @ S2 @ var_X)\
                                - ca.trace(I_lambs.T @ sum_ij_G[:, k_idx, :] @ I_v @ var_X) == 0)

        # v variables quadratic constraints
        # v^T Qi v + ai^T v = 0 
        for name in v_names[1:]:
            vars = name.split("_")
            if len(vars) == 1: continue
            pref_v, v = "_".join(vars[:-1]), vars[-1]
            # print(f"{name=}, {pref_v=}, {v=}")
            ai = - one_hot(v_size, name2idx[name])
            # include both permutations for the product 
            Qi = symm_prod_one_hot((v_size, v_size), name2idx[v], name2idx[pref_v])
            opti.subject_to( ca.trace(I_v.T @ Qi @ I_v @ var_X)+ ai.T @ I_v @ var_x == 0)

        # lambda >= 0 constraints
        opti.subject_to( I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1)))
        # diag(P) >= 0 constraints
        vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
        Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
        np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
        # P_diag_constraints = [Q]
        # print( Q.shape, vec_P.shape, dim_G)
        opti.subject_to( Q @ I_P @ var_x >= np.zeros((dim_G, 1)) )

        opts = {}
        if not verbose:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        ca_vars = {name : var_x[idx] for name, idx in name2idx.items()}
        opti.minimize( -sympy_expression_to_casadi(self.obj, ca_vars, opti))
        opti.solver('ipopt', opts)
        sol = opti.solve() # QCQP for solving CircuitOpt
        assert sol.stats()['success'], print(sol.stats())
        self.opti = opti
        self.name2idx = name2idx
        self.v_names = v_names
        self.vars = {"x": sol.value(var_x)}
        return dict_parameters_ciropt(sol, ca_vars), sol, sp_exp


    def solve_gp_canonical_X(self, verbose=True, debug=False, time_limit=1000, ftol=1e-9, heur=0.001, method=0, bounds=None, **kwargs):
        # formulate problem explicitly as QCQP using x and matrix X
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]

        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)

        v_size = len(v_names)
        x_size = len(v_names) + total_I_size +  dim_G * (dim_G + 1) // 2
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
                        "P"  : [v_size + total_I_size, x_size - 1]} 

        I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
        I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
        I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (I_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs.shape[0], dim_F, v_size))

        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs.shape[0], I_v.shape[0])
            M1 = obj_F[k : k+1, :] @ I_v
            M2 = I_lambs.T @ sum_ij_F[:, k, :] @ I_v
            print(type(M1), type(var_X), type(M2))
            model.addConstr(  (M1 @ var_x).item()  - gp_trace(M2 @ var_X) == 0)

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (I_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
                M1 = obj_G[k_idx : k_idx+1, :] @ I_v
                M2 = S1.T @ S2
                M3 = I_lambs.T @ sum_ij_G[:, k_idx, :] @ I_v
                model.addConstr(  (M1 @ var_x).item()  \
                                + gp_trace(M2 @ var_X) \
                                - gp_trace(M3 @ var_X) == 0)

        # v variables quadratic constraints
        # v^T Qi v + ai^T v = 0 
        for name in v_names[1:]:
            vars = name.split("_")
            if len(vars) == 1: continue
            pref_v, v = "_".join(vars[:-1]), vars[-1]
            # print(f"{name=}, {pref_v=}, {v=}")
            ai = - one_hot(v_size, name2idx[name])
            # include both permutations for the product 
            Qi = symm_prod_one_hot((v_size, v_size), name2idx[pref_v], name2idx[v])
            M1 = (I_v.T @ Qi @ I_v)
            M2 = ai.T @ I_v
            model.addConstr( gp_trace(M1 @ var_X) + (M2 @ var_x).item() == 0)
            # model.addConstr( gp_trace((I_v.T @ Qi2 @ I_v) @ var_X) + ((ai.T @ I_v) @ var_x).item() == 0)

        # lambda >= 0 constraints
        model.addConstr( I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1)))
        # diag(P) >= 0 constraints
        vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
        Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
        np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
        # P_diag_constraints = [Q]
        # print( Q.shape, vec_P.shape, dim_G)
        M1 = Q @ I_P
        model.addConstr( M1 @ var_x >= np.zeros((dim_G, 1)) )

        if not verbose:
            model.Params.LogToConsole = 0
        model.update()
        gp_vars = {name : var_x[name2idx[name]].item() for name in core_vars}
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
        self.vars = {"x": var_x.X}
        return dict_parameters_ciropt_gp(model, gp_vars), model, sp_exp
   
  
    def bounds_sdp_relax_all(self, verbose=True, cvx_solver=cp.CLARABEL, debug=False, bounds=None, **kwargs):
        # formulate problem explicitly as QCQP using x and matrix X
        # variable x = vec(v, lambda, P)
        # and relax it to convex SDP to get bounds on the variables
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        
        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
        # bounds_names = sorted(['alpha', 'beta', 'h', 'b', 'd', 'P', "lamb"])
        bounds_names = sorted(['alpha', 'beta', 'P', "lamb"])

        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)
        v_size = len(v_names)
        x_size = len(v_names) + total_I_size +  dim_G * (dim_G + 1) // 2
        
        # bounds
        bounds_vars = cp.Variable((len(bounds_names)))
        bnames2idx = {bname:idx for idx, bname in enumerate(bounds_names)}
        # SDP variables
        var_x = cp.Variable((x_size, 1))
        W = cp.Variable((x_size + 1, x_size + 1), symmetric=True)
        var_X = cp.Variable((x_size, x_size), symmetric=True)
        constraints = [ W >> 0, \
                       var_X >> 0, 
                      ]

        constraints += [ var_x[name2idx["b"]] >= 0.05, \
                         var_x[name2idx["h"]] >= 0.01, \
                         var_x[name2idx["d"]] >= 0]
        # constraints to encode X \succeq xx^T
        constraints += [W[ : x_size, : x_size] == var_X, \
                        W[x_size : x_size+1, : x_size] == var_x.T, \
                        W[ : x_size, x_size : x_size+1] == var_x, \
                        W[x_size, x_size] == 1]
        # implied linear constraints
        diag_W = cp.diag(W)
        diag_X = cp.diag(var_X)
        constraints += [ diag_W >= np.zeros((x_size + 1)), \
                        W >= -(cp.outer(np.ones((x_size + 1)), diag_W) + cp.outer(diag_W, np.ones((x_size + 1))) ) / 2 , \
                        W <=  (cp.outer(np.ones((x_size + 1)), diag_W) + cp.outer(diag_W, np.ones((x_size + 1))) ) / 2, \
                        # is true when equality holds X = xx^T
                        diag_X >= np.zeros((x_size)), \
                        var_X >= -(cp.outer(np.ones((x_size)), diag_X) + cp.outer(diag_X, np.ones((x_size))) ) / 2 , \
                        var_X <=  (cp.outer(np.ones((x_size)), diag_X) + cp.outer(diag_X, np.ones((x_size))) ) / 2 ]

        constraints += [ var_x[name2idx["alpha"]] >= -1, \
                        var_x[name2idx["alpha"]] <= 1 , \
                        var_x[name2idx["beta"]] >= -1, \
                        var_x[name2idx["beta"]] <= 1, \
                        var_x[0] == 1 ]

        assert v_k_list[-1] == "FG_d", print(v_k_list)

        vec_indices = { "v"   : [0, v_size - 1],\
                        "lamb": [v_size, v_size + total_I_size - 1 ], \
                        "P"  : [v_size + total_I_size, x_size - 1]} 

        I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
        I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
        I_P = get_vec_var(var_x, "P", vec_indices, matrix=True)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (I_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs.shape[0], dim_F, v_size))
        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs.shape[0], I_v.shape[0]), \
            print(obj_F[k : k+1, :].shape, I_v.shape, sum_ij_F[:, k, :].shape, I_lambs.shape) 
            constraints += [ cp.sum(obj_F[k : k+1, :] @ I_v @ var_x)  - cp.trace(I_lambs.T @ sum_ij_F[:, k, :] @ I_v @ var_X) == 0]

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (I_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                S1, S2 = get_PPt_matrix(var_x, vec_indices, k1, k2)
                constraints += [ cp.sum(obj_G[k_idx : k_idx+1, :] @ I_v @ var_x)  \
                                + cp.trace(S1.T @ S2 @ var_X) \
                                - cp.trace((I_lambs.T @ (sum_ij_G[:, k_idx, :] @ I_v)) @ var_X) == 0 ]

        # v variables quadratic constraints
        # v^T Qi v + ai^T v = 0 
        for name in v_names[1:]:
            vars = name.split("_")
            if len(vars) == 1: continue
            pref_v, v = "_".join(vars[:-1]), vars[-1]
            ai = -one_hot(v_size, name2idx[name])
            # include both permutations for the product 
            Qi = symm_prod_one_hot((v_size, v_size), name2idx[v], name2idx[pref_v])
            constraints += [ cp.trace(I_v.T @ Qi @ I_v @ var_X) + cp.sum(ai.T @ I_v @ var_x) == 0 ]

        # lambda >= 0 constraints
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
                        bounds_vars[bnames2idx["lamb"]] <= cp.pnorm(I_lambs @ var_x, 0.999)]
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


    def solve_cvx_dccp(self, verbose=True, debug=False, bounds=None, **kwargs):
        # solve QCQP problem using convex-concave procedure
        # using x=[vec(v), vec(lamb)] and Z
        # every matrix can be decomposed onto a difference of two PSD matrices
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        
        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])

        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)
        v_size = len(v_names)
        x_size = len(v_names) + total_I_size
        
        # variables
        var_x = cp.Variable((x_size, 1))
        var_X = cp.Variable((x_size, x_size), symmetric=True)
        var_Z = cp.Variable((dim_G, dim_G), symmetric=True)
        constraints = [ var_Z >> 0, var_x[0] == 1, \
                        var_X == var_x @ var_x.T
                      ]
        cvx_vars = { "Z": var_Z,  "x": var_x, "X":var_X}

        constraints += [ var_x[name2idx["b"]] >= 0.001, \
                         var_x[name2idx["h"]] >= 0.001, \
                         var_x[name2idx["d"]] >= 0]

        constraints += [ -1 <= var_x[name2idx["alpha"]], 
                               var_x[name2idx["alpha"]] <= 1 , 
                         -1 <= var_x[name2idx["beta"]], 
                               var_x[name2idx["beta"]] <= 1, 
                        ]

        assert v_k_list[-1] == "FG_d", print(v_k_list)

        vec_indices = { "v"   : [0, v_size - 1],\
                        "lamb": [v_size, v_size + total_I_size - 1 ]} 

        I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
        I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)
        # bounds
        constraints = cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lambs)

        constraints += [I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1))]

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (I_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs.shape[0], dim_F, v_size))
        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs.shape[0], I_v.shape[0]) 
            Q_F = I_lambs.T @ (sum_ij_F[:, k, :] @ I_v)
            q_F = obj_F[k : k+1, :] @ I_v
            # Q_F_plus, Q_F_minus = co.matrix_to_diff_psd(Q_F)
            constraints += [ q_F @ var_x - cp.trace(Q_F @ var_X) == 0]

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (I_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                Q_G = I_lambs.T @ (sum_ij_G[:, k_idx, :] @ I_v)
                q_G = obj_G[k_idx : k_idx+1, :] @ I_v
                # Q_G_plus, Q_G_minus = co.matrix_to_diff_psd(Q_G)
                constraints += [ q_G @ var_x + var_Z[k1, k2] - cp.trace( Q_G @ var_X) == 0]

        # v variables quadratic constraints
        # v^T Qi v + ai^T v = 0 
        for name in v_names[1:]:
            vars = name.split("_")
            if len(vars) == 1: continue # monomial with degree 1
            pref_v, v = "_".join(vars[:-1]), vars[-1]
            # ai = - one_hot(v_size, name2idx[name])
            # include both permutations for the product 
            # Qi = symm_prod_one_hot((v_size, v_size), name2idx[pref_v], name2idx[v]) 
            i_idx, j_idx = name2idx[pref_v], name2idx[v]
            constraints += [ 0.5 * (var_X[i_idx, j_idx] + var_X[j_idx, i_idx]) == var_x[name2idx[name]] ]

        # lambda >= 0 constraints
        constraints += [ I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1))]

        obj = -sympy_expression_to_cvx(self.obj, var_x, name2idx)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        print("problem is DCP:", prob.is_dcp())   # false
        print("problem is DCCP:", dccp.is_dccp(prob))  # true
        prob.solve(method='dccp', verbose=verbose, max_iter=200)
        print(f"{prob.status=}")
        self.prob = prob
        self.name2idx = name2idx
        self.v_names = v_names
        self.var_x = var_x
        self.bounds_vars = {name:bounds_vars.value[b_idx] for b_idx, name in enumerate(bounds_names) }
        self.bounds_vars["P"] = np.sqrt(self.bounds_vars["Z"])
        return self.bounds_vars, prob, sp_exp


    def bounds_sdp_relax(self, verbose=True, var_bound="Z", debug=False, bounds=None, cvx_solver=cp.CLARABEL, **kwargs):
        # formulate problem explicitly as QCQP using x and matrix X and Z
        # keep Z=PP^T as it is, not aggregate into x = vec(p, lambda)
        # and relax it to convex SDP to get bounds on the variables
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        
        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
        bounds_names = sorted(['alpha', 'beta', 'Z', "lamb"])

        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)
        v_size = len(v_names)
        x_size = len(v_names) + total_I_size
        
        # bounds
        bounds_vars = cp.Variable((len(bounds_names)), nonneg=True)
        bnames2idx = {bname:idx for idx, bname in enumerate(bounds_names)}
        # SDP variables
        var_x = cp.Variable((x_size, 1))
        W = cp.Variable((x_size + 1, x_size + 1), symmetric=True)
        var_X = cp.Variable((x_size, x_size), symmetric=True)
        var_Z = cp.Variable((dim_G, dim_G), symmetric=True)
        constraints = [ W >> 0, var_Z >> 0 ]
        cvx_vars = { "Z": var_Z,  "x": var_x, "X": var_X}

        constraints += [ var_x[name2idx["b"]] >= 0.001, \
                         var_x[name2idx["h"]] >= 0.001, \
                         var_x[name2idx["d"]] >= 0]
        # constraints to encode X >> xx^T
        constraints += [W[ : x_size, : x_size] == var_X, \
                        W[x_size : x_size+1, : x_size] == var_x.T, \
                        W[ : x_size, x_size : x_size+1] == var_x, \
                        W[x_size, x_size] == 1]
        # implied linear constraints for PSD matrix W
        diag_W = cp.diag(W)
        constraints += [ diag_W >= np.zeros((x_size + 1)), \
                        cp.abs(W) <= 0.5 * (cp.outer(np.ones((x_size + 1)), diag_W) + cp.outer(diag_W, np.ones((x_size + 1))) ) 
                        ]

        constraints += [ -1 <= var_x[name2idx["alpha"]], 
                               var_x[name2idx["alpha"]] <= 1 , 
                         -1 <= var_x[name2idx["beta"]], 
                               var_x[name2idx["beta"]] <= 1, 
                         var_x[0] == 1 
                        ]

        assert v_k_list[-1] == "FG_d", print(v_k_list)

        vec_indices = { "v"   : [0, v_size - 1],\
                        "lamb": [v_size, v_size + total_I_size - 1 ]} 

        I_v = get_vec_var(var_x, "v", vec_indices, matrix=True)
        I_lambs = get_vec_var(var_x, "lamb", vec_indices, matrix=True)

        constraints = cvx_add_bounds(constraints, bounds, cvx_vars, name2idx, var_x, I_lambs)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (I_lambs.shape[0], dim_F, v_size), print(sum_ij_F.shape, (I_lambs.shape[0], dim_F, v_size))
        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == I_v.shape[0] and sum_ij_F[:, k, :].shape == (I_lambs.shape[0], I_v.shape[0]), \
            print(obj_F[k : k+1, :].shape, I_v.shape, sum_ij_F[:, k, :].shape, I_lambs.shape) 
            constraints += [ cp.sum(obj_F[k : k+1, :] @ I_v @ var_x)  - cp.trace(I_lambs.T @ sum_ij_F[:, k, :] @ I_v @ var_X) == 0]

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (I_lambs.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (I_lambs.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                constraints += [ cp.sum(obj_G[k_idx : k_idx+1, :] @ I_v @ var_x)  \
                                + var_Z[k1, k2] \
                                - cp.trace((I_lambs.T @ (sum_ij_G[:, k_idx, :] @ I_v)) @ var_X) == 0 ]

        # v variables quadratic constraints
        # v^T Qi v + ai^T v = 0 
        for name in v_names[1:]:
            vars = name.split("_")
            if len(vars) == 1: continue
            pref_v, v = "_".join(vars[:-1]), vars[-1]
            ai = - one_hot(v_size, name2idx[name])
            # include both permutations for the product 
            Qi = symm_prod_one_hot((v_size, v_size), name2idx[pref_v], name2idx[v]) 
            constraints += [ cp.trace(I_v.T @ Qi @ I_v @ var_X) + cp.sum(ai.T @ I_v @ var_x) == 0 ]

        # lambda >= 0 constraints
        constraints += [ I_lambs @ var_x >= np.zeros((I_lambs.shape[0], 1))]

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

        constraints += [  bounds_vars[bnames2idx["Z"]] <= cp.trace(var_Z), \
                          bounds_vars[bnames2idx["lamb"]] <= cp.pnorm(I_lambs @ var_x, 0.999),
                          bounds_vars[bnames2idx["alpha"]] == var_x[name2idx["alpha"]],
                          bounds_vars[bnames2idx["beta"]] == var_x[name2idx["beta"]],   ]

        obj = - cp.sum(bounds_vars[bnames2idx[var_bound]])
        prob = cp.Problem(cp.Minimize(obj), constraints)
        # 'SCS', 'CVXOPT','SDPA', cp.CLARABEL, cp.MOSEK
        prob.solve(solver=cvx_solver, verbose=verbose)
        print(f"{prob.status=}")
        self.prob = prob
        self.name2idx = name2idx
        self.v_names = v_names
        self.var_x = var_x
        self.bounds_vars = {name:bounds_vars.value[b_idx] for b_idx, name in enumerate(bounds_names) }
        self.bounds_vars["P"] = np.sqrt(self.bounds_vars["Z"])
        return self.bounds_vars, prob, sp_exp


    def solve_cvx_fix_discr_sdp(self, params=None, cvx_solver=cp.CLARABEL, verbose=True, debug=False, **kwargs):
        # for fixed alpha, beta, h, b, d 
        # solve the corresponding SDP
        dim_G = Point.counter
        dim_F = Expression.counter 
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        
        core_vars = sorted(['alpha', 'beta', 'h', 'b', 'd'])
        assert sorted(list(params.keys())) == core_vars

        sp_exp, total_I_size = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)[:2]

        v_coeffs, v_names, name2idx, v_k_list = sp_v_coeff_matrix(sp_exp, core_vars)
        if debug:   assert set(list(name2idx.keys())) == set(v_names)
        lambda_size = total_I_size
        v_value = vars_to_vector(v_names, params)
        v_size = len(v_names)
        
        # SDP variables
        var_lambda = cp.Variable((lambda_size, 1), nonneg=True)
        Z = cp.Variable((dim_G, dim_G), symmetric=True)
        constraints = [ Z >> 0 ]
        assert v_k_list[-1] == "FG_d", print(v_k_list)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (lambda_size, dim_F, v_size)
        constraints += [ cp.vec(obj_F @ v_value, order="C") == cp.vec(var_lambda.T @ (sum_ij_F @ v_value).squeeze(), order="C")]

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (lambda_size, dim_G*dim_G, v_size)
        vec_Z = cp.vec(Z, order="C")
        constraints += [cp.vec(obj_G @ v_value, order="C") + vec_Z == cp.vec(var_lambda.T @ (sum_ij_G @ v_value).squeeze(), order="C") ]

        prob = cp.Problem(cp.Minimize(0), constraints)
        # 'SCS', 'CVXOPT','SDPA', cp.CLARABEL, cp.MOSEK
        prob.solve(solver=cvx_solver, verbose=verbose)
        print(f"{prob.status=}")
        self.prob = prob
        self.name2idx = name2idx
        self.v_names = v_names
        self.vars = {"Z" : Z.value, "lambdas" : var_lambda.value}
        return self.vars, prob, sp_exp
   

    def solve(self, solver="ca", **kwargs):
        if solver == "gp":
            return self.solve_gp(**kwargs)
        elif solver == "gp_canonical_X":
            return self.solve_gp_canonical_X(**kwargs)
        elif solver == "bounds_sdp_relax":
            return self.bounds_sdp_relax(**kwargs)
        elif solver == "ca":
            return self.solve_ca(**kwargs)
        elif solver == "ca_canonical":
            return self.solve_ca_canonical(**kwargs)
        elif solver == "ca_canonical_X":
            return self.solve_ca_canonical_X(**kwargs)
        elif solver == "cvx_fix_discr_sdp":
            return self.solve_cvx_fix_discr_sdp(**kwargs)
        elif solver == "cvx_dccp":
            return self.solve_cvx_dccp(**kwargs)
        