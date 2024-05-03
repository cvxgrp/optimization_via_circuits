import casadi as ca
import numpy as np
import sympy as sp
import cvxpy as cp
import scipy

from collections import OrderedDict

from ciropt.point import Point
from ciropt.expression import Expression
from ciropt.function import Function
from ciropt.constraint import Constraint
from ciropt.utils import *
from ciropt.sympy_parsing import *
from ciropt.sympy_to_solvers import *
from ciropt.extra_methods import *

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
        self.list_of_constraints = list()


    def _init_discretization_parameters(self):
        self.b = sp.symbols('b')
        self.d = sp.symbols('d')
        self.h = sp.symbols('h')
        self.alpha = sp.symbols('alpha')
        self.beta = sp.symbols('beta')
        self.gamma = sp.symbols('gamma')
        self.delta = sp.symbols('delta')
        self.discretization_params = sorted(['alpha', 'beta', 'h', 'b', 'd', 'gamma', 'delta'])


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
    

    def add_constraint(self, constraint):
        assert isinstance(constraint, Constraint)
        self.list_of_constraints.append(constraint)

    
    def set_performance_metric( self,  perf_metric):
        """
        The goal is to show that descent lemma holds, ie, for all one step transitions
        perf_metric is nonpositive for
            perf_metric = E_2 - (E_1 - Delta_1)
        Delta_1 = b * (x - x^\star) * (y - y_star) + d * (R * \|i_R - i_R^\star\|^2_2)
        """
        assert isinstance(perf_metric, Expression)
        self.perf_metric = perf_metric
        assert check_degree_expression(self.perf_metric, 2)


    def _expression_to_matrix(self, expression, dim_G, dim_F, sp_v=None):
        """
        Convert expression into matrix representation as a function of F and G
        Return Fweights and Gweights such that 
                            expression = trace(F.T * Fweights) + trace(G.T * Gweights)
            Fweights: 
                np.array(dim_F) with each entry being
            Gweights: 
                np.array(dim_G, dim_G) symmetric matrix
                each entry being a polynomial of type sympy
        """
        Fweights = np.zeros((dim_F), dtype=object)
        Gweights = np.zeros((dim_G, dim_G), dtype=object)
        for key, sp_weight in expression.decomposition_dict.items():
            if not isinstance(sp_weight, sp.Basic) or sp.simplify(sp_weight).free_symbols == set():
                sp_weight = float(sp_weight)
            if type(key) == Expression: # Function values are stored in F
                assert key.get_is_leaf()
                Fweights[key.counter] += sp_weight
            elif type(key) == tuple: # Inner products between points are stored in G
                point1, point2 = key
                assert point1.get_is_leaf() and point2.get_is_leaf()
                # symmetric matrix
                Gweights[point1.counter, point2.counter] += sp_weight/2
                Gweights[point2.counter, point1.counter] += sp_weight/2
        return Fweights, Gweights
    

    def circuit_symbolic_matrices(self, list_of_leaf_functions, dim_G, dim_F):
        """
        Extract coefficient matrices for primal variable F (function values) and primal variable G (Gram matrix)
        that appear in objective, interlolating constraints and extra constraints
            coefficients are linear in p = (monomials of h, alpha, beta, b, d, gamma) 
        sp_exp: OrderedDict() 
                1) F/Gweights for inequalities
                    if there are extra inequalities (besides the interpolating ones), then I(extra ineq.)=1
                    sp_exp[(0, 0, i, 0)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                        ith extra inequality
                    and
                    sp_exp[(0, f_idx + I(extra ineq.), i, j)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                        (i,j)th interpolating inequality for function f_idx
                2) F/Gweights for equalities
                    sp_exp[(1, 0, i, 0)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                        ith equality for function f_idx
                3) F/Gweights for objective
                    sp_exp["FG_d"] = {"F" : Fweights_d, "G" : Gweights_d}

        Call p_coeffs, p_names, name2idx, _ = sp_v_coeff_matrix(sp_exp, discretization_params) to decompose
        coefficients onto vector of monomials p_names and with vector of constants p_coeffs, which dot product
        recovers the expression
        """
        sum_ij_La = np.zeros(dim_F, dtype=object)
        sum_ij_AC  = np.zeros((dim_G, dim_G), dtype=object)
        sp_exp_ineq, sp_exp_eq = OrderedDict(), OrderedDict()
        total_I_size = 0; shift_f_idx = 0; total_eq_size = 0
        # add extra constraints
        if self.list_of_constraints != list():
            ineq_idx = 0; eq_idx = 0
            for expr in self.list_of_constraints:
                # expression is a linear combination of entries in G and F 
                Fweights_ij, Gweights_ij = self._expression_to_matrix(expr.expression, dim_G, dim_F)
                if expr.equality_or_inequality == "inequality":
                    prefix_name = "lamb0"
                    total_I_size += 1
                    sp_exp_ineq[(0, 0, ineq_idx, 0)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                    lamb_ij = sp.symbols(prefix_name + "|%d.%d|"%(ineq_idx, 0))
                    ineq_idx += 1
                elif expr.equality_or_inequality == "equality":
                    prefix_name = "nu0"
                    total_eq_size += 1
                    sp_exp_eq[(1, 0, eq_idx, 0)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                    lamb_ij = sp.symbols(prefix_name + "|%d.%d|"%(eq_idx, 0))
                    eq_idx += 1
                sum_ij_La += lamb_ij * Fweights_ij
                sum_ij_AC += lamb_ij * Gweights_ij
            if ineq_idx >= 1: shift_f_idx = 1
        # add interpolating inequalities for each f_i
        for f_idx, function in enumerate(list_of_leaf_functions):
            function.add_class_constraints()
            size_I_function = len(function.list_of_points)
            counter = 0
            prefix_name = "lamb%d"%(f_idx + shift_f_idx)
            for i in range(size_I_function):
                for j in range(size_I_function):
                    if i == j: continue 
                    lamb_ij = sp.symbols(prefix_name + "|%d.%d|"%(i,j))
                    f_interpolation_ineq_ij = function.list_of_class_constraints[counter]
                    # expression is a linear combination of entries in G and F
                    Fweights_ij, Gweights_ij = self._expression_to_matrix(f_interpolation_ineq_ij, dim_G, dim_F)
                    sp_exp_ineq[(0, f_idx + shift_f_idx, i, j)] = {"F" : Fweights_ij, "G" : Gweights_ij}
                    counter += 1
                    assert sum_ij_La.shape == Fweights_ij.shape and sum_ij_AC.shape == Gweights_ij.shape
                    sum_ij_La += lamb_ij * Fweights_ij
                    sum_ij_AC += lamb_ij * Gweights_ij
            total_I_size += counter
            assert len(function.list_of_class_constraints) == size_I_function * (size_I_function - 1)
        sp_exp_ineq.update(sp_exp_eq)
        # ordered dictionary with FGweights for inequalities, FGweights for equalities, FGweights for objective
        sp_exp = sp_exp_ineq
        assert len(sp_exp) == total_I_size + total_eq_size
        Fweights_d, Gweights_d = self._expression_to_matrix(self.perf_metric, dim_G, dim_F) 
        sp_exp["FG_d"] = {"F" : Fweights_d, "G" : Gweights_d}
        assert sum_ij_La.shape == Fweights_d.shape and sum_ij_AC.shape == Gweights_d.shape
        return sp_exp, total_I_size, total_eq_size, sum_ij_La, sum_ij_AC
        
    
    def solve_ipopt(self, verbose=True, debug=False, init_vals=None, bounds=None, **kwargs):
        """
        Use ipopt to find proofs: 
            solve the nonconvex problem by directly dualizing the Grammian formulation, 
            ie, each expression is polynomial in discretization parameters linear in G and F 
            without introducing the dummy variables for the discretization parameters
        """
        dim_G = Point.counter
        dim_F = Expression.counter 
        print(f"{dim_G=}, {dim_F=}")
        opti = ca.Opti()        
        ca_vars = { 'b': opti.variable(),
                    'd': opti.variable(),
                    'h': opti.variable(),
                    'gamma': opti.variable(),
                    'delta': opti.variable(),
                    'alpha': opti.variable(),
                    'beta': opti.variable(), 
                    'P': opti.variable(dim_G, dim_G) }
        P = ca.tril(ca_vars["P"])
        opti.subject_to( ca.diag(P) >= np.zeros((dim_G, 1)) )
        ca_add_bounds(opti, bounds, ca_vars, set())
        opti.subject_to( ca_vars["b"] >= 0 ); opti.subject_to( ca_vars["d"] >= 0 )
        opti.subject_to( ca_vars["h"] >= 0 )
        # opti.subject_to( ca_vars["alpha"] >= -100 ); opti.subject_to( ca_vars["alpha"] <= 100 )
        # opti.subject_to( ca_vars["beta"] >= -100 ); opti.subject_to( ca_vars["beta"] <= 100 )

        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        
        sp_exp, total_I_size, total_eq_size, sum_ij_La, sum_ij_AC = self.circuit_symbolic_matrices(list_of_leaf_functions, dim_G, dim_F)
        shift_f_idx = 0; ineq_size = 0; eq_size = 0
        if self.list_of_constraints != list():
            eq_size = total_eq_size
            ineq_size = len(self.list_of_constraints) - total_eq_size
            if ineq_size >= 1:
                shift_f_idx = 1
                lamb = opti.variable(ineq_size, 1)
                ca_vars["lamb0"] = lamb
                opti.subject_to( lamb >= np.zeros((ineq_size, 1)) )
            if eq_size >= 1:
                nu = opti.variable(eq_size, 1)
                ca_vars["nu0"] = nu

        for f_idx, function in enumerate(list_of_leaf_functions):
            size_I_function = len(function.list_of_points)
            lamb = opti.variable(size_I_function, size_I_function)
            ca_vars["lamb%d"%(f_idx + shift_f_idx)] = lamb
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

        if init_vals is not None:
            for name, var_value in init_vals.items():
                opti.set_initial(ca_vars[name], var_value)

        opti.minimize( - sympy_expression_to_casadi(self.obj, ca_vars, opti))
        opti.solver('ipopt', opts)
        self.vars = ca_vars

        print(f"Ipopt total # of variables = {ca_dict_total_variable_size(ca_vars)}")
        print(f"Actual # of variables = {total_I_size + total_eq_size + dim_F + dim_G*(dim_G+1)//2 + len(self.discretization_params)}")

        try:
            sol = opti.solve() # QCQP for solving CircuitOpt
        except:
            if debug:
                self.ca_vars = ca_vars
            print("Could not find a solution using Ipopt")
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
    
  
    def solve_ipopt_qcqp(self, verbose=True, init_values=None, x0=None, bounds=None, extra_dim=150, debug=False, **kwargs):
        """
        Use ipopt to find proofs and formulate problem explicitly as QCQP: 
            introduce the dummy variables for the discretization parameters
            to make each expression bilinear in discretization parameters and entries of G and F,
            with variables vec(v), vec(lamb), vec(nu), and vec(P)
        """
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
        opti = ca.Opti()
        var_x2 = opti.variable(x_size + extra_dim, 1)
        var_x = var_x2[:x_size]
        opti.subject_to(var_x[0] == 1)
        ca_vars = {"x": var_x}

        print(f"Ipopt total # of variables = {np.prod(var_x2.size())}")
        print(f"Actual # of variables = {x_size}")

        ca_add_bounds(opti, bounds, ca_vars, name2idx)
        assert v_k_list[-1] == "FG_d", print(v_k_list)

        vec_indices = { "v"   : [0, v_size - 1],\
                        "lamb": [v_size, v_size + total_I_size - 1 ], \
                        "nu": [v_size + total_I_size, v_size + total_I_size + total_eq_size - 1 ], \
                        "P"  : [v_size + total_I_size + total_eq_size, x_size - 1]} 
        # vectorized variables
        vec_v = get_vec_var(var_x, "v", vec_indices)
        vec_P = get_vec_var(var_x, "P", vec_indices)

        init_ca_ipopt(init_values, opti, vec_indices, var_x)
        vec_indices["lamb_nu"] =  [v_size, v_size + total_I_size + total_eq_size - 1 ]
        vec_lambs_nus = get_vec_var(var_x, "lamb_nu", vec_indices)

        opti.subject_to(var_x[name2idx["b"]] >= 1e-7)
        opti.subject_to(var_x[name2idx["h"]] >= 1e-7)
        opti.subject_to(var_x[name2idx["d"]] >= 1e-7)
        opti.subject_to(var_x[name2idx["gamma"]] >= 1e-7)

        # matrix coefficient for variable F is 0
        obj_F = np.concatenate([v_coeffs["F"][-1], np.zeros((dim_F, v_size - v_coeffs["F"][-1].shape[1]))], axis=1)
        sum_ij_F = stack_vectors(v_coeffs["F"][:-1], v_size)
        assert sum_ij_F.shape == (vec_lambs_nus.shape[0], dim_F, v_size), print(sum_ij_F.shape, (vec_lambs_nus.shape[0], dim_F, v_size))
        res = np.zeros((dim_F, 1))
        for k in range(dim_F):
            assert obj_F[k : k+1, :].shape[1] == vec_v.shape[0] and sum_ij_F[:, k, :].shape == (vec_lambs_nus.shape[0], vec_v.shape[0]) 
            opti.subject_to(  obj_F[k : k+1, :] @ vec_v - vec_lambs_nus.T @ sum_ij_F[:, k, :] @ vec_v == 0)

        # matrix coefficient for variable G is 0
        obj_G = np.concatenate([v_coeffs["G"][-1], np.zeros((dim_G*dim_G, v_size - v_coeffs["G"][-1].shape[1]))], axis=1)
        sum_ij_G = stack_vectors(v_coeffs["G"][:-1], v_size)
        assert sum_ij_G.shape == (vec_lambs_nus.shape[0], dim_G*dim_G, v_size), print(sum_ij_G.shape, (vec_lambs_nus.shape[0], dim_G*dim_G, v_size))
        for k1 in range(dim_G):
            for k2 in range(dim_G):
                k_idx = k1 * dim_G + k2
                P_k1P_k2 = get_PPt_element(vec_P, k1, k2)
                opti.subject_to(  obj_G[k_idx : k_idx+1, :] @ vec_v + P_k1P_k2 - vec_lambs_nus.T @ sum_ij_G[:, k_idx, :] @ vec_v == 0)

        # v variables quadratic constraints
        # v^T Qi v + ai^T v = 0 
        for name in v_names[1:]:
            vars = name.split("_")
            if len(vars) == 1: continue
            pref_v, v = "_".join(vars[:-1]), vars[-1]
            ai = - one_hot(v_size, name2idx[name])
            # include both permutations for the product 
            Qi = symm_prod_one_hot(v_size, name2idx[v], name2idx[pref_v])
            opti.subject_to( vec_v.T @ Qi @ vec_v + ai.T @ vec_v == 0)

        # lambda >= 0 constraints
        vec_lambs = get_vec_var(var_x, "lamb", vec_indices)
        opti.subject_to( vec_lambs >= np.zeros(vec_lambs.shape))
        # diag(P) >= 0 constraints
        vec_diag_P_idx = (np.cumsum(np.arange(dim_G + 1))[1:] - 1).reshape(-1, 1)
        Q = np.zeros((dim_G, dim_G * (dim_G + 1) // 2))
        np.put_along_axis(Q, vec_diag_P_idx, 1, axis=1)
        # P_diag_constraints = [Q]
        opti.subject_to( Q @ vec_P >= np.zeros((dim_G, 1)) )

        opts = {'ipopt.max_iter':50000,} #"ipopt.tol": 1e-4, "ipopt.constr_viol_tol": 1e-4, "ipopt.dual_inf_tol": 1e-2} 
        if not verbose:
            opts.update({'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
        ca_vars = {name : var_x[idx] for name, idx in name2idx.items()}
        opti.minimize( - sympy_expression_to_casadi(self.obj, ca_vars, opti))
        if x0 is not None:
            opti.set_initial(var_x, x0)
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
    

    solve_ipopt_qcqp_matrix = solve_ipopt_qcqp_matrix

    solve_gp = solve_gp

    solve_gp_qcqp_matrix = solve_gp_qcqp_matrix

    solve_qcqp_sni = solve_qcqp_sni

    solve_dccp = solve_dccp

    bounds_sdp_relax_all = bounds_sdp_relax_all

    solve_sdp_relax = solve_sdp_relax


    def solve_bisection_b(self, freq=20, params=None, max_iters=30, cvx_solver=cp.CLARABEL, **kwargs):
        # given discretization parameters in params
        # do bisection on b by checking if the resulting SDP subject to params and b
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

        b = params["b"] if "b" in params else 10.
        for t in range(max_iters):
            params["b"] = b
            try:
                prob = self.solve_fix_discr_sdp(params=params, cvx_solver=cvx_solver, inputs=inputs, verbose=False)[1]
                if t % freq == 0: print(f"{t=}, {b=}, {prob.status=}")
                if prob.status == 'optimal':
                    return params, b
            except: pass
            b /= 2
        return params, b


    def solve_fix_discr_sdp(self, params=None, cvx_solver=cp.CLARABEL, verbose=True, debug=False, inputs=None, **kwargs):
        # for fixed alpha, beta, h, b, d, gamma 
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
        assert v_k_list[-1] == "FG_d", print(v_k_list)

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
   

    def solve(self, solver="ipopt", **kwargs):
        if solver == "gp":
            return self.solve_gp(**kwargs)
        elif solver == "gp_qcqp_matrix":
            return self.solve_gp_qcqp_matrix(**kwargs)
        elif solver == "sdp_relax":
            return self.solve_sdp_relax(**kwargs)
        elif solver == "bounds_sdp_relax_all":
            return self.bounds_sdp_relax_all(**kwargs)
        elif solver == "ipopt":
            return self.solve_ipopt(**kwargs)
        elif solver == "ipopt_qcqp":
            return self.solve_ipopt_qcqp(**kwargs)
        elif solver == "ipopt_qcqp_matrix":
            return self.solve_ipopt_qcqp_matrix(**kwargs)
        elif solver == "fix_discr_sdp":
            return self.solve_fix_discr_sdp(**kwargs)
        elif solver == "dccp":
            return self.solve_dccp(**kwargs)
        elif solver == "qcqp_sni":
            return self.solve_qcqp_sni(**kwargs)
        elif solver == "bisection_b":
            return self.solve_bisection_b(**kwargs)
        