

# Optimization Algorithm Design via Electric Circuits
 
This repository accompanies the [manuscript](XXX).

We present a ***novel methodology for optimization algorithm design***, 
using ideas from electric RLC circuits and the performance estimation problem. 
Our methodology provides a quick and systematic recipe for designing new, provably convergent optimization algorithms, including distributed optimization algorithms. 

In this repository, we provide `ciropt` package implementing this methodology.
With `ciropt`, you can easily explore a wide range of algorithms that have a convergence guarantee.
 

## Installation
To install `ciropt` 1) activate virtual environment, 2) clone the repo, 3) from inside the directory run 
```python3
pip install -e .
```
Requirements
* python >= 3.10
* pepit >= 0.2.1
* numpy >= 1.26
* scipy >= 1.11.3
* casadi >= 3.6.4
* sympy >= 1.12


# Process of optimization algorithm design

## High level process
1. Start with the optimization problem. 

$$
\begin{array}{ll}
\text{minimize}& f(x)\\
\text{subject to}& x\in \mathcal{R}(E^\intercal ).
\end{array}
$$

2. Create the static interconnect (SI) respresenting the optimality conditions.

3. Replace wires of SI with RLC components to derive the ***admissible*** dynamic interconnect (DI) that relaxes in equillibrium to SI.

4. Using `ciropt` find discretization parameters for DI that produce convergent algorithm.

5. Use the resulting algorithm to solve the original problem. 

Note that in step 3, there are infinitely many admissible DIs that can be designed. 
Each of these, when discretized, results in a different optimization algorithm. 
Feel free to experiment with various DIs to discover new algorithms suitable for your problem at hand.


## Hello world
See the `examples/hello_world.ipynb` notebook or explanation below.

1. As a hello world example we consider the problem given below, where $f$
is convex function.

$$
\begin{array}{ll}
\text{minimize}& f(x).
\end{array}
$$

2. The optimality condition for this problem is to find $x$ such that
$0 \in \partial f(x)$. The corresponding SI for this condition follows, see
[circuit](./examples/figures/hello_world_si.pdf).

3. We consider the following admissible DI, see 
[circuit](./examples/figures/hello_world_di.pdf).

4. Now let's discretize this DI using `ciropt`.


**Step 1.** Define a problem.
```python3
import ciropt as co

problem = co.CircuitOpt()
```

**Step 2.** Define function class, in this example $f$ is convex and nondifferentiable, i.e., $\mu=0$ and $M=\infty$.
```python3
f = co.def_function(problem, mu, M)
```

**Step 3.** Define optimal points.
```python3
x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
```

**Step 4.** Define values for the RLC components and
discretization parameters, here for simplicity 
we take $\alpha=0$ and $\beta=1$.
```python3
R, C = 1, 10
h, eta = problem.h, problem.eta
```

**Step 5.** Define the one step transition of the discretized V-I relations.
```python3
z_1 = problem.set_initial_point()
e2_1 = problem.set_initial_point()
x_1 = co.proximal_step(z_1, f, R/2)[0]
y_1 = (2 / R) * (z_1 - x_1)
e1_1 = (e2_1 - R * y_1) / 2
v_C1_1 = e2_1 / 2 - z_1
v_C2_1 = e2_1

e2_2 = e2_1  -  h / (2 * R * C) *  (R * y_1 + 3 * e2_1)  
z_2 = z_1  -  h / (4 * R * C) *  (5 * R * y_1 + 3 * e2_1)
x_2 = co.proximal_step(z_2, f, R/2)[0]
y_2 = (2 / R) * (z_2 - x_2)
v_C1_2 = e2_2 / 2 - z_2
v_C2_2 = e2_2 
```

**Step 6.** Define dissipative term and set the objective to maximize descent coefficients.
Solve the final problem.

```python
E_1 = (C/2) * (v_C1_1 + x_star)**2 + (C/2) * (v_C2_1)**2
E_2 = (C/2) * (v_C1_2 + x_star)**2 + (C/2) * (v_C2_2)**2
Delta_1 = eta * (x_1 - x_star) * (y_1 - y_star) 

problem.set_performance_metric(E_2 - (E_1 - Delta_1))
params = problem.solve()[:1]
``` 

The resulting provably convergent algorithm is 

$$
\begin{align*}
x^k &= \mathbf{prox}_{(1/2) f}(z^k ),\quad  y^k=2(z^k-x^k)\\
w^{k+1} &= w^k - 0.33(y^k + 3w^k) \\
z^{k+1} &= z^k - 0.16(5 y^k + 3w^k).
\end{align*}
$$

5. Solve your problem using new algorithm. 
We consider the primal problem

$$
\begin{array}{ll}
\text{minimize} & f(x) = \sum_i
\begin{cases}
(x_i - c_i)^2 & |x_i-c_i| \leq 1 \\
2(x_i-c_i)-1 & |x_i-c_i| > 1
\end{cases} \\
\text{subject to} & Ax = b,
\end{array}
$$

and solve the dual problem

$$
\begin{array}{ll}
\text{maximize} & g(y) = -f^*(-A^T y ) - b^Ty.
\end{array}
$$

We apply our discretization to solve the dual problem $g(y)$.
Since $f$ is CCP and $2$-smooth (as a huber loss), then $f^*$ is $1/2$-strongly convex. We rescale $A$ to have $g$ $1/2$-strongly convex.


The relative error versus iteration is plotted [here](./examples/figures/simple_hello_wrld.pdf).

## Example notebooks
See notebooks in `examples/` folder
that show how to use `ciropt` to discretize various circuits.

Centralized methods:
* gradient descent, proximal gradient, proximal point method, Nesterov acceleration
* primal decomposition, dual decomposition, Douglas-Rachford splitting, proximal decomposition, Davis-Yin splitting         

Decentralized methods: 
* DGD
* diffusion 
* DADMM
* PG-EXTRA

Please consult our [manuscript](XXX) for the details of mentioned problems. 


## Tips
Since we are using Ipopt which does not solve the nonconvex optimization problem to global optimality, here are some tips we found useful.
1. Vary resistances, capacitances, inductances if the Ipopt does not find a proof for given values. 
2. Try changing smoothness of your function class from $M=\infty$ to bounded value.
3. Consider the full dissipating term

$$\mathcal{E}_2- \mathcal{E}_1 +  \eta\langle x^1-x^\star, y^1-y^\star\rangle + \rho\|i_\mathcal{R}^1\|_{D_\mathcal{R}}^2.$$ 

4. Vary the mixing weight $w$ in the objective $w\eta + \rho$, by setting `problem.obj = eta + w*rho`, which increases the descent in energy.
5. Change solvers in the `problem.solve` from `"ipopt", "ipopt_qcqp", "ipopt_qcqp_matrix"`.