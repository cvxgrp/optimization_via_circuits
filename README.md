

# Optimization Algorithm Design via Electric Circuits
 
This repository accompanies the [manuscript](XXX).

We present a ***novel methodology for optimization algorithm design***, 
using ideas from electric RLC circuits and the performance estimation problem. 
Our methodology provides a quick and systematic recipe for designing new, provably convergent optimization algorithms, including distributed optimization algorithms. 

In this repository, we provide `ciropt` package implementing this methodology.
With `ciropt`, you can easily explore a wide range of algorithms that have a convergence guarantee.

The structure of this repo borrows a lot from [PEPit](https://pepit.readthedocs.io/en/latest/quickstart.html) package.
This is intended to easily verify that the discretization is dissipative
using other package. 

## Installation
To install `ciropt` 1) activate virtual environment, 2) clone the repo, 3) from inside the directory run 
```python3
python setup.py install
```
Requirements
* python >= 3.10
* numpy >= 1.26
* scipy >= 1.11.3
* casadi >= 3.6.4
* sympy >= 1.12
* pepit >= 0.2.1


## Designing convergent optimization algorithms 

### High level process
1. Start with the optimization problem. 
```math
\begin{array}{ll}
\mbox{minimize}& f(x)\\
    \mbox{subject to}& x\in \mathcal{R}(E^\intercal )
\end{array}
```

2. Create the static interconnect (SI) respresenting the optimality conditions.

3. Replace wires of SI with RLC components to derive the ***admissible*** dynamic interconnect (DI) that relaxes in equillibrium to SI.

4. Using `ciropt` find discretization parameters for DI that produce convergent algorithm.

5. Use the resulting algorithm to solve the original problem. 

Note that in step 3, there are infinitely many admissible DIs that can be designed. 
Each of these, when discretized, results in a different optimization algorithm. 
Feel free to experiment with various DIs to discover new algorithms suitable for your problem at hand.


### Hello world

1. As a hello world example we consider the simplest problem given below.
```math
\begin{array}{ll}
\mbox{minimize}& f(x)\\
    \mbox{subject to}& x
\end{array}
```
2. The optimality condition for this problem are to find $x$ such that
$0 \in \nabla f(x)$. The corresponding SI for this condition is below.
[View circuit](examples/figures/hello_world_si.pdf)
<iframe src="assets/tikz_drawing.pdf" width="100%" height="600px">
    This browser does not support PDFs. Please download the PDF to view it: <a href="examples/figures/hello_world_si.pdf">Download PDF</a>.
</iframe>

3. We consider the following admissible DI, see [circuit](examples/figures/hello_world_di.pdf).

4. Now let's discretize this DI using `ciropt`.


**Step 1.** Define a problem.
```python3
import ciropt as co

problem = co.CircuitOpt()
```


**Step 2.** Define function classes for each $f_i$. In this example there is only single function.
```python3
f = co.define_function(problem, mu, L_smooth, package )
```

**Step 3.** Define optimal points.
```python3
x_star, y_star, f_star = func.stationary_point(return_gradient_and_function_value=True)
```

**Step 4.** Define discretization parameters, for simplicity take $\alpha=0$ and $\beta=1$.
```python3
h, b, d = problem.h, problem.b, problem.d
```

**Step 5.** Define values for RLC
and one step transition for discretized V-I relations
of given DI.
```python3
R, C = 1, 10

z_1 = problem.set_initial_point()
e2_1 = problem.set_initial_point()
x_1 = co.proximal_step(z_1, f, R/2)[0]
y_1 = (2 / R) * (z_1 - x_1)


e2_2 = e2_1  -  h / (2 * R * C) *  (R * y_1 + 3 * e2_1)  
z_2 = z_1  -  h / (4 * R * C) *  (5 * R * y_1 + 3 * e2_1)
x_2 = co.proximal_step(z_2, f, R/2)[0]
y_2 = (2 / R) * (z_2 - x_2)

v_C1_1 = e2_1 / 2 - z_1
v_C1_2 = e2_2 / 2 - z_2
v_C2_1 = e2_1; v_C2_2 = e2_2 
e1_1 = (e2_1 - R * y_1) / 2
```

**Step 6.** Define dissipative term and set the objective to maximize descent coefficients.
Solve the final problem.

```python
E_1 = (C/2) * (v_C1_1 + x_star)**2 + (C/2) * (v_C2_1)**2
E_2 = (C/2) * (v_C1_2 + x_star)**2 + (C/2) * (v_C2_2)**2
Delta_1 = b * (x_1 - x_star) * (y_1 - y_star) \
        + d * (1/R) * ((e1_1)**2 + (e1_1 - e2_1)**2 + (e2_1)**2)


problem.set_performance_metric(E_2 - (E_1 - Delta_1))
problem.obj = problem.b + problem.d * 1.1

params = problem.solve(solver="ipopt")[:1]
``` 



The resulting provably convergent algorithm is 
```math
\begin{align*}
x^k &= \prox_{(1/2) f}(z^k ),\quad  y^k=2(z^k-x^k)\\
w^{k+1} &= w^k - 0.315(y^k + 3w^k) \\
z^{k+1} &= z^k - 0.1575(5 y^k + 3w^k)
\end{align*}
```

5. Run this algorithm to our problem. 



We provide a guideline on how to use our automatic discretization package in the 
[hello world notebook](https://github.com/cvxgrp/mlr_fitting/tree/main/examples/hello_world.ipynb). 


## Example notebooks
We have [example notebooks](https://github.com/cvxgrp/optimization_via_circuits/tree/main/examples) 
that show how to use `ciropt` to discretize various circuits.

Centralized methods:
* gradient descent, see [notebook](x) 
* proximal gradient, see [notebook](x)  
* proximal point method, see [notebook](x)           

Decentralized methods: (for underlying graph 1 -- 2 -- 3)
* DGD, see [notebook](https://github.com/cvxgrp/optimization_via_circuits/blob/main/examples/dgd.ipynb) 
* diffusion, see [notebook](https://github.com/cvxgrp/optimization_via_circuits/blob/main/examples/diffusion.ipynb) 
* DADMM, see [notebook](https://github.com/cvxgrp/optimization_via_circuits/blob/main/examples/decentralized_admm_line3.ipynb) 
* PG-EXTRA, see [notebook](https://github.com/cvxgrp/optimization_via_circuits/blob/main/examples/pg_extra_line3.ipynb) 

Please consult our [manuscript](XXX) for the details of mentioned problems. 


## Tips
Since we are using Ipopt which does not solve the nonconvex optimization problem to global optimality, here are some tips we found useful.
1. Vary resistances, capacitances, inductances if the Ipopt does not find a proof for given values. 
2. Try changing smoothness of your function class from $M=\infty$ to bounded value.
3. Vary the mixing weight `w` in the objective `wb + d`.