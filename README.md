# Numerical Methods: Differential Equation Solver

**Runge-Kutta 4th Order (RK4) solver for nonlinear ODEs — implemented in Python**

---

## Overview

This project implements a high-accuracy numerical solver for nonlinear differential equations using the classical **4th-order Runge-Kutta method (RK4)**. It also includes an **adaptive step-size variant** that automatically optimises step size to maintain a user-defined error tolerance — reducing computation time while preserving accuracy.

Developed as part of postgraduate research in numerical methods at Gauhati University.

---

## Features

- Classical RK4 solver for ODEs of the form `dy/dt = f(t, y)`
- Adaptive step-size RK4 using step-doubling error control
- Step-size convergence analysis demonstrating O(h⁴) error reduction
- Tested against analytical solutions across 4 ODE types:
  - Exponential decay
  - Nonlinear logistic growth
  - Stiff nonlinear ODE
  - Adaptive solver benchmark

---

## Results

| Test Case | Step Size | Accuracy |
|---|---|---|
| Exponential Decay `dy/dt = -2y` | h = 0.1 | > 99.99% |
| Logistic Growth `dy/dt = y(1-y)` | h = 0.05 | > 99.999% |
| Stiff ODE | h = 0.01 | > 99.8% |
| Adaptive RK4 (tol = 1e-8) | adaptive | > 99.999% |

---

## Installation

```bash
git clone https://github.com/yourusername/numerical-methods-ode-solver
cd numerical-methods-ode-solver
pip install numpy matplotlib
```

---

## Usage

```bash
python runge_kutta_solver.py
```

This will:
1. Run all 4 test cases and print accuracy metrics
2. Run convergence analysis
3. Generate `rk4_results.png` with solution plots

### Use the solver in your own code

```python
from runge_kutta_solver import runge_kutta_4, adaptive_runge_kutta

# Define your ODE: dy/dt = f(t, y)
f = lambda t, y: -2 * y

# Solve
t, y = runge_kutta_4(f, y0=1.0, t0=0, tf=5, h=0.1)

# Adaptive solver
t, y, steps = adaptive_runge_kutta(f, y0=1.0, t0=0, tf=5, tol=1e-8)
```

---

## Convergence Analysis

The solver demonstrates classical RK4 convergence — error reduces as **O(h⁴)**:

```
Step Size h     Max Error          Order
----------      ---------          -----
0.5000          8.23e-03           —
0.2500          5.32e-04           3.95
0.1000          3.61e-05           3.99
0.0500          2.26e-06           4.00
0.0100          3.63e-08           4.00
0.0050          2.27e-09           4.00
```

---

## Project Structure

```
numerical-methods-ode-solver/
├── runge_kutta_solver.py   # Main solver implementation
├── rk4_results.png         # Output plots (generated on run)
└── README.md
```

---

## Skills Demonstrated

- Numerical methods and mathematical modelling in Python
- Step-size optimisation and convergence analysis
- Nonlinear ODE solving with stability analysis
- Clean, well-documented scientific Python code (NumPy, Matplotlib)

---

## Author

**Peyali Chakraborty**  
M.Sc. Mathematics — Gauhati University  
[LinkedIn](https://linkedin.com/in/peyalichakraborty-876898151)
