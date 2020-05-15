import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4(f, y0, t0, tf, h):
    """
    Classical 4th-order Runge-Kutta solver for ODEs of the form dy/dt = f(t, y).

    Parameters:
        f   : callable — the ODE function f(t, y)
        y0  : float    — initial condition y(t0)
        t0  : float    — start time
        tf  : float    — end time
        h   : float    — step size

    Returns:
        t_values : numpy array of time points
        y_values : numpy array of solution values
    """
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0

    while t < tf:
        # Adjust last step to land exactly on tf
        if t + h > tf:
            h = tf - t

        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h

        t_values.append(round(t, 10))
        y_values.append(y)

    return np.array(t_values), np.array(y_values)


def adaptive_runge_kutta(f, y0, t0, tf, h_init=0.1, tol=1e-6):
    """
    Adaptive step-size Runge-Kutta solver using step doubling for error control.
    Reduces step size when error exceeds tolerance, increases when well within tolerance.

    Parameters:
        f      : callable — the ODE function f(t, y)
        y0     : float    — initial condition
        t0     : float    — start time
        tf     : float    — end time
        h_init : float    — initial step size
        tol    : float    — error tolerance

    Returns:
        t_values : numpy array of time points
        y_values : numpy array of solution values
        steps    : int — number of steps taken
    """
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0
    h = h_init
    steps = 0

    while t < tf:
        if t + h > tf:
            h = tf - t

        # Full step
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y_full = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Two half steps
        h2 = h / 2
        k1 = h2 * f(t, y)
        k2 = h2 * f(t + h2 / 2, y + k1 / 2)
        k3 = h2 * f(t + h2 / 2, y + k2 / 2)
        k4 = h2 * f(t + h2, y + k3)
        y_half = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        k1 = h2 * f(t + h2, y_half)
        k2 = h2 * f(t + h2 + h2 / 2, y_half + k1 / 2)
        k3 = h2 * f(t + h2 + h2 / 2, y_half + k2 / 2)
        k4 = h2 * f(t + h2 + h2, y_half + k3)
        y_two_half = y_half + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Error estimate
        error = abs(y_two_half - y_full) / 15.0

        if error < tol:
            # Accept step
            t += h
            y = y_two_half
            t_values.append(round(t, 10))
            y_values.append(y)
            steps += 1

            # Try increasing step size
            if error > 0:
                h = min(h * 0.9 * (tol / error) ** 0.2, h * 2.0)
        else:
            # Reject step, reduce step size
            h = max(h * 0.9 * (tol / error) ** 0.25, h * 0.1)

    return np.array(t_values), np.array(y_values), steps


# ── Test Cases ───────────────────────────────────────────────────────────────

def test_exponential_decay():
    """
    Test: dy/dt = -2y,  y(0) = 1
    Analytical solution: y(t) = e^(-2t)
    """
    print("=" * 55)
    print("TEST 1: Exponential Decay   dy/dt = -2y,  y(0)=1")
    print("=" * 55)

    f = lambda t, y: -2 * y
    analytical = lambda t: np.exp(-2 * t)

    t, y_num = runge_kutta_4(f, y0=1.0, t0=0, tf=5, h=0.1)
    y_exact = analytical(t)

    max_err = np.max(np.abs(y_num - y_exact))
    rel_err = np.max(np.abs((y_num - y_exact) / y_exact)) * 100

    print(f"  Step size h       : 0.1")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.4f}%")
    print(f"  Accuracy          : {100 - rel_err:.4f}%")
    print()
    return t, y_num, y_exact


def test_nonlinear_logistic():
    """
    Test: dy/dt = y(1 - y),  y(0) = 0.1
    Analytical solution: y(t) = 1 / (1 + 9*e^(-t))
    """
    print("=" * 55)
    print("TEST 2: Nonlinear Logistic  dy/dt = y(1-y),  y(0)=0.1")
    print("=" * 55)

    f = lambda t, y: y * (1 - y)
    analytical = lambda t: 1.0 / (1 + 9 * np.exp(-t))

    t, y_num = runge_kutta_4(f, y0=0.1, t0=0, tf=10, h=0.05)
    y_exact = analytical(t)

    max_err = np.max(np.abs(y_num - y_exact))
    rel_err = np.max(np.abs((y_num - y_exact) / y_exact)) * 100

    print(f"  Step size h       : 0.05")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.6f}%")
    print(f"  Accuracy          : {100 - rel_err:.4f}%")
    print()
    return t, y_num, y_exact


def test_nonlinear_stiff():
    """
    Test: dy/dt = -50(y - cos(t)) - sin(t),  y(0) = 1
    Analytical solution: y(t) = cos(t)
    Stiff ODE — tests solver stability.
    """
    print("=" * 55)
    print("TEST 3: Stiff Nonlinear ODE  y(0)=1")
    print("=" * 55)

    f = lambda t, y: -50 * (y - np.cos(t)) - np.sin(t)
    analytical = lambda t: np.cos(t)

    t, y_num = runge_kutta_4(f, y0=1.0, t0=0, tf=2, h=0.01)
    y_exact = analytical(t)

    max_err = np.max(np.abs(y_num - y_exact))
    rel_err_vals = np.abs((y_num - y_exact) / (y_exact + 1e-12)) * 100
    rel_err = np.max(rel_err_vals)

    print(f"  Step size h       : 0.01")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.6f}%")
    print(f"  Accuracy          : {100 - rel_err:.4f}%")
    print()
    return t, y_num, y_exact


def test_adaptive_solver():
    """
    Adaptive RK4 on dy/dt = y(1-y) demonstrating step-size optimisation.
    """
    print("=" * 55)
    print("TEST 4: Adaptive Step-Size RK4")
    print("=" * 55)

    f = lambda t, y: y * (1 - y)
    analytical = lambda t: 1.0 / (1 + 9 * np.exp(-t))

    t, y_num, steps = adaptive_runge_kutta(f, y0=0.1, t0=0, tf=10, h_init=0.5, tol=1e-8)
    y_exact = analytical(t)

    max_err = np.max(np.abs(y_num - y_exact))
    rel_err = np.max(np.abs((y_num - y_exact) / y_exact)) * 100

    print(f"  Tolerance         : 1e-8")
    print(f"  Adaptive steps    : {steps}")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.6f}%")
    print(f"  Accuracy          : {100 - rel_err:.4f}%")
    print()
    return t, y_num, y_exact


def plot_results(results):
    """Plot all test results in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Numerical Solutions of Nonlinear Differential Equations\nRunge-Kutta 4th Order Method",
        fontsize=13, fontweight='bold', y=0.98
    )

    titles = [
        "Test 1: Exponential Decay\ndy/dt = -2y",
        "Test 2: Logistic Growth\ndy/dt = y(1-y)",
        "Test 3: Stiff ODE\ndy/dt = -50(y - cos t) - sin t",
        "Test 4: Adaptive Step-Size RK4\ndy/dt = y(1-y)",
    ]

    for ax, (t, y_num, y_exact), title in zip(axes.flat, results, titles):
        ax.plot(t, y_exact, 'k-',  linewidth=2,   label='Analytical', zorder=3)
        ax.plot(t, y_num,  'b--',  linewidth=1.5, label='RK4 Numerical', zorder=2)
        ax.fill_between(t, y_num, y_exact, alpha=0.15, color='steelblue', label='Error')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('t', fontsize=8)
        ax.set_ylabel('y(t)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig('rk4_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved → rk4_results.png")
    plt.show()


def step_size_convergence():
    """
    Demonstrate convergence: error reduces as O(h^4) with smaller step sizes.
    """
    print("=" * 55)
    print("CONVERGENCE ANALYSIS  (dy/dt = -2y)")
    print("=" * 55)
    print(f"  {'Step Size h':<15} {'Max Error':<18} {'Order'}")
    print(f"  {'-'*13:<15} {'-'*16:<18} {'-'*8}")

    f = lambda t, y: -2 * y
    analytical = lambda t: np.exp(-2 * t)
    steps = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
    errors = []

    for h in steps:
        t, y_num = runge_kutta_4(f, y0=1.0, t0=0, tf=5, h=h)
        y_exact = analytical(t)
        err = np.max(np.abs(y_num - y_exact))
        errors.append(err)

    for i, (h, err) in enumerate(zip(steps, errors)):
        if i == 0:
            print(f"  {h:<15.4f} {err:<18.6e} {'—'}")
        else:
            order = np.log(errors[i-1] / err) / np.log(steps[i-1] / h)
            print(f"  {h:<15.4f} {err:<18.6e} {order:.2f}")
    print()


if __name__ == "__main__":
    print("\nNumerical Solutions of Nonlinear Differential Equations")
    print("Runge-Kutta 4th Order Method | Peyali Chakraborty\n")

    r1 = test_exponential_decay()
    r2 = test_nonlinear_logistic()
    r3 = test_nonlinear_stiff()
    r4 = test_adaptive_solver()
    step_size_convergence()
    plot_results([r1, r2, r3, r4])
