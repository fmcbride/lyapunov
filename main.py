import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.stats as stats


# METHODS


def gsr(m):
    # performs a Gram-Schmidt Reorthonormalization on matrix m
    # m must be non-singular
    # Orthogonalization:
    def proj(u, v):
        return (np.dot(u, v) / np.dot(u, u)) * u

    dim = len(m[0])
    for i in range(1, dim):
        for j in range(0, i):
            w = m[i]
            m[i] = m[i] - proj(m[j], w)
    # Normalization:
    for k in range(dim):
        m[k] = m[k] / np.linalg.norm(m[k])
    return m


def volume(vectors):
    # calculates the volume of a n-parallelotope in R^m space, m>=n spanned by vectors
    # uses Gram determinant
    n = len(vectors)
    matrix = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i):
            matrix[i, j] = np.dot(vectors[i], vectors[j])
    matrix = matrix + matrix.transpose()
    for k in range(n):
        matrix[k, k] = np.dot(vectors[k], vectors[k])
    return np.sqrt(np.linalg.det(matrix))


def main(system, x=None, breakin_cont=100, breakin_disc=1000, lyap_breakin=2000, cycles=10000, eps=1E-4, per_orb=10,
         digits=3, display=False):
    # iterates forward in time to allow transients to decay
    if not x:
        x = system.ic
    print('Finding attractor', end='...')
    if system.discrete:
        # computes breakin_disc iterations
        per_orb = 1
        if display:
            breakin_disc *= 1000
        points = np.zeros((system.dim, breakin_disc))
        points[..., 0] = x
        for i in range(1, breakin_disc):
            points[..., i] = system.func(points[..., i - 1])
        diameter = 0
        for k in range(system.dim):
            diameter += (np.max(points[k]) - np.min(points[k])) ** 2
            diameter = np.sqrt(diameter)
        x = points[..., -1]
        t = 0
    else:
        # computes trajectory over breakin_cont time interval
        # uses 4th order Runge-Kutta method with 5th order error correction
        # computes pseudoperiod
        system.apex.terminal = False
        system.apex.direction = -1
        if display:
            outputs = solve_ivp(system.func, (0, breakin_cont), x, max_step=0.001, events=system.apex)
        else:
            outputs = solve_ivp(system.func, (0, breakin_cont), x, events=system.apex)
        x = outputs.y[..., -1]
        t = outputs.t[-1]
        if display:
            points = outputs.y
        # defines pseudoperiod as highest mode of KDE of pseudoorbit durations
        pseudoorbit_times = np.ediff1d(outputs.t_events)
        kernel = stats.gaussian_kde(pseudoorbit_times)
        height = kernel.pdf(pseudoorbit_times)
        pseudoperiod = pseudoorbit_times[np.argmax(height)]
        print('Pseudoperiod: {}'.format(pseudoperiod))
        # finds approximate diameter of attractor set and scales epsilon by it
        diameter = 0
        for k in range(system.dim):
            diameter += (np.max(outputs.y[k]) - np.min(outputs.y[k])) ** 2
        diameter = np.sqrt(diameter)
    eps *= diameter
    if display:
        point_count = len(points[0])
        start = int(point_count / 5)
        points = points[..., start:]
        if system.discrete:
            if system.dim == 2:
                plt.plot(points[0], points[1], '.k', markersize=0.1)
                plt.xlabel(system.ax_names[0])
                plt.ylabel(system.ax_names[1])
            else:
                plt.plot(points[0], points[-1], '.k', markersize=0.1)
                plt.xlabel(system.ax_names[0])
                plt.ylabel(system.ax_names[-1])
        else:
            if system.dim == 2:
                plt.plot(points[0], points[1], 'k', linewidth=0.5)
                plt.xlabel(system.ax_names[0])
                plt.ylabel(system.ax_names[1])
            else:
                plt.plot(points[0], points[-1], 'k', linewidth=0.5)
                plt.xlabel(system.ax_names[0])
                plt.ylabel(system.ax_names[-1])
        plt.title('{} System'.format(system.title))
        plt.show()

    def lyap_step(x, t, err_mat):
        # err_mat should be orthonormal
        err_mat *= eps
        if system.discrete:
            x_new = system.func(x)
            for v in range(system.dim):
                err_mat[v] = system.func(x + err_mat[v]) - x_new
            t_new = t + 1
        else:
            ode_outs = solve_ivp(system.func, (t, pseudoperiod / per_orb + t), x, atol=eps / 100)
            x_new = ode_outs.y[..., -1]
            t_new = ode_outs.t[-1]
            for v in range(system.dim):
                ode_outs = solve_ivp(system.func, (t, pseudoperiod / per_orb + t), x + err_mat[v], atol=eps / 100)
                err_mat[v] = ode_outs.y[..., -1] - x_new
        # compute growth factors in each direction
        growth_factor = np.zeros(system.dim)
        for d in range(system.dim):  # this may or may not be a problem for non-autonomous systems
            spanning_vectors = err_mat[:d + 1]
            growth_factor[d] = volume(spanning_vectors / eps)
        # reorthonomalize error matrix
        err_mat = gsr(err_mat / eps)
        return x_new, t_new, err_mat, growth_factor

    def extract_exponents(sums):
        exps = sums
        for i in range(system.dim - 1, 0, -1):
            exps[i] -= sums[i - 1]
        return exps

    print('Initial Perturbation Size: {}'.format(eps))
    print('Initializing Lyapunov Finder', end='...')
    perturb = np.eye(system.dim)
    for i in range(lyap_breakin):
        x, t, perturb, error = lyap_step(x, t, perturb)
    print('Calculating Lyapunov Exponents...')
    abs_change = 1
    t_0 = t
    lyap_totals = np.zeros(system.dim)
    lyap_sum = np.zeros(system.dim)
    count = 0
    if system.discrete:
        print('Iterations:', end=' ')
    else:
        print('Orbits:', end=' ')
    while count < (cycles * per_orb) and abs_change > 10 ** (-digits):
        x, t, perturb, growth_factor = lyap_step(x, t, perturb)
        lyap_totals += np.log(growth_factor)
        lyap_sum = lyap_totals / (t - t_0)
        count += 1
        if not count % (cycles * per_orb / 20):
            print(count // per_orb, end='...')
    print('\n Lyapunov Spectrum for {} System:'.format(system.__class__.__name__))
    print(np.round(extract_exponents(lyap_sum), decimals=3))


# SYSTEMS


class Lorenz:
    def __init__(self, s=16, r=45.92, b=4):
        def f(t, w):
            x = w[0]
            y = w[1]
            z = w[2]
            dxdt = s * (y - x)
            dydt = x * (r - z) - y
            dzdt = x * y - b * z
            return np.array([dxdt, dydt, dzdt])

        def g(t, w):
            return s * (w[1] - w[0])

        self.func = f
        self.apex = g

    discrete = False
    autonomous = True
    dim = 3
    ic = np.array([10, 10, 1])
    ax_names = ['x', 'y', 'z']


class Rossler:
    def __init__(self, a=0.15, b=0.2, c=10):
        def f(t, w):
            x = w[0]
            y = w[1]
            z = w[2]
            dxdt = -(y + z)
            dydt = x + a * y
            dzdt = b + z * (x - c)
            return np.array([dxdt, dydt, dzdt])

        def g(t, w):
            return -(w[1] + w[2])

        self.func = f
        self.apex = g

    discrete = False
    autonomous = True
    dim = 3
    ic = np.array([1, 1, 0])
    ax_names = ['x', 'y', 'z']
    title = 'Rössler'


class Henon:
    def __init__(self, a=1.4, b=0.3):
        def f(w):
            x = w[0]
            y = w[1]
            x_new = 1 - a * x ** 2 + y
            y_new = b * x
            return np.array([x_new, y_new])

        self.func = f

    discrete = True
    dim = 2
    ic = np.array([0, 0])
    ax_names = ['x', 'y']
    title = 'Henón'


class Predator_Prey_Discrete:
    def __init__(self, a=2.8, g=7, v=0.5):
        def f(w):
            n = w[0]
            p = w[1]
            eaten = g * n * p / (n + 1)
            return np.array([n + a * n * (1 - n) - eaten, v * eaten])

        self.func = f

    discrete = True
    dim = 2
    ic = np.array([0.1, 0.1])
    ax_names = ['N', 'P']
    title = 'Discrete Predator-Prey'


class Predator_Prey_Seasonal:
    def __init__(self, d=0.58, a=1.5, k=2.2, g=2.5, v=0.5, m=0.4):
        def f(t, w):
            x = w[0]
            y = w[1]
            dxdt = a * x * (1 + d * np.sin(2 * np.pi * t / 8) - x / k) - g * x * y / (x + 1)
            dydt = v * g * x * y / (x + 1) - m * y
            return np.array([dxdt, dydt])

        def apx(t, w):
            x = w[0]
            y = w[1]
            return a * x * (1 + d * np.sin(2 * np.pi * t / 8) - x / k) - g * x * y / (x + 1)

        self.func = f
        self.apex = apx

    discrete = False
    autonomous = False
    dim = 2
    ic = np.array([0.5, 1])
    ax_names = ['N', 'P ']
    title = 'Seasonally-Forced Predator-Prey'


class Chua:
    def __init__(self, a=15.6, b=28, m0=-1.143, m1=-0.714):
        def f(t, w):
            x = w[0]
            y = w[1]
            z = w[2]
            h = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
            dxdt = a * (y - x - h)
            dydt = x - y + z
            dzdt = -b * y
            return np.array([dxdt, dydt, dzdt])

        def g(t, w):
            x = w[0]
            y = w[1]
            h = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
            return a * (y - x - h)

        self.func = f
        self.apex = g

    discrete = False
    autonomous = True
    dim = 3
    ic = np.array([0.5, 0, 0])
    ax_names = ['x', 'y', 'z']
    title = 'Chua Circuit'


main(Predator_Prey_Seasonal(), display=True)
