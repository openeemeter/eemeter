import numpy as np

from copy import deepcopy as copy
import numba

# import jax.numpy as jnp
# from jax import jit
# from jax import jacfwd, jacrev


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


@numba.jit(nopython=True, cache=numba_cache)
def fdcoeffF(k, xbar, x):
    # Produces finite difference stencil based on Fornberg's method
    # Generation of Finite Difference Formulas on Arbitrarily Spaced Grids, Mathematics of Computation 51(1988)
    #   pp. 699-706, doi:10.1090/S0025-5718-1988-0935077-0
    #
    # https://rjleveque.github.io/amath585w2020/notebooks/html/fdstencil.html

    # shortcut for common finite difference grids
    if (k == 1) and (xbar == 0):
        # backward finite difference
        if np.array_equal(x, [-2, -1, 0]):
            return np.array([0.5, -2, 3 / 2])

        # central finite difference
        elif np.array_equal(x, [-1, 0, 1]):
            return np.array([-0.5, 0, 0.5])
        elif np.array_equal(x, [-2, -1, 0, 1, 2]):
            return np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        elif np.array_equal(x, [-3, -2, -1, 0, 1, 2, 3]):
            return np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])
        elif np.array_equal(x, [-4, -3, -2, -1, 0, 1, 2, 3, 4]):
            return np.array(
                [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
            )

        # foward finite difference
        elif np.array_equal(x, [0, 1, 2]):
            return np.array([-3 / 2, 2, -1 / 2])
        elif np.array_equal(x, [0, 1, 2, 3, 4]):
            return np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])
        elif np.array_equal(x, [0, 1, 2, 3, 4, 5, 6]):
            return np.array([-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6])

    n = len(x) - 1
    if k > n:
        raise ValueError("*** len(x) must be larger than k")

    m = k  # for consistency with Fornberg's notation
    c1 = 1.0
    c4 = x[0] - xbar
    C = np.zeros((n + 1, m + 1))
    C[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - xbar
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    C[i, s] = c1 * (s * C[i - 1, s - 1] - c5 * C[i - 1, s]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                C[j, s] = (c4 * C[j, s] - s * C[j, s - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2

    # C contains all coefficients for nth derivatives up to k
    C = C[:, -1]  # last column of C

    # if really close to zero, assign to zero
    C = np.where(np.abs(C) < 1e-14, 0, C)

    return C


default_dx = np.sqrt(np.finfo(float).eps)


def numerical_jacobian(fcn, x, dx=default_dx, order=2, fd_type="central"):
    # order : order of accuracy, since we're dealing with models, 2 is best

    if order % 2 != 0:
        raise ValueError("'order' must be an even int")

    jac = np.empty_like(x)
    for i in range(len(x)):
        # this is not a pretty way to handle list/non-list
        if isinstance(dx, float):
            dxi = dx
        else:
            dxi = dx[i]

        if isinstance(fd_type, str):
            fd_type_i = fd_type
        else:
            fd_type_i = fd_type[i]

        if fd_type_i == "backward":
            grid = np.array(range(int(-order), 1))
        elif fd_type_i == "central":
            grid = np.array(range(int(-order / 2), int(order / 2) + 1))
        elif fd_type_i == "forward":
            grid = np.array(range(0, int(order) + 1))

        x_grid = x[i] + grid * dxi
        fd_coeffs = fdcoeffF(1, 0, grid)

        deriv = 0
        for fdi, xi in zip(fd_coeffs, x_grid):
            if fdi == 0:
                continue

            x_deriv = copy(x)
            x_deriv[i] = xi

            deriv += fdi * fcn(x_deriv)

        jac[i] = deriv / dxi  # since this is first order deriv, dxi^n otherwise

    return jac[None, :]


if __name__ == "__main__":
    fcn = (
        lambda x: 1.5 * float(x[0]) ** 2
        + 1 / 5 * float(x[0])
        + 2.5 * x[1] ** 2
        - 7.5 * x[1]
        - 1e6
    )

    x = [1.0, 1.0]

    print(numerical_jacobian(fcn, x, dx=[1e-6, 1], order=2))
