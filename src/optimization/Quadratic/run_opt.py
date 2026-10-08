"""Ensemble optimization (EnOpt) of a quadratic function, with and without constraints.

    minimize   f(x) = 0.5 * ||x - 1||^2

The unconstrained minimum is x = (1, 1). The constrained run adds

    x_0 + x_1 = 3    (equality)
    x_0 <= 0         (inequality)

which moves the minimum to x = (0, 3). The constraints are handled with an
exterior penalty function (EPF): PET hands the objective an ``epf`` dict holding
the penalty factor ``r``, and the objective writes the penalty it adds back into
that dict, so PET can tell when the constraints are satisfied.
"""

import numpy as np
from pathlib import Path
from input_output import read_config
from popt import EnOpt, GaussianEnsemble

HERE = Path(__file__).parent


def objective(x, epf=None, **kwargs):
    """Quadratic objective, evaluated for every column of x (shape: controls x members)."""
    f = 0.5 * np.sum((x - 1.0) ** 2, axis=0)

    if epf:
        epf['penalty'] = penalty(x, epf['r'])
        f = f + epf['penalty']

    return f


def penalty(x, r):
    """Exterior penalty: zero when the constraints hold, growing quadratically outside."""
    c_eq = np.sum(x, axis=0) - 3.0     # x_0 + x_1 = 3
    c_iq = np.maximum(x[0], 0.0)       # x_0 <= 0
    return 0.5 * r * (c_eq ** 2 + c_iq ** 2)


def main():

    # ---------------------------------------------------------------
    # Unconstrained optimization
    # ---------------------------------------------------------------
    kwopt, _, kwens = read_config.read(HERE / 'config.toml')

    # No simulator: the ensemble evaluates the objective directly
    ensemble = GaussianEnsemble(kwens, None, objective)

    res = EnOpt.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function,
        jac=ensemble.gradient,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        **kwopt,
    )
    print(f'\nUnconstrained: x = {np.round(res.x, 3)}  (true optimum [1, 1])\n')



    # ---------------------------------------------------------------
    # Constrained optimization (EPF)
    # ---------------------------------------------------------------
    kwopt, _, kwens = read_config.read(HERE / 'config_epf.toml')

    ensemble = GaussianEnsemble(kwens, None, objective)

    res = EnOpt.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function,
        jac=ensemble.gradient,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        **kwopt,
    )
    print(f'\nConstrained: x = {np.round(res.x, 3)}  (true optimum [0, 3])\n')


if __name__ == '__main__':
    main()
