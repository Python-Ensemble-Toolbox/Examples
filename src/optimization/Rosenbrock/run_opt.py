"""Minimizing the Rosenbrock function with an ensemble gradient and with the analytical gradient.

    minimize   f(x) = (1 - x_0)^2 + 100 * (x_1 - x_0^2)^2

The minimum is x = (1, 1), at the end of a long, curved, nearly flat valley,
which makes the function a classic test for optimizers.

Two runs start from x = (-2, -2):

    1. EnOpt, which estimates the gradient from an ensemble of perturbed controls.
       It only needs function values, so it also works when the gradient is unknown,
       as for a reservoir simulator.
    2. BFGS line search with the analytical gradient.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import rosen, rosen_der
from input_output import read_config
from popt import EnOpt, LineSearch, GaussianEnsemble

HERE = Path(__file__).parent


def objective(x, **kwargs):
    """Rosenbrock function for every column of x (shape: controls x members).

    The ensemble passes extra keywords to the objective, which rosen does not take.
    """
    return rosen(x)


def main():

    # ---------------------------------------------------------------
    # EnOpt (ensemble gradient)
    # ---------------------------------------------------------------
    kwopt, _, kwens = read_config.read(HERE / 'config_enopt.toml')

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
    print(f'\nEnOpt: x = {np.round(res.x, 3)} after {res.nit} iterations  (true optimum [1, 1])\n')



    # ---------------------------------------------------------------
    # BFGS (analytical gradient)
    # ---------------------------------------------------------------
    res = LineSearch.minimize(
        x0=np.array([-2.0, -2.0]),
        fun=rosen,
        jac=rosen_der,
        method='BFGS',
    )
    print(f'\nBFGS: x = {np.round(res.x, 3)} after {res.nit} iterations  (true optimum [1, 1])\n')


if __name__ == '__main__':
    main()
