"""Ensemble optimization of a quadratic function, with and without constraints.

    minimize   f(x) = 0.5 * ||x - 1||^2

The unconstrained minimum is x = (1, 1). The constrained run adds

    x_0 + x_1 = 3    (equality)
    x_0 <= 0         (inequality)

which moves the minimum to x = (0, 3). Three cases are run:

1. EnOpt without the constraints.
2. EnOpt with the constraints handled by an exterior penalty function (EPF):
   PET hands the objective an ``epf`` dict holding the penalty factor ``r``, and
   the objective writes the penalty it adds back into that dict, so PET can tell
   when the constraints are satisfied.
3. EnSQP (ensemble-based sequential quadratic programming) with the constraints
   handled by the optimizer: the objective returns the constraint values with
   the objective value, and the gradient and the constraint Jacobian are
   estimated from the same ensemble.
"""

import numpy as np
from pathlib import Path
from input_output import read_config
from popt import EnOpt, EnSQP, GaussianEnsemble

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


def objective_with_constraints(x, **kwargs):
    """Quadratic objective and its constraints, in the form EnSQP takes them.

    The constraints are returned as one row per constraint and one column per member.
    An equality holds when ``c_eq = 0``, an inequality when ``c_iq >= 0``.
    """
    return {
        'f': 0.5 * np.sum((x - 1.0) ** 2, axis=0),
        'c_eq': np.sum(x, axis=0, keepdims=True) - 3.0,    # x_0 + x_1 = 3
        'c_iq': -x[:1],                                    # x_0 <= 0
    }


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
    print(f'\nConstrained (EPF): x = {np.round(res.x, 3)}  (true optimum [0, 3])\n')



    # ---------------------------------------------------------------
    # Constrained optimization (EnSQP)
    # ---------------------------------------------------------------
    kwopt, _, kwens = read_config.read(HERE / 'config_ensqp.toml')

    ensemble = GaussianEnsemble(kwens, None, objective_with_constraints)

    res = EnSQP.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function_and_constraints,
        jac=ensemble.gradient,
        con_jac=ensemble.constraint_jacobian,
        hess=ensemble.lagrangian_hessian,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        **kwopt,
    )
    print(f'\nConstrained (EnSQP): x = {np.round(res.x, 3)}  (true optimum [0, 3]), '
          f'constraint violation = {res.violation:.1e}\n')


if __name__ == '__main__':
    main()
