"""Two ensemble-based optimizers on a small 2D field: EnOpt and sequential Monte Carlo (SmcOpt).

A 100x100 field with two water injectors and one producer, run from 1994 to 1999. The
controls are the bottom-hole pressures (BHP) of the three wells, constant over the run,
and the goal is to maximize the net present value (NPV). The case runs in a few minutes.

    1. EnOpt estimates the gradient from an ensemble of perturbed controls and steps
       along it.
    2. SmcOpt draws samples around the controls, weights them by their NPV, and moves
       the controls towards the best ones. It needs no gradient, and also keeps track
       of the best sample it has seen.

The simulator is OPM Flow, and the NPV comes from subsurface.cost_functions.npv.
Run from this folder, then run plot_res.py to compare the two runs.
"""

import numpy as np
from input_output import read_config
from popt import EnOpt, SmcOpt, GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from subsurface.cost_functions.npv import npv


def main():

    # ---------------------------------------------------------------
    # EnOpt
    # ---------------------------------------------------------------
    kwopt, kwsim, kwens = read_config.read('config_enopt.toml')

    ensemble = GaussianEnsemble(kwens, flow(kwsim), npv)

    res = EnOpt.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function,
        jac=ensemble.gradient,
        hess=ensemble.hessian,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        **kwopt,
    )
    print(f'\nEnOpt: NPV {-float(res.fun):.3f} million USD after {res.nit} iterations\n')



    # ---------------------------------------------------------------
    # SmcOpt
    # ---------------------------------------------------------------
    kwopt, kwsim, kwens = read_config.read('config_smcopt.toml')

    ensemble = GaussianEnsemble(kwens, flow(kwsim), npv)

    res = SmcOpt.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function,
        sens=ensemble.calc_ensemble_weights,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        **kwopt,
    )
    print(f'\nSmcOpt: NPV {-float(np.mean(res.fun)):.3f} million USD after {res.nit} iterations '
          f'(best sample {-res.best_func:.3f})\n')


if __name__ == '__main__':
    main()
