"""Robust ensemble optimization (EnOpt) on a 2D field with two injectors and one producer.

The permeability is uncertain, so the controls are optimized for five geological
models at once: every control vector is simulated on all five, and the objective
is their mean net present value (NPV). Each gradient perturbation is simulated on
one model, two perturbations per model.

Two runs, each starting from a constant strategy over five years (30-day steps):

    1. Bottom-hole pressure (BHP) controls: injectors at 250 bar, producer at 150 bar.
    2. Rate controls: injectors at 300 Sm3/day, producer at 100 Sm3/day.

The simulator is OPM Flow, and the NPV comes from subsurface.cost_functions.npv.
Run from this folder, then run plot_res.py to compare the two runs.
"""

import numpy as np
from input_output import read_config
from popt import EnOpt, GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from subsurface.cost_functions.npv import npv


def main():

    # Initial strategy, one value per 30-day step (the two injectors interleaved)
    np.savez(
        'initial_controls.npz',
        injbhp=np.full(120, 250.0),
        prodbhp=np.full(60, 150.0),
        injrate=np.full(120, 300.0),
        prodrate=np.full(60, 100.0),
    )

    # ---------------------------------------------------------------
    # BHP controls
    # ---------------------------------------------------------------
    kwopt, kwsim, kwens = read_config.read('config_bhp.toml')

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
    print(f'\nBHP: mean NPV {-np.mean(res.fun):.2f} million USD after {res.nit} iterations\n')



    # ---------------------------------------------------------------
    # Rate controls
    # ---------------------------------------------------------------
    kwopt, kwsim, kwens = read_config.read('config_rate.toml')

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
    print(f'\nRate: mean NPV {-np.mean(res.fun):.2f} million USD after {res.nit} iterations\n')


if __name__ == '__main__':
    main()
