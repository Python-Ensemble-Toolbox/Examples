"""BFGS line search on a 2D five-spot field, with the gradient estimated by an ensemble.

One producer sits in the centre of a 50x50 field and one water injector in each
corner. The controls are the injection rates of the four injectors, one value per
year for eight years, and the goal is to maximize the net present value (NPV).

The NPV comes from the OPM Flow reservoir simulator, which provides no gradient, so
the ensemble estimates it: it perturbs the rates, runs the simulator for each
perturbation, and fits a gradient to how the NPV changes.

The optimizer minimizes, so obj_scaling = -1e9 in config.toml turns the NPV into
a negative value in billion USD.

Run from this folder, then run plot_res.py to plot the NPV against iteration.
"""

from input_output import read_config
from popt import LineSearch, GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from subsurface.cost_functions.npv import npv


def main():
    kwopt, kwsim, kwens = read_config.read('config.toml')

    ensemble = GaussianEnsemble(kwens, flow(kwsim), npv)

    res = LineSearch.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function,
        jac=ensemble.gradient,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        method='BFGS',
        **kwopt,
    )
    print(f'\nNPV: {-res.fun:.4f} billion USD after {res.nit} iterations\n')


if __name__ == '__main__':
    main()
