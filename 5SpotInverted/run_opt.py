import sys
import os
import glob
import shutil
import logging
from glob import glob
import numpy as np

from popt.loop.optimize import Optimize
from popt.loop.ensemble_gaussian import GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from input_output import read_config
from popt.update_schemes.enopt import EnOpt
from popt.cost_functions.npv import npv

from plot_optim import plot_obj_func, plot_state

np.random.seed(101122)


def main():
    # Check if folder contains any En_ files, and remove them!
    for folder in glob('En_*'):
        shutil.rmtree(folder)
    for file in glob('optimize_result_*'):
        os.remove(file)

    ko, kf, ke = read_config.read_toml('init_optim.toml')

    sim = flow(kf)
    ensemble = GaussianEnsemble(ke, sim, npv)
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    EnOpt(ensemble.function, x0, args=(cov,), jac=ensemble.gradient, hess=ensemble.hessian, bounds=bounds, **ko)

    # Plotting
    plot_obj_func()
    plot_state(4)


if __name__ == '__main__':
    main()
