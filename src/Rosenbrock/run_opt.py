import glob
import shutil
from glob import glob
import numpy as np
import os
import sys

from popt.loop.optimize import Optimize
from popt.loop.ensemble_gaussian import GaussianEnsemble
from simulator.simple_models import noSimulation
from input_output import read_config
from popt.update_schemes.enopt import EnOpt
from popt.cost_functions.rosenbrock import rosenbrock
from plot_optim import plot_obj_func, plot_state

np.random.seed(101122)


def main():
    # Check if folder contains any En_ files, and remove them!
    for folder in glob('En_*'):
        shutil.rmtree(folder)
    for file in glob('optimize_result_*'):
        os.remove(file)

    ko, kf, ke = read_config.read_toml('init_optim.toml')

    dimension = 100  # dimension of Rosenbrock function

    # select starting point
    startmean = np.array([-2]*dimension)
    np.savez('init_mean.npz', startmean)

    sim = noSimulation(kf)
    ensemble = GaussianEnsemble(ke, sim, rosenbrock)
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()
    EnOpt(ensemble.function, x0, args=(cov,), jac=ensemble.gradient, hess=ensemble.hessian, bounds=bounds, **ko)

    # Post-processing
    plot_obj_func()
    plot_state(2)


if __name__ == '__main__':
    main()
