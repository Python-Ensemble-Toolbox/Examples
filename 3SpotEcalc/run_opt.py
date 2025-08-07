import sys
import os
import glob
import shutil
import logging
from glob import glob
import numpy as np

from popt.loop.ensemble import Ensemble
from popt.loop.optimize import Optimize
from subsurface.multphaseflow.opm import flow
from input_output import read_config
from popt.update_schemes.enopt import EnOpt
from popt.cost_functions.ecalc_npv import ecalc_npv
from plot_optim import plot_obj_func

np.random.seed(101122)


def main():

    # Check if folder contains any En_ files, and remove them!
    for folder in glob('En_*'):
        shutil.rmtree(folder)
    for file in glob('optimize_result_*'):
        os.remove(file)

    ko, kf, ke = read_config.read_toml('init_optim.toml')

    # Get variables
    sim = flow(kf)
    ensemble = Ensemble(ke, sim, ecalc_npv)
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Run EnOpt
    EnOpt(ensemble.function, x0, args=(cov,), jac=ensemble.gradient, hess=ensemble.hessian, bounds=bounds, **ko)

    # Post-processing: enopt_plot
    plot_obj_func()


if __name__ == '__main__':
    main()
