import os
import glob
import shutil
import logging
from glob import glob
import numpy as np
from scipy.optimize import minimize

from popt.loop.optimize import Optimize
from popt.loop.ensemble_gaussian import GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from input_output import read_config
from popt.update_schemes.enopt import EnOpt
from popt.update_schemes.smcopt import SmcOpt
from popt.cost_functions.npv import npv
from popt.misc_tools import optim_tools as ot

from enopt_plot import plot_obj_func

np.random.seed(101122)


def main():
    # Check if folder contains any En_ files, and remove them!
    for folder in glob('En_*'):
        shutil.rmtree(folder)
    for file in glob('optimize_result_*'):
        os.remove(file)

    # Read inputfile
    ko, kf, ke = read_config.read_toml('init_optim.toml')

    # Initialize
    sim = flow(kf)
    ensemble = GaussianEnsemble(ke, sim, npv)
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Example using ensemble gradient approximation with the conjugate gradient (CG) method from scipy.minimize
    #res = minimize(ensemble.function, x0, args=(cov,), method='CG', jac=ensemble.gradient, tol=ko['tol'],
    #               callback=ot.save_optimize_results, bounds=bounds, options=ko)
    #print(res)

    # Example using EnOpt
    EnOpt(ensemble.function, x0, args=(cov,), jac=ensemble.gradient, hess=ensemble.hessian, bounds=bounds, **ko)

    # Example using the sequential Monte Carlo method
    #ko['savedata'] += ["best_state", "best_func"]
    #SmcOpt(ensemble.function, x0, args=(cov,), sens=ensemble.calc_ensemble_weights, bounds=bounds, **ko)

    # Example calling EnOpt through scipy.minimize (this does exactly the same as running EnOpt)
    #minimize(ensemble.function, x0, args=(cov,), method=EnOpt, jac=ensemble.gradient, hess=ensemble.hessian,
    #         bounds=bounds, options=ko)

    # Post-processing: enopt_plot
    plot_obj_func()


if __name__ == '__main__':
    main()
