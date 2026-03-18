import sys
import os
import glob
import shutil
import logging
from glob import glob
import numpy as np
import datetime as dt
import csv

from popt.loop.optimize import Optimize
from popt.loop.ensemble_gaussian import GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from input_output import read_config
from popt.update_schemes.enopt import EnOpt
from popt.cost_functions.npv import npv

from plot_optim import plot_obj_func

np.random.seed(270623)


def main():
    # Check if folder contains any En_ files, and remove them!
    for folder in glob('En_*'):
        shutil.rmtree(folder)
    for file in glob('optimize_result_*'):
        os.remove(file)

    # Set initial pressure
    init_injbhp = 250.0 * np.ones(120)
    init_prodbhp = 150.0 * np.ones(60)
    np.savez('init_injbhp.npz', init_injbhp)
    np.savez('init_prodbhp.npz', init_prodbhp)

    # Set dates
    report_dates = []
    for index in range(60):
        report_dates.append(dt.datetime(2029, 1, 1) + dt.timedelta(30*(index+1)))
    with open('report_dates.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list(report_dates))

    ko, kf, ke = read_config.read_toml('init_optim_bhp.toml')

    sim = flow(kf)
    ensemble = GaussianEnsemble(ke, sim, npv)
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    EnOpt(ensemble.function, x0, args=(cov,), jac=ensemble.gradient, hess=ensemble.hessian, bounds=bounds, **ko)

    # Plot
    plot_obj_func()


if __name__ == '__main__':
    main()
