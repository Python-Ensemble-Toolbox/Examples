"""Generate the prior ensemble and the synthetic seismic data for the SPE11B case.

The log-permeability of each rock type (SATNUM 1-7) is a Gaussian field with its own
mean, variance and correlation. One draw is the truth and the others are the prior
ensemble (prior.npz). The truth is simulated with OPM Flow followed by the rock-physics
model in config.toml, and the observations are its change in bulk impedance at each
survey, plus Gaussian noise of 10% of the vintage's spread (bulkimp_<i>.npz, data.csv,
var.csv). The true field is saved to true_permx.npz for plotting.
"""

import numpy as np
import pandas as pd
from geostat.decomp import Cholesky
from input_output import read_config
from subsurface.multphaseflow.flow_rock import flow_rock

NX, NZ = 83, 58     # grid cells along x and depth

# Per rock type: log-permeability mean [ln mD], variance, correlation length [cells], anisotropy, angle [deg]
ROCKS = [
    (-2.289, 0.01, 1, 1, 0),
    (4.618, 0.09, 40, 4, 45),
    (5.311, 0.09, 40, 4, 45),
    (6.228, 0.09, 40, 4, 45),
    (6.921, 0.09, 40, 4, 45),
    (7.614, 0.09, 40, 4, 45),
    (-11.56, 0.01, 1, 1, 0),
]


def read_satnum(file='model/SATNUM.INC'):
    """The rock type of every cell, in the order Flow reads them (x fastest); expands n*value."""
    values = []
    for token in open(file).read().split('SATNUM', 1)[1].split('/')[0].split():
        if '*' in token:
            count, value = token.split('*')
            values += [int(value)] * int(count)
        else:
            values.append(int(token))
    return np.array(values)


def draw_fields(ne):
    """ne + 1 log-permeability fields: column 0 is the truth, the rest the prior ensemble."""
    rock = read_satnum()
    fields = np.zeros((rock.size, ne + 1))
    for number, (mean, var, length, aniso, angle) in enumerate(ROCKS, start=1):
        cov = Cholesky().gen_cov2d(NX, NZ, var, length, aniso, angle, 'sph')
        draws = Cholesky().gen_real((mean - var / 2) * np.ones(rock.size), cov, ne + 1)
        fields[rock == number] = draws[rock == number]
    return fields


def main():
    np.random.seed(10)
    _, kwsim, kwens = read_config.read('config.toml')

    fields = draw_fields(kwens['ne'])
    np.savez('true_permx.npz', permx=fields[:, 0])
    np.savez('prior.npz', permx=fields[:, 1:])
    np.savez('overburden.npz', obvalues=400.0)    # overburden pressure [bar], the same in every cell

    # Simulate the truth: one record per survey
    sim = flow_rock(kwsim)
    sim.redund_sim = None
    sim.setup_fwd_run()
    records = sim.run_fwd_sim({'permx': fields[:, 0]}, 0)

    # Noisy observations, one file per vintage, and their variances
    data, var = [], []
    for vintage, record in enumerate(records):
        value = record['bulkimp']
        std = 0.1 * np.std(value)
        np.savez(f'bulkimp_{vintage}.npz', bulkimp=value + std * np.random.standard_normal(value.shape))
        data.append(f'bulkimp_{vintage}.npz')
        var.append(['abs', float(std ** 2)])

    index = pd.Index(kwsim['reportpoint'], name=kwsim['reporttype'])
    pd.DataFrame({'bulkimp': data}, index=index).to_csv('data.csv')
    pd.DataFrame({'bulkimp': var}, index=index).to_csv('var.csv')


if __name__ == '__main__':
    main()
