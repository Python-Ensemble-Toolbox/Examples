"""Generate the synthetic observations for one 3dBox case: data.csv and var.csv.

Run it from the case folder, e.g.

    cd tiny
    python ../make_data.py

It draws a true log-permeability field on the case's grid and runs the case's own
simulator on it, as configured in config.toml: OPM Flow, or OPM Flow with rock physics
when the simulator section has a pem block (flowrock). The observations are the
simulated data plus Gaussian noise: 10% of the value (at least 0.005) for well rates,
and 10% of each vintage's spread for seismic data, which is saved to bulkimp_<i>.npz.
"""

import numpy as np
import pandas as pd
from geostat.gaussian_sim import fast_gaussian
from input_output import read_config
from misc import grdecl


def simulator(kwsim):
    """The case's simulator: OPM Flow, with rock physics if the config has a pem block."""
    if 'pem' in kwsim:
        from subsurface.multphaseflow.flow_rock import flow_rock
        return flow_rock(kwsim)
    from subsurface.multphaseflow.opm import flow
    return flow(kwsim)


def main():
    np.random.seed(10)
    _, kwsim, _ = read_config.read('config.toml')

    # The truth: log-permeability around 3.5 with a correlation length of 10 cells
    dims = grdecl.read('grid/Grid.grdecl')['DIMENS']
    permx = 3.5 + fast_gaussian(dims, np.array([1]), np.array([10, 10, 10])).flatten()

    # Simulate it, one record per report point
    sim = simulator(kwsim)
    sim.redund_sim = None
    sim.setup_fwd_run()
    records = sim.run_fwd_sim({'permx': permx}, 0)
    truth = pd.DataFrame.from_records(records, index=pd.Index(kwsim['reportpoint'], name=kwsim['reporttype']))

    # Noisy observations and their variances
    data = truth.astype(object).copy()
    var = pd.DataFrame(index=truth.index, columns=truth.columns, dtype=object)
    vintage = 0
    for label in truth.index:
        for column in truth.columns:
            value = truth.at[label, column]
            if np.ndim(value) == 0:          # a well rate
                if value is None or np.isnan(value):
                    continue
                std = max(0.1 * abs(value), 0.005)
                data.at[label, column] = value + std * np.random.standard_normal()
            else:                            # a seismic vintage, saved to its own file
                std = 0.1 * np.std(value)
                file = f'{column}_{vintage}.npz'
                np.savez(file, **{column: value + std * np.random.standard_normal(value.shape)})
                data.at[label, column] = file
                vintage += 1
            var.at[label, column] = ['abs', float(std ** 2)]

    data.to_csv('data.csv')
    var.to_csv('var.csv')


if __name__ == '__main__':
    main()
