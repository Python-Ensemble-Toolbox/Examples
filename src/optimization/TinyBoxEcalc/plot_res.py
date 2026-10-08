"""Plot the NPV and the total CO2 emissions against iteration, for the run saved in results/.

The result files hold each iteration's controls and NPV, but not its emissions, so the
controls of every iteration are simulated once more, as one ensemble, with the same
simulator as run_opt.py (OPM Flow followed by eCalc).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from input_output import read_config
from popt import GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from subsurface.facilities.ecalc import Ecalc
from subsurface.cost_functions.npv import npv


def total_co2(sim_data):
    """Total CO2 [ton] of each member: the emission rate [ton/day] times the days in each report step."""
    days = pd.DatetimeIndex(sim_data.index).to_series().diff().dt.days.to_numpy()[1:]
    rates = np.array([np.atleast_1d(r) for r in sim_data['FU_CO2R']])[1:]   # (report steps, members)
    return days @ rates


# Controls of every iteration, one per column
files = sorted(Path('results').glob('optimize_result_*.npz'), key=lambda f: int(f.stem.split('_')[-1]))
controls = np.array([np.load(f)['x'] for f in files]).T

kwopt, kwsim, kwens = read_config.read('config.toml')
ensemble = GaussianEnsemble(kwens, Ecalc(flow(kwsim), ecalc_config='ecalc_config.yaml'), npv)

npv_values = -ensemble.function(controls)           # billion USD
co2 = total_co2(ensemble.sim_data) / 1e3            # thousand tons

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
ax1.plot(npv_values, marker='o', color='cadetblue')
ax1.set_ylabel('NPV [billion USD]')
ax2.plot(co2, marker='o', color='indianred')
ax2.set_ylabel('total CO$_2$ emissions [kt]')
for ax in (ax1, ax2):
    ax.set_xlabel('iteration')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
