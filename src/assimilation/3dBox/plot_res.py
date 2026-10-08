"""Plot the results of a 3dBox case. Run it from the case folder, after run_assim.py:

    python ../plot_res.py

Left: the data misfit of every member after each assimilation step. Right: the oil
and water rates of the producers, prior and posterior ensembles against the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from misc.structures import PETDataFrame

WELLS = ['PRO1', 'PRO2', 'PRO3']

# Data misfit of each member, one file per step (step 0 is the prior)
files = sorted(Path('results').glob('assimilation_result_*.npz'), key=lambda f: int(f.stem.split('_')[-1]))
misfit = [np.load(f)['ensemble_misfit'] for f in files]

# Observations with their standard deviations, and the prior and posterior forecasts
data = pd.read_csv('data.csv', index_col=0)
std = pd.read_csv('var.csv', index_col=0).map(lambda v: np.sqrt(float(v.strip('[]').split(',')[1])) if isinstance(v, str) else np.nan)
forecasts = {}
for name in ['prior', 'posterior']:
    forecast = PETDataFrame.from_pickle(f'results/{name}_forecast.pkl')
    forecast.is_ensemble = True
    forecasts[name] = forecast

fig = plt.figure(figsize=(13, 5.5))
grid = fig.add_gridspec(2, 4)

ax = fig.add_subplot(grid[:, 0])
ax.boxplot(misfit, positions=range(len(misfit)))
ax.set_yscale('log')
ax.set_xlabel('assimilation step')
ax.set_ylabel('data misfit')

for row, key in enumerate(['WOPR', 'WWPR']):
    for col, well in enumerate(WELLS):
        ax = fig.add_subplot(grid[row, col + 1])
        column = f'{key}:{well}'
        for (name, forecast), color in zip(forecasts.items(), ['tab:blue', 'tab:orange']):
            members = np.asarray(forecast[column].tolist())
            ax.fill_between(data.index, members.min(axis=1), members.max(axis=1), color=color, alpha=0.4, label=name)
        ax.errorbar(data.index, data[column], yerr=2 * std[column], fmt='o', color='k', markersize=3,
                    capsize=2, label=r'data $\pm 2\sigma$')
        ax.set_title(column, fontsize=10)
        ax.grid(alpha=0.25)
        if row == 1:
            ax.set_xlabel('days')
        if col == 0:
            ax.set_ylabel('Sm$^3$/day')

fig.axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
