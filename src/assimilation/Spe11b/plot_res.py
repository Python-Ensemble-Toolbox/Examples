"""Plot the results of run_assim.py.

Top: the true log-permeability, and the mean of the prior and posterior ensembles. Bottom:
the observed change in bulk impedance at the last survey, and the posterior mean prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from misc.structures import PETDataFrame

NX, NZ = 83, 58


def section(values):
    """A field on the grid (x fastest, depth from the top) as an image."""
    return np.reshape(values, (NZ, NX))


true = np.load('true_permx.npz')['permx']
prior = np.load('prior.npz')['permx'].mean(axis=1)
posterior = np.load('results/posterior_state_estimate.npz')['permx'].mean(axis=1)

observed = np.load('bulkimp_4.npz')['bulkimp']
forecast = PETDataFrame.from_pickle('results/posterior_forecast.pkl')
forecast.is_ensemble = True
predicted = np.asarray(forecast['bulkimp'].iloc[-1]).mean(axis=1)

fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
for ax, field, title in zip(axes[0], [true, prior, posterior], ['true', 'prior mean', 'posterior mean']):
    image = ax.imshow(section(field), vmin=0, vmax=8, cmap='viridis', aspect='auto')
    ax.set_title(f'log-permeability, {title}', fontsize=10)
fig.colorbar(image, ax=axes[0].tolist(), label='ln mD')

limit = np.abs(observed).max()
for ax, field, title in zip(axes[1], [observed, predicted], ['observed', 'posterior mean']):
    image = ax.imshow(section(field), vmin=-limit, vmax=limit, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'impedance change, {title}', fontsize=10)
fig.colorbar(image, ax=axes[1].tolist(), label='after 25 years')
axes[1, 2].axis('off')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
