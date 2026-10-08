"""Plot the NPV of every geological model, and their mean, against iteration for the runs saved by run_opt.py."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def npv_per_model(folder):
    """NPV for each geological model at each iteration, shape (iterations, models).

    'fun' holds the negative NPV in million USD, one value per model.
    """
    files = sorted(Path(folder).glob('optimize_result_*.npz'), key=lambda f: int(f.stem.split('_')[-1]))
    return np.array([-np.load(f)['fun'] for f in files])


fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)

for ax, folder, title, color in zip(axes, ['results_bhp', 'results_rate'], ['BHP controls', 'rate controls'], ['C0', 'C1']):
    npv = npv_per_model(folder)
    ax.plot(npv, color=color, alpha=0.35, linewidth=1)
    ax.plot(npv.mean(axis=1), color=color, marker='o', linewidth=2.5, label='mean')
    ax.plot([], [], color=color, alpha=0.35, linewidth=1, label='geological models')
    ax.set_title(title)
    ax.set_xlabel('iteration')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right')

axes[0].set_ylabel('NPV [million USD]')

fig.tight_layout()
fig.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
