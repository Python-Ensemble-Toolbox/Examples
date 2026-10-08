"""Plot the NPV against iteration for the EnOpt and SmcOpt runs saved by run_opt.py."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load(folder, key):
    """One value per iteration; the saved objective values are the negative NPV in million USD."""
    files = sorted(Path(folder).glob('optimize_result_*.npz'), key=lambda f: int(f.stem.split('_')[-1]))
    return [-float(np.mean(np.load(f)[key])) for f in files]


fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(load('results_enopt', 'fun'), marker='o', label='EnOpt')
ax.plot(load('results_smcopt', 'fun'), marker='s', label='SmcOpt')
ax.plot(load('results_smcopt', 'best_func'), color='C1', linestyle='--', label='SmcOpt, best sample')
ax.grid(alpha=0.25)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_xlabel('iteration')
ax.set_ylabel('NPV [million USD]')
ax.legend()

fig.tight_layout()
fig.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
