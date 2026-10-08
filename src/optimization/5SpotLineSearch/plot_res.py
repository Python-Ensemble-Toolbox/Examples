"""Plot the NPV against iteration, from the files run_opt.py saves in results/."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# One file per iteration; 'fun' is the negative NPV in billion USD
files = sorted(Path('results').glob('optimize_result_*.npz'), key=lambda f: int(f.stem.split('_')[-1]))
npv = [-float(np.load(f)['fun']) for f in files]

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(npv, marker='o', color='cadetblue')
ax.grid(alpha=0.25)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_xlabel('iteration')
ax.set_ylabel('NPV [billion USD]')

# Relative increase from the starting point on the right axis
increase = lambda v: 100 * (v - npv[0]) / npv[0]
inverse = lambda p: npv[0] * (1 + p / 100)
right = ax.secondary_yaxis('right', functions=(increase, inverse))
right.set_ylabel('relative increase [%]', color='cadetblue')
right.tick_params(axis='y', colors='cadetblue')

fig.tight_layout()
fig.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
