import matplotlib.pyplot as plt
import numpy as np

# Get NPV
iterations = 20
npv = []
for i in range(iterations+1):
    file = f'results/optimize_result_{i}.npz'
    npv.append(-np.load(file)['fun'])


fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(np.arange(len(npv)), npv, marker='o', linestyle='-', color='cadetblue')
ax.grid(alpha=0.25)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_xlim(-0.5, iterations+0.5)
ax.set_xlabel('iteration')
ax.set_ylabel('NPV [Billion $]')

def f(x):
    return 100*(x-npv[0])/npv[0]
secaxy = ax.secondary_yaxis('right', functions=(f, f))
secaxy.tick_params(axis='y', colors='cadetblue')
secaxy.set_ylabel('relative increase [%]', color='cadetblue')

plt.tight_layout()
plt.show()
fig.savefig('results.png', dpi=300, bbox_inches='tight')