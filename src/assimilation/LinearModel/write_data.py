"""Write the synthetic observations of the linear model: data.csv and var.csv.

The true state is a draw from N(0, I) on 150 cells, and the data are its values at
every fifth cell, observed with unit variance.
"""

import numpy as np
import pandas as pd

np.random.seed(10)
truth = np.random.multivariate_normal(np.zeros(150), np.eye(150))

positions = pd.Index(range(5, 150, 5), name='position')
pd.DataFrame({'value': truth[positions]}, index=positions).to_csv('data.csv')
pd.DataFrame({'value': [['abs', 1.0]] * len(positions)}, index=positions).to_csv('var.csv')
