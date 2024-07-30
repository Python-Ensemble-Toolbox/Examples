from popt.misc_tools.basic_tools import *
from geostat.gaussian_sim import fast_gaussian
import numpy as np

models = []
for m in range(25):
	x = np.log(200) + fast_gaussian(np.array([100, 100]), np.array([.3]), np.array([50, 20]))
	p = np.exp(x)
	models.append(p)

models = np.array(models).T
np.savez('geo_models_lowvar.npz', models=models)

models = []
for m in range(25):
	x = np.log(200) + fast_gaussian(np.array([100, 100]), np.array([3]), np.array([50, 20]))
	p = np.exp(x)
	models.append(p)

models = np.array(models).T
np.savez('geo_models_highvar.npz', models=models)