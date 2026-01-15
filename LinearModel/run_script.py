from pipt.loop.assimilation import Assimilate
from simulator.simple_models import lin_1d
from input_output import read_config
from pipt import pipt_init
from ensemble.ensemble import Ensemble
import matplotlib.pyplot as plt
from geostat.decomp import Cholesky

import numpy as np
np.random.seed(10)

param = 'permx'

#kd, kf, _ = read_config.read_toml('init_esmda.toml')
kd, kf = read_config.read_txt('init.pipt')
sim = lin_1d(kf)

# np.random.seed(10)
# en = Ensemble(kd,sim)
# en.calc_prediction(save_prediction='test1')

np.random.seed(10)
analysis = pipt_init.init_da(kd, kd, sim)
assimilation = Assimilate(analysis)
assimilation.run()

# Post process - compare with analytic solution
# ---------------------------------------------

# calculate true KF mean and standard dev
geostat = Cholesky()

G = np.zeros((len(assimilation.ensemble.sim.l_prim),assimilation.ensemble.prior_info[param]['nx']))
for i in range(len(assimilation.ensemble.sim.l_prim)):
    G[i,assimilation.ensemble.sim.true_order[1][i]] = 1

Cm = geostat.gen_cov2d(assimilation.ensemble.prior_info[param]['nx'],assimilation.ensemble.prior_info[param]['ny'],
                   assimilation.ensemble.prior_info[param]['variance'][0],assimilation.ensemble.prior_info[param]['corr_length'][0],
                   assimilation.ensemble.prior_info[param]['aniso'][0],assimilation.ensemble.prior_info[param]['angle'][0],
                   assimilation.ensemble.prior_info[param]['vario'][0])

if hasattr(assimilation.ensemble, 'full_cov'):
    if len(assimilation.ensemble.full_cov_data.shape)==1:
        Cd = np.diag(assimilation.ensemble.full_cov_data)
    else:
        Cd = assimilation.ensemble.full_cov_data
    dobs =  assimilation.ensemble.full_real_obs_data.mean(axis=1)
else:
    if len(assimilation.ensemble.cov_data.shape)==1:
        Cd = np.diag(assimilation.ensemble.cov_data)
    else:
        Cd = assimilation.ensemble.cov_data
    try:
        dobs = assimilation.ensemble.obs_data_vector
    except:
        dobs = assimilation.ensemble.vecObs  

CxGT = np.dot(Cm, G.T)
GCxGT = np.dot(G,CxGT)
C_inv = np.linalg.inv(GCxGT + Cd)
K = np.dot(CxGT, C_inv)  # Kalman gain

# Analytic posterior
x_prior = assimilation.ensemble.prior_info[param]['mean']
x_post = x_prior + np.dot(K, dobs)
Cx_post = Cm - np.dot(K, CxGT.T)
Sx_post = np.sqrt(np.diag(Cx_post))

# Plot the results
x_post_ens = assimilation.ensemble.state[param].mean(axis=1)
Sx_prior = np.sqrt(assimilation.ensemble.prior_info[param]['variance'])*np.ones(x_prior.size)
Sx_post_ens = assimilation.ensemble.state[param].std(ddof=1,axis=1)
plt.figure();plt.plot(x_prior,'g');plt.plot(x_post_ens);plt.plot(x_post,'r');
plt.legend(['Prior','Posterior', 'Analytic']); plt.xlabel('Index'); plt.ylabel('State mean');
plt.figure();plt.plot(Sx_prior,'g');plt.plot(Sx_post_ens);plt.plot(Sx_post,'r');
plt.legend(['Prior','Posterior', 'Analytic']); plt.xlabel('Index'); plt.ylabel('State std');
plt.show()
