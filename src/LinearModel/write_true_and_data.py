from simulator.simple_models import lin_1d
from input_output import read_config
import numpy as np
import csv

kd, kf = read_config.read_txt('init.pipt')
sim = lin_1d(kf)

np.random.seed(10)
# Gen true model
sim.setup_fwd_run()
state = {'m':np.random.multivariate_normal(np.zeros(150),np.eye(150))}
sim.run_fwd_sim(state,0)

f = open('true_data.csv','w',newline='')
writer1 = csv.writer(f)
g = open('var.csv','w',newline='')
writer2 = csv.writer(g)

for indx in sim.l_prim:
    writer1.writerow(sim.pred_data[indx]['value'])
    writer2.writerow(['ABS', 1])

f.close()
g.close()