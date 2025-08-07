from pipt.loop.assimilation import Assimilate
from subsurface.multphaseflow.opm import flow
from input_output import read_config
from pipt import pipt_init
from ensemble.ensemble import Ensemble

# fix the seed for reproducability
import numpy as np
np.random.seed(10)

#kd, kf, ke = read_config.read_toml('3D_ESMDA.toml')  # Run with ESMDA and toml input format
kd, kf = read_config.read_txt('3D_ES.pipt')  # Run with ES and plain text input format
ke = kd
sim = flow(kf)

#en = Ensemble(kd,sim)
#en.calc_prediction(save_prediction='prior_prediction')

analysis = pipt_init.init_da(kd, ke, sim)
assimilation = Assimilate(analysis)
assimilation.run()

