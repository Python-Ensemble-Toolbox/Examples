from pipt.loop.assimilation import Assimilate
from subsurface.multphaseflow.flow_rock import flow_rock
from input_output import read_config
from pipt import pipt_init
from ensemble.ensemble import Ensemble

# fix the seed for reproducability
import numpy as np
np.random.seed(10)

kd, kf, ke = read_config.read_toml('3D_ES.toml')
sim = flow_rock(kf)

analysis = pipt_init.init_da(kd, ke, sim)
assimilation = Assimilate(analysis)
assimilation.run()

