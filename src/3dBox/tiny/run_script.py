from pipt.loop.assimilation import Assimilate
from subsurface.multphaseflow.opm import flow
from subsurface.multphaseflow.eclipse import ecl_100
from input_output import read_config
from pipt import pipt_init
from ensemble.ensemble import Ensemble
from misc import grdecl
# fix the seed for reproducibility
import sys
import numpy as np
np.random.seed(10)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                         message="resdata vectors are deprecated")

# ── settings ──────────────────────────────────────────────────────────────
SIMULATOR = 'eclipse'      # 'eclipse' (ecl_100) or 'flow' (OPM)
TRUE_CASE = 'tiny'        # which 3dBox grid this case uses
REGENERATE_TRUTH = True   # True -> rebuild ../data/*.pkl before assimilating
# ──────────────────────────────────────────────────────────────────────────

if REGENERATE_TRUTH:
    # ../data is not an importable package, so put it on the path first. The
    # import lives in here because it pulls in mat73/geostat/resdata/mako,
    # which are only needed when actually regenerating.
    sys.path.insert(0, '../data')
    import setup as true_case  # ../data/setup.py
    true_case.main(SIMULATOR, model=TRUE_CASE)

kd, kf, ke = read_config.read_toml('3D_ESMDA.toml')  # Run with ESMDA and toml input format
#kd, kf = read_config.read_txt('3D_ES.pipt')  # Run with ES and plain text input format
#ke = kd

# infer the grid size from the grid file so RUNFILE.mako need not hardcode it.
# these reach the template via _runMako, which merges input_dict['mako_kwargs']
# into the Mako context. vapoil is required by the PVTG table in include/pvt.txt
nx, ny, nz = (int(d) for d in grdecl.read('grid/Grid.grdecl')['DIMENS'])
kf['mako_kwargs'] = {'nx': nx, 'ny': ny, 'nz': nz, 'vapoil': True}

if SIMULATOR == 'eclipse':
    kf['parallel'] = 1  # adjusting for one eclipse license
    sim = ecl_100(kf)
else:
    sim = flow(kf)

#en = Ensemble(kd,sim)
#en.calc_prediction(save_prediction='prior_prediction')

analysis = pipt_init.init_da(kd, ke, sim)
assimilation = Assimilate(analysis)
assimilation.run()