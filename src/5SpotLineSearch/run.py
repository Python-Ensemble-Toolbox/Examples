import numpy  as np
import pandas as pd
import copy

# Set initial rate values
rate_max = 500
rate_min = 0
rate_var = 0.05**2 * rate_max**2
rate_info = [
    ['mean', 8*[200]],
    ['var', rate_var],
    ['limits', rate_min, rate_max],
]

# Economic constants for NPV
npv_parameters = [
    ['wop', 400],    # oil price [USD/Sm³]
    ['wgp', 0.4],    # gas price [USD/Sm³]
    ['wwp', 20],     # cost of water disposal  [USD/Sm³]
    ['wwi', 10],     # cost of water injection [USD/Sm³]
    ['disc', 0.08]
]

# Report dates for simulator
report = pd.date_range('2000-02-01', '2008-01-01', freq='MS').to_pydatetime().tolist()

# Ensmble options
en_options = {
    'ne': 50,
    'state': ['rate_inj1', 'rate_inj2', 'rate_inj3', 'rate_inj4'],
    'transform': True,
    'prior_rate_inj1': copy.deepcopy(rate_info),
    'prior_rate_inj2': copy.deepcopy(rate_info),
    'prior_rate_inj3': copy.deepcopy(rate_info),
    'prior_rate_inj4': copy.deepcopy(rate_info),
}

# Simulator options
sim_options = {
    'npv_const': npv_parameters,
    'parallel': 5,
    'runfile': '5SPOT',
    'reportpoint': report,
    'reporttype' : 'dates',
    'datatype': ['fopt', 'fgpt', 'fwpt', 'fwit']
}

# imports from PET
from popt.loop.ensemble_gaussian import GaussianEnsemble
from popt.update_schemes.linesearch import LineSearch
from popt.cost_functions.npv import npv
from simulator.opm import flow

# Define objective function (NPV)
NPV = lambda *args, **kwargs: -npv(*args, **kwargs)/1e9

if __name__ == '__main__':

    # Set random seed
    np.random.seed(10_08_1997)

    # Ensemble initialization
    ensemble = GaussianEnsemble(en_options, flow(sim_options), NPV)

    # Get initial state
    x0  = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Define function and gradient for optimization
    func = lambda x,*args: ensemble.function(x,*args)
    grad = lambda x,*args: ensemble.gradient(x,*args)/np.diag(args[0])

    # Optimization with the BFGS method
    options = {
        'save_folder': 'results',
        'step_size': 1.0,
        'maxiter': 20
    }

    res = LineSearch(
        fun=func,
        x=x0,
        jac=grad,
        method='BFGS',
        args=(cov,),
        bounds=bounds,
        **options
    )
    print(res)