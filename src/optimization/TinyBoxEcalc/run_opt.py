"""Line search (steepest descent) with the cost of CO2 emissions, computed by eCalc.

The reservoir is TinyBox from the PET tutorials: a 10x10x2 grid with oil, water and
gas, three water injectors and three producers, run for eleven years from 2022. Its
permeability is the history-matched field from the PET TinyBox tutorial. The controls
are the bottom-hole pressures (BHP) of the six wells, constant over the run, and the
goal is to maximize the net present value (NPV). The gradient is estimated by an
ensemble of perturbed controls.

The NPV includes the cost of the CO2 the facility emits. The simulator is OPM Flow
followed by eCalc: for every member, eCalc computes the facility's CO2 emissions from
the production, using the model in ecalc_config.yaml (water injection pumps and fixed
loads, a gas export compressor, all powered by gas turbines), and adds them to the
simulated data as FU_CO2R. The NPV prices the CO2 at wem USD/ton.

Run from this folder, then run plot_res.py to plot the NPV against iteration.
"""

from input_output import read_config
from popt import LineSearch, GaussianEnsemble
from subsurface.multphaseflow.opm import flow
from subsurface.facilities.ecalc import Ecalc
from subsurface.cost_functions.npv import npv


def main():
    kwopt, kwsim, kwens = read_config.read('config.toml')

    # The reservoir simulation, then eCalc on its production
    simulator = Ecalc(flow(kwsim), ecalc_config='ecalc_config.yaml')

    ensemble = GaussianEnsemble(kwens, simulator, npv)

    res = LineSearch.minimize(
        x0=ensemble.get_state(),
        fun=ensemble.function,
        jac=ensemble.gradient,
        args=(ensemble.get_cov(),),
        bounds=ensemble.get_bounds(),
        method='GD',
        **kwopt,
    )
    print(f'\nNPV: {-float(res.fun):.3f} billion USD after {res.nit} iterations, BHP = {res.x.round(1)} bar\n')


if __name__ == '__main__':
    main()
