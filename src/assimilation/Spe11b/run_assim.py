"""ES on the SPE11B CO2-storage case: estimate the log-permeability from time-lapse seismic.

A 2D vertical section (83 x 58 cells) where CO2 is injected through two wells for 25
years. The data are the change in bulk impedance since the start, after every five years
of injection, simulated with OPM Flow followed by a rock-physics model (flow_rock) from a
synthetic true field. The prior is drawn separately for each of the seven rock types.

Run from this folder:

    python make_data.py     # the prior ensemble and the synthetic observations
    python run_assim.py     # the assimilation (about an hour)
    python plot_res.py      # permeability fields and seismic data
"""

from input_output import read_config
from pipt import ES
from subsurface.multphaseflow.flow_rock import flow_rock


def main():
    kwda, kwsim, kwens = read_config.read('config.toml')

    res = ES.assimilate(kwda, kwens, flow_rock(kwsim))
    print(f'\nData misfit: {res.prior_data_misfit:.1f} -> {res.data_misfit:.1f}\n')


if __name__ == '__main__':
    main()
