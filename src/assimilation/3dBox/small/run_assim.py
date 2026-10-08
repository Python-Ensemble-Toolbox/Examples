"""ES on the small 3D box: estimate the log-permeability from well rates.

A 40x20x5 reservoir with oil, water and gas, three water injectors and three producers.
The data are the oil and water rates of the producers and the injection rate of INJ1,
every 400 days for 4000 days, simulated with OPM Flow from a synthetic true field.

Run from this folder:

    python ../make_data.py    # the synthetic observations, data.csv and var.csv
    python run_assim.py       # the assimilation
    python ../plot_res.py     # data misfit, and the forecasts against the data
"""

from input_output import read_config
from pipt import ES
from subsurface.multphaseflow.opm import flow


def main():
    kwda, kwsim, kwens = read_config.read('config.toml')

    res = ES.assimilate(kwda, kwens, flow(kwsim))
    print(f'\nData misfit: {res.prior_data_misfit:.1f} -> {res.data_misfit:.1f}\n')


if __name__ == '__main__':
    main()
