"""ES on the large 3D box with seismic data: estimate the log-permeability from well
rates and time-lapse bulk impedance.

A 60x60x5 reservoir with oil, water and gas, three water injectors and three producers.
The data are the oil and water rates of the producers and the injection rate of INJ1,
every 400 days for 4000 days, and the change in bulk impedance since the start at day
2000 and day 4000. The simulator is OPM Flow followed by a rock-physics model (flow_rock),
and the seismic data are wavelet compressed before they are assimilated.

Run from this folder:

    python ../make_data.py    # the synthetic observations, data.csv and var.csv
    python run_assim.py       # the assimilation
    python ../plot_res.py     # data misfit, and the forecasts against the data
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
