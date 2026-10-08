# 3D box

A reservoir with oil, water and gas, three water injectors and three producers, at four
grid sizes. The estimated parameter is the log-permeability, and the data are the oil and
water rates of the producers and the injection rate of INJ1, every 400 days for 4000 days.

- tiny: 10x10x2 grid cells, ES-MDA
- small: 40x20x5 grid cells, ES
- medium: 60x60x3 grid cells, ES
- large: 60x60x5 grid cells, ES
- flowrock: the large grid with time-lapse seismic (bulk impedance) as well, from a
  rock-physics model, wavelet compressed before it is assimilated; ES

All cases use the same wells, schedule and prior, and the simulator is OPM Flow. Each case
folder holds its own grid, template (RUNFILE.mako), configuration and synthetic data.

Run from a case folder:

```sh
python ../make_data.py    # the synthetic observations, data.csv and var.csv
python run_assim.py       # the assimilation
python ../plot_res.py     # data misfit, and the forecasts against the data
```
