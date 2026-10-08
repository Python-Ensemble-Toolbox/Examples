# SPE11B: history matching of a CO2-storage case with time-lapse seismic

A 2D vertical section (83 x 58 cells) based on the SPE11B benchmark, where CO2 is
injected through two wells for 25 years. The data are the change in bulk impedance
since the start, after every five years of injection, and the estimated parameter is
the log-permeability, with a prior drawn separately for each of the seven rock types.

The deck was generated with pyopmspe11 (https://github.com/OPM/pyopmspe11). Unlike the
benchmark, the run starts injecting at once, without the 1000-year period that brings
the reservoir to equilibrium first, so a simulation takes about two minutes.

Run from this folder:

```sh
python make_data.py     # the prior ensemble and the synthetic observations
python run_assim.py     # the assimilation (about an hour)
python plot_res.py      # permeability fields and seismic data
```
