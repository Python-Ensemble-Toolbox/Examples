# 3D box case

The case consists of four fidelity levels you can run.

- large: 60x60x5 grid cells
- medium: 60x60x3 grid cells
- small: 40x20x5 grid cells
- tiny: 10x10x2 grid cells

- flowrock: based on the large model, but including bulk impedance data and wavelet compression. The case also demonstrates use of the toml input formal. 

All examples have the same well pattern and the same boundary/initial conditions. Run the setup.py file in the data folder to generate data from the relevant case.  
The true (**data/**) and **tiny/** cases can run on Eclipse (current default) and have been tested on Windows. To use OPM Flow instead, set `simulator = 'flow'` in `tiny/run_script.py` (and `SIMULATOR` in `data/setup.py` when running it directly).