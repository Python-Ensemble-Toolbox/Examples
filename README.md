# Examples
Folder containing example cases for PIPT and POPT

**Data Assimilation Cases**
- LinearModel   : Fast test for data assimilation methods that does not require an external simulator.
- 3dBox         : Data assimilation of a 3D reservoir with 3 injectors and 3 producers. Four different fidelity levels. Set up with ES. The tiny case can run on Eclipse/Windows (see `src/assimilation/3dBox/README.md`).
- SPE11b        : Data assimilation using synthetic impedance data for the SPE11b case

**Optimization Cases**
- 3Spot         : Simple case to test optimization methods that runs in less than one minute. The test requires OPM or Eclipse. 
- 3SpotRobust   : Simple case to test optimization methods including multiple geo-models.
- TinyBoxEcalc  : Optimize the well pressures of TinyBox (oil, gas and water) for NPV, including the cost of CO2 emissions computed by eCalc.
- Rosenbrock    : Minimize the Rosenbrock function (https://en.wikipedia.org/wiki/Rosenbrock_function) in any dimension
- Quadratic     : Minimize a quadratic function on the form ||x-b||<sub>A</sub><sup>2</sup>, in any dimension
 
**Installation**

Inside the Examples folder, run

    python3 -m pip install -e .

- Note: all examples also require installation of PET to run (see the PET folder for instructions).
- The dot is needed to point to the current directory.
- The -e option installs PET such that changes to it take effect immediately (without re-installation).
