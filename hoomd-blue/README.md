# HOOMD-blue scripts

- `hpmc.py`: This script runs a NVT ensemble Monte Carlo simulation for hard-spheres in 3D. It includes the `QuickCompress` method to compress an initial system to a target packing fraction as well as a move updater for adjusting the acceptance ratio.
- `hs-npt.py`: This script contains the necessary code to run a NPT ensemble Monte Carlo simulation of pure hard-spheres in 2 dimensions. The extension to 3D is straightforward, because the only thing that needs to be changed is how the initial configuration is created.
- `analysis.py`: A simple script to plot the change in density for the NPT simulation.

## Dependencies

The Python dependencies for these scripts are:

- `hoomd`
- `numpy`
- `matplotlib`
- `gsd`
