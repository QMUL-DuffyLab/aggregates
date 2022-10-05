Simulating different models of non-photochemical quenching (NPQ) in LHCII aggregates, and whether the differences make the models distinguishable from experimentally-measurable quantities.

The heavy lifting is done in Fortran with OpenMPI so you will need a Fortran compiler and OpenMPI installed.
Python 3.9 or newer also required for a couple of useful argparse things.
The Python code will compile the Fortran for you and the actual compiler shouldn't matter, it's all F2008 conforming (at least I think so!).

run `python main.py args` to run a sweep across all fluences / excitation densities.
Required parameters are `-m` (model), `-l` (lattice), `-q` (quencher density).
Other parameters have defaults given in `defaults.py`; you can check other arguments (and their defaults) with `python main.py -h`.

After each simulation a two-step fit is done: first the tail is fit to get the decay rate(s), then a reconvolution fit with those rate(s) fixed to get the amplitude(s) and IRF shift.
By default this is done for one exponential and two exponentials.
To redo these fits first change whatever needs changing, then run the same command as before but with the `--fit_only` option.
