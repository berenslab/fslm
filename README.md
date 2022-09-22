# Feature Selection through Likelihood Marginalization (FSLM)

This repository contains an updated, refactored version of FSLM and some utils to improve the usability. The modules were structured and obsolete pieces of code were removed. If you are just looking to reproduce the results from [Efficient identification of informative features in
simulation-based inference](https://openreview.net/forum?id=AYQI3rlp9tW), see [here](https://github.com/berenslab/fslm_repo).

`fslm` is a PyTorch package for efficient feature selection in Simulation-based Inference (SBI). It builds on top of the already existing [`sbi` library](https://github.com/mackelab/sbi) and adds additional functionality for the purpose of feature selection. Along with the main code for `fslm`, this repository also contains different utilities to conduct experiments. This includes a simple database to store and retrieve samples/posteriors/metadata, a small task engine to distribute function calls across a process pool, a set of tools to simulate and extract features from Hodgkin-Huxley (HH) models, solutions for using SBI in conjunction with NaN data, some toy examples, as well as functions and metrics to analyse and plot the results.

## Installation

`fslm` requires Python 3.6 or higher. With `conda` installed on the system, an environment for
installing `fslm` can be created as follows:
```commandline
# Create an environment for fslm (indicate Python 3.6 or higher); activate it
$ conda create -n fslm_env python=3.8 && conda activate fslm_env
```

Then create a clone of the repository on your local system with:
```commandline
$ pip clone https://github.com/berenslab/fslm fslm
$ cd fslm
```

`fslm` along with the included utils can then be installed via the provided `setup.py` by invoking:
```commandline
$ pip install .
```

To test the installation, drop into a python prompt and run
```python
from experiment_utils.toymodels import LGM
lgm = LGM()
lgm.sample_joint()
```
