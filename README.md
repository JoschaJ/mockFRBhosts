# mockFRBhosts

The goal of `mockFRBhosts` is to understand how limiting the visibility and follow-up of FRB host
galaxies is for their cosmological and astrophysical applications. It was released with a paper
on the same topic detailed below. The package provides a function that uses `frbpoppy` to generate
a population of FRBs for a given radio telescope. For each FRB a host galaxy is drawn from a data
base that was generated with `GALFORM`. The galaxies magnitude in different photometric surveys
is calculated and the number of bands in which it is detected. Functions are also provided to
calculate the follow-up time in a 10-m optical telescope required to do photometry or spectroscopy.
The package also provides a simple interface to bayesian inference methods via MCMC simulations
provided in the `FRB` package.

The core package is supplemented by notebooks and simulated data to reproduce all plots in the
paper.

## Structure
The package is structured as follows.
```
mockFRBhosts
├── mockFRBhosts
│   ├── __init__.py
│   ├── generate_FRBs.py
│   ├── observable.py
│   ├── mcmc_simulations.py
│   └── survey_overlaps.py
├── paper_notebooks
│   ├── generate_FRBs.ipynb
│   ├── host_follow_up.ipynb
│   ├── obseravable_all_telescopes.ipynb
│   ├── run_mcmcs.ipynb
│   ├── MCMC_plots.ipynb
│   └── survey_overlaps.py
├── Simulated_FRBs
│   ├── ...
├── Posteriors
│   ├── ...
├── example.ipynb
├── README.md
└── setup.py
```
All the core functions are in mockFRBhosts. If you want to use the package you can start with the
example notebook, which a simplified case for a single survey. The more sophisticated ways of
reproducing all figures published in the paper are contained in paper_notebooks.

## Installation

frbpoppy, FRB,

MCMC simulations : pymc3, numba, theano

## Cite