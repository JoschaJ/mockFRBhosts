# mockFRBhosts

The goal of `mockFRBhosts` is to understand how limiting the visibility and follow-up of FRB host
galaxies is for their cosmological and astrophysical applications. It was released with a paper
on the same topic detailed below. The package provides a function that uses `frbpoppy` to generate
a population of FRBs for a given radio telescope. For each FRB, a host galaxy is drawn from a data
base that was generated with `GALFORM`. The galaxies' magnitudes in different photometric surveys
are calculated and the number of bands in which they are detected. Functions are also provided to
calculate the follow-up time in a 10-m optical telescope required to do photometry or spectroscopy.
The package also provides a simple interface to Bayesian inference methods via MCMC simulations
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
All the core functions are in mockFRBhosts. If you want to use the package, you can start with the
example notebook, which is a simplified case for a single survey. The more sophisticated ways of
reproducing all figures published in the paper are contained in paper_notebooks.

## Installation
`mockFRBhosts` depends on the following packages
```
python >= 3.6
numpy <= 1.22
scipy
pandas
matplotlib
seaborn
astropy
corner
FRB
frbpoppy
ne2001
```
If you want to use the MCMC simulations, you further need:
```
pymc3
numba
theano
```
### Anaconda
If you use Anaconda, you can either create a fresh environment or install it in an existing environment.
To create a new environment with all packages run
```
conda create -c conda-forge --name mockFRBs python=3.10 "numpy<=1.22" theano pymc3 scipy pandas matplotlib seaborn astropy corner jupyterlab numba
```
To install them within the currently active environment use
```
conda install -c conda-forge "numpy<=1.22" theano pymc3 scipy pandas matplotlib seaborn astropy corner jupyterlab numba
```

### FRB specific packages
The following FRB specific software needs to be installed from git. Go to the location where
you want to install these (e.g. ~/git/) and run the following code block.
```
git clone https://github.com/FRBs/ne2001
cd ne2001
pip install .
cd ..

git clone https://github.com/davidgardenier/frbpoppy
cd frbpoppy
sed -i -e 's/2.20.0./2.20.0/g' setup.py  # newer setuptools versions don't allow the dot at the end
python setup.py develop
cd ..

git clone https://github.com/FRBs/FRB
cd FRB
python setup.py develop
cd ..

git clone https://github.com/JoschaJ/mockFRBhosts
cd mockFRBhosts
python setup.py develop
```


## Cite
Will be added when published.