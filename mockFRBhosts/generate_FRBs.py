#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions to generate intrinsic pulse widths.

Created on Tue May  9 17:47:00 2023

@author: jjahns
"""
import types
import numpy as np

from scipy.stats import lognorm

from frbpoppy.w_dists import calc_w_arr
from frbpoppy import CosmicPopulation, Survey, SurveyPopulation


def lognorm_mu(mu, sigma, w_min=0., shape=1, z=0):
    """Generate a sample from a lognormal distribution.

    Use mu and sigma instead of mean and stddev, which frbpoppy uses.
    """
    draw_min = lognorm.cdf(w_min, s=sigma, loc=0, scale=mu)

    # Draw from uniform distribution and invert it. Note that scipys
    # lognorm is differently defined from numpy.randoms.
    rng = np.random.default_rng()
    uni = rng.uniform(low=draw_min, high=1., size=shape)
    w_int = lognorm.ppf(uni, s=sigma, loc=0, scale=mu).astype(np.float32)
    w_arr = calc_w_arr(w_int, z=z)
    return w_int, w_arr


def gen_w(self):
    """Generate pulse widths [ms].

    Modified to give redshift to function.
    """
    shape = self.w_shape()
    z = self.frbs.z
    self.frbs.w_int, self.frbs.w_arr = self.w_func(shape, z)

    # From combined distribution inputs
    if self._transpose_w:
        self.frbs.w_int = self.frbs.w_int.T
        self.frbs.w_arr = self.frbs.w_arr.T


def generate_frbs(survey_model, beam_model='gaussian', z_model='sfr', n_srcs=1e7, z_max=1.5,
                  bol_lum_low=1e40, specif_lum_high=1e40, w_min=0.):
    """Generate a number of FRBs.

    Most values are from  the build in "optimal" population.
    This is just a convenience function to access the parameters we need.

    spec_lums in erg/s/Hz
    """
    spectral_index = -0.65
    # Generate an observed FRB population
    if isinstance(survey_model, str):
        survey = Survey(survey_model)
    else:
        survey = survey_model
    survey.set_beam(model=beam_model)
    ra_min, ra_max, dec_min, dec_max = survey.ra_min, survey.ra_max, survey.dec_min, survey.dec_max

    cosmic_pop = CosmicPopulation(n_srcs)
    cosmic_pop.set_dist(model=z_model, z_max=z_max)  # , alpha=-2.2)
    cosmic_pop.set_direction(min_ra=ra_min, max_ra=ra_max, min_dec=dec_min, max_dec=dec_max)
    cosmic_pop.set_dm(mw=False, igm=False, host=False)   # We don't care about DM

    # Emission frequency range.
    freq_low, freq_high = 100e6, 50e9
    cosmic_pop.set_emission_range(low=freq_low, high=freq_high)  # Default frequency

    # Luminosity function.
    freq_factor = (freq_high**(1+spectral_index) - freq_low**(1+spectral_index)
                   ) / 1.3e9**spectral_index
    luminosity_low = bol_lum_low  # bol_lum_low * freq_factor
    luminosity_high = specif_lum_high * freq_factor
    cosmic_pop.set_lum(model='powerlaw', low=luminosity_low, high=luminosity_high, power=-1.05)

    # Use our own function for the widths.
    cosmic_pop.w_shape = lambda: cosmic_pop.n_srcs
    cosmic_pop.w_func = lambda shape, z: lognorm_mu(mu=5.49, sigma=np.log(2.46), w_min=w_min,
                                                    shape=shape, z=z)

    # To change the funtion in the Class we need this "types" from
    # stackoverflow. Would be good to put it in frbpoppy.
    # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
    cosmic_pop.gen_w = types.MethodType(gen_w, cosmic_pop)

    cosmic_pop.set_si(model='constant', value=spectral_index)
    cosmic_pop.generate()

    survey_pop = SurveyPopulation(cosmic_pop, survey)  # , scat=True, scin=True) These take long

    return cosmic_pop, survey_pop
