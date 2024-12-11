#!/usr/bin/env python
import sys
import os
import glob
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import astropy.table as at

from numpy.lib.recfunctions import append_fields
from scipy.stats import linregress
import pandas as pd
from collections import OrderedDict

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax import random
import jax
import jax.numpy as jnp

numpyro.set_host_device_count(4)
jax.config.update('jax_enable_x64',True)

def get_data(file_loc='data'):
    suffix = '_abmag'
    ilaph_version = '5'
    magsys = 'ABmag'
    ref = 'FMAG'  # which photometry package should be used to compute zeropoints
    mintime = 0.7 # mininum exposure length to consider for computing zeropoints

    marker    = ['o',      'd',     '*']       # markers to use for each standard in plots
    use_stars = ['GD-153', 'G191B2B', 'GD-71'] # what stars are to be used to get zeropoints


    # cut out some fields we do not need to make indexing the data frame a little easier
    dref = 'ERRMAG'
    drop_fields = ['X', 'Y', 'BCKGRMS', 'SKY', 'FITS-FILE']

    mag_table = OrderedDict() # stores combined magnitudes and zeropoints in each passband

    mags1   = at.Table.read('{}/cycle_22_20_obs.txt'.format(file_loc), format='ascii')
    
    mags2   = at.Table.read('{}/cycle_25_obs.txt'.format(file_loc,ilaph_version), format='ascii')

    all_mags = at.vstack([mags1,mags2])
    all_mags[ref] -= 30.
    mask = (all_mags[dref] < 0.5) & (np.abs(all_mags[ref]) < 50) & (all_mags['EXPTIME'] >= mintime)
    nbad = len(all_mags[~mask])

    all_mags = all_mags[mask]
    all_mags.rename_column('OBJECT-NAME','objID')
    all_mags.rename_column('FILTER','pb')

    cycle_flag = [ 0 if  x>58000 else 1 if x > 56700 else 2 for x in all_mags['MJD'] ]
    cycle_flag = np.array(cycle_flag)
    all_mags['cycle'] = cycle_flag
    
    for pb in ['F275W','F336W','F475W','F625W','F775W','F160W']:
        mask = (all_mags['pb'] == pb)
        mag_table[pb] = all_mags[mask].to_pandas()
    # init some structure to store the results for each passband
    return mag_table





def select_stars(tab,include_these=[],file_loc=''):

    for _,pb in enumerate(tab):

        sample_names = tab[pb]['objID']

        name_map = at.Table.read('name_map.dat', names=['old','new'], format='ascii.no_header')


        name_map = dict(zip(name_map['old'], name_map['new']))

        for i, n in enumerate(sample_names):
            if n.startswith('SDSS-J'):
                n = n.split('.')[0].replace('-','')
            elif n.startswith('WD'):
                n = n.replace('WD-','wd').split('-')[0].split('+')[0]
            else:
                pass
            n = n.lower().replace('-','')
            n = name_map.get(n, n)

            if n[0]=='a':
                n = n[:-2]
            sample_names.at[i]=n
        if len(include_these)>0:
            tab[pb]= tab[pb][sample_names.isin(include_these)]
    return tab




def load_phot(file_loc,names):

    mag_table=get_data(file_loc)


    mag_table = select_stars(mag_table,include_these=names,file_loc=file_loc)
    result_table = OrderedDict()


        # drop some fields we do not need from the results
    derop_fields = ['hdi_3%','hdi_97%','mcse_mean','mcse_sd','ess_bulk','ess_tail','r_hat']
        # keep a track of all_objects
    all_objects = set()
        # and variable names
        #var_names = ['zeropoint', 'c20_offset', 'sig_intrinsic', 'nu']
    var_names = ['zeropoint', 'sig_intrinsic', 'nu']
    nvar = len(var_names)

    use_stars = ['gd153', 'g191b2b', 'gd71']
    ref = 'FMAG'  # which photometry package should be used to compute zeropoints
    dref = 'ERRMAG'




    sample_names = np.array([])
    standard_names = np.array([])

    max_stand=0
    max_samp = 0


    for i, pb in enumerate(mag_table):



            all_mags = mag_table[pb]
            mask = all_mags['objID'].isin(use_stars)


            sample_mags   = all_mags.copy()

            # the standards are "special" - they have apparent magnitudes from a
            # model and are used to set the zeropoint for everything else
            standard_mags = all_mags[mask].copy()

            # what are the unique stars
            standards  = standard_mags['objID'].unique()

            standard_names= np.unique(np.append(standard_names,standards))

    

            samples  = sample_mags['objID'].unique()

            sample_names = np.unique(np.append(sample_names,samples))

            max_stand = np.max([max_stand,len(standard_mags)])

            max_samp= np.max([max_samp,len(sample_mags)])




    samp_mags = np.zeros((6,max_samp))+1000
    samp_ids = np.zeros((6,max_samp)).astype(int)+1000
    samp_err = np.zeros((6,max_samp))+1000

    stand_mags = np.zeros((6,max_stand))+1000
    stand_ids = np.zeros((6,max_stand)).astype(int)+1000
    stand_err = np.zeros((6,max_stand))+1000


    cycle_samp_ids = np.zeros((6,max_samp)) + 10000
    cycle_stand_ids = np.zeros((6,max_stand)) + 10000

    nstandards = len(standard_names)
    standard_ind = list(range(nstandards))
    standard_map = dict(zip(standard_names, standard_ind))

    
    nsamples= len(sample_names)
    sample_ind = list(range(nsamples))
    sample_map = dict(zip(names, sample_ind))



    zpt_est= np.zeros(6)

    for i, pb in enumerate(mag_table):
    

            all_mags = mag_table[pb]
            mask = all_mags['objID'].isin(use_stars)


            sample_mags   = all_mags.copy()


            standard_mags = all_mags[mask].copy()


    

            stand_ids[i,:len(standard_mags)] = [standard_map[x] for x in standard_mags['objID']]
            stand_mags[i,:len(standard_mags)] = standard_mags[ref]
            stand_err[i,:len(standard_mags)] = standard_mags[dref]

            samp_ids[i,:len(sample_mags)]=[sample_map[x] for x in sample_mags['objID']]
            samp_mags[i,:len(sample_mags)] = sample_mags[ref]
            samp_err[i,:len(sample_mags)] = sample_mags[dref]




            cycle_stand_ids[i,:len(standard_mags)] =standard_mags['cycle']
            cycle_samp_ids[i,:len(sample_mags)] =sample_mags['cycle']

    


    return jnp.asarray(stand_ids),jnp.asarray(samp_ids),jnp.asarray(stand_mags),jnp.asarray(samp_mags),jnp.asarray(stand_err),jnp.asarray(samp_err),jnp.asarray(cycle_stand_ids),jnp.asarray(cycle_samp_ids)
