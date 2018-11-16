#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Al dummy to callibrate density. Corrected reconstruction shoud show the density of aluminum = 2.7 g/cm3
"""
#%% Imports:

from flexdata import display

from flextomo import project

from flexcalc import process

#%% Compute spectrum:

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

# Compute the spectrum of the system using data @ path:
# It is assumed that the data provided has a sample made of a single material (see 'compound' parameter)

energy, spectrum = process.data_to_spectrum(path, compound = 'Al', density = 2.7, energy_bin = 100)

#%% Read data, reconstruct uncorrected:

# Read:
proj, meta = process.process_flex(path, sample = 2, skip = 2)

# Reconstruct:
vol = project.init_volume(proj)

project.FDK(proj, vol, meta['geometry'])

display.display_slice(vol, title = 'Uncorrected FDK')

a,b = process.histogram(vol, rng = [-0.05, 0.15])

#%% Calibrate: 

proj = process.equivalent_density(proj, meta, energy, spectrum, compound = 'Al', density = 2.7) 

#%% Reconstruct corrected:

vol = project.init_volume(proj)

project.FDK(proj, vol, meta['geometry'])

display.display_slice(vol, title = 'Corrected FDK')

a,b = process.histogram(vol, rng = [-1, 4])
        
