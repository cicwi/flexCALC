#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test finding the detector tilt example.
"""
#%% Imports:

import numpy

from flexdata import display

from flextomo import project

from flexcalc import process

#%% Read

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'
proj, geom = process.process_flex(path, sample = 4, skip = 4)

#%% Reconstruct uncorrected:

vol = project.init_volume(proj)
project.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK Corrected')

#%% Use optimize_rotation_center:

# Optimization by scanning -3..+3 degrees with 7 steps:
guess = process.optimize_modifier(numpy.linspace(-3, 3, 7), proj, geom, samp = [10, 1, 1], key = 'det_roll', preview = False)

#%% Reconstruct corrected:

vol = project.init_volume(proj)
project.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK Corrected')
