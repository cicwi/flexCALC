#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test finding the rotation centre example.
"""
#%% Imports:
from flexdata import display
from flexdata import geometry

from flextomo import phantom
from flextomo import projector
from flexcalc import process
import numpy

#%% Simulate data with shifted rotation axis
    
# Initialize images:    
proj = numpy.zeros([1, 64, 128], dtype = 'float32')

# Define a simple projection geometry:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.1, ang_range = [0, 360])
geom['axs_tan'] = -0.123

# Create phantom and project into proj:
vol = phantom.abstract_nudes([1, 128, 128], geom, 6)
display.slice(vol, title = 'Phantom')

# Forward project:
projector.forwardproject(proj, vol, geom)
display.slice(proj, title = 'Sinogram')

#%% Use optimize_rotation_center:

# Unmodified geometry:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.1, ang_range = [0, 360])

# Optimization:    
process.optimize_rotation_center(proj, geom, subscale = 4, preview = True)

#%% Reconstruct:

vol = projector.init_volume(proj)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK Corrected')