#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize tilt of the detector.
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
proj = numpy.zeros([128, 128, 128], dtype = 'float32')

# Define a simple projection geometry:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.1, ang_range = [0, 360])
geom['det_roll'] = 1.45

# Create phantom and project into proj:
#vol = phantom.random_spheroids([128, 128, 128], geom, 8)
vol = phantom.cuboid([128, 128, 128], geom, 3,3,3)
display.slice(vol, title = 'Phantom')

# Forward project:
projector.forwardproject(proj, vol, geom)
display.slice(proj, dim = 1, title = 'Projection')

#%% Use optimize_rotation_center:

# Unmodified geometry:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.1, ang_range = [0, 360])

vol = projector.init_volume(proj)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [0, 2], title = 'FDK: uncorrected')

#%% Optimization:
vals = numpy.linspace(0., 3., 7)
process.optimize_modifier(vals, proj, geom, samp = [1, 1, 1], key = 'det_roll', metric = 'highpass')

#%% Reconstruct:

vol = projector.init_volume(proj)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [0, 2], title = 'FDK: corrected')