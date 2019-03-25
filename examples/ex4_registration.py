#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SImulate two datasets corresponsing to different orientations of the object.
Register them and use in a multi-axis reconstruction.
"""
#%% Imports

from flexdata import geometry
from flexdata import display

from flextomo import projector
from flextomo import phantom

from flexcalc import process

import numpy

#%% Create volume and forward project into two orhogonal scans:

# Define a simple projection geometry:
geom_a = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.1, ang_range = [0, 360])

# The saceon dataset willl be rotated and translated:
geom_b = geom_a.copy()
geom_b.parameters['vol_rot'] = [0.0,0.0,90.0]
geom_b.parameters['vol_tra'] = [0, -0.5, 0]

# Create a phantom:
vol = phantom.random_spheroids([100, 100, 100], geom_a, 10)

display.slice(vol, dim = 1, title = 'Phantom')

#%% Forward model:

# Initialize images:    
proj_a = numpy.zeros([128, 32, 128], dtype = 'float32')
proj_b = numpy.zeros([128, 32, 128], dtype = 'float32')

# Forward project:
projector.forwardproject(proj_a, vol, geom_a)
projector.forwardproject(proj_b, vol, geom_b)

display.slice(proj_a, dim = 1, title = 'Proj A')
display.slice(proj_b, dim = 1, title = 'Proj B')

#%% Preview reconstructions:
geom_b = geom_a.copy()

# First volume:
vola = projector.init_volume(proj_a)
projector.FDK(proj_a, vola, geom_a)

# Second volume:
volb = projector.init_volume(proj_b)
projector.FDK(proj_b, volb, geom_b)

display.projection(vola, dim = 1,  title = 'Volume A')
display.projection(volb, dim = 1, title = 'Volume B')

#%% Register:
R, T = process.register_volumes(vola, volb, subsamp = 1, use_moments = True, use_CG = True)

# Convert to a new geometry:
geom_r = geom_a.copy()
geom_r.from_matrix(R, T)
    
# Show the result of registration:
volb = projector.init_volume(proj_b)
projector.FDK(proj_b, volb, geom_r)
process.equalize_intensity(vola, volb)

display.projection(vola, dim = 1, title = 'Volume A')
display.projection(volb, dim = 1, title = 'Volume B (registered)')

#%% Use Multi-axis PWLS:

vola = projector.init_volume(proj_a)

projector.settings.subsets = 10
projector.PWLS([proj_a, proj_b], vola, [geom_a, geom_r], iterations = 10)

display.projection(vola, dim = 1,  title = 'Volume combined PWLS')