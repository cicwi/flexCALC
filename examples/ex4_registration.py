#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Register two datasets and use them in a combined PWLS-type reconstruction.
"""
#%% Imports

from flexdata import geometry
from flexdata import display

from flextomo import projector
from flextomo import phantom

from flexcalc import analyse

import transforms3d
import numpy

#%% Create volume and forward project into two orhogonal scans:
# Define a simple projection geometry:
geom_a = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.1, ang_range = [0, 360])
geom_b = geom_a.copy()
geom_b.parameters['axs_pitch'] = 90

# Create phantom and project into proj:
#vol = phantom.random_spheroids([128, 128, 128], geom_a, number = 10)
vol = phantom.spheroid([128,]*3, geom_a, 3, 1, 0.5)
display.slice(vol, title = 'Phantom')

#%%

T, R = analyse.moments_orientation(vol)
#geom_reg = process.transform_to_geometry(R, T, geom_a)
transforms3d.euler.mat2euler(R.T, axes = 'sxyz')

#%%
# Initialize images:    
proj_a = numpy.zeros([128, 16, 128], dtype = 'float32')
proj_b = numpy.zeros([128, 16, 128], dtype = 'float32')

# Forward project:
projector.forwardproject(proj_a, vol, geom_a)
projector.forwardproject(proj_b, vol, geom_b)

display.slice(proj_a, dim = 1, title = 'Proj A')
display.slice(proj_b, dim = 1, title = 'Proj B')

#%%







#%% Preview reconstructions:
geom1 = meta1['geometry']
geom2 = meta2['geometry']

# First volume:
vol1 = project.init_volume(proj1, geom1)
project.FDK(proj1, vol1, geom1)

# Second volume:
vol2 = project.init_volume(proj2, geom2)
project.FDK(proj2, vol2, geom2)

display.max_projection(vol1, dim = 1,  title = 'Volume 1')
display.max_projection(vol2, dim = 1, title = 'Volume 2')

#%% Register:

# Get rotation and translation based on two datasets:
R, T = process.register_astra_geometry(proj1, proj2, geom1, geom2)    

# Convert to a new geometry:
geom2_reg = process.transform_to_geometry(R, T, geom2)
    
# Show the result of registration:
vol2 *= 0
project.FDK(proj2, vol2, geom2_reg)
display.max_projection(vol2, dim = 1,  title = 'Volume 2 - registered')

#%% FDK doens't compute the correct intensity in rotated system of coordinates. Here we will correct for that:

process.equalize_intensity(vol1, vol2)
display.max_projection(vol2, dim = 1,  title = 'Volume 2 - registered')

#%% Use Multi-axis PWLS:

vol1 *= 0

project.settings['block_number'] = 20
project.MULTI_PWLS([proj1, proj2], vol1, [geom1, geom2_reg], iterations = 20)
    
display.max_projection(vol1, dim = 1,  title = 'Volume combined PWLS')