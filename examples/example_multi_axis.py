#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model two datasets and use them in a combined PWLS-type reconstruction.
"""
#%% Imports

from flexdata import display

from flextomo import project

from flexcalc import process

#%% Read data:

path1 = '/ufs/ciacc/flexbox/test_data/pcb/pcb_horizontal_filt/'
path2 = '/ufs/ciacc/flexbox/test_data/pcb/pcb_vertical_filt/'

bining = 2

proj1, meta1 = process.process_flex(path1, sample = bining, skip = bining) 
proj2, meta2 = process.process_flex(path2, sample = bining, skip = bining) 

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