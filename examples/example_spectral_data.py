#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Here we will load the spectral data from JINR.
"""

#%% Imports

from flexdata import io
from flexdata import geometry
from flexdata import array
from flexdata import display
from flextomo import project

import numpy

#%% Read the data 

path = '/ufs/ciacc/flexbox/JINR_data/projections/SPos0/Energy150000/'    
proj = io.read_stack(path, '00')

# Transpose for ASTRA compatibility:
proj = array.raw2astra(proj) 

# Get rid of funny values:
proj[(~numpy.isfinite(proj)) | (proj < 0.1)] = 1  
proj[proj > 1] = 1

# Da Holy LOG:
proj = -numpy.log(proj)

display.slice(proj, title = 'Sinogram')
display.slice(proj, dim = 1, title = 'Projection')

#%% Create geometry:

# We don't have a dedicated parser for JINR geometry description file.
# so we will simply make an ASTRA geometry by hand

'''
Mode = cone beam
Last angle = 360�
Pixel size = 0.055 mm
Centre of rotation = 1034.82
Source object distance = 120 mm
Source detector distance = 220 mm
Direction of rotation = counter-clockwise
Vertical centre = 203
Horizontal centre = 822
Tilt = 0�
Skew = 0�
Voxel size = 0.029145 mm
'''

src2obj = 120
det2obj = 220 - src2obj
img_pixel = 0.029145
det_pixel = 0.055
ang_range = [360, 0]
ang_count = 360

hrz_cntr = proj.shape[2] / 2
vrt_cntr = proj.shape[0] / 2

geom = geometry.circular(src2obj, det2obj, det_pixel, ang_range = ang_range)

# Source position:
geom['src_tan'] = (822 - hrz_cntr) * det_pixel
geom['src_ort'] = -(203 - vrt_cntr) * det_pixel

# Centre of rotation in mm:
geom['axs_tan'] = (1034.82 - hrz_cntr - geom['src_tan'] / det_pixel) * img_pixel + geom['src_tan']

# Recon 10 slices:
vol = numpy.zeros([10, 1200, 1200], dtype = 'float32')

project.FDK(proj, vol, geom)

display.projection(vol, dim = 0, bounds = [], title = 'FDK')

#%% Save geometry:

import os
    
io.write_astra(os.path.join(path, 'projection.geom'), proj.shape, geom)