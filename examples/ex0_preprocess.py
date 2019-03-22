#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load, preprocess and reconstruct a FlexRay data.
"""

#%% Imports:

from flexdata import display
from flexdata import data

from flextomo import projector
from flexcalc import process

#%% Read (short version for FlexRay data)

path = 'D:\data\skull'
proj, geom = process.process_flex(path, sample = 4, skip = 4)

#%% Read (longer, more general approach)

# Read:
path = 'D:\data\skull'
dark = data.read_stack(path, 'di00', sample = 4)
flat = data.read_stack(path, 'io00', sample = 4)    
proj = data.read_stack(path, 'scan_', sample = 4, skip = 4)
geom = data.read_flexraylog(path)   

# Process:
proj = process.preprocess(proj, flat, dark, transpose = [1, 0, 2], updown = True)
display.slice(proj, dim = 0, bounds = [], title = 'Sinogram')

#%% Reconstruct uncorrected:

vol = projector.init_volume(proj)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK Corrected')

