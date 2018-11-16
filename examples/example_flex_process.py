#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a standard CT scan. Use flexCALC.process to preprocess the data.
"""
#%% Imports

from flexcalc import process

from flexdata import display

from flextomo import project

#%% Read data:
    
path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

proj, meta = process.process_flex(path, skip = 4, sample = 4)
 
#%% FDK:

vol = project.init_volume(proj)
project.FDK(proj, vol, meta['geometry'])

display.display_slice(vol, bounds = [], title = 'FDK')
