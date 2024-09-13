#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate polychromatic data, apply beam hardening correction.
"""
#%% Imports

from flexdata import geometry  # Define geometry and display
from flexdata import display

from flextomo import phantom   # Simulate
from flextomo import model
from flextomo import projector # Reconstruct

from flexcalc import analyze
from flexcalc import process

import numpy

#%% Short version of the spectral modeling:

path = '/export/scratch2/kostenko/archive/Natrualis/al_70kv_1mmbrass/'
proj, geom = process.process_flex(path, sample = 2, skip = 2)

display.slice(proj, dim = 0, bounds = [], title = 'Sinogram')

#%% Reconstruct uncorrected:

vol = projector.init_volume(proj)
#vol = numpy.zeros((220, 550, 550), dtype = 'float32')
#projector.settings.subsets = 10
#projector.settings.bounds = (0, 10)
#projector.settings.preview = True
#projector.SIRT(proj, vol, geom, iterations = 20)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'Uncorrected')

#%% Reconstruct:

vol_rec = numpy.zeros_like(vol)
projector.FDK(proj, vol_rec, geom)

display.slice(vol_rec, title = 'Uncorrected FDK')
display.plot(vol_rec[0, 64], title = 'Cross section')

#%% Estimate system spectrum:

print('Calibrating spectrum...')
energy, spec = analyze.calibrate_spectrum(proj, vol, geom, compound = 'Al', density = 2.7)

#meta = {'energy':e, 'spectrum':s, 'description':geom.description}
#data.write_toml(os.path.join(path, 'spectrum.toml'), meta)

#%% Beam Hardening correction based on the estimated spectrum:

# Correct data:
proj_cor = process.equivalent_density(proj,  geom, energy, spec, compound = 'Al', density = 2.7)

# Reconstruct:
vol_rec = numpy.zeros_like(vol)
projector.FDK(proj_cor, vol_rec, geom)

# Display:
display.slice(vol_rec, title = 'Corrected FDK')
display.plot(vol_rec[0, 64])