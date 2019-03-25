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

# Geomtery:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.2, ang_range = [0, 360])

# This is our phantom:
vol = phantom.cuboid([1, 128, 128], geom, 8,8,8)  
display.slice(vol , title = 'Phantom') 

# Spectrum of the scanner:   
kv = 90
filtr = {'material':'Cu', 'density':8, 'thickness':0.1}
detector = {'material':'Si', 'density':5, 'thickness':1} 
E, S = model.effective_spectrum(kv = kv, filtr = filtr, detector = detector)  

# Display:
display.plot(E, S, title ='Effective spectrum')   

# Materials list that corresponds to the number of labels in our phantom:  
mats = [{'material':'Al', 'density':2.7},]
        
# Sinogram that will be simultated:
counts = numpy.zeros([1, 128, 128], dtype = 'float32')

# Simulate:
model.forward_spectral(vol, counts, geom, mats, E, S, n_phot = 1e6)
proj = -numpy.log(counts)

# Display:
display.slice(proj, title = 'Modelled sinogram')  

#%% Reconstruct:
    
vol_rec = numpy.zeros_like(vol)
projector.FDK(proj, vol_rec, geom)

display.slice(vol_rec, title = 'Uncorrected FDK')
display.plot(vol_rec[0, 64], title = 'Crossection')       

#%% Estimate system spectrum:

print('Callibrating spectrum...')    
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