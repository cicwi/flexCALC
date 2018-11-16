#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simulation including phase-contrast effect.
We will simulate conditions close to micro-CT of a sea shell.
"""
#%% Imports

from flexdata import io
from flexdata import display

from flextomo import project
from flextomo import phantom

from flexcalc import spectrum
from flexcalc import resolution

import numpy

#%% Create volume and forward project:
    
# Initialize images:    
x = 1                 #  resolution multiplier
h = 512 * x           # volume size

vol = numpy.zeros([1, h, h], dtype = 'float32')
proj = numpy.zeros([1, 361, h], dtype = 'float32')

# Define a simple projection geometry:
src2obj = 100     # mm
det2obj = 100     # mm   
det_pixel = 0.001 / x # mm (1 micron)

geometry = io.init_geometry(src2obj, det2obj, det_pixel, [0, 360])

# Create phantom (150 micron wide, 15 micron wall thickness):
vol = phantom.sphere(vol.shape, geometry, 0.08)     
vol -= phantom.sphere(vol.shape, geometry, 0.07)     

# Show:
display.display_slice(vol, title = 'Phantom')

# Project:
project.forwardproject(proj, vol, geometry)

#%%
# Get the material refraction index:
c = spectrum.find_nist_name('Calcium Carbonate')    
rho = c['density'] / 10

energy = 30 # KeV
n = spectrum.material_refraction(energy, 'CaCO3', rho)

#%% Proper Fresnel propagation for phase-contrast:
   
# Create Contrast Transfer Functions for phase contrast effect and detector blurring    
phase_ctf = resolution.get_ctf(proj.shape[::2], 'fresnel', [det_pixel, energy, src2obj, det2obj])

sigma = det_pixel 
phase_ctf *= resolution.get_ctf(proj.shape[::2], 'gaussian', [det_pixel, sigma * 1])

# Electro-magnetic field image:
proj_i = numpy.exp(-proj * n )


# Field intensity:
proj_i = resolution.apply_ctf(proj_i, phase_ctf) ** 2

display.display_slice(proj_i, title = 'Projections (phase contrast)')

#%% Reconstruct with phase contrast:
    
vol_rec = numpy.zeros_like(vol)

project.FDK(-numpy.log(proj_i), vol_rec, geometry)
display.display_slice(vol_rec, title = 'FDK')  
    
#%% Invertion of phase contrast based on dual-CTF model:
    
# Propagator (Dual CTF):
alpha = numpy.imag(n) / numpy.real(n)
dual_ctf = resolution.get_ctf(proj.shape[::2], 'dual_ctf', [det_pixel, energy, src2obj, det2obj, alpha])
dual_ctf *= resolution.get_ctf(proj.shape[::2], 'gaussian', [det_pixel, sigma])

# Use inverse convolution to solve for blurring and pci
proj_inv = resolution.deapply_ctf(proj_i, dual_ctf, epsilon = 0.1)

# Depending on epsilon there is some lof frequency bias introduced...
proj_inv /= proj_inv.max()

display.display_slice(proj_inv, title = 'Inverted phase contrast')   

# Reconstruct:
vol_rec = numpy.zeros_like(vol)
project.FDK(-numpy.log(proj_inv), vol_rec, geometry)
display.display_slice(vol_rec, title = 'FDK')   
