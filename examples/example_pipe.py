#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:34:43 2018

@author: kostenko

Test of the flexPipe. We will load the 6 tiles of the ivory ball (downsampled),
merge detector tiles, apply beam-hardening, FDK, make STL, compress the images.
"""
#%% Imports

from flexcalc import pipe
     
#%% Create a Pipe:

# Pipe will use a particular folder for swap files:
pipe = pipe.Pipe(memmap_path = '/export/scratch3/kostenko/flexbox_scratch/')

#pipe.ignore_warnings(True)

# Pre-processing:
binning = 2                                                           # Use binning to accelerate the test run 
pipe.process_flex(sampling = binning, skip = binning)  # Pre-process the dataset

pipe.read_all_meta(sampling = binning)                                 # Load all metadata to compute tile positioning later

pipe.bh_correction(path = '../', compound ='Al', density = 2.7)        # Apply beam hardening and density callibration based on aluminum

pipe.display(dim = 1)                                                  # Show the result of BH correction

# Merge:
pipe.merge_detectors(memmap = True)                                    # Merge datasets into a continuous dataset when the source position is constant 

pipe.display(dim = 1)                                                  # Show the result of merging the data 
pipe.ramp(width = [10, 10], dim = 2, mode = 'linear')                  # To avoid border artifacts - pad and ramp (not really needed with this data)

# Find rotation:
pipe.find_rotation(subscale = 4)                                       # Correct for deviations in rotation centre if needed 

# Reconstruct:
pipe.FDK()                                                             # Reconstruct using FDK 

pipe.memmap()                                                          # Move RAM to memmap to free up memory

pipe.display(dim = 0, display_type = 'slice')

#pipe.merge_volume(memmap = True)                                      # Not really needed here since FDK will output a single volume. Use it when a stack of volumes is produced

# Post-processing:
pipe.auto_crop()                                                       # Cut the air out 

#pipe.marker_normalization(2.7)                                        # There are no density markers in this data unfortunately...
pipe.soft_threshold(mode = 'otsu')                                     # Kill small values. 'otsu' mode is more agressive than the 'histogram' mode.

pipe.display(dim = 1, display_type = 'max_projection')                 # Show
pipe.display(dim = 2, display_type = 'max_projection')

# Write to disk:
pipe.history_to_meta()                                                 # Create a history record
pipe.write_flexray(folder = '../fdk_nobin', dim=0, compress='zip')     # Save slices with zip compression

# Downsample and write to disk:
pipe.bin()                                                             # Produce a binned volume with integer 8-bit precision as a preview
pipe.cast2type(dtype = 'uint8', bounds = [0, 50])
pipe.history_to_meta()
pipe.write_flexray(folder = '../fdk_bin', dim=0, compress='zip')

# Make and STL preview and write to disk:
pipe.bin()                                                             # Make an STL file 
pipe.make_stl(file = '../surface.stl', preview = True)

#%% Add data to processing que and run:

pipe.add_data('/ufs/ciacc/flexbox/test_data/ivory/t*')                 # Where the data is... 

pipe.run()                                                             # Run Lola Run! 

pipe.flush()