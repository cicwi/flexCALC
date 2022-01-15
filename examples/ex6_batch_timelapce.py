# -*- coding: utf-8 -*-
"""
Example of batch processing. 
There are three parts in this example: scheduling and runtime of input operations, scheduling and runtime of reconstruction and restoration and restart of a crashed process.

@author: alex
"""
#%% Imports

from flexcalc import pipeline
from flexcalc import process
from numpy import linspace

#%% Initialize and schedule (1):

P = pipeline.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = True)

# Load data:
path = '/ufs/ciacc/flexbox/timelapce/day*'

# Binning:
b = 2

# Raed data:
P.read_data(path, 'scan_', sampling = b)

# Apply flatfield and log:
P.flatlog(flats = 'io0', darks = 'di0', sample = b)

# Rotate 90 degrees using process.rotate:
P.generic(process.rotate, angle = -90, axis = 1)

# Display:
P.display('slice', dim = 1, title = 'Projections')

# Optimize detector roll:
P.optimize(linspace(-0.5, 0.5, 7), key = 'det_roll')

# Reconstruct:
P.FDK()

P.display('slice', dim = 2, title = 'FDK')

# Register volumes:
P.registration(subsamp=1)

P.display('slice', dim = 2, title = 'Volume')

# Reduce size and save data:
P.cast2type(dtype = 'uint8', bounds = [0.01, 1])

#P.autocrop()

P.write_data('./fdk', 'vol')

P.display('max_projection', dim = 2, title = 'Volume')

P.draw_nodes()

#%% Runtime

P.run()
