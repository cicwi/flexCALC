# -*- coding: utf-8 -*-
"""
Example of batch processing. 
There are three parts in this example: scheduling and runtime of input operations, scheduling and runtime of reconstruction and restoration and restart of a crashed process.

@author: alex
"""
#%% Imports

from flexcalc import batch
from flexcalc import process
from numpy import linspace

#%% Initialize and schedule (1):

lola = batch.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = True)

# Load data:
path = '/ufs/ciacc/flexbox/timelapce/day*'

# Binning:
b = 2

# Raed data:
lola.read_data(path, 'scan_', sampling = b, flipdim = True)

# Apply flatfield and log:
lola.flatlog(flats = 'io', darks = 'di', sample = b, flipdim = True)

# Rotate 90 degrees using process.rotate:
lola.generic(process.rotate, angle = -90, axis = 1)

# Display:
lola.display('slice', dim = 1, title = 'Projections')

# Optimize detector roll:
lola.optimize(linspace(-0.5, 0.5, 7), key = 'det_roll')

# Reconstruct:
lola.FDK()

lola.display('slice', dim = 2, title = 'FDK')

# Register volumes:
lola.registration(subsamp=1)

lola.display('slice', dim = 2, title = 'Volume')

# Reduce size and save data:
lola.cast2type(dtype = 'uint8', bounds = [0.01, 1])

#lola.autocrop()

lola.write_data('./fdk', 'vol')

lola.display('max_projection', dim = 2, title = 'Volume')

lola.draw_nodes()

#%% Runtime

lola.run()