# -*- coding: utf-8 -*-
"""
Example of batch processing. 
There are three parts in this example: scheduling and runtime of input operations, scheduling and runtime of reconstruction and restoration and restart of a crashed process.

@author: alex
"""
#%% Imports

from flexcalc import batch
#from numpy import linspace

#%% Initialize and schedule (1):

lola = batch.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = True)

# Load data:
path = '/ufs/ciacc/flexbox/tiling/t*'
lola.read_data(path, 'scan_', sampling = 2)

# Apply flatfield and log:
lola.flatlog(flats = 'io', darks = 'di', sample = 2)

# Display:
lola.display('slice', dim = 1, title = 'Projections')

# Marge projections:
lola.merge('projections')

lola.display('slice', dim = 1, title = 'Merged')

# Optimize detector shift:
#lola.optimize(linspace(-0.5, 0.5, 7), key = 'det_tan')

# Reconstruct:
lola.FDK()

lola.display('slice', dim = 2, title = 'FDK')
lola.display('slice', dim = 0, title = 'FDK')

# Merge volumes:
lola.merge('volume')

# Reduce size and save data:
lola.cast2type(dtype = 'uint8')
lola.autocrop()
lola.write_data('../fdk', 'vol')

lola.display('max_projection', dim = 2, title = 'Volume')

# Visualize nodes:
lola.draw_nodes()

#%% Runtime:

lola.run()

#%% Restore node tree after crash and repeat:

masha = batch.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = False)
masha.restore_nodes()
masha.draw_nodes()
masha.report()

masha.run()

#masha.cleanup()