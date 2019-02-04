# -*- coding: utf-8 -*-
"""
Example of batch processing. 
There are three parts in this example: scheduling and runtime of input operations, scheduling and runtime of reconstruction and restoration and restart of a crashed process.

@author: alex
"""
#%% Imports

from flexcalc import batch

#%% Initialize and schedule (1):

lola = batch.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = True)

# Load data:
path = '/ufs/ciacc/SeeThroughMuseum/naturalis/dubois_collection/dec_2017/skull_cap/high_res/t*'
lola.read_data(path, 'scan_', sampling = 8, dtype = 'uint16', shape = (1024, 1024), flipdim = True)

# Apply flatfield and log:
lola.flatlog(flats = 'io', darks = 'di', sample = 8, flipdim = True)

# Marge projections:
lola.merge('projections')

# Display:
lola.display('slice', dim = 1, title = 'Projections')

# Visualize nodes:
lola.draw_nodes()

#%% Runtime (1)

# Run:
lola.run()

#%% Optimize the detector centre, FDK, pos-process, write data (2):

# Optimize detector shift:
from numpy import linspace
lola.optimize(linspace(-0.3, 0.3, 7), key = 'det_hrz')

# Reconstruct:
lola.FDK()

# Merge volumes:
lola.crop(dim = 0, width = [10, 10])
lola.merge('volume')

lola.display('slice', dim = 2, title = 'Volume')
lola.display('slice', dim = 0, title = 'Volume')

# Reduce size and save data:
lola.cast2type(dtype = 'uint8')
lola.autocrop()

lola.write_data('../fdk', 'vol')

lola.display('max_projection', dim = 2, title = 'Volume')

lola.draw_nodes()

#%% Runtime (2)

lola.run()

#%% Restore node tree after crash and repeat (3):

masha = batch.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = False)
masha.restore_nodes()
masha.draw_nodes()
masha.report()

masha.run()

#masha.cleanup()