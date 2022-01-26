# -*- coding: utf-8 -*-
"""
Example of batch processing. 
There are three parts in this example: scheduling and runtime of input operations, scheduling and runtime of reconstruction and restoration and restart of a crashed process.

@author: alex
"""
#%% Imports

from flexcalc import pipeline
#from numpy import linspace

#%% Initialize and schedule (1):

P = pipeline.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = True)

# Load data:
path = '/ufs/ciacc/flexbox/tiling/t*'
P.read_data(path, 'scan_', sampling = 2, correct='cwi-flexray-2019-04-24')

# Apply flatfield and log:
P.flatlog(flats = 'io_0', darks = 'di_0', sample = 2)

# Display:
P.display('slice', dim = 1, title = 'Projections')

# Merge projections:
P.merge('projections')

P.display('slice', dim = 1, title = 'Merged')

# Optimize detector shift:
#P.optimize(linspace(-0.5, 0.5, 7), key = 'det_tan')

# Reconstruct:
P.FDK()

P.display('slice', dim = 2, title = 'FDK')
P.display('slice', dim = 0, title = 'FDK')

# Merge volumes:
P.merge('volume')

# Reduce size and save data:
P.cast2type(dtype = 'uint8')
P.autocrop()
P.write_data('../fdk', 'vol')

P.display('max_projection', dim = 2, title = 'Volume')

# Visualize nodes:
#P.draw_nodes()

#%% Runtime:

P.run()

#%% Restore node tree after crash and repeat:

# If the pipeline execution crashed (due to e.g., an out of memory error),
# it is possible to resume execution since the current state is recorded in
# the scratch directory.

#Q = pipeline.scheduler('/export/scratch3/kostenko/scratch/', clean_scratch = False)
#Q.restore_nodes()
#Q.draw_nodes()
#Q.report()
#Q.run()
#Q.cleanup()
