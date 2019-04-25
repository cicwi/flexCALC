"""
@author: Alex Kostenko
This module contains data analysis routines.
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import numpy

from scipy import ndimage
from scipy import signal

from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d
    
from flexdata import display
from flexdata import data

import transforms3d

from flextomo import phantom
from flextomo import projector
from flextomo import model

from flexdata.data import logger

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def get_background(array, mode = 'histogram'):
    '''
    Get the background intensity.
    '''
    # Compute air if needed:
    if mode == 'histogram':
        x, y = histogram(array[::2,::2,::2], log = True, plot = False)
        
        # Make sure there are no 0s:
        y = numpy.log(y + 1)    
        y = ndimage.filters.gaussian_filter1d(y, sigma=1)
        
        # Find air maximum:
        air_index = numpy.argmax(y)
        
        flat = x[air_index]
    
    elif mode == 'sides':    
        
        # Get the intensity of the sides:
        left = array[:, :, 0:10].mean(2)
        right = array[:, :, 0:10].mean(2)
        
        left = left.max(1)
        right = left.max(1)
        
        linfit = interp1d([0,1], numpy.vstack([left, right]).T, axis=1)
        
        flat = linfit(numpy.linspace(0, 1, array.shape[2]))
        flat = flat.astype(array.dtype)
    else:
        logger.error('Unknown mode: ' + mode)
        
    return flat  

def histogram(data, nbin = 256, rng = [], plot = True, log = False):
    """
    Compute histogram of the data.
    """
    
    #print('Calculating histogram...')
    data2 = data[::2, ::2, ::2]
    
    if rng == []:
        mi = min(data2.min(), 0)
        
        ma = numpy.percentile(data2, 99.99)
    else:
        mi = rng[0]
        ma = rng[1]

    y, x = numpy.histogram(data2, bins = nbin, range = [mi, ma])
    
    # Set bin values to the middle of the bin:
    x = (x[0:-1] + x[1:]) / 2

    if plot:
        display.plot(x, y, semilogy = log, title = 'Histogram')
    
    return x, y

def intensity_range(data):
    """
    Compute intensity range based on the histogram.
    
    Returns:
        a: position of the highest spike (typically air)
        b: 99.99th percentile
        c: center of mass of the histogram
    """
    # 256 bins should be sufficient for our dynamic range:
    x, y = histogram(data, nbin = 256, plot = False)
    
    # Smooth and find the first and the third maximum:
    y = ndimage.filters.gaussian_filter(numpy.log(y + 0.1), sigma = 1)
    
    # Air:
    a = x[numpy.argmax(y)]
    
    # Most of the other stuff:
    b = numpy.percentile(data, 99.99) 
    
    # Compute the center of mass excluding the high air spike +10% and outlayers:
    y = y[(x > a + (b-a)/10) & (x < b)]    
    x = x[(x > a + (b-a)/10) & (x < b)]
          
    c = numpy.sum(y * x) / numpy.sum(y)  
    
    return [a, b, c] 
    
def centre(array):
    """
    Compute the centre of the square of mass.
    """
    data2 = array[::2, ::2, ::2].copy().astype('float32')**2
    
    M00 = data2.sum()
            
    return [moment2(data2, 1, 0) / M00 * 2, moment2(data2, 1, 1) / M00 * 2, moment2(data2, 1, 2) / M00 * 2]

def moment3(array, order, center = numpy.zeros(3), subsample = 1):
    '''
    Compute 3D image moments $mu_{ijk}$.
    
    Args:
        data(array): 3D dataset
        order(int): order of the moment
        center(array): coordinates of the center
        subsample: subsampling factor - 1,2,4...
        
    Returns:
        float: image moment
    
    '''
    # Create central indexes:
    shape = array.shape
       
    array_ = array[::subsample, ::subsample, ::subsample].copy()
    
    for dim in range(3):
        if order[dim] > 0:
            
            # Define moment:
            m = numpy.arange(0, shape[dim], dtype = numpy.float32)
            m -= center[dim]
                
            data.mult_dim(array_, m[::subsample] ** order[dim], dim)    
            
    return numpy.sum(array_) * (subsample**3)
    
def moment2(array, power, dim, centered = True):
    """
    Compute 2D image moments (weighed averages) of the data. 
    
    sum( (x - x0) ** power * data ) 
    
    Args:
        power (float): power of the image moment
        dim (uint): dimension along which to compute the moment
        centered (bool): if centered, center of coordinates is in the middle of array.
        
    """
    # Create central indexes:
    shape = array.shape

    # Index:        
    x = numpy.arange(0, shape[dim])    
    if centered:
        x -= shape[dim] // 2
    
    x **= power
    
    if dim == 0:
        return numpy.sum(x[:, None, None] * array)
    elif dim == 1:
        return numpy.sum(x[None, :, None] * array)
    else:
        return numpy.sum(x[None, None, :] * array)

def bounding_box(array):
    """
    Find a bounding box for the volume based on intensity (use for auto_crop).
    """
    # Avoid memory overflow!
    #data = data.copy()
    data2 = array[::2, ::2, ::2].copy().astype('float32')
    data2 = data.bin(data2)
    
    data2[data2 < binary_threshold(data2, mode = 'otsu')] = 0
    
    integral = numpy.float32(data2).sum(0)
    
    # Filter noise:
    integral = ndimage.gaussian_filter(integral, 3)
    mean = numpy.mean(integral[integral > 0])
    integral[integral < mean / 10] = 0
    
    # Compute bounding box:
    rows = numpy.any(integral, axis=1)
    cols = numpy.any(integral, axis=0)
    b = numpy.where(rows)[0][[0, -1]]
    c = numpy.where(cols)[0][[0, -1]]
    
    integral = numpy.float32(data2).sum(1)
        
    # Filter noise:
    integral = ndimage.gaussian_filter(integral, 3)
    mean = numpy.mean(integral[integral > 0])
    integral[integral < mean / 10] = 0
    
    # Compute bounding box:
    rows = numpy.any(integral, axis=1)
    a = numpy.where(rows)[0][[0, -1]]
    
    # Add a safe margin:
    a_int = (a[1] - a[0]) // 20
    b_int = (b[1] - b[0]) // 20
    c_int = (c[1] - c[0]) // 20
    
    a[0] = a[0] - a_int
    a[1] = a[1] + a_int
    b[0] = b[0] - b_int
    b[1] = b[1] + b_int
    c[0] = c[0] - c_int
    c[1] = c[1] + c_int
    
    a[0] = max(0, a[0] * 4)
    a[1] = min(array.shape[0], a[1] * 4)
    
    b[0] = max(0, b[0] * 4)
    b[1] = min(array.shape[1], b[1] * 4)
    
    c[0] = max(0, c[0] * 4)
    c[1] = min(array.shape[2], c[1] * 4)
    
    return a, b, c

def binary_threshold(data, mode = 'histogram', threshold = 0):
    '''
    Compute binary threshold. Use 'histogram, 'otsu', or 'constant' mode.
    '''
    
    import matplotlib.pyplot as plt
    
    print('Applying binary threshold...')
    
    if mode == 'otsu':
        threshold = threshold_otsu(data[::2,::2,::2])    
        
    elif mode == 'histogram':
        x, y = histogram(data[::2,::2,::2], log = True, plot = False)
        
        # Make sure there are no 0s:
        y = numpy.log(y + 1)    
        y = ndimage.filters.gaussian_filter1d(y, sigma=1)
        
        plt.figure()
        plt.plot(x, y)
            
        # Find air maximum:
        air_index = numpy.argmax(y)
        
        print('Air found at %0.3f' % x[air_index])
    
        # Find the first shoulder after air peak in the histogram spectrum:
        x = x[air_index:]
        
        yd = abs(numpy.diff(y))
        yd = yd[air_index:]
        y = y[air_index:]
        
        # Minimum derivative = Saddle point or extremum:
        ind = signal.argrelextrema(yd, numpy.less)[0][0]
        min_ind = signal.argrelextrema(y, numpy.less)[0][0]
    
        plt.plot(x[ind], y[ind], '+')
        plt.plot(x[min_ind], y[min_ind], '*')
        plt.show()
        
        # Is it a Saddle point or extremum?
        if abs(ind - min_ind) < 2:    
            threshold = x[ind]         
    
            print('Minimum found next to the air peak at: %0.3f' % x[ind])        
        else:            
            # Move closer to the air peak since we are looking at some other material             
            threshold = x[ind] - abs(x[ind] - x[0]) / 4 
    
            print('Saddle point found next to the air peak at: %0.3f' % x[ind])        
            
    elif mode == 'constant':
        pass        
            
    else: raise ValueError('Wrong mode parameter. Can be histogram or otsu.')
    
    print('Threshold value is %0.3f' % threshold)
    
    return threshold

def find_marker(array, geometry, d = 5):
    """
    Find a marker in 3D volume by applying a circular kernel with an inner diameter d [mm].
    """
    # TODO: it fail sometimes when the marker is adjuscent to something...
    
    #data = data.copy()
    # First subsample data to avoid memory overflow:
    data2 = array[::4, ::4, ::4]
    data2[data2 < 0] = 0
    
    r = d / 4
        
    # Get areas with significant density:
    t = binary_threshold(data2, mode = 'otsu')
    threshold = numpy.float32(data2 > t)
    
    # Create a circular kernel (take into account subsampling of data2):
    kernel = -0.5 * phantom.sphere(data2.shape, geometry, r * 2, [0,0,0])
    kernel += phantom.sphere(data2.shape, geometry, r, [0,0,0])

    kernel[kernel > 0] *= (2**3 - 1)
    
    print('Computing feature sizes...')
    
    # Map showing the relative size of feature
    data.convolve_kernel(threshold, kernel)
    
    A = threshold
    
    A[A < 0] = 0
    A /= A.max()
    
    display.max_projection(A, dim = 0, title = 'Feature size.')
    
    print('Estimating local variance...')
    
    # Now estimate the local variance:
    B = ndimage.filters.laplace(data2) ** 2    
    B /= (numpy.abs(data2) + data2.max()/100)
    
    # Make sure that boundaries don't affect variance estimation:
    threshold = threshold == 0
    
    threshold = ndimage.morphology.binary_dilation(threshold)
    
    B[threshold] = 0
    B = numpy.sqrt(B)
    B = ndimage.filters.gaussian_filter(B, 4)
    B /= B.max()
    
    display.max_projection(B, dim = 0, title = 'Variance.')
    
    # Compute final weight:    
    A -= B
    
    # Make it dependent on absolote intensity: (could be dependent on distance from some value....)
    A *= numpy.sqrt(data2)
    #A -= numpy.sqrt((data2 - density)**2 + density / 10)
    
    print('A.max', A.max())
    
    print('A.mean', A[A > 0].mean())
    
    index = numpy.argmax(A)
    
    # Display:
    display.max_projection(A, dim = 0, title = 'Marker map')
    
    # Coordinates:
    a, b, c = numpy.unravel_index(index, A.shape)
    
    # Upsample:
    a *= 4
    b *= 4
    c *= 4
    
    print('Found the marker at:', a, b, c)
    
    return a, b, c
    
def moments_orientation(data, subsample = 1):
    '''
    Find the center of mass and the intensity axes of the image.
    
    Args:
        data(array): 3D input
        subsample: subsampling factor to to make it faster
        
    Returns:
        T, R: translation vector to the center of mass and rotation matrix to intensity axes 
    
    '''
    # find centroid:
    m000 = moment3(data, [0, 0, 0], subsample = subsample)
    m100 = moment3(data, [1, 0, 0], subsample = subsample)
    m010 = moment3(data, [0, 1, 0], subsample = subsample)
    m001 = moment3(data, [0, 0, 1], subsample = subsample)

    # Somehow this system of coordinates and the system of ndimage.interpolate require negation of j:
    T = [m100 / m000, m010 / m000, m001 / m000]
    
    # find central moments:
    mu200 = moment3(data, [2, 0, 0], T, subsample = subsample) / m000
    mu020 = moment3(data, [0, 2, 0], T, subsample = subsample) / m000
    mu002 = moment3(data, [0, 0, 2], T, subsample = subsample) / m000
    mu110 = moment3(data, [1, 1, 0], T, subsample = subsample) / m000
    mu101 = moment3(data, [1, 0, 1], T, subsample = subsample) / m000
    mu011 = moment3(data, [0, 1, 1], T, subsample = subsample) / m000
    
    # construct covariance matrix and compute rotation matrix:
    M = numpy.array([[mu200, mu110, mu101], [mu110, mu020, mu011], [mu101, mu011, mu002]])

    #Compute eigen vecors of the covariance matrix and sort by eigen values:
    vec = numpy.linalg.eig(M)[1].T
    lam = numpy.linalg.eig(M)[0]    
    
    # Here we sort the eigen values:
    ind = numpy.argsort(lam)
    
    # Matrix R is composed of basis vectors:
    R = numpy.array(vec[ind[::-1]])
    
    # Makes sure our basis always winds the same way:
    R[2, :] = numpy.cross(R[0, :], R[1, :])     
    
    # Centroid:
    T = numpy.array(T) - numpy.array(data.shape) // 2
    
    return T, R

def calibrate_spectrum(projections, volume, geometry, compound = 'Al', density = 2.7, threshold = None, iterations = 1000, n_bin = 10):
    '''
    Use the projection stack of a homogeneous object to estimate system's 
    effective spectrum.
    Can be used by process.equivalent_thickness to produce an equivalent 
    thickness projection stack.
    Please, use conventional geometry. 
    ''' 
    
    #import random
    
    # Find the shape of the object:                                                    
    if threshold:
        t = binary_threshold(volume, mode = 'constant', threshold = threshold)
        
        segmentation = numpy.float32()
    else:
        t = binary_threshold(volume, mode = 'otsu')
        segmentation = numpy.float32(volume > t)
        
    display.slice(segmentation, dim=0,title = 'Segmentation')        
        
    # Crop:    
    #height = segmentation.shape[0]   
    #w = 15

    #length = length[height//2-w:height//2 + w, : ,:]    
    
    # Forward project the shape:                  
    print('Calculating the attenuation length.')  
    
    length = numpy.zeros_like(projections)    
    length = numpy.ascontiguousarray(length)
    projector.forwardproject(length, segmentation, geometry)
        
    projections[projections < 0] = 0
    intensity = numpy.exp(-projections)
    
    # Crop to avoid cone artefacts:
    height = intensity.shape[0]//2
    window = 10
    intensity = intensity[height-window:height+window,:,:]
    length = length[height-window:height+window,:,:]
    
    # Make 1D:
    intensity = intensity[length > 0].ravel()
    length = length[length > 0].ravel()
    
    lmax = length.max()
    lmin = length.min()    
    
    print('Maximum reprojected length:', lmax)
    print('Minimum reprojected length:', lmin)
    
    print('Selecting a random subset of points.')  
    
    # Rare the sample to avoid slow times:
    #index = random.sample(range(length.size), 1000000)
    #length = length[index]
    #intensity = intensity[index]
    
    print('Computing the intensity-length transfer function.')
    
    # Bin number for lengthes:
    bin_n = 128
    bins = numpy.linspace(lmin, lmax, bin_n)
    
    # Sample the midslice:
    #segmentation = segmentation[height//2-w:height//2 + w, : ,:]    
    #projections_ = projections[height//2-w:height//2 + w, : ,:]
    
    
    #import flexModel
    #ctf = flexModel.get_ctf(length.shape[::2], 'gaussian', [1, 1])
    #length = flexModel.apply_ctf(length, ctf)  
            
    # TODO: Some cropping might be needed to avoid artefacts at the edges
    
    #flexUtil.slice(length, title = 'length sinogram')
    #flexUtil.slice(projections_, title = 'apparent sinogram')
        
    # Rebin:
    idx  = numpy.digitize(length, bins)
    
    # Rebin length and intensity:        
    length_0 = bins + (bins[1] - bins[0]) / 2
    intensity_0 = [numpy.median(intensity[idx==k]) for k in range(bin_n)]
    
    # In case some bins are empty:
    intensity_0 = numpy.array(intensity_0)
    length_0 = numpy.array(length_0)
    length_0 = length_0[numpy.isfinite(intensity_0)]
    intensity_0 = intensity_0[numpy.isfinite(intensity_0)]
    
    # Get rid of tales:
    rim = len(length_0) // 20
    length_0 = length_0[rim:-rim]    
    intensity_0 = intensity_0[rim:-rim]    
    
    # Dn't trust low counts!
    length_0 = length_0[intensity_0 > 0.05]
    intensity_0 = intensity_0[intensity_0 > 0.05]
    
    # Get rid of long rays (they are typically wrong...)   
    #intensity_0 = intensity_0[length_0 < 35]    
    #length_0 = length_0[length_0 < 35]    
    
    # Enforce zero-one values:
    length_0 = numpy.insert(length_0, 0, 0)
    intensity_0 = numpy.insert(intensity_0, 0, 1)
    
    #flexUtil.plot(length_0, intensity_0, title = 'Length v.s. Intensity')
        
    print('Intensity-length curve rebinned.')
        
    print('Computing the spectrum by Expectation Maximization.')
    
    volts = geometry.description.get('voltage')
    if not volts: volts = 100
    
    energy = numpy.linspace(5, max(100, volts), n_bin)
    
    mu = model.linear_attenuation(energy, compound, density)
    exp_matrix = numpy.exp(-numpy.outer(length_0, mu))
    
    # Initial guess of the spectrum:
    spec = model.bremsstrahlung(energy, volts) 
    spec *= model.scintillator_efficiency(energy, 'Si', rho = 5, thickness = 0.5)
    spec *= model.total_transmission(energy, 'H2O', 1, 1)
    spec *= energy
    spec /= spec.sum()
    
    #spec = numpy.ones_like(energy)
    #spec[0] = 0
    #spec[-1] = 0
    
    norm_sum = exp_matrix.sum(0)
    spec0 = spec.copy()
    #spec *= 0
    
    # exp_matrix at length == 0 is == 1. Sum of that is n_bin
    
    # EM type:   
    for ii in range(iterations): 
        frw = exp_matrix.dot(spec)

        epsilon = frw.max() / 100
        frw[frw < epsilon] = epsilon
        
        spec = spec * exp_matrix.T.dot(intensity_0 / frw) / norm_sum
        
        # Make sure that the total count of spec is 1 - that means intensity at length = 0 is equal to 1
        spec = spec / spec.sum()
        
    print('Spectrum computed.')
        
    #flexUtil.plot(length_0, title = 'thickness')
    #flexUtil.plot(mu, title = 'mu')
    #flexUtil.plot(_intensity, title = 'synth_counts')
    
    # synthetic intensity for a check:
    _intensity = exp_matrix.dot(spec)
    
    import matplotlib.pyplot as plt
    
    # Display:   
    plt.figure()
    plt.semilogy(length[::200], intensity[::200], 'b.', lw=4, alpha=.8)
    plt.semilogy(length_0, intensity_0, 'g--')
    plt.semilogy(length_0, _intensity, 'r-', lw=3, alpha=.6)
    
    #plt.scatter(length[::100], -numpy.log(intensity[::100]), color='b', alpha=.2, s=4)
    plt.axis('tight')
    plt.title('Log intensity v.s. absorption length.')
    plt.legend(['raw','binned','solution'])
    plt.show() 
    
    # Display:
    plt.figure()
    plt.plot(energy, spec, 'b')
    plt.plot(energy, spec0, 'r:')
    plt.axis('tight')
    plt.title('Calculated spectrum')
    plt.legend(['computed','initial guess'])
    plt.show() 
            
    
    return energy, spec