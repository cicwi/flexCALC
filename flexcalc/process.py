#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Kostenko
This module contains calculation routines for processing of the data.
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy
import time

from scipy import ndimage
import scipy.ndimage.interpolation as interp

import transforms3d

from tqdm import tqdm
import SimpleITK as sitk

from skimage import feature
from stl import mesh
from skimage import measure
    
from flexdata import data
from flexdata import display
from flextomo import projector
from flextomo import model
from . import analyze

from flexdata.data import logger

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def process_flex(path, sample = 1, skip = 1, memmap = None, index = None, proj_number = None):
    '''
    Read and process the array.
    
    Args:
        path:  path to the flexray array
        sample:
        skip:
        memmap:
        index:
        proj_number (int): force projection number (treat lesser numbers as missing)
        
    Return:
        proj: min-log projections
        meta: meta array
        
    '''
    # Read:    
    print('Reading...')
    
    #index = []
    proj, flat, dark, geom = data.read_flexray(path, sample = sample, skip = skip, memmap = memmap, proj_number = proj_number)
    
    # Prepro:            
    proj = preprocess(proj, flat, dark)
        
    '''
    index = numpy.array(index)
    index //= skip
    
    if (index[-1] + 1) != index.size:
        print(index.size)
        print(index[-1] + 1)
        print('Seemes like some files were corrupted or missing. We will try to correct thetas accordingly.')
        
        thetas = numpy.linspace(geom['range'][0], geom['range'][1], index[-1]+1)
        thetas = thetas[index]
        
        geom['thetas'] = thetas
        
        import pylab
        pylab.plot(thetas, thetas ,'*')
        pylab.title('Thetas')
    '''
    
    print('Done!')
    
    return proj, geom
      
def preprocess(array, flats = None, darks = None, mode = 'sides', dim = 1):
    '''
    Apply flatfield correction based on availability of flat- and dark-field.
    
    Args:
        flats (ndarray): divide by flats
        darks (ndarray): subtract darks
        mode (str): "sides" to use maximum values of the detector sides to estimate the flat field or a mode of intensity distribution with "single".     
        dim  (int): dimension that represents the projection number
    '''          
    logger.print('Pre-processing...')
    
    # Cast to float if int:
    if (array.dtype.kind == 'i') | (array.dtype.kind == 'u'):    
            # In case array is mapped on disk, we need to rewrite the file as another type:
            new = array.astype('float32')    
            array = data.rewrite_memmap(array, new)    
            
    if darks is not None:
        
        darks = darks.astype('float32')
        
        if darks.ndim > 2:
            darks = darks.mean(dim)
        
        # Subtract:
        data.add_dim(array, -darks, dim)
        
    else:
        
        darks = numpy.zeros(1, dtype = 'float32')
            
    if flats is not None:
        
        flats = flats.astype('float32')
        
        if flats.ndim > 2:
            flats = flats.mean(dim)
        
        # Subtract:
        data.add_dim(flats, -darks, dim) 
        data.mult_dim(array, 1 / flats, dim)
        
    numpy.log(array, out = array)
    array *= -1
    
    # Fix nans and infs after log:
    array[~numpy.isfinite(array)] = 0
    
    return array

def residual_rings(array, kernel=[3, 3]):
    '''
    Apply correction by computing outlayers .
    '''
    # Compute mean image of intensity variations that are < 5x5 pixels
    logger.print('Our best agents are working on the case of the Residual Rings. This can take years if the kernel size is too big!')

    tmp = numpy.zeros(array.shape[::2])
    
    for ii in tqdm(range(array.shape[1]), unit = 'images'):                 
        
        block = array[:, ii, :]

        # Compute:
        tmp += (block - ndimage.filters.median_filter(block, size = kernel)).sum(1)
        
    tmp /= array.shape[1]
    
    logger.print('Subtract residual rings.')
    
    for ii in tqdm(range(array.shape[1]), unit='images'):                 
        
        block = array[:, ii, :]
        block -= tmp

        array[:, ii, :] = block 
    
    logger.print('Residual ring correcion applied.')

def generate_stl(array, geometry):
    """
    Make a mesh from a volume.
    """
    # Segment the volume:
    threshold = array > analyze.binary_threshold(array, mode = 'otsu')
    
    # Close small holes:
    print('Filling small holes...')
    threshold = ndimage.morphology.binary_fill_holes(threshold, structure = numpy.ones((3,3,3)))

    print('Generating mesh...')
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes_lewiner(threshold, 0)

    print('Mesh with %1.1e vertices generated.' % verts.shape[0])
    
    # Create stl:    
    stl_mesh = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh.vectors = verts[faces] * geometry.voxel
    
    return stl_mesh

def soft_threshold(array, mode = 'histogram', threshold = 0):
    """
    Removes values smaller than the threshold value.
    Args:
        array (ndarray)  : data array (implicit)
        mode (str)       : 'histogram', 'otsu' or 'constant'
        threshold (float): threshold value if mode = 'constant'
    """
    # Avoiding memory overflow:
    thresh = analyze.binary_threshold(array, mode, threshold)
    
    for ii in range(array.shape[0]):
        
        img = array[ii, :, :]
        img[img < thresh] = 0
        
        array[ii, :, :] = img

def hard_threshold(array, mode = 'histogram', threshold = 0):
    """
    Returns a binary map based on the threshold value.
    Args:
        array (ndarray)  : data array (implicit)
        mode (str)       : 'histogram', 'otsu' or 'constant'
        threshold (float): threshold value if mode = 'constant'
    """
    # Avoiding memory overflow:
    thresh = analyze.binary_threshold(array, mode, threshold)
    
    binary = numpy.zeros(array.shape, dtype = 'bool')
    for ii in range(array.shape[0]):
        binary[ii, :, :] = array[ii, :, :] < thresh

def affine(array, matrix, shift):
    """
    Apply 3x3 rotation matrix and shift to a 3D arrayset.
    """
    print('Applying affine transformation.')
    time.sleep(0.3)
    
    pbar = tqdm(unit = 'Operations', total=1) 
   
    # Compute offset:
    T0 = numpy.array(array.shape) // 2
    T1 = numpy.dot(matrix, T0 + shift)

    array = ndimage.interpolation.affine_transform(array, matrix, offset = T0-T1, order = 1)

    pbar.update(1)
    pbar.close()      
    
    return array
    
def scale(array, factor, order = 1):
    '''
    Scales the volume via interpolation.
    '''
    print('Applying scaling.')
    time.sleep(0.3)
    
    pbar = tqdm(unit = 'Operations', total=1) 
    
    array = ndimage.interpolation.zoom(array, factor, order = order)
    
    pbar.update(1)
    pbar.close()      
    
    return array  

def autocrop(array, geom = None):
    '''
    Auto_crop the volume and update the geometry record.
    '''
    a,b,c = analyze.bounding_box(array)
        
    sz = array.shape
    
    print('Old dimensions are: ' + str(sz))
           
    array = data.crop(array, 0, [a[0], sz[0] - a[1]], geom)
    array = data.crop(array, 1, [b[0], sz[1] - b[1]], geom)
    array = data.crop(array, 2, [c[0], sz[2] - c[1]], geom)
    
    print('New dimensions are: ' + str(array.shape))
    
    return array

def allign_moments(array, axis = 0):
    '''
    Compute orientations of the volume intensity moments and allign them with XYZ.
    Align the primary moment with vertical axis - use axis == 0.
    '''
    
    print('Alligning volume moments.')
    
    # Moments orintations:
    T, R = analyze.moments_orientation(array, 8)
    
    if axis == 0:
        R_90 = R.T.dot(transforms3d.euler.euler2mat(0, 0, 0))
    elif axis == 1:
        R_90 = R.T.dot(transforms3d.euler.euler2mat(numpy.pi / 2, 0, 0))
    elif axis == 2:    
        R_90 = R.T.dot(transforms3d.euler.euler2mat(numpy.pi / 2, numpy.pi / 2, 0))
  
    # Apply transformation:
    return affine(array, R_90, [0,0,0])
    
def rotate(array, angle, axis = 0):
    '''
    Rotates the volume via interpolation.
    '''
    print('Applying rotation.')
    time.sleep(0.3)    
    sz = array.shape[axis]
    
    if angle == 90:
       ax = [0,1,2]
       ax.remove(axis)
       return numpy.rot90(array, axes=ax) 
    elif angle == -90:
       ax = [0,1,2]
       ax.remove(axis)
       return numpy.rot90(array, k =3, axes=ax) 
        
    pbar = tqdm(unit = 'slices', total=sz) 
    for ii in range(sz):     
        
        sl = data.anyslice(array, ii, axis)
        
        array[sl] = ndimage.interpolation.rotate(array[sl], angle, reshape=False)
        
        pbar.update(1)
    pbar.close()
        
    return array
        
def translate(array, shift, order = 1):
    """
    Apply a 3D tranlation.
    """
    print('Applying translation.')
    time.sleep(0.3)

    pbar = tqdm(unit = 'Operation', total=1) 
    
    #ndimage.interpolation.shift(array, shift, output = array, order = order)
    # for some reason may return zeros in this implementation
    array = ndimage.interpolation.shift(array, shift, order = order)    
    
    pbar.update(1)
    pbar.close()

    return array


def _itk2mat_(transform, shape):
    """
    Transform ITK to matrix and a translation vector.
    """
    
    # transform contains information about the centre of rptation, rotation and translation
    # We need to convert this to a rotation matrix and single translation vector
    # here we go,,,
    
    T = -numpy.array(transform.GetParameters()[3:][::-1])
    euler = -numpy.array(transform.GetParameters()[:3])
    R = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], axes='szyx')
    
    # Centre of rotation:
    centre = (transform.GetFixedParameters()[:3][::-1] - T)
    T0 = centre - numpy.array(shape) // 2
    
    # Add rotated vector pointing to the centre of rotation to total T
    T = T - numpy.dot(T0, R) + T0
    
    return T, R
    
def _mat2itk_(R, T, shape):
    """
    Initialize ITK transform from a rotation matrix and a translation vector
    """       
    centre = numpy.array(shape, dtype = float) // 2
    euler = transforms3d.euler.mat2euler(R, axes = 'szyx')    

    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(True)
        
    transform.SetTranslation(-T[::-1])
    transform.SetCenter((centre + T)[::-1])    

    transform.SetRotation(-euler[0], -euler[1], -euler[2])    
    
    return transform    
   
def _moments_registration_(fixed, moving):
    """
    Register two volumes using image moments.
    
        Args:
        fixed (array): fixed 3D array
        moving (array): moving 3D array
        
    Returns:
        moving will be altered in place.
        
        Ttot: translation vector
        Rtot: rotation matrix
        Tfix: position of the fixed volume

    """
    # Positions of the volumes:
    Tfix, Rfix  = analyze.moments_orientation(fixed)
    Tmov, Rmov  = analyze.moments_orientation(moving)
    
    # Total rotation and shift:
    Rtot = numpy.dot(Rmov, Rfix.T)
    Ttot = Tfix - numpy.dot(Tmov, Rtot)

    # Apply transformation:
    moving_ = affine(moving.copy(), Rtot, Ttot)
    
    # Solve ambiguity with directions of intensity axes:    
    Rtot, Ttot = _find_best_flip_(fixed, moving_, Rfix, Tfix, Rmov, Tmov, use_CG = False)
    
    return Ttot, Rtot, Tfix
    
def _itk_registration_(fixed, moving, R_init = None, T_init = None, shrink = [4, 2, 1, 1], smooth = [8, 4, 2, 0]):
    """
    Carry out ITK based volume registration (based on Congugate Gradient).
    
    Args:
        fixed (array): fixed 3D array
        moving (array): moving 3D array
        
    Returns:
        moving will be altered in place.
        
        T: translation vector
        R: rotation matrix
        
    """
    #  Progress bar    
    pbar = tqdm(unit = 'Operations', total=1) 
    
    # Initial transform:
    if R_init is None:
        R_init = numpy.zeros([3,3])
        R_init[0, 0] = 1
        R_init[1, 1] = 1
        R_init[2, 2] = 1
        
    if T_init is None:
        T_init = numpy.zeros(3)    
    
    # Initialize itk images:
    fixed_image =  sitk.GetImageFromArray(fixed)
    moving_image = sitk.GetImageFromArray(moving)
    
    # Regitration:
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Initial centering transform:
    transform = _mat2itk_(R_init, T_init, fixed.shape)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsPowell()
    #registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=200, convergenceMinimumValue=1e-10, convergenceWindowSize=10)
    #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations = 100)
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations = 100)
    #registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = shrink)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = smooth)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(transform, inPlace=False)

    transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    
    pbar.update(1)
    pbar.close()
    
    #print("Final metric value: ", registration_method.GetMetricValue())
    print("Optimizer`s stopping condition: ", registration_method.GetOptimizerStopConditionDescription())

    # This is a bit of woodo to get to the same definition of Euler angles and translation that I use:
    T, R = _itk2mat_(transform, moving.shape)
            
    #moving_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    #moving = sitk.GetArrayFromImage(moving_image)    
        
    #flexUtil.projection(fixed - moving, dim = 1, title = 'native diff')  
    
    return T, R, registration_method.GetMetricValue()

def _generate_flips_(Rfix):
    """
    Generate number of rotation and translation vectors.
    """    
    # Rotate the moving object around it's main axes:
    R = [numpy.eye(3),]
    
    # Axes:
    for ii in range(3):    
        #R.append(transforms3d.euler.axangle2mat(Rfix[ii], numpy.pi))
        # Angles:
        for jj in range(3):
            R.append(transforms3d.euler.axangle2mat(Rfix[ii], (jj+1) * numpy.pi/2))
    
    return R
                    
def register_volumes(fixed, moving, subsamp = 2, use_moments = True, use_CG = True, use_flips = False, threshold = 'otsu'):
    '''
    Registration of two 3D volumes.
    
    Args:
        fixed (array): reference volume
        moving (array): moving/slave volume
        subsamp (int): subsampling of the moments computation
        use_itk (bool): if True, use congugate descent method after aligning the moments
        treshold (str): can be None, 'otsu' or 'histogram' - defines the strategy for removing low intensity noise
        
    Returns:
        
    '''        
    if fixed.shape != moving.shape: raise IndexError('Fixed and moving volumes have different dimensions:', fixed.shape, moving.shape)
    
    print('Using image moments to register volumes.')
        
    # Subsample volumes:
    fixed_0 = fixed[::subsamp,::subsamp,::subsamp].copy()
    moving_0 = moving[::subsamp,::subsamp,::subsamp].copy()
    
    if threshold:
        # We use Otsu here instead of binary_threshold to make sure that the same 
        # threshold is applied to both images:
        
        threshold = analyze.threshold_otsu(numpy.append(fixed_0[::2, ::2, ::2], moving_0[::2, ::2, ::2]))
        fixed_0[fixed_0 < threshold] = 0
        moving_0[moving_0 < threshold] = 0
        
    display.max_projection(fixed_0, title = 'Preview: fixed volume')
    display.max_projection(moving_0, title = 'Preview: moving volume')
        
    L2 = norm(fixed_0 - moving_0)
    print('L2 norm before registration: %0.2e' % L2)
    
    if use_moments:
        
        print('Running moments registration.')
        
        # Progress:
        pbar = tqdm(unit = 'Operations', total=1) 
    
        # Positions of the volumes:
        Tfix, Rfix  = analyze.moments_orientation(fixed_0)
        Tmov, Rmov  = analyze.moments_orientation(moving_0)
               
        # Total rotation and shift:
        #Rtot = numpy.dot(Rmov, Rfix.T)
        #Rtot = Rmov.T.dot(Rfix)

        #Ttot = Tfix - numpy.dot(Tmov, Rtot)
        
        Rtot, Ttot = _find_best_flip_(fixed_0, moving_0, Rfix, Tfix, Rmov, Tmov, use_CG = use_flips)
        
        pbar.update(1)
        pbar.close()
    
    else:
        # Initial transform:
        Rtot = numpy.zeros([3,3])
        Rtot[0, 0] = 1
        Rtot[1, 1] = 1
        Rtot[2, 2] = 1
        
        # Positions of the volumes:
        Tfix, Rfix  = analyze.moments_orientation(fixed_0)
        Tmov, Rmov  = analyze.moments_orientation(moving_0)
        
        Ttot = Tfix - Tmov#numpy.zeros(3)
            
    # Refine registration using ITK optimization:
    if use_CG:
        
        print('Running ITK optimization.')
        
        #Rtot = Rmov.T.dot(Rfix)
        #Rtot = Rmov.dot(Rfix.T)
        #Ttot = Tfix - Tmov.dot(Rtot)

        # Find flip with or without CG:
        #Rtot, Ttot = _find_best_flip_(fixed_0, moving_0, Rfix, Tfix, Rmov, Tmov, use_CG = use_flips)
        
        # Show the result of moments registration:
        L2 = norm(fixed_0 - affine(moving_0.copy(), Rtot, Ttot))
        print('L2 norm after moments registration: %0.2e' % L2)
        time.sleep(0.1)    
        
        # Run CG with the best result:
        Ttot, Rtot, L = _itk_registration_(fixed_0, moving_0, Rtot, Ttot, shrink = [8, 2, 1], smooth = [8, 2, 0])               
            
    # Apply transformation:
    L2 = norm(fixed_0 - affine(moving_0.copy(), Rtot, Ttot))
    print('L2 norm after registration: %0.2e' % L2)
            
    print('Found shifts:', Ttot * subsamp)
    print('Found Euler rotations:', transforms3d.euler.mat2euler(Rtot))        
    
    return Rtot, Ttot * subsamp 
        
def register_astra_geometry(proj_fix, proj_mov, geom_fix, geom_mov, subsamp = 1):
    """
    Compute a rigid transformation that makes sure that two reconstruction volumes are alligned.
    Args:
        proj_fix : projection data of the fixed volume
        proj_mov : projection data of the fixed volume
        geom_fix : projection data of the fixed volume
        geom_mov : projection data of the fixed volume
        
    Returns:
        geom : geometry for the second reconstruction volume
    """
    
    print('Computing a rigid tranformation between two datasets.')
    
    # Find maximum vol size:
    sz = numpy.array([proj_fix.shape, proj_mov.shape]).max(0)    
    sz += 10 # for safety...
    
    vol1 = numpy.zeros(sz, dtype = 'float32')
    vol2 = numpy.zeros(sz, dtype = 'float32')
    
    projector.settings.bounds = [0, 10]
    projector.settings.subsets = 10
    projector.settings['mode'] = 'sequential'
    
    projector.FDK(proj_fix, vol1, geom_fix)    
    projector.SIRT(proj_fix, vol1, geom_fix, iterations = 5)
    
    projector.FDK(proj_mov, vol2, geom_mov)
    projector.SIRT(proj_mov, vol2, geom_mov, iterations = 5)
    
    display.slice(vol1, title = 'Fixed volume preview')
    display.slice(vol1, title = 'Moving volume preview')
    
    # Find transformation between two volumes:
    R, T = register_volumes(vol1, vol2, subsamp = subsamp, use_moments = True, use_CG = True)
    
    return R, T

def _find_best_flip_(fixed, moving, Rfix, Tfix, Rmov, Tmov, use_CG = True, sample = 2):
    """
    Find the orientation of the moving volume with the mallest L2 distance from the fixed volume, 
    given that there is 180 degrees amiguity for each of three axes.
    
    Args:
        fixed(array): 3D volume
        moving(array): 3D volume
        centre(array): corrdinates of the center of rotation
        area(int): radius around the center of rotation to look at
        
    Returns:
        (array): rotation matrix corresponding to the best flip
    
    """
    fixed = fixed[::sample, ::sample, ::sample].astype('float32')
    moving = moving[::sample, ::sample, ::sample].astype('float32')
    
    # Apply filters to smooth erors somewhat:
    fixed = ndimage.filters.gaussian_filter(fixed, sigma = 1)
    moving = ndimage.filters.gaussian_filter(moving, sigma = 1)
    
    # Generate flips:
    Rs = _generate_flips_(Rfix)
    
    # Compute L2 norms:
    Lmax = numpy.inf
    
    # Appliy flips:
    for ii in range(len(Rs)):
        
        Rtot_ = Rmov.T.dot(Rfix).dot(Rs[ii])
        Ttot_ = (Tfix - numpy.dot(Tmov, Rtot_)) / sample
        
        if use_CG:
            
            Ttot_, Rtot_, L = _itk_registration_(fixed, moving, Rtot_, Ttot_, shrink = [2,], smooth = [4,]) 
        
        mo_ = affine(moving, Rtot_, Ttot_)                  
    
        L = norm(fixed - mo_)
        
        if Lmax > L:
            Rtot = Rtot_.copy()
            Ttot = Ttot_.copy()
            Lmax = L
            
            print('We found better flip(%u), L ='%ii, L)
            display.projection(fixed - mo_, title = 'Diff (%u). L2 = %f' %(ii, L))
    
    return Rtot, Ttot * sample
                
def equalize_intensity(master, slave, mode = 'percentile'):
    """
    Compute 99.99th percentile of two volumes and use it to renormalize the slave volume.
    """
    if mode == 'percentile':
        m = numpy.percentile(master, 99.99) 
        s = numpy.percentile(slave, 99.99) 
        
        slave *= (m / s)
    elif mode == 'histogram':
        
        a1, b1, c1 = analyze.intensity_range(master[::2, ::2, ::2])
        a2, b2, c2 = analyze.intensity_range(slave[::2, ::2, ::2])
        
        slave *= (c1 / c2)
        
    else: raise Exception('Unknown mode:' + mode)
    
def interpolate_lines(proj):
    '''
    Interpolate values of the horizontal read out lines of the flexray flat panel detector.
    '''
    
    lines = numpy.ones(proj.shape[0::2], dtype = bool)    
    
    sz = proj.shape[0]
    
    if sz == 1536:
        lines[125::256, :] = 0
        lines[126::256, :] = 0    
    else:
        step = sz // 12
        lines[(step-1)::step*2, :] = 0    

    interpolate_holes(proj, lines, kernel = [1,1])   
          
def interpolate_holes(array, mask2d, kernel = [1,1]):
    '''
    Fill in the holes, for instance, saturated pixels.
    
    Args:
        mask2d: holes are zeros. Mask is the same for all projections.
        kernel: size of the interpolation kernel
    '''
    mask_norm = ndimage.filters.gaussian_filter(numpy.float32(mask2d), sigma = kernel)
    #flexUtil.slice(mask_norm, title = 'mask_norm')
    
    sh = array.shape[1]
    
    for ii in tqdm(range(sh), unit='images'):    
            
        array[:, ii, :] = array[:, ii, :] * mask2d           

        # Compute the filler:
        tmp = ndimage.filters.gaussian_filter(array[:, ii, :], sigma = kernel) / mask_norm      
                                              
        #flexUtil.slice(tmp, title = 'tmp')

        # Apply filler:                 
        array[:, ii, :][~mask2d] = tmp[~mask2d]
         
def interpolate_zeros(array, kernel = [1,1], epsilon = 1e-9):
    '''
    Fill in zero volues, for instance, blank pixels.
    
    Args:
        kernel: Size of the interpolation kernel
        epsilon: if less than epsilon -> interpolate
    '''
    sh = array.shape[1]
    
    for ii in tqdm(range(sh), unit='images'):    
           
        mask = array[:, ii, :] > epsilon 
        mask_norm = ndimage.filters.gaussian_filter(numpy.float32(mask), sigma = kernel)
        
        # Compute the filler:
        tmp = ndimage.filters.gaussian_filter(array[:, ii, :], sigma = kernel) / mask_norm      
        
        # Apply filler:                 
        array[:, ii, :][~mask] = tmp[~mask]        
        
def expand_medipix(array):
    '''
    Get the correct image size for a MEDIPIX data (fill in extra central lines)
    '''
    # Bigger array:
    sz = numpy.array(array.shape)
    sz[0] += 4
    sz[2] += 4
    new = numpy.zeros(sz, dtype = array.dtype)
    
    for ii in range(array.shape[1]):
        
        img = numpy.insert(array[: ,ii, :], 257, -1, axis = 0)
        img = numpy.insert(img, 256, -1, axis = 0)
        img = numpy.insert(img, 256, -1, axis = 0)
        img = numpy.insert(img, 255, -1, axis = 0)
    
        img = numpy.insert(img, 257-2, -1, axis = 1)
        img = numpy.insert(img, 256-2, -1, axis = 1)
        img = numpy.insert(img, 256-2, -1, axis = 1)
        img = numpy.insert(img, 255-2, -1, axis = 1)
        
        new[: ,ii, :] = img
        
    mask = img >= 0
    interpolate_holes(new, mask, kernel = [1,1])        
        
    return new  

def _parabolic_min_(values, index, space):    
    '''
    Use parabolic interpolation to find the extremum close to the index value:
    '''
    if (index > 0) & (index < (values.size - 1)):
        # Compute parabolae:
        x = space[index-1:index+2]    
        y = values[index-1:index+2]

        denom = (x[0]-x[1]) * (x[0]-x[2]) * (x[1]-x[2])
        A = (x[2] * (y[1]-y[0]) + x[1] * (y[0]-y[2]) + x[0] * (y[2]-y[1])) / denom
        B = (x[2]*x[2] * (y[0]-y[1]) + x[1]*x[1] * (y[2]-y[0]) + x[0]*x[0] * (y[1]-y[2])) / denom
            
        x0 = -B / 2 / A  
        
    else:
        
        x0 = space[index]

    return x0    
    
def norm(array, type = 'L2'):
    """
    Compute L2 norm of the array.
    """
    return numpy.sqrt(numpy.mean((array)**2))    
    
def _sample_FDK_(projections, geometry, sample):
    '''
    Compute a subsampled version of FDK
    '''
    geometry_ = geometry.copy()
    projections_ = projections[::sample[0], ::sample[2], ::sample[2]]
    
    # Apply subsampling to detector and volume:    
    vol_sample = [sample[0], sample[1], sample[2]]
    det_sample = [sample[0], sample[2], sample[2]]
    
    geometry_['vol_sample'] = vol_sample
    geometry_['det_sample'] = det_sample
    
    volume = projector.init_volume(projections_)
    
    # Do FDK without progress_bar:
    projector.settings.progress_bar = False
    projector.FDK(projections_, volume, geometry_)
    projector.settings.progress_bar = True
    
    return volume
    
def _modifier_l2cost_(projections, geometry, subsample, value, key, metric = 'gradient', preview = False):
    '''
    Cost function based on L2 norm of the first derivative of the volume. Computation of the first derivative is done by FDK with pre-initialized reconstruction filter.
    '''
    geometry_ = geometry.copy()
    
    geometry_[key] = value

    vol = _sample_FDK_(projections, geometry_, subsample)
    
    vol[vol < 0] = 0
    
    # Crop to central part:
    sz = numpy.array(vol.shape) // 8 + 1
    if vol.shape[0] < 3:
        vol = vol[:, sz[1]:-sz[1], sz[2]:-sz[2]]
    else:
        vol = vol[sz[0]:-sz[0], sz[1]:-sz[1], sz[2]:-sz[2]]
    
    #vol /= vol.max()

    l2 = 0
    
    for ii in range(vol.shape[0]):
        if metric == 'gradient':
            grad = numpy.gradient(numpy.squeeze(vol[ii, :, :]))
            grad = (grad[0] ** 2 + grad[1] ** 2)         
        
            l2 += numpy.mean(grad[grad > 0])
            
        elif metric == 'highpass':
        
            l2 += numpy.mean((ndimage.gaussian_filter(vol[ii, :, :], 2) - vol[ii, :, :])**2)
            
        elif metric == 'correlation':
            
            im = vol[ii, :, :]
            im = numpy.pad(im, ((im.shape[0],0), (im.shape[1],0)), mode = 'constant')
            im = numpy.fft.fft2(im)
            im = numpy.abs(numpy.fft.ifft2(im*numpy.conj(im)))
            #l2 += im[0,0]
            l2 += numpy.abs(numpy.mean(im * numpy.conj(im)))
            
        else:
            raise Exception('Unknown metric: ' + metric)
        
    if preview:
        display.slice(vol, title = 'Guess = %0.2e, L2 = %0.2e'% (value, l2))    
            
    return -l2    
    
def optimize_modifier(values, projections, geometry, samp = [1, 1, 1], key = 'axs_tan', metric = 'correlation', update = True, preview = False):  
    '''
    Optimize a geometry modifier using a particular sampling of the projection array.
    '''  
    maxiter = values.size
    
    # Valuse of the objective function:
    func_values = numpy.zeros(maxiter)    
    
    print('Starting a full search from: %0.3f' % values.min(), 'to %0.3f'% values.max())
    
    time.sleep(0.3) # To print TQDM properly
    
    ii = 0
    for val in tqdm(values, unit = 'point'):
        
        func_values[ii] = _modifier_l2cost_(projections, geometry, samp, val, key, metric = metric, preview = preview)
        
        ii += 1          
        
    min_index = func_values.argmin()    
    
    display.plot2d(values, func_values, title = 'Objective: ' + key)
    
    guess = _parabolic_min_(func_values, min_index, values)  
    
    if update:
        geometry[key] = guess
    
    print('Optimum found at %3.3f' % guess)
    
    return guess
        
def optimize_modifier_multires(projections, geometry, step, guess = None, subscale = 1, key = 'axs_tan', metric = 'correlation', preview = False):
    '''
    
    '''        
    print('The initial guess is %0.3f mm' % guess)
    
    # Downscale the array:
    while subscale >= 1:
        
        # Check that subscale is 1 or divisible by 2:
        if (subscale != 1) & (subscale // 2 != subscale / 2): ValueError('Subscale factor should be a power of 2! Aborting...')
        
        print('Subscale factor %1d' % subscale)    

        # We will use constant subscale in the vertical direction but vary the horizontal subscale:
        samp =  [5 * subscale, subscale, subscale]

        # Create a search space of 5 values around the initial guess:
        trial_values = numpy.linspace(guess - step * subscale, guess + step * subscale, 5)
        
        guess = optimize_modifier(trial_values, projections, geometry, samp, key = key, preview = preview)
                
        subscale = subscale // 2
    
    print('Old value:%0.3f' % geometry[key], 'new value: %0.3f' % guess)          
    geometry[key] = guess
    
    return guess

def optimize_rotation_center(projections, geometry, guess = None, subscale = 1, centre_of_mass = False, metric = 'highpass', preview = False):
    '''
    Find a center of rotation. If you can, use the center_of_mass option to get the initial guess.
    If that fails - use a subscale larger than the potential deviation from the center. Usually, 8 or 16 works fine!
    '''
    
    # Usually a good initial guess is the center of mass of the projection array:
    if  guess is None:  
        if centre_of_mass:
            
            print('Computing centre of mass...')
            guess = analyze.centre(projections)[2] * geometry.voxel[2]
        
        else:
        
            guess = geometry['axs_tan']
        
    guess = optimize_modifier_multires(projections, geometry, step = geometry.voxel[2], guess = guess, 
                                       subscale = subscale, key = 'axs_tan', metric = metric, preview = preview)
    
    return guess

def find_shift(volume_m, volume_s):
    
    if volume_m.max() == 0 or volume_s.max() == 0:
        return (0,0,0)
        
    # Find intersection:
    #mask_m = binary_threshold(volume_m, mode = 'otsu')
    #mask_s = binary_threshold(volume_s, mode = 'otsu')
    
    sect = volume_m[::2,::2,::2] * volume_s[::2,::2,::2]
    
    if sect.max() == 0:
        print('WARNING! Find shift fails bacause of no intersecting regions.')
        
    a,b,c = analyze.bounding_box(sect)
    a *= 2
    b *= 2
    c *= 2
    
    # Compute cross-correlation:
    vol_m = volume_m[a[0]:a[1], b[0]:b[1], c[0]:c[1]]
    vol_s = volume_s[a[0]:a[1], b[0]:b[1], c[0]:c[1]]
    
    vol_m = ndimage.filters.laplace(vol_m)
    vol_s = ndimage.filters.laplace(vol_s)
    
    #display.slice(vol_m, dim = 0, title = 'Master volume')
    #display.slice(vol_s, dim = 0, title = 'Slave volume')
    
    #display.slice(vol_m- vol_s, dim = 0, title = 'Diff before')
    
    vol_m = numpy.fft.fftn(vol_m)
    vol_m = -vol_m.conjugate()
    
    vol_s = numpy.fft.fftn(vol_s)
    
    vol_m *= vol_s
    vol_m = numpy.abs(numpy.fft.ifftn(vol_m))
    vol_m = numpy.fft.fftshift(vol_m)
    shift = numpy.unravel_index(numpy.argmax(vol_m), dims = vol_m.shape) - numpy.array(vol_m.shape)//2
        
    return shift

def _find_shift_(array_ref, array_slave, offset, dim = 1):    
    """
    Find a small 2D shift between two 3d images.
    """ 
    shifts = []
    
    # Look at a few slices along the dimension dim:
    for ii in numpy.arange(0, array_slave.shape[dim], 10):
        
        # Take a single slice:
        sl = data.anyslice(array_ref, ii, dim)    
        im_ref = numpy.squeeze(array_ref[sl]).copy()
        sl = data.anyslice(array_slave, ii, dim)    
        im_slv = numpy.squeeze(array_slave[sl]).copy()
        
        # Make sure that the array we compare is the same size:.        
        if (min(offset) < 0):
            raise Exception('Offset is too small!')
            
        if (offset[1] + im_slv.shape[1] > im_ref.shape[1])|(offset[0] + im_slv.shape[0] > im_ref.shape[0]):            
            raise Exception('Offset is too large!')
            
            # TODO: make formula for smaller total size of the total array
            
        im_ref = im_ref[offset[0]:offset[0] + im_slv.shape[0], offset[1]:offset[1] + im_slv.shape[1]]
            
        # Find common area:        
        no_zero = (im_ref * im_slv) != 0

        if no_zero.sum() > 0:
            im_ref *= no_zero
            im_slv *= no_zero
            
            # Crop:
            im_ref = im_ref[numpy.ix_(no_zero.any(1),no_zero.any(0))]    
            im_slv = im_slv[numpy.ix_(no_zero.any(1),no_zero.any(0))]                

            #flexUtil.slice(im_ref - im_slv, title = 'im_ref')
                                  
            # Laplace is way better for clipped objects than comparing intensities!
            im_ref = ndimage.laplace(im_ref)
            im_slv = ndimage.laplace(im_slv)
            
            #display.slice(im_ref, title = 'im_ref')
            #display.slice(im_slv, title = 'im_slv')
        
            # Shift registration with subpixel accuracy (skimage):
            shift, error, diffphase = feature.register_translation(im_ref, im_slv, 10)
                        
            shifts.append(shift)

    shifts = numpy.array(shifts)            
    
    if shifts.size == 0:        
        shift = [0, 0]
        
    else:
        # prune around mean:
        mean = numpy.mean(shifts, 0)    
        
        error = (shifts - mean[None, :])
        
        error = numpy.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
        mean = numpy.sqrt(mean[None, 0]**2 + mean[None, 1]**2)
        
        shifts = shifts[error < mean]

        if shifts.size == 0:
            
            shift = [0, 0]
            
        else:
            
            # total:        
            shift = numpy.mean(shifts, 0)    
            std = numpy.std(shifts, 0)
            
            shift_norm = numpy.sqrt(shift[0]**2+shift[1]**2)
            std_norm = numpy.sqrt(std[0]**2+std[1]**2)
    
            print('Found shift:', shift, 'with STD:', std)
            
            # Check that std is at least 2 times less than the shift estimate:
            if (std_norm > shift_norm / 2)|(shift_norm < 1):    
                    print('Bad shift. Discarding it.')
                    shift = [0, 0]
                
    return shift 

def _append_(total, new, x_offset, y_offset, pad_x, pad_y, base_dist, new_dist):
    """
    Append a new image to total via interpolation:
    """
    
    # Pad to match sizes:
    new = numpy.pad(new.copy(), ((0, pad_y), (0, pad_x)), mode = 'constant')  
    
    # Apply shift:
    if (x_offset != 0) | (y_offset != 0):   
        
        # Shift image:
        new = interp.shift(new, [y_offset, x_offset], order = 1)
    
    # Create distances to edge:
    return ((base_dist * total) + (new_dist * new)) / norm
    
def append_tile(array, geom, tot_array, tot_geom):
    """
    Append a tile to a larger arrayset.
    Args:
        
        array: projection stack
        geom: geometry descritption
        tot_array: output array
        tot_geom: output geometry
        
    """ 
        
    print('Stitching a tile...')               
    
    # Assuming all projections have equal number of angles and same pixel sizes
    total_shape = tot_array.shape[::2]
    det_shape = array.shape[::2]
    
    if tot_geom['det_pixel'] != geom['det_pixel']:
        raise Exception('This array has different detector pixels! %u v.s. %u. Aborting!' % (geom['det_pixel'], tot_geom['det_pixel']))
    
    if tot_array.shape[1] != array.shape[1]:
        raise Exception('This array has different number of projections from the others. %u v.s. %u. Aborting!' % (array.shape[1], tot_array.shape[1]))
    
    total_size = tot_geom.detector_size(total_shape)
    det_size = geom.detector_size(det_shape)
                    
    # Offset from the left top corner:
    y0, x0 = tot_geom.detector_centre()   
    y, x = geom.detector_centre()
    
    x_offset = ((x - x0) + total_size[1] / 2 - det_size[1] / 2) / geom.pixel[1]
    y_offset = ((y - y0) + total_size[0] / 2 - det_size[0] / 2) / geom.pixel[0]
    
    # Round em up!            
    x_offset = int(numpy.round(x_offset))                   
    y_offset = int(numpy.round(y_offset))                   
                
    # Pad image to get the same size as the total_slice:        
    pad_x = tot_array.shape[2] - array.shape[2]
    pad_y = tot_array.shape[0] - array.shape[0]  
    
    # Collapce both arraysets and compute residual shift
    shift = _find_shift_(tot_array, array, [y_offset, x_offset])
    
    x_offset += shift[1]
    y_offset += shift[0]
           
    # Precompute weights:
    base0 = (tot_array[:, ::100, :].mean(1)) != 0
    
    new0 = numpy.zeros_like(base0)
    # Shift image:
    new0[:det_shape[0], :det_shape[1]] = 1.0
    new0 = interp.shift(new0, [y_offset, x_offset], order = 1)
    #new0[y_offset:int(y_offset+det_shape[0]), x_offset:int(x_offset + det_shape[1])] = 1.0
    
    base_dist = ndimage.distance_transform_bf(base0)    
    new_dist =  ndimage.distance_transform_bf(new0)    
     
    # Trim edges to avoid interpolation errors:
    base_dist -= 1    
    new_dist -= 1
    
    base_dist *= base_dist > 0
    new_dist *= new_dist > 0
    norm = (base_dist + new_dist)
    norm[norm == 0] = numpy.inf
    
    time.sleep(0.5)
    
    # Apply offsets:
    for ii in tqdm(range(tot_array.shape[1]), unit='img'):   
        
        # Pad to match sizes:
        new = numpy.pad(array[:, ii, :], ((0, pad_y), (0, pad_x)), mode = 'constant')  
        
        # Apply shift:
        if (x_offset != 0) | (y_offset != 0):   
            
            # Shift image:
            new = interp.shift(new, [y_offset, x_offset], order = 1)
                    
        # Add two images in a smart way:
        base = tot_array[:, ii, :]  
        
        # Create distances to edge:
        tot_array[:, ii, :] = ((base_dist * base) + (new_dist * new)) / norm
        
def append_volume(array, geom, tot_array, tot_geom, ramp = 10):
    """
    Append a volume array to a larger arrayset.
    Args:
        
        array: projection stack
        geom: geometry descritption
        tot_array: output array
        tot_geom: output geometry
        
    """ 
    print('Stitching a volume block...')               
    
    display.slice(array, dim = 2, title = 'Append')
    display.slice(tot_array, dim = 2, title = 'Total')
    
    # Offset (pixel precision):   
    offset = numpy.array(geom['vol_tra']) / geom['img_pixel'] - numpy.array(array.shape) / 2
    offset -= numpy.array(tot_geom['vol_tra']) / tot_geom['img_pixel'] - numpy.array(tot_array.shape) / 2
    offset = numpy.round(offset).astype('int')
    
    # Create a slice of the big arrayset:
    s0 = slice(offset[0], offset[0] + array.shape[0])
    s1 = slice(offset[1], offset[1] + array.shape[1])
    s2 = slice(offset[2], offset[2] + array.shape[2])
    
    # Writable view on the total array:
    w_array = tot_array[s0, s1, s2]
    
    # Find shift:
    shift = find_shift(w_array, array)
    
    print('Found shift of :' + str(shift))
    
    if numpy.abs(shift).max() > 0: array = translate(array, -shift, order = 1)   
    print('Computing weigths.')
    # Ramp weight:
    weight = numpy.ones(array.shape, dtype = 'float16')
    weight = data.ramp(weight, 0, [ramp, ramp], mode = 'linear')
    weight = data.ramp(weight, 1, [ramp, ramp], mode = 'linear')
    weight = data.ramp(weight, 2, [ramp, ramp], mode = 'linear')
    
    # Weight can be 100% where no prior array exists:
    for ii in range(weight.shape[0]):
        weight[ii][w_array[ii] == 0] = 1
    print('Adding volumes.')
    # Apply weights and add (save memory):
    array *= weight
    
    weight -= 1
    weight *= -1
    
    w_array *= weight
    w_array += array                   
    
    display.slice(tot_array, dim = 2, title = 'Total')
    
def equivalent_density(projections, geometry, energy, spectr, compound, density, preview = False):
    '''
    Transfrom intensity values to projected density for a single material array
    '''
    # Assuming that we have log array!

    print('Generating the transfer function.')
    
    if preview:
        display.plot2d(energy, spectr, semilogy=False, title = 'Spectrum')
    
    # Attenuation of 1 mm:
    mu = model.linear_attenuation(energy, compound, density)
    
    # Make thickness range that is sufficient for interpolation:
    #m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    #img_pix = geometry['det_pixel'] / m
    img_pix = geometry['img_pixel']

    thickness_min = 0
    thickness_max = max(projections.shape) * img_pix * 2
    
    print('Assuming thickness range:', [thickness_min, thickness_max])
    thickness = numpy.linspace(thickness_min, thickness_max, max(projections.shape))
    
    exp_matrix = numpy.exp(-numpy.outer(thickness, mu))
        
    synth_counts = exp_matrix.dot(spectr)
    
    #flexUtil.plot(thickness, title = 'thickness')
    #flexUtil.plot(mu, title = 'mu')
    #flexUtil.plot(synth_counts, title = 'synth_counts')
    
    if preview:
        display.plot2d(thickness,synth_counts, semilogy=True, title = 'Attenuation v.s. thickness [mm].')
        
    synth_counts = -numpy.log(synth_counts)
    
    print('Callibration attenuation range:', [synth_counts[0], synth_counts[-1]])
    print('array attenuation range:', [projections.min(), projections.max()])

    print('Applying transfer function.')    
    
    time.sleep(0.5) # Give time to print messages before the progress is created
    
    for ii in tqdm(range(projections.shape[1]), unit = 'img'):
        
        projections[:, ii, :] = numpy.array(numpy.interp(projections[:, ii, :], synth_counts, thickness * density), dtype = 'float32') 
               
    return projections
