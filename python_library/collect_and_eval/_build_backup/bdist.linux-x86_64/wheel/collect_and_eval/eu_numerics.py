"""
Module defining general numerical and data processing functions.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import sys
#import pdb


#----------------------------------------------------------------------
def spline2d(r, z, dat, *args, **kwargs):
    """
    Wrapper for scipy.interpolate.RectBivariateSpline.
    """
    return RectBivariateSpline(r, z, dat, *args, **kwargs)


#----------------------------------------------------------------------
def interp_nan(dat):
    """
    Replace NaN-s with linear interpolation in 1D numpy arrays.
    """
    if ( np.isnan(np.sum(dat)) ):
        nanmask = np.isnan(dat)
        nonzero = lambda z: z.nonzero()[0]
        dat[nanmask] = np.interp( nonzero(nanmask), nonzero(~nanmask), dat[~nanmask] )

    return dat


#----------------------------------------------------------------------
def iterate_dims(dat, axis, function, *args, **kwargs):
    """
    Apply function on selected axis iterating over all other dimensions 
    of the input array dat. Can be used e.g. with smooth, interpolate, etc.
    """
    # Make selected axis the last one (i.e. the fastest changing index in C-like row major order):
    tmp = dat.swapaxes(axis, len(dat.shape)-1)
    saveshape = tmp.shape
 
    # Flatten array into 1D. After the swapaxes operation the values along the selected
    # axis will be contiguous in the 1D array.
    tmp = tmp.ravel()
 
    # Apply function for each section of the 1D array representing the selected axis, taking into
    # account if the function changes the length of the input array.
    lenaxis = dat.shape[axis]
    nsections = int(len(tmp) / lenaxis)
    try:
        newlen = len(function(tmp[0:lenaxis], *args, **kwargs))
    except TypeError:
        newlen = 1
    out = np.zeros(newlen * nsections)
 
    for i in range(nsections):
        out[i*newlen:(i+1)*newlen] = function(tmp[i*lenaxis:(i+1)*lenaxis], *args, **kwargs)
 
    # Reshape output array to have the same number of dimensions as dat:
    newshape = saveshape[0:-1] + (newlen,)
    out = out.reshape(newshape)
 
    # Swap axes back into original position:
    out = out.swapaxes(axis, len(dat.shape)-1) 

    return out


#----------------------------------------------------------------------
def iterate_dims_coord(dat, axis, function, *args, coord=None, newcoord=None, **kwargs):
    """
    Apply function on selected axis iterating over all other dimensions 
    of the input array dat. Can be used e.g. with smooth, interpolate, etc.

    The optional coord and newcoord arguments are passed to function to
    enable handling non-constant dimension arrays.
    """
    # Make selected axis the last one (i.e. the fastest changing index in C-like row major order):
    tmp = dat.swapaxes(axis, len(dat.shape)-1)
    saveshape = tmp.shape
 
    # Flatten array into 1D. After the swapaxes operation the values along the selected
    # axis will be contiguous in the 1D array.
    tmp = tmp.ravel()
 
    # Calculate the number of sections the axis fits in the flattened array.
    lenaxis = dat.shape[axis]
    nsections = int(len(tmp) / lenaxis)

    if (coord is not None) and (newcoord is not None):

        newcoord, nnc = to_iterable(newcoord, array=True)

        if dat.shape != coord.shape:
            raise IOError('dat and coord must be the same shape.')

        tmp_coord = coord.swapaxes(axis, len(coord.shape)-1)
        tmp_coord = tmp_coord.ravel()

        if newcoord.squeeze().ndim < 2:
            # Scalar or 1D
            tmp_newcoord = np.tile(newcoord, nsections)
            newlen = nnc
        else:
            tmp_newcoord = newcoord.swapaxes(axis, len(newcoord.shape)-1)
            tmp_newcoord = tmp_newcoord.ravel()
            newlen = newcoord.shape[axis]

        out = np.zeros(newlen * nsections)

        for i in range(nsections):
            # It is assumed that coord and newcoord are the second and third non-keyword arguments.
            # It is also assumed that there are no arbitrary non-keyword arguments.
            out[i*newlen:(i+1)*newlen] = function(tmp[i*lenaxis:(i+1)*lenaxis], tmp_coord[i*lenaxis:(i+1)*lenaxis], tmp_newcoord[i*newlen:(i+1)*newlen], **kwargs)

    else:
        try:
            newlen = len(function(tmp[0:lenaxis], *args, **kwargs))
        except TypeError:
            newlen = 1

        out = np.zeros(newlen * nsections)

        for i in range(nsections):
            out[i*newlen:(i+1)*newlen] = function(tmp[i*lenaxis:(i+1)*lenaxis], *args, **kwargs)
 
    # Set the shape tuple of the output array:
    newshape = saveshape[0:-1] + (newlen,)

    # Reshape output array to have the same number of dimensions as dat:
    out = out.reshape(newshape)
 
    # Swap axes back into original position:
    out = out.swapaxes(axis, len(dat.shape)-1) 

    return out


#----------------------------------------------------------------------
def interp(dat, x, newx, order=3, ext=0):
    """
    Create an interpolating spline object for the 1D array dat(x), and 
    evaluate it on newx. 
    """
    #x = np.sort(x)

    dat, ndat = to_iterable(dat)
    newx, nnewx = to_iterable(newx)

    if len(dat) < order+1:
        raise IOError('Data array must be longer than %s for spline interpolation with order %s.' %(order+1, order))  
    if len(dat) != len(x):
        raise IOError('dat and x arrays must be the same length for spline interpolation.')
    if len(newx) < 1:
        raise IOError('No values specified for evaluating spline.')

    spl = InterpolatedUnivariateSpline(x, dat, k=order, ext=ext)
    out = spl(newx)

    return out


#----------------------------------------------------------------------
def interp_dim1(dat, x, newx, order=3, ext=0):
    """
    Interpolate a 2D data array along dim1 (2nd dimension). Both the original
    x grid and newgrid can be either 1D or 2D (varying along dim0).
     
    """
    try:
        tmp = dat.shape
        tmp = x.shape
        tmp = newx.shape
    except AttributeError:
        print('Input arguments dat, x and newx have to be numpy arrays. Aborting.')
        sys.exit()

    if (x.ndim == 1 and x.shape[0] == dat.shape[1]) or (x.shape == (1, dat.shape[1])):
        # Repeat rows of 1D x array to form the right size 2D array.
        x = np.tile(x, (dat.shape[0],1))
    elif x.shape == dat.shape:
        pass
    else:
        print('Input arguments dat and x have inconsistent size. Aborting.')
        #sys.exit()

    if (newx.ndim == 1) or (newx.ndim == 2 and newx.shape[0] == 1):
        # Repeat rows of 1D newx array to form the right size 2D array.
        newx = np.tile(newx, (dat.shape[0],1))
    elif newx.ndim == 2 and newx.shape[0] == dat.shape[0]:
        pass
    else:
        print('Input arguments dat and newx have inconsistent size. Aborting.')
        sys.exit()

    out = np.zeros(newx.shape)

    for i in range(dat.shape[0]):
        out[i,:] = interp(dat[i,:], x[i,:], newx[i,:], order=order, ext=ext)

    return out


#----------------------------------------------------------------------
def apply_func_dim1(dat, x, func, *args, **kwargs):
    """
    Apply function along dim1 (2nd dimension) of a 2D array. The x grid
    used in the function call can be either 1D or 2D (varying along dim0).
     
    """
    try:
        tmp = dat.shape
        tmp = x.shape
    except AttributeError:
        print('Input arguments dat and  x have to be numpy arrays. Aborting.')
        sys.exit()

    if (x.ndim == 1 and x.shape[0] == dat.shape[1]) or (x.shape == (1, dat.shape[1])):
        # Repeat rows of 1D x array to form the right size 2D array.
        x = np.tile(x, (dat.shape[0],1))
    elif x.shape == dat.shape:
        pass
    else:
        print('Input arguments dat and x have inconsistent size. Aborting.')
        sys.exit()

    try:
        newlen = len(func(dat[0,:], x[0,:], *args, **kwargs))
    except TypeError:
        newlen = 1
    out = np.zeros((dat.shape[0], newlen))

    for i in range(dat.shape[0]):
        out[i,:] = func(dat[i,:], x[i,:], *args, **kwargs)

    return out


#----------------------------------------------------------------------
def to_iterable(dat, array=False):
    """
    Convert scalar to list or numpy array.
    """
    if type(dat) is not np.ndarray:
        try:
            ndat = len(dat)
        except TypeError:
            dat = [dat]
            ndat = 1
    
        if array:
            dat = np.array(dat)

    else:
        if dat.ndim == 0:
            dat = dat[np.newaxis]

        ndat = dat.size

    return dat, ndat


#----------------------------------------------------------------------
def intersect(a, b, c, d, debug=0):
    """
    Find the coordinates of the intersection of two straight line segments.
    a, b, c, d are each a list of [x,y] coordinates of 4 points. 
    Line1 is defined by (a,b), line2 is defined by (c,d). 
    """

    x = 1234567.
    y = 1234567.

    denom = (d[1]-c[1]) * (b[0]-a[0]) - (d[0]-c[0]) * (b[1]-a[1])

    if (abs(denom) < 1.e-6):
        ier = -1
        if debug > 0:
            print('Line segments are parallel.')
    else:
        u1 = ( (d[0]-c[0]) * (a[1]-c[1]) - (d[1]-c[1]) * (a[0]-c[0]) ) / denom
        u2 = ( (b[0]-a[0]) * (a[1]-c[1]) - (b[1]-a[1]) * (a[0]-c[0]) ) / denom

        if ((u1 >= 0) and (u1 <= 1) and (u2 >= 0) and (u2 <= 1)):
            if debug > 0:
                print('Line segments intersect.')
            x = a[0] + u1 * (b[0]-a[0])
            y = a[1] + u1 * (b[1]-a[1])
            ier = 0
        else:
            if debug > 0:
                print('Line segments do not intersect.')
            ier = -2

    return x, y, ier


#----------------------------------------------------------------------
def smooth(x, window_len=3, window='flat'):
    """
    Smoothing function based on convolution of the signal with the selected window,
    from www.scipy.org/Cookbook/SignalSmooth.
    """

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')
  
    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')
  
    if window_len < 3:
        return x
  
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.")
  
    # Mirror and append a window_len long part of x at the beginning and end:
    s=np.r_[ x[window_len-2::-1], x, x[-1:-window_len:-1] ]
  
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
  
    # Convolution when the two signals overlap:
    #y = np.convolve(w/w.sum(), s, mode='valid')
    # Remove extra points at the beginning and end: (symmetric with odd window width, asymetric with even)
    y = np.convolve(w/w.sum(), s, mode='valid')[int(window_len/2):-int((window_len-1)/2)]
  
    return y


#----------------------------------------------------------------------
def deriv(y, x, method=2):

    if (method==1):
        """
        Derivative of y(x) calculated with first order centered differences.
        Only works with equidistant grids.
        """
        if all(np.diff(np.diff(x)) < 1e-6):
            yprime = np.gradient(y,np.diff(x)[0])
        else:
            print('Grid not equidistant, derivative method 2 is used.')
            method = 2

    if (method==2):
        """
        Derivative of y(x) array calculated with Lagrangian interpolation polynomial.
        See IDL deriv documentation for details of the formula. 
        The input arguments x and y have to be the same length 1D arrays. 
        """
        ny = np.size(y)
        yprime = np.zeros(ny)
        looparray = np.arange(1,ny-1)
        for i in looparray:
            x01 = x[i-1] - x[i]
            x02 = x[i-1] - x[i+1]
            x12 = x[i] - x[i+1]

            # centre:
            yprime[i] = y[i-1]*x12/(x01*x02) + y[i]*(1/x12-1/x01) - y[i+1]*x01/(x02*x12)
            if (i==1): #first point
                yprime[i-1] = y[i-1]*(x01+x02)/(x01*x02) - y[i]*x02/(x01*x12) + y[i+1]*x01/(x02*x12)
            elif (i==ny-2): #last point
                yprime[i+1] = -y[i-1]*x12/(x01*x02) + y[i]*x02/(x01*x12) - y[i+1]*(x02+x12)/(x02*x12)

    return yprime


#----------------------------------------------------------------------
def select_interval_indices(dat, intervals):
    """
    Select points from ordered 1D array dat that fall into a list of
    intervals defined as [[start1, start2, ...], [end1, end2, ...]]
    """
    sorted = all(dat[i] <= dat[i+1] for i in range(len(dat)-1))
    if not sorted:
        print('Input array not sorted. Aborting.')
        return

    sel = []
    isel = []
    for i in range(len(intervals[0])):
        ilow = np.where(dat > intervals[0][i])[0][0]
        iupp = np.where(dat < intervals[1][i])[0][-1]
        for j in range(ilow, iupp):
            sel.append(t[j])
            isel.append(j)

    return np.array(sel), np.array(isel)


#------------------------------------------------------------------------------
def find_nearest(array, value):
    """
    Finds element nearest to value in array.
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


#------------------------------------------------------------------------------
def mirror_profile(dat, insert_midpoint=True):
    """
    Concatenate 1D data array with its mirror.
    """
    nd = len(dat)
    if insert_midpoint:
        out = np.zeros(2 * nd + 1)
        out[0:nd] = np.flip(dat, axis=0)
        out[nd] = dat[0]
        out[nd+1:] = dat
    else:
        out = np.zeros(2 * nd)
        out[0:nd] = np.flip(dat)
        out[nd:] = dat

    return out


#------------------------------------------------------------------------------
def compute_status_flag(dat, sigma, r=None, z=None, minval=None, abs_error_threshold=None, rel_error_threshold=None, rmax=3.9, rmin=0., zmax=2., zmin=-2., rrange=None, mask=None):
    """
    Compute status flag for each data point based on their error bar.
    rrange: a list of major radius ranges. If present, takes precedence
    over rmin and rmax.
    """
    if dat.shape != sigma.shape:
        raise IOError('Shape of input arrays must be the same.')

    sflag = np.zeros(dat.shape)

    if mask is not None:
        try:
            sflag[mask > 0] = 1
        except:
            print('compute_status_flag (warning): Invalid mask array.')

    # Points which are nan
    sflag[np.isnan(dat)] = 1

    # Points with relative error larger than input threshold value:
    if rel_error_threshold is not None:
        sflag[sigma / np.abs(dat) > rel_error_threshold] = 1

    # Points with absolute error larger than input threshold value:
    if abs_error_threshold is not None:
        sflag[sigma > abs_error_threshold] = 1

    # Points with values lower than input minimum:
    if minval is not None:
        sflag[np.abs(dat) < minval] = 1

    # Points outside region of interest:
    if r is not None:
        if dat.shape != r.shape:
            raise IOError('Shape of input arrays must be the same.')
        if rrange == None:
            sflag[r > rmax] = 1
            sflag[r < rmin] = 1
        else:
            sflag_tmp = np.ones(dat.shape)
            for i,rng in enumerate(rrange):
                rmin_tmp = min(rng)
                rmax_tmp = max(rng)
                sflag_tmp[np.logical_and(r < rmax_tmp, r > rmin_tmp)] = 0
            sflag[sflag_tmp == 1] = 1

    if z is not None:
        if dat.shape != z.shape:
            raise IOError('Shape of input arrays must be the same.')
        sflag[z > zmax] = 1
        sflag[z < zmin] = 1

    return sflag


#------------------------------------------------------------------------------
def compute_weights(dat, sflag=None, weights=1, dim=None):
    """
    Set the fitting weights for the dat array taking into account the status flag.    
    If dim is provided, the weights are only applied along the selected dimension.
    """
    weights, nweights = to_iterable(weights, array=True)

    if dim is None:
        weights_array = np.zeros(dat.shape)
    else:
        weights_array = np.zeros(dat.shape[dim])

    try:
        # Automatic broadcasting if weights is a scalar, same size as weights_array, or same size as weights_array.shape[1].
        weights_array = weights_array + weights
    except ValueError:
        # Computed manually if weights is same size as weights_array.shape[0].
        if weights.shape[0] == weights_array.shape[0]:
            weights_array = weights_array + np.tile(weights, (weights.shape[1],1)).transpose()
    except:
        raise IOError('Shape of data and weights arrays not compatible.')

    # Set weight to 0 where the status flag is raised:
    if sflag is not None:
        weights_array[np.where(sflag > 0)] = 0

    return weights_array



