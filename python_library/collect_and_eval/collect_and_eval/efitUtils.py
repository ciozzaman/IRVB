import math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.constants as constants
import pandas as pd
from shapely.geometry import LineString,Polygon
from . import eu_numerics as eun

# -----------------------------------
# Miscellaneous functions
# -----------------------------------
# def range1(start, end):
#     ran = range(start, end+1)
#
#     return ran[0], ran[-1]
#
# def myround(a, decimals=1):
#     return np.around(a-10**(-(decimals+5)), decimals=decimals)

def string_is_number(string_test):

    try:
        float(string_test)
    except ValueError:
        return False

    return True

def writeColumns(fp, a, nc, fmt):
    i=a.size
    t=tuple(a.reshape(1, -1)[0])
    print((i,nc,fmt))
    ks=0
    while True:
      if i >= nc:
        k=nc
      else:
        k=i
      ke=ks+k
      fstr = k*fmt
      if k > 0:
        print(fstr%( t[ks:ke]), file=fp)
      if i < nc:
        break
      i=i-nc
      ks=ks+nc

def toPrecision(x,p):
    """
    returns a string representation of x formatted with a precision of p
    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

def isOutlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", The ASQC Basic References in Quality Control:
    Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def interp_array(old_time, new_time, old_data, scale_factor=1.0):
    data_interp = np.zeros(len(new_time))
    data_interp[:] = np.nan

    # We can only interpolate in the time frame of the new data
    time_cp = np.copy(new_time)
    ind_keep = np.where((time_cp < np.max(old_time)) & (time_cp > np.min(old_time)))
    time_cp = time_cp[ind_keep]

    # Interpolate
    interp_fn = interpolate.interp1d(old_time, old_data)
    data_interp[ind_keep] = interp_fn(time_cp)

    # Scale
    data_interp *= scale_factor

    return data_interp

# -----------------------------------
# Functions to determine things relating to the plasma
# -----------------------------------
def find_flattop(time, ip, ip_thresh=0.0, flattop_perc_thresh=0.95, smoothing_window=20, return_indices=False, timemargin=0.0):

    ind_ip, = np.where(ip > ip_thresh)

    if len(ind_ip) == 0:
        return (None, None)

    ip_smooth = pd.Series(ip[ind_ip]).rolling(window=smoothing_window, center=False).mean()
    ip_max = np.max(ip_smooth)
    ind_ft, = np.where(ip_smooth > ip_max*flattop_perc_thresh)

    if len(ind_ft) == 0:
        return (None, None)

    ind_ft_start = ind_ip[ind_ft[0]]
    ind_ft_end = ind_ip[ind_ft[-1]]
    time_ft_start = time[ind_ft_start]
    time_ft_end = time[ind_ft_end]

    # Adjust for timemargin if given
    if timemargin > 0.0:
        ind_ft_start = np.argmin(time - (time_ft_start + timemargin))
        ind_ft_end = np.argmin(time - (time_ft_end - timemargin))
        time_ft_start = time[ind_ft_start]
        time_ft_end = time[ind_ft_end]

    if return_indices:
        return (ind_ft_start, ind_ft_end)

    return (time_ft_start, time_ft_end)


def get_boundary_classification(ed, time_index, snd_psin_thresh=0.03):
    limloc = 'SNB'  # Default to lower single null

    # Hardcoded aeqdsk values (shouldn't really be hardcoded...)
    # limloc: IN,OUT,TOP,BOT for limited plasma, SNT,SNB,DN for diverted plasmas
    if ed.data["separatrixGeometry"].boundaryClassificationInteger.values[time_index] == 0:
        # Determine if it is limited on inner, outer, top or bottom of the machine
        lim_contact_r = ed.data["separatrixGeometry"].limiterR.values[time_index]
        lim_contact_z = ed.data["separatrixGeometry"].limiterZ.values[time_index]

        # Accurate definition of IN, OUT, TOP, BOT would depend on how these are defined for a given machine
        # So make a guess
        # Check if we are close to the minimum/maximum R of the limiting surface.
        lim_min_r = ed.data["geometry"]["limiter"].rValues.values.min()
        lim_max_r = ed.data["geometry"]["limiter"].rValues.values.min()
        if np.absolute(lim_contact_r - lim_min_r) < 0.1:
            limloc = 'IN'
        elif np.absolute(lim_contact_r - lim_max_r) < 0.1:
            limloc = 'OUT'
        elif lim_contact_z < 0.0:
            limloc = 'BOT'
        else:
            limloc = 'TOP'
    else:
        # For diverted plasmas, check psi difference between the x-points (if more than one is found)
        try:
            xpoint_count = ed.data["separatrixGeometry"].dndXpointCount.values[time_index]
            xp1_r = ed.data["separatrixGeometry"].dndXpoint1R.values[time_index]
            xp1_z = ed.data["separatrixGeometry"].dndXpoint1Z.values[time_index]
            xp2_r = ed.data["separatrixGeometry"].dndXpoint2R.values[time_index]
            xp2_z = ed.data["separatrixGeometry"].dndXpoint2Z.values[time_index]
        except AttributeError:
            # Relying on xpointCount is unreliable for MAST because only the x-points considered to be on
            # the boundary (ie. CDND) are counted. But try anyway.
            xpoint_count = ed.data["separatrixGeometry"].xpointCount.values[time_index]
            xp1_r = ed.data["separatrixGeometry"].xpointR.values[time_index, 0]
            xp1_z = ed.data["separatrixGeometry"].xpointZ.values[time_index, 0]
            xp2_r = ed.data["separatrixGeometry"].xpointR.values[time_index, 1]
            xp2_z = ed.data["separatrixGeometry"].xpointZ.values[time_index, 1]
            if np.isclose(xp1_r, 0):
                xp1_r = np.nan
                xp1_z = np.nan
            if np.isclose(xp2_r, 0):
                xp2_r = np.nan
                xp2_z = np.nan

        if xpoint_count < 2 and not np.isnan(xp1_r) and xp1_z > 0.0:
            limloc = 'SNT'
        elif xpoint_count < 2 and not np.isnan(xp2_r) and xp2_z < 0.0:
            limloc = 'SNB'
        else:
            # Get psi-norm values of the two x-points
            func = eun.spline2d(ed.data["profiles2D"].r[time_index, :].values,
                                ed.data["profiles2D"].z[time_index, :].values,
                                ed.data["profiles2D"].psiNorm[time_index, :, :].values,
                                kx=3, ky=3)
            psin_xp1 = func(xp1_r, xp1_z)
            psin_xp2 = func(xp2_r, xp2_z)

            if psin_xp1 < psin_xp2:
                primary_z = xp1_z
            else:
                primary_z = xp2_z

            psin_diff = np.absolute(psin_xp1 - psin_xp2)

            if psin_diff > snd_psin_thresh and primary_z > 0.0:
                limloc = 'SNT'
            elif psin_diff > snd_psin_thresh and primary_z < 0.0:
                limloc = 'SNB'
            else:
                limloc = 'DN'

    return limloc

# -----------------------------------
# Helper functions for plotting
# -----------------------------------
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
    cmap : The matplotlib colormap to be altered
    start : Offset from lowest point in the colormap's range.
    Defaults to 0.0 (no lower ofset). Should be between
    0.0 and `midpoint`.
    midpoint : The new center of the colormap. Defaults to
    0.5 (no shift). Should be between 0.0 and 1.0. In
    general, this should be  1 - vmax/(vmax + abs(vmin))
    For example if your data range from -15.0 to +5.0 and
    you want the center of the colormap at 0.0, `midpoint`
    should be set to  1 - 5/(5 + 15)) or 0.75
    stop : Offset from highets point in the colormap's range.
    Defaults to 1.0 (no upper ofset). Should be between
    `midpoint` and 1.0.
    '''
    cdict = {
    'red': [],
    'green': [],
    'blue': [],
    'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
    np.linspace(0.0, midpoint, 128, endpoint=False),
    np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap

def get_svg_path_text(xpoints, ypoints):
    path = "M {},{}".format(xpoints[0], ypoints[0])

    for xpoint, ypoint in zip(xpoints[1:], ypoints[1:]):
        path += " L{},{}".format(xpoint, ypoint)

    path += " z"

    return path

# -----------------------------------
# General data manipulation functions
# -----------------------------------
def find_index_flatline_zero(dataArray, errorAbs, axis=0):
    data_average = np.nanmean(dataArray, axis=axis)
    data_std = np.nanstd(dataArray, axis=axis)
    error_average = np.nanmean(errorAbs, axis=axis)
    ind_zero, = np.where((np.absolute(data_average) < error_average) & (data_std < error_average))

    return ind_zero

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def make2DGrid(psinorm_2d, xvals, yvals):
    fMap = interpolate.Rbf(xvals, yvals,function='multiquadric',epsilon=1e-20)
    var_grid=fMap(np.transpose(psinorm_2d))

    return var_grid

def polygonLineIntersection(polygon_r, polygon_z, line_start, line_end):

    coordTuple = list(zip(polygon_r, polygon_z))
    polygonBoundary = Polygon(coordTuple)

    try:
        line = LineString([line_start, line_end])
        intersectPoints = polygonBoundary.intersection(line)
    except:
        return None

    return intersectPoints

def findLOSTemperatures(rcoords_pressure, pressure, rcoords_density, density, temperatures):
    if rcoords_pressure.shape != rcoords_density.shape: 
        #raise Exception('Pressure and density arrays must be the same shape.')
        print('findLOSTemperatures: pressure and density arrays must be the same shape. Returning.')
        return []
    if not np.allclose(rcoords_pressure, rcoords_density):
        #raise Exception('findLOSTemperatures only implemented for pressure & density on measured at same points.')
        print('findLOSTemperatures: only implemented for pressure & density measured at the same points. Returning.')
        return []

    rCoords = rcoords_pressure
    temperature_vals = pressure / density / constants.elementary_charge

    radius_vals = []

    # set the edge channels to zero temperature and remove any bad channels further inside.
    i = -1
    while np.isnan(temperature_vals[i]):
        i = i - 1
    temperature_vals[i + 1:-1] = 0
    finite_vals = ~np.isnan(temperature_vals)
    rCoords = rCoords[finite_vals]
    temperature_vals = temperature_vals[finite_vals]

    for k, edgeTemperat in enumerate(temperatures):
        yreduced = temperature_vals - edgeTemperat

        tck = interpolate.splrep(rCoords, yreduced)
        roots = interpolate.sproot(tck)

        if roots.size == 1:
            radius_vals.append(roots[0])

    return radius_vals

def get_vertices_efit_element(r, z, dr, dz, ang1, ang2, close_path=False):
    if ang1 == np.nan:
        ang1 = 0.0
    if ang2 == np.nan:
        ang2 = 0.0

    if np.abs(ang1) > 0.0:
        tangle1 = np.tan(ang1 * np.pi / 180.0)
    else:
        tangle1 = 0.0

    if np.abs(ang2) > 0.0:
        tangle2 = 1.0 / np.tan(ang2 * np.pi / 180.0)
    else:
        tangle2 = 0.0

    rr = [r - dr / 2. - dz / 2. * tangle2,
          r + dr / 2. - dz / 2. * tangle2,
          r + dr / 2. + dz / 2. * tangle2,
          r - dr / 2. + dz / 2. * tangle2]
    zz = [z - dz / 2. - dr / 2. * tangle1,
          z - dz / 2. + dr / 2. * tangle1,
          z + dz / 2. + dr / 2. * tangle1,
          z + dz / 2. - dr / 2. * tangle1]

    if close_path:
        rr.append(rr[0])
        zz.append(zz[0])

    return (rr,zz)

# -----------------------------------
# Functions acting on xarray DataArray with no dependence on efitplotData class structure
# -----------------------------------
def fillinf(dataArray, fillvalue):
    isinf = np.isinf(dataArray.values)
    dataArray.values[np.where(isinf)] = fillvalue

# -----------------------------------
# Functions acting on xarray DataSet which depend on the efitplotData class structure
# -----------------------------------
def chi2totalcalc(dataSet, norm=False, weighted=False):

    # Total chi2 as a function of time
    if len(dataSet.chi2.dims) == 1:
        if norm and weighted:
            dataArray = dataSet.chi2WN.fillna(0.0)
        elif weighted:
            dataArray = dataSet.chi2W.fillna(0.0)
        else:
            dataArray = dataSet.chi2.fillna(0.0)
    else:
        sum_dims = [x for x in dataSet.chi2.dims if x != 'time']

        if norm and weighted:
            dataArray = dataSet.chi2WN.fillna(0.0).sum(dim=sum_dims)
        elif weighted:
            dataArray = dataSet.chi2W.fillna(0.0).sum(dim=sum_dims)
        else:
            dataArray = dataSet.chi2.fillna(0.0).sum(dim=sum_dims)

    return dataArray

def errorsCalc(dataSet, computedname="computed", targetname="target"):
    dataSet = dataSet.assign(error=(dataSet.data_vars[targetname] - dataSet.data_vars[computedname]))
    dataSet = dataSet.assign(absError=(np.abs(dataSet.data_vars[targetname] - dataSet.data_vars[computedname])))
    dataSet = dataSet.assign(percError=(200.0 * dataSet.error / (dataSet.data_vars[targetname] + dataSet.data_vars[computedname])))

    return dataSet

def chi2calc(dataSet, computedname="computed", targetname="target", sigmaname="sigma"):
    if sigmaname not in dataSet.data_vars.keys():
        sigmaname = "sigmas"

    dataSet = dataSet.assign(parameters = np.count_nonzero(dataSet.weights))
    dataSet = dataSet.assign(chi2=( np.abs(dataSet.data_vars[targetname] - dataSet.data_vars[computedname])
                                      / np.abs(dataSet.data_vars[sigmaname]) )**2 )
    dataSet = dataSet.assign(chi2W=( dataSet.weights * np.abs(dataSet.data_vars[targetname] - dataSet.data_vars[computedname])
                                      / np.abs(dataSet.data_vars[sigmaname])) ** 2 )
    dataSet = dataSet.assign(chi2WN= 1.0 / dataSet.parameters *
                                     (dataSet.weights * np.abs(dataSet.data_vars[targetname] - dataSet.data_vars[computedname])
                                    / np.abs(dataSet.data_vars[sigmaname])) ** 2)

    # Set infinite values to 0
    fillinf(dataSet.chi2, 0.0)
    fillinf(dataSet.chi2W, 0.0)
    fillinf(dataSet.chi2WN, 0.0)

    return dataSet
