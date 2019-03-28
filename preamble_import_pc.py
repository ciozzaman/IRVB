# Created 03/12/2018
# Fabio Federici

import numpy as np
from scipy import misc
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy.optimize import curve_fit
from scipy.optimize import newton_krylov	# added 2018-11-13 to replace Fenics for heat transfer simulations
# import matplotlib
# matplotlib.use('Agg')	# this line allows to save output from matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm	# added 2018-11-17 to allow logarithmic scale plots
import math
import statistics as s
import csv
import scipy.stats
import os,sys
import xarray as xr	# added 2018-11-05 to allow importing data from SOLPS

os.chdir("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval")
import matplotlib.animation as animation
import warnings
from scipy.interpolate import RectBivariateSpline,interp2d
import peakutils
import pickle
import collections
import collect_and_eval as coleval
import copy
from scipy.interpolate import interp1d


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


#This function will generate something similar to thermal radiation
def therm(x, *params):

	A = params[0]
	off = params[1]
	c = params[2]
	# y0 = params[3]
	#n = params[4]
	return A*((np.array(x)+off)**4-300.**4)-c*(np.array(x))

# #This function will generate a polinomial of n order
# def polygen(n):
# 	def polyadd(x, *params):
# 		temp=0
# 		for i in range(n):
# 			temp+=params[i]*x**i
# 		return temp
# 	return polyadd
#
# 	A = params[0]
# 	off = params[1]
# 	c = params[2]
# 	# y0 = params[3]
# 	#n = params[4]
# 	return A*((np.array(x)+off)**4-300.**4)-c*(np.array(x))

# This function will generate a line

def line(x, *params):

	m = params[0]
	q = params[1]
	return m*x+q

def line_throug_zero(x, *params):

	m = params[0]
	# q = params[1]
	return m*x


def costant(x, *params):
	import numpy as np

	c = params[0]
	len=np.ones((np.shape(x)))
	return c*len

def square_wave(x, *params):

	A = params[0]
	phase = params[1]
	freq = params[2]
	if ((x*freq+phase) % 1)>=0.5:
		onoff=1
	else:
		onoff=0
	return A*onoff
