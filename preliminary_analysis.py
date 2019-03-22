import numpy as np
from scipy import misc
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import statistics as s
import csv
import scipy.stats
import os,sys

os.chdir("/home/ffederic/work/python_library/collect_and_eval")
import collect_and_eval as coleval
import matplotlib.animation as animation
import warnings
from scipy.interpolate import RectBivariateSpline,interp2d
import peakutils



# script is to estimate the count/temperature dempendance of the IR camera
#
# IR camera calibration with hot NUC plate
# 07/03/2018 FF/MR
#
# Multimeter	Fluke 179
# Thermocouple Bead temperature probe / K thermocouple (1) 4jpl6 -30/300°C
# IR camera FLIR SC7000
# Hoven APEXA Appliance number SW39 0000072208
#
# Heated up for 10 min at 70°C
#
# between camera lens and NUC plate of 600mm


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


# files=['hotNUC1.npy','hotNUC2.npy','hotNUC3.npy','hotNUC4.npy','hotNUC5.npy','coldNUC1.npy']


#integration time 1ms, frequency 50Hz	      !!! ! ! B A D   D A T A ! ! !!!
# temperature1=[38.4,35.4,33.7,32.6,31.6,30.6,29.6,28.6,27.6,26.6,25.6]
# #files1=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000011']
temperature1=[33.7,32.6,31.6,30.6,29.6,28.6,27.6,26.6,25.6]
files1=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000011']

#integration time 1ms, frequency 50Hz
# temperature2=[44.0,42.9,41.9,40.9,39.9,38.9,37.9,36.9,35.9,34.9,33.9,32.9,31.9,30.9,29.9,28.9,27.9,26.9,25.9,24.9]
# #files2=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000031']
temperature2=[41.9,40.9,39.9,38.9,37.9,36.9,35.9,34.9,33.9,32.9,31.9,30.9,29.9,28.9,27.9,26.9,25.9,24.9]
files2=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000031']


#integration time 1ms, frequency 50Hz
# temperature3=[41.8,40.8,39.3,38.8,37.8,36.8,35.8,34.8,33.8,32.8,31.8,30.8,29.8,28.8,27.8]
# #files3=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000015']
temperature3=[39.3,38.8,37.8,36.8,35.8,34.8,33.8,32.8,31.8,30.8,29.8,28.8,27.8]
files3=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000015']


#integration time 1ms, frequency 50Hz
# temperature4=[44.3,43.3,41.1,38.8,37.2,36.2,34.6,32.9,32.2,31.2,30.4,29.6,29.0,28.4,27.8]
# #files4=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000030']
#temperature4=[41.1,38.8,37.2,36.2,34.6,32.9,32.2,31.2,30.4,29.6,29.0,28.4,27.8]
#files4=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000030']
temperature4=[41.1,38.8,36.2,34.6,32.9,32.2,31.2,30.4,29.6,29.0,28.4,27.8]
files4=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000030']


#integration time 1ms, frequency 50Hz
# temperature5=[45.1,44.1,43.1,42.1,41.1,40.1,39.1,38.1,37.1,36.1,35.1,34.1,33.1,32.1,31.1,30.1,29.1,28.1]
# #files5=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000048']
temperature5=[43.1,42.1,41.1,40.1,39.1,38.1,37.1,36.1,35.1,34.1,33.1,32.1,31.1,30.1,29.1,28.1]
files5=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000048']


#integration time 1ms, frequency 50Hz
# temperature6=[13.3,13.5,13.7,14.5,14.9,15.6,16.3,16.9,17.6,18.2,18.9,19.6,20.2,20.8,21.4]
# #files6=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000063']
temperature6=[13.7,14.5,14.9,15.6,16.3,16.9,17.6,18.2,18.9,19.6,20.2,20.8,21.4]
files6=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000063']


#integration time 2ms, frequency 383Hz
# temperature7=[16,16.2,16.8,17.2,17.7,18.2,18.7,19.2,19.7,20.2,20.7,21.2,21.6,22]
# #files7=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000014']
temperature7=[16.8,17.2,17.7,18.2,18.7,19.2,19.7,20.2,20.7,21.2,21.6,22]
files7=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000014']


#integration time 2ms, frequency 383Hz
# temperature8=[36.8,36.1,35.5,35.0,34.5,33.8,33.0,32.3,31.4,30.6,30.0,29.2,28.5,28.1,24.9,24.6,24.3,23.9,23.6,23.3]
# #files8=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000041']
temperature8=[33.8,33.0,32.3,31.4,30.6,30.0,29.2,28.5,28.1,24.9,24.6,24.3,23.9,23.6,23.3]
files8=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000041']


#integration time 2ms, frequency 383Hz
#temperature10=[39.6,37.4,36.8,36.0,35.1,34.1,32.7,31.5,30.2,29.6,29.5,29.0,28.6,28.1,27.7,27.3,26.9,26.6,26.4,26.2,26.0]
temperature10=[29.6,29.5,29.0,28.6,28.1,27.7,27.3,26.9,26.6,26.4,26.2,26.0]
#files10=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000034']
files10=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000034']

#integration time 1ms, frequency 383Hz
temperature11=[31.8,30.8,29.9,29.2,28.7,28.3,27.9,27.5,27.2,26.8,26.5,26.3,26.1,25.9]
files11=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000035']

#integration time 2ms, frequency 383Hz
#temperature12=[41.3,40.1,39.1,38.1,37.1,36.4,35.7,34.9,33.8,33.3,32.5,31.8,30.9,30.1,29.5,28.8,28.1,27.6,27.1,26.5,26.1]
temperature12=[30.1,29.5,28.8,28.1,27.6,27.1,26.5,26.1]
#files12=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000081']
files12=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000081']

#integration time 1ms, frequency 383Hz
#temperature13=[44.6,43.8,43.0,42.5,41.9,40.8,39.6,38.6,37.6,36.8,36.0,35.3,34.2,33.5,32.9,32.2,31.5,30.6,29.8,28.6,28.4,27.7,27.4,26.6,26.2]
temperature13=[43.8,43.0,42.5,41.9,40.8,39.6,38.6,37.6,36.8,36.0,35.3,34.2,33.5,32.9,31.5,30.6,29.8,28.6,28.4,27.7,27.4,26.6,26.2]
# temperature13=[31.5,30.6,29.8,28.9,28.4,27.7,27.4,26.6,26.2]
#files13=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000080']
files13=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000080']
# files13=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000080']

#integration time 1ms, frequency 383Hz
# temperature14=[43.3,42.2,40.9,40.1,39.5,39.0,38.5,38.0,37.2,35.9,34.9,34.0,33.3,32.5,31.8,30.9,30.3,29.6,29.0,28.6,28.1,27.7,27.3,26.9,26.5,26.2,25.8,25.4,25.0,24.5,24.1]
# files14=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000053']
temperature14=[42.2,40.9,40.1,39.5,39.0,38.5,38.0,37.2,35.9,34.9,34.0,33.3,32.5,31.8,30.9,30.3,29.6,29.0,28.6,28.1,27.7,27.3,26.9,26.5,26.2,25.8,25.4,25.0,24.5,24.1]
files14=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000053']


#integration time 2ms, frequency 383Hz
# temperature15=[36.5,35.4,34.5,33.7,32.9,32.1,31.4,30.6,29.9,29.3,28.8,28.4,27.8,27.5,27.1,26.7,26.4,26.0,25.6,25.2,24.8,24.3,23.9]
# files15=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000054']
temperature15=[34.5,33.7,32.9,32.1,31.4,30.6,29.9,29.3,28.8,28.4,27.8,27.5,27.1,26.7,26.4,26.0,25.6,25.2,24.8,24.3,23.9]
files15=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000054']


#integration time 1ms, frequency 383Hz
# temperature16=[41.2,40.6,39.8,38.8,38.2,37.5,36.9,36.3,35.4,34.7,34.0,33.3,32.7,32.1,31.6,31.0,30.4,29.8,29.2,28.6,27.9,27.5,26.8,26.4,26.1,25.7,25.4,25.0,24.6]
# files16=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000082','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000102']
# temperature16=[40.6,39.8,38.8,38.2,37.5,36.9,36.3,35.4,34.7,34.0,33.3,32.7,32.1,31.6,31.0,30.4,29.8,29.2,28.6,27.9,27.5,26.8,26.4,26.1,25.7,25.4,25.0,24.6]
# files16=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000082','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000102']
temperature16=[40.6,39.8,38.8,38.2,37.5,36.9,36.3,35.4,34.7,34.0,33.3,32.7,32.1,31.6,31.0,30.4,29.8,29.2,28.6,27.9,27.5,26.8,26.4,25.7,25.4,25.0,24.6]
files16=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000082','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000102']


#integration time 2ms, frequency 383Hz
# temperature17=[35.7,35.1,34.3,33.7,33.0,32.3,31.9,31.3,30.7,30.1,29.4,28.9,28.3,27.7,27.2,26.6,25.6,25.1,24.8,24.4]
# files17=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000081','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000083','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000085','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000087','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000103']
temperature17=[34.3,33.7,33.0,32.3,31.9,31.3,30.7,30.1,29.4,28.9,28.3,27.7,27.2,26.6,25.6,25.1,24.8,24.4]
files17=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000081','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000083','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000085','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000087','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000103']


#integration time 1ms, frequency 383Hz
# temperature18=[15.7,16.2,16.8,17.3,17.8,18.3,19.1,19.5,20.0,20.5,21.0,21.5,22.1,22.4,22.8,23.2,23.4,23.8]
# files18=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000104','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000106','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000114','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000117','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000119','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000121','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000123','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000125','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000127','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000129','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000131','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000133','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000135','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000137','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000139']
temperature18=[16.8,17.3,17.8,18.3,19.1,19.5,20.0,20.5,21.0,21.5,22.1,22.4,22.8,23.2,23.4,23.8]
files18=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000114','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000117','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000119','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000121','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000123','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000125','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000127','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000129','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000131','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000133','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000135','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000137','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000139']


#integration time 2ms, frequency 383Hz
# temperature19=[15.9,16.5,17.0,17.6,18.0,18.6,19.3,19.8,20.2,20.8,21.3,21.7,22.2,22.7,22.9,23.2,23.6]
# files19=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000105','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000107','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000115','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000118','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000120','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000122','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000124','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000126','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000128','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000130','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000132','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000134','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000136','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000138']
temperature19=[17.0,17.6,18.0,18.6,19.3,19.8,20.2,20.8,21.3,21.7,22.2,22.7,22.9,23.2,23.6]
files19=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000115','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000118','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000120','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000122','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000124','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000126','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000128','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000130','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000132','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000134','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000136','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000138']


#integration time 1ms, frequency 383Hz
#temperature20=[16.2,16.7,17.2,17.7,18.3,18.9,19.5,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0,23.4]
#files20=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000140','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000142','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000144','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000146','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000148','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000150','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000152','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000154','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000156','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000158','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000160','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000162','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000164','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000166','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000168','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000170']
# temperature20=[16.2,16.7,17.2,17.7,18.9,19.5,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0,23.4]
# files20=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000140','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000142','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000144','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000146','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000150','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000152','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000154','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000156','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000158','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000160','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000162','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000164','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000166','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000168','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000170']
temperature20=[17.2,17.7,18.9,19.5,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0,23.4]
files20=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000144','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000146','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000150','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000152','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000154','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000156','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000158','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000160','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000162','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000164','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000166','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000168','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000170']


#integration time 2ms, frequency 383Hz
# temperature21=[16.4,16.9,17.5,18.0,18.6,19.2,19.8,20.3,20.8,21.2,21.6,22.0,22.4,22.8,23.2,23.6]
# files21=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000141','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000143','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000145','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000147','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000149','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000151','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000153','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000155','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000157','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000159','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000161','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000163','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000165','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000167','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000169','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000171']
temperature21=[17.5,18.0,18.6,19.2,19.8,20.3,20.8,21.2,21.6,22.0,22.4,22.8,23.2,23.6]
files21=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000145','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000147','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000149','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000151','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000153','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000155','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000157','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000159','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000161','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000163','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000165','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000167','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000169','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000171']


#integration time 1ms, frequency 994Hz, partial frame
temperature22=[40.1,39.5,39.0,38.3,37.8,37.2,36.6,35.8,35.3,34.8,34.1,33.5,33.1,32.7,32.3,31.9,31.5,31.1,30.7,30.3,29.7,29.2,28.6,28.1,27.5,27.0,26.5,26.0,25.5,24.8]
files22=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000002','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000003','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000004','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000005','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000006','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000007','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000008','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000009','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000010','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000011','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000012','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000013','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000014','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000015','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000016','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000017','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000018','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000019','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000020','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000021','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000022','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000023','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000024','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000025','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000026','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000027','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000028','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000029','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000030','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000031']

#integration time 1ms, frequency 994Hz, partial frame (width 320 height 64 x offset 0 y offset 96)
# temperature23=[15.8,16.1,16.6,17.7,18.0,18.1,18.5,18.7,18.9,19.3,19.7,20.0,20.3,20.6,20.9,21.2,21.5,21.8,22.1,22.4,23.0]
# files23=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000032','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000033','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000034','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000036','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000037','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000038','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000039','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000040','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000041','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000042','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000043','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000044','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000045','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000046','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000047','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000048','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000049','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000050','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000051','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000052','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000053']
temperature23=[16.1,16.6,17.7,18.0,18.1,18.5,18.7,18.9,19.3,19.7,20.0,20.3,20.6,20.9,21.2,21.5,21.8,22.1,22.4,23.0]
files23=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000033','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000034','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000036','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000037','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000038','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000039','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000040','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000041','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000042','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000043','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000044','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000045','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000046','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000047','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000048','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000049','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000050','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000051','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000052','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000053']


#integration time 0.5ms, frequency 383Hz
temperature24=[39.2,38.3,37.2,36.4,35.6,34.9,34.3,33.7,33.1,32.5,31.9,31.3,30.7,30.1,29.5,28.9,28.3,27.6,27.1,26.5,25.9,25.2]
files24=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000044']


#integration time 1.5ms, frequency 383Hz
temperature25=[38.7,37.7,36.8,35.9,35.3,34.6,34.0,33.4,32.8,32.2,31.6,31.0,30.4,29.8,29.2,28.6,28.0,27.7,27.4,26.8,26.2,25.6,25.0]
files25=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000045']


#integration time 0.5ms, frequency 383Hz
# temperature26=[15.0,15.4,15.9,16.4,16.9,17.6,18.0,18.5,18.8,19.3,19.7,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0]
# files26=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000037']
temperature26=[15.9,16.4,16.9,17.6,18.0,18.5,18.8,19.3,19.7,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0]
files26=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000037']


#integration time 1.5ms, frequency 383Hz
# temperature27=[15.2,15.7,16.2,16.7,17.3,17.8,18.2,18.6,19.1,19.5,19.9,20.4,20.8,21.4,21.6,22.0,22.4,22.8]
# files27=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000036']
temperature27=[15.7,16.2,16.7,17.3,17.8,18.2,18.6,19.1,19.5,19.9,20.4,20.8,21.4,21.6,22.0,22.4,22.8]
files27=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000036']


#integration time 0.5ms, frequency 383Hz
temperature28=[41.1,40.0,39.0,38.0,37.1,36.0,35.2,34.4,33.6,32.9,32.2,31.5,30.8,30.1,29.4,28.8,28.2,27.4,26.8,26.2,25.6,25.0,24.4]
files28=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000082']


#integration time 1.5ms, frequency 383Hz
temperature29=[40.5,39.5,38.5,37.4,36.7,35.6,34.8,34.0,33.3,32.5,31.9,31.1,30.4,29.8,29.1,28.5,27.7,27.1,26.5,25.9,25.3,24.7,24.2]
files29=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000081','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000083']


#integration time 0.5ms, frequency 383Hz
# temperature30=[16.5,17.0,17.5,18.0,18.4,18.8,19.3,19.7,20.1,20.5,20.9,21.3,21.7,22.2,22.6,23.0]
# files30=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000085','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000087','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000103','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000105','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000107','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000115']
temperature30=[17.5,18.0,18.4,18.8,19.3,19.7,20.1,20.5,20.9,21.3,21.7,22.2,22.6,23.0]
files30=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000103','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000105','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000107','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000115']


#integration time 1.5ms, frequency 383Hz
# temperature31=[16.3,16.7,17.3,17.7,18.2,18.6,19.1,19.5,19.9,20.3,20.7,21.1,21.5,22.0,22.4,22.8]
# files31=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000102','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000104','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000106','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000114']
temperature31=[16.7,17.3,17.7,18.2,18.6,19.1,19.5,19.9,20.3,20.7,21.1,21.5,22.0,22.4,22.8]
files31=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000102','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000104','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000106','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000114']



##  FILES RELATIVE TO THE LASER MEASUREMENTS

reflaserpower=[5,4.97,3.97,2.97,2.03,1.52,0.99,0.62,0.28,0.06,0.03,0]
reflaserpower=np.multiply(0.001,np.flip(reflaserpower,0))
reflaserfvoltage=[3,1.4,1.2,1,0.8,0.7,0.6,0.5,0.45,0.4,0.35,0]
reflaserfvoltage=np.flip(reflaserfvoltage,0)

# laser 08/03/2018 1ms 50Hz
laser1=['/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000011']
voltlaser1=[2,1.6,1.4,1.2,1,0.8,0.6,0.4,0.3,0.2,2]
freqlaser1=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,10]

# laser 09/03/2018 2ms 383Hz
laser2=['/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000015']
voltlaser2=[1.4,1.2,1,0.8,0.7,0.6,0.5,0.45,0.4,0.35,1.2,1.2,1.2,1.2,1.2]
freqlaser2=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60]


# Laser experiments 25/07/2018 1ms 383Hz focused laser straight on pinhole out of aberration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser10=['/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000014']
voltlaser10=[0.05,0.1,0.25,0.35,0.5,0.6,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser10=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser10=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 25/07/2018 1ms 383Hz partially defocused laser straight on pinhole out of aberration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser11=['/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000028']
voltlaser11=[0.05,0.1,0.25,0.35,0.5,0.6,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser11=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser11=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 25/07/2018 1ms 994Hz (width=320, height=32, xoffset=0, yoffset=64, invert (V flip) selected) partially defocused laser straight on pinhole out of aberration with low duty cycle
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# NOTE I CAN'T USE THIS DATA, I DON'T HAVE ITS REFERENCE FRAME!
laser12=['/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000036','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000037']
voltlaser12=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser12=[10,50,100,10,50,100,10,50,100]
dutylaser12=[0.02,0.02,0.02,0.05,0.05,0.05,0.1,0.1,0.1]


# Laser experiments 02/08/2018 1ms 383Hz focused laser further out of aberration, away from the feature given by calibration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser13=['/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000007']
voltlaser13=[0.05,0.1,0.25,0.35,0.5,0.6,0.7]
freqlaser13=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]
dutylaser13=[0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# Laser experiments 02/08/2018 1ms 383Hz partially defocused laser position on the foil as high as possible
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser14=['/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000014']
voltlaser14=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser14=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]
dutylaser14=[0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 02/08/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum1=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000004']
vacuumframerate1=[383,994,383,994]


reflaserpower1=[4.14,4.14,0,0]
reflaserpower1=np.multiply(0.001,np.flip(reflaserpower1,0))
reflaserfvoltage1=[10,0.505,0.012,0]
reflaserfvoltage1=np.flip(reflaserfvoltage1,0)

vacuumtest1=['/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000007-001_03_30_27_189','/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000008-001_03_49_16_589','/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000009-001_03_59_09_189','/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000010-001_04_13_48_036']


# Laser experiments 08/08/2018 1ms 383Hz focused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser15=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000015']
voltlaser15=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser15=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser15=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]


# Laser experiments 08/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) partially defocused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser16=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000030']
voltlaser16=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser16=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser16=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]


# Laser experiments 08/08/2018 1ms 383Hz focused laser as low as possible close to the corner
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser17=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000036','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000037','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000039','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000040','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000042','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000043']
voltlaser17=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser17=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser17=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 08/08/2018 1ms 383Hz focused laser as high as possible close to the side
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser18=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000044','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000045','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000046','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000048','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000049','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000050','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000053','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000054','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000055','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000056']
voltlaser18=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser18=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser18=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 08/08/2018 1ms 383Hz focused laser as close as possible to the aberration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser19=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000057','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000059','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000060','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000061','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000062','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000063','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000065','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000066','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000067','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000069']
voltlaser19=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser19=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser19=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 08/08/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum2=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000029','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000030','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000031','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000032','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000033','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000034','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000035']
vacuumframerate2=[383,994,994,994,994,994,994,994,994,994,383,994,994,994,994]



# Laser experiments 13/08/2018 1ms 383Hz fully defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser20=['/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000009']
voltlaser20=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser20=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser20=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 13/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) fully defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser21=['/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000018']
voltlaser21=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser21=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser21=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 13/08/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum3=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000029']
vacuumframerate3=[383,994,994,994,994,994,994,994,383,994,994,994,994,994,994,994,383,383,383,994,994,994,383,994,994,994,994,994,994,994]




# Laser experiments 20/08/2018 1ms 383Hz focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser22=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000013']
voltlaser22=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser22=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser22=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 20/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser23=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000036','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000037']
voltlaser23=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser23=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90,150,180,210,240,270,300,330,360,390]
dutylaser23=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser24=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000039','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000040','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000042','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000043','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000044','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000045','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000046','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000048','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000049','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000050','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000053','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000054','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000055','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000056','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000057','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000059','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000060','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000061','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000062']
voltlaser24=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser24=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90,140,190,240,290,340,390,440,490,540,590,640,690]
dutylaser24=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 20/08/2018 2ms 383Hz focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser25=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000093','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000094','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000095','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000096','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000097','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000098','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000099','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000100','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000101','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000102','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000103','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000104','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000105']
voltlaser25=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser25=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser25=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) focused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser26=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000063','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000065','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000066','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000067','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000069','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000070','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000071','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000072','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000073','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000074','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000075','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000076','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000077']
voltlaser26=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser26=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser26=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) focused laser right in the pinhole at half power
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser27=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000078','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000079','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000080','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000081','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000082','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000083','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000084','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000085','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000086','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000087','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000088','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000089','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000090','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000091','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000092']
voltlaser27=[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
freqlaser27=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser27=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) partially defocused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser28=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000106','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000107','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000108','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000109','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000110','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000111','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000112','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000113','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000114','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000115','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000116','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000117','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000118','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000119','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000120']
voltlaser28=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser28=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser28=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) partially defocused laser right in the pinhole at half power
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser29=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000121','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000122','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000123','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000124','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000125','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000126','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000127','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000128','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000129','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000130','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000131','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000132','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000133','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000134','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000135']
voltlaser29=[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
freqlaser29=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser29=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]


# Laser experiments 20/08/2018 2ms 383Hz fully defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser30=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000136','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000137','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000138','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000139','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000140','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000141','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000142','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000143','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000144']
voltlaser30=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser30=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser30=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) fully defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser31=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000145','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000146','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000147','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000148','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000149','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000150','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000151','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000152','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000153','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000154','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000155','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000156','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000157','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000158','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000159','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000160','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000161','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000162','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000163','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000164']
voltlaser31=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser31=[0.2,0.5,1,3,5,10,30,60,90,140,190,240,290,340,390,440,409,540,590,640]
dutylaser31=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 20/08/2018 1ms 383Hz focused laser as high and left as possible
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser32=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000165','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000166','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000167','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000168','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000169','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000170','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000171','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000172','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000173','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000174','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000175','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000176','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000177']
voltlaser32=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser32=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser32=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]




# Laser experiments 20/08/2018 1ms+2ms+0.5ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum4=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000029','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000030','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000031','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000032','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000034','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000035','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000036','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000037','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000038','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000039','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000040','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000041','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000042','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000043','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000044','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000045']
vacuumframerate4=[383,383,383,383,994,994,994,994,994,994,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,383,383,383,383,1976,1976,1976,1976,1976,1976,1976,383,383,383,1976,1976,1976,1976,1976,383,383,383,383]


# Laser experiments 23/08/2018 1ms 383Hz fully defocused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser33=['/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000012']
voltlaser33=[0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser33=[0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser33=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]



# Laser experiments 23/08/2018 1ms+0.5ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum5=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000020']
vacuumframerate5=[383,383,383,994,994,994,994,994,994,994,1976,1976,1976,1976,1976,1976]



# Laser experiments 25/0810/2018 1ms 383Hz focused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser34=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000013']
voltlaser34=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser34=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser34=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]


# Laser experiments 25/10/2018 1ms 383Hz fully defocused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser35=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000024']
voltlaser35=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser35=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser35=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]





##  IRVB FOIL PARAMETERS

foilemissivity=[0.87915237,0.84817978,0.76761355,0.73948514,0.81449329,0.86221733,0.81863479,0.83238318,0.75014256,0.82455539,0.87319584,0.84327053,0.8539331,0.83438179,0.82835241,0.83600738,0.8822808,0.84797073,0.75170152,0.87531842,0.86392473,0.89127889,0.89074805,0.86384333,0.82690635,0.88410513,0.84639702,0.78673421,0.84965535,0.8552314,0.82101809,0.81856018,0.80005979,0.89601878,0.83453686,0.91161225,0.88091159,0.81225521,0.87672616,0.8256423,0.85326844,0.85851042,0.83989414,0.84155748,0.84208017,0.8138739,0.87013654,0.83342386,0.84799913,0.94714049,0.86117643,0.87311947,0.86511982,0.94298577,0.94470884,0.88453373,0.84725634,0.84136077,0.91724797,0.88871259,0.84423068,0.84649853,0.93356454] #[au]
foilthickness=[1.49682E-06,1.37544E-06,1.6084E-06,1.63786E-06,1.41477E-06,0.000001314,1.33445E-06,1.43023E-06,0.000001259,1.06779E-06,1.1723E-06,1.36939E-06,1.31647E-06,1.43612E-06,1.53133E-06,1.39534E-06,0.000001335,1.48462E-06,1.65731E-06,1.35984E-06,1.36457E-06,1.29692E-06,0.000001389,1.39388E-06,1.47688E-06,1.0365E-06,1.01075E-06,1.18426E-06,1.18164E-06,1.28718E-06,1.35841E-06,1.41908E-06,1.48667E-06,1.27384E-06,1.43041E-06,1.27027E-06,0.000001233,1.43669E-06,1.36793E-06,1.46373E-06,1.35276E-06,1.19541E-06,1.2682E-06,0.000001148,1.04091E-06,1.01677E-06,1.03126E-06,1.09143E-06,1.2775E-06,1.20106E-06,1.35034E-06,1.42827E-06,1.3802E-06,1.21383E-06,1.17713E-06,0.000001358,1.28267E-06,1.11787E-06,1.02187E-06,1.01785E-06,1.07125E-06,1.00346E-06,8.67195E-07] #[m]


foilemissivity=np.reshape(foilemissivity,(7,9))
foilemissivity=np.flip(foilemissivity,0)
#foilemissivity=foilemissivity.transpose()
foilthickness=np.reshape(foilthickness,(7,9))
foilthickness=np.flip(foilthickness,0)
#foilthickness=foilthickness.transpose()


Ptthermalconductivity=71.6 #[W/(m·K)]
Ptspecificheat=133 #[J/(kg K)]
Ptdensity=21.45*1000 #[kg/m3]
Ptthermaldiffusivity=Ptthermalconductivity/(Ptspecificheat*Ptdensity)    #m2/s
sigmaSB=5.6704e-08 #[W/(m2 K4)]
zeroC=273.15 #K

# temperature=[temperature1,temperature2,temperature3,temperature4,temperature5,temperature6,temperature7,temperature8]
# files=[files1,files2,files3,files4,files5,files6,files7,files8]

if False:

	# 14/05/2018 LINES TO IMPORT DATA FROM CVS/FIT FORMAT FO THE REGULAR .npy

	files=[laser11[4:],laser12]
	files=np.concatenate(files)
	for file in files:
		coleval.collect_subfolderfits(file)
		coleval.evaluate_back(file)


	for file in vacuumtest1[2:4]:
		# filenames=coleval.all_file_names(files,'npy')
		# data=np.load(os.path.join(pathfiles,filenames))
		coleval.movie(file,100,1,xlabel=('Vertical pixesl'),ylabel=('Horizontal pixels'),barlabel=('Counts'))


	file_ref='/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018'
	files=[laser15,laser16,laser17,laser18,laser19]
	files=np.concatenate(files)
	for file in files:
		coleval.search_background_timestamp(file,file_ref)


	files=[vacuum3[13],vacuum3[16],vacuum3[20]]
	files=coleval.flatten_full(files)
	for file in files:
		coleval.collect_subfolder(file)
		coleval.save_timestamp(file)
		coleval.evaluate_back(file)


	file_ref='/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018'
	files=[laser18[3]]
	files=coleval.flatten_full(files)
	for file in files:
		coleval.collect_subfolder(file)
		coleval.save_timestamp(file)
		coleval.search_background_timestamp(file,file_ref)


if False:

	# VERY FIRST BIT OF CODE, TO READ MULTIPLE DATA STATISTICS AND PLOT THE RESULTS

	temperature=[temperature11,temperature13,temperature16]
	files=[files11,files13,files16]
	meancounttot=[]
	meancountstdtot=[]
	for j in range(len(files)):
		meantemp=[]
		meandev=[]
		for i in range(len(files[j])):
			# print('folder',files[j][i])
			data=coleval.collect_stat(files[j][i])
			meantemp.append(data[2][0])
			meandev.append(data[2][1])
		# print('len(meantemp)',len(meantemp))
		# print('len(meandev)',len(meandev))
		# print('len(temperature[j])',len(temperature[j]))
		meancounttot.append(meantemp)
		meancountstdtot.append(meandev)

	for i in range(len(temperature)):
		plt.errorbar(temperature[i],meancounttot[i],yerr=meancountstdtot[i],label='dataset'+str(i+1))

		# guess=[0.00001,273.,1000.]
		# # temp1,temp2 = curve_fit(polinomial, temperature[j], meancount, p0=guess, sigma=meancountstd, maxfev=100000000)
		# # plt.plot(temperature[j],polinomial(temperature[j],*temp1))
		# print('Counts from experiment'+str(j+1)+'fitted with the equation A*((x+off)**4+300**4)')
		# # print('Counts from experiment'+str(j+1)+'fitted with the equation A*((x+off)**4+T0**4)')
		# print('Parameters are: A='+str(round(temp1[0],8))+' , off='+str(round(temp1[1],2))+' , T0='+str(round(temp1[2],2)))
		# print('Parameters are: A='+str(round(temp1[0],8))+' , off='+str(round(temp1[1],2)))
	# 	meancounttot.append(meancount)
	# 	meancountstdtot.append(meancountstd)
	#
	# np.save('meancounttot',data)

	# guess=[1,1]
	# temp1,temp2 = curve_fit(line, np.stack((temperature[1],temperature[5])), np.stack((meancounttot[1],meancounttot[5])), p0=guess, sigma=np.stack((meancountstdtot[1],meancountstdtot[5])), maxfev=100000000)

	# temp1=np.interp(temperature[0],temperature[1],meancounttot[1])
	# ratio=meancounttot[0]/temp1
	# plt.figure()
	# plt.plot(temperature[0],ratio)
	plt.legend()
	plt.legend(loc='best')
	plt.show()

elif False:

	# 27/03/2018 THIS IS USELESS, i DON'T NEED CONVERSION FROM TEMP TO COUNT, I NEED THE OPPOSITE!

	temperature=[temperature2,temperature3,temperature4,temperature5,temperature6]
	files=[files2,files3,files4,files5,files6]
	meancounttot=[]
	meancountstdtot=[]
	for j in range(len(files)):
		meantemp=[]
		meandev=[]
		for i in range(len(files[j])):
			# print('folder',files[j][i])
			data=coleval.collect_stat(files[j][i])
			meantemp.append(data[0])
			meandev.append(data[1])
		# print('len(meantemp)',len(meantemp))
		# print('len(meandev)',len(meandev))
		# print('len(temperature[j])',len(temperature[j]))
		meancounttot.append(np.array(meantemp))
		meancountstdtot.append(np.array(meandev))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	# print('temperature[1]',temperature[1])
	# print('meancounttot[1][:,0,0]',meancounttot[1][:,0,0])
	# print('meancountstdtot[1][:,0,0]',meancountstdtot[1][:,0,0])
	nmax=5
	shapex=np.shape(meancounttot[0][0,:,:])[0]
	shapey=np.shape(meancounttot[0][0,:,:])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	done=0
	for n in range(2,nmax+1):
		# n=4	#                  ORDER OF POLINOMIAL ! ! ! !
		guess=np.ones(n)
		# print('guess',guess)
		# shapex=np.shape(meancounttot[0][0,:,:])[0]
		# shapey=np.shape(meancounttot[0][0,:,:])[1]
		guess,temp2=curve_fit(coleval.polygen(n), temperature[1], meancounttot[1][:,0,0], p0=guess, sigma=meancountstdtot[1][:,0,0], maxfev=100000000)

		# print('guess,temp2',guess,temp2)
		# print('shapex,shapey',shapex,shapey)

		coeff=np.zeros((shapex,shapey,n))
		errcoeff=np.zeros((shapex,shapey,n))

		# for i in range(len(temperature)):
		for j in range(shapex):
			for k in range(shapey):
				# print('j,k',j,k)
				y=[]
				yerr=[]
				for i in range(len(meancounttot)):
					y.append(meancounttot[i][:,j,k])
					yerr.append(meancountstdtot[i][:,j,k])
				temp1,temp2=curve_fit(coleval.polygen(n), np.concatenate(temperature), np.concatenate(y), p0=guess, sigma=np.concatenate(yerr), maxfev=100000000)
				coeff[j,k,:]=temp1
				errcoeff[j,k,:]=np.sqrt(np.diagonal(temp2))
				score[n-2,j,k]=rsquared(np.concatenate(y),coleval.polygen(n)(np.concatenate(temperature),*temp1))
				if (j==int(shapex/2)) & (k==int(shapey/2)):
					if done==0:
						plt.figure()
						plt.errorbar(np.concatenate(temperature),np.concatenate(y),yerr=np.concatenate(yerr),label='dataset',fmt='o')
						done+=1
					plt.plot(np.sort(np.concatenate(temperature)),coleval.polygen(n)(np.sort(np.concatenate(temperature)),*temp1),label='fit grade='+str(n)+' R2='+str(np.around(rsquared(np.concatenate(y),coleval.polygen(n)(np.concatenate(temperature),*temp1)),decimals=3)))
					plt.legend()
					plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
					plt.legend(loc='best')
		np.save('coeffpolydeg'+str(n)+'int1ms',coeff)
		np.save('errcoeffpolydeg'+str(n)+'int1ms',errcoeff)

		print('for a polinomial of degree '+str(n)+' the R^2 score is '+str(np.sum(score[n-2])))

	# plt.errorbar(np.concatenate(temperature),np.concatenate(y),yerr=np.concatenate(yerr),label='dataset',fmt='o')
	# plt.plot(np.concatenate(temperature),coleval.polygen(n)(np.concatenate(temperature),*temp1),'+')
	# plt.legend()
	# plt.legend(loc='best')
	# plt.show()
	#
	# for i in range(n):
	# 	plt.figure()
	# 	plt.pcolor(coeff[:,:,i])
	# 	plt.colorbar()

	plt.show()

elif False:

	# 27/03/2018 THIS IS USELESS, i DON'T NEED CONVERSION FROM TEMP TO COUNT, I NEED THE OPPOSITE!

	temperature=[temperature7,temperature8]
	files=[files7,files8]
	meancounttot=[]
	meancountstdtot=[]
	for j in range(len(files)):
		meantemp=[]
		meandev=[]
		for i in range(len(files[j])):
			# print('folder',files[j][i])
			data=coleval.collect_stat(files[j][i])
			meantemp.append(data[0])
			meandev.append(data[1])
		# print('len(meantemp)',len(meantemp))
		# print('len(meandev)',len(meandev))
		# print('len(temperature[j])',len(temperature[j]))
		meancounttot.append(np.array(meantemp))
		meancountstdtot.append(np.array(meandev))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	# print('temperature[1]',temperature[1])
	# print('meancounttot[1][:,0,0]',meancounttot[1][:,0,0])
	# print('meancountstdtot[1][:,0,0]',meancountstdtot[1][:,0,0])
	nmax=5
	shapex=np.shape(meancounttot[0][0,:,:])[0]
	shapey=np.shape(meancounttot[0][0,:,:])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	done=0
	for n in range(2,nmax+1):
		# n=4	#                  ORDER OF POLINOMIAL ! ! ! !
		guess=np.ones(n)
		# print('guess',guess)
		# shapex=np.shape(meancounttot[0][0,:,:])[0]
		# shapey=np.shape(meancounttot[0][0,:,:])[1]
		guess,temp2=curve_fit(coleval.polygen(n), temperature[1], meancounttot[1][:,0,0], p0=guess, sigma=meancountstdtot[1][:,0,0], maxfev=100000000)

		# print('guess,temp2',guess,temp2)
		# print('shapex,shapey',shapex,shapey)

		coeff=np.zeros((shapex,shapey,n))
		errcoeff=np.zeros((shapex,shapey,n))

		# for i in range(len(temperature)):
		for j in range(shapex):
			for k in range(shapey):
				# print('j,k',j,k)
				y=[]
				yerr=[]
				for i in range(len(meancounttot)):
					y.append(meancounttot[i][:,j,k])
					yerr.append(meancountstdtot[i][:,j,k])
				temp1,temp2=curve_fit(coleval.polygen(n), np.concatenate(temperature), np.concatenate(y), p0=guess, sigma=np.concatenate(yerr), maxfev=100000000)
				coeff[j,k,:]=temp1
				errcoeff[j,k,:]=np.sqrt(np.diagonal(temp2))
				score[n-2,j,k]=rsquared(np.concatenate(y),coleval.polygen(n)(np.concatenate(temperature),*temp1))
				if (j==int(shapex/2)) & (k==int(shapey/2)):
					if done==0:
						plt.figure()
						plt.errorbar(np.concatenate(temperature),np.concatenate(y),yerr=np.concatenate(yerr),label='dataset',fmt='o')
						done+=1
					plt.plot(np.sort(np.concatenate(temperature)),coleval.polygen(n)(np.sort(np.concatenate(temperature)),*temp1),label='fit grade='+str(n)+' R2='+str(np.around(rsquared(np.concatenate(y),coleval.polygen(n)(np.concatenate(temperature),*temp1)),decimals=3)))
					plt.legend()
					plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
					plt.legend(loc='best')
		np.save('coeffpolydeg'+str(n)+'int2ms',coeff)
		np.save('errcoeffpolydeg'+str(n)+'int2ms',errcoeff)

		print('for a polinomial of degree '+str(n)+' the R^2 score is '+str(np.sum(score[n-2])))

	# plt.errorbar(np.concatenate(temperature),np.concatenate(y),yerr=np.concatenate(yerr),label='dataset',fmt='o')
	# plt.plot(np.concatenate(temperature),coleval.polygen(n)(np.concatenate(temperature),*temp1),'+')
	# plt.legend()
	# plt.legend(loc='best')
	# plt.show()
	#
	# for i in range(n):
	# 	plt.figure()
	# 	plt.pcolor(coeff[:,:,i])
	# 	plt.colorbar()

	plt.show()


elif False:

	# LINES TO CALCULATE THE PARAMETERS FOR 1ms INTEGRATION TIME

	path='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'

	temperature=[temperature2,temperature3,temperature4,temperature5,temperature6]
	files=[files2,files3,files4,files5,files6]
	int=1	#ms
	nmax=3

	coleval.builf_poly_coeff(temperature,files,int,path,nmax)


elif False:

	#THESE ARE THE COMMANDS TO ANALYSE MULTIPLE SET OF DATA CO EVALUATE IF THE PARAMETERS HAVE BEEN CALCULATED RIGHT
	# FOR THE CASE 1ms AS INTEGRATION TIME

	path='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'

	temperature=[temperature2,temperature3,temperature4,temperature5,temperature6]
	files=[files2,files3,files4,files5,files6]
	int=1	#ms
	nmax=5

	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)

	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	for n in range(2,nmax+1):
		coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
		errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
		for j in range(shapex):
			for k in range(shapey):
				# if ((j == 0) & ((k==134) | (k==135) | (k==136) | (k==137))):
				if ((j == 128) & (k==160)):

					x=np.array(meancounttot[:,j,k])
					xerr=np.array(meancountstdtot[:,j,k])

					yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2

					score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))

					plt.figure(k)

					plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
						# done+=1
					plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='polynomial coefficients='+str(n)+' R2='+str(np.around(score[n-2,j,k],decimals=3)))
					plt.legend()
					plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
					plt.legend(loc='best')
					plt.xlabel('Counts [au]')
					plt.ylabel('Temperature [°C]')



		# print('for a polinomial of degree '+str(n)+' the R^2 score is '+str(np.sum(score[n-2])))

	# plt.errorbar(np.concatenate(temperature),np.concatenate(y),yerr=np.concatenate(yerr),label='dataset',fmt='o')
	# plt.plot(np.concatenate(temperature),coleval.polygen(n)(np.concatenate(temperature),*temp1),'+')
	# plt.legend()
	# plt.legend(loc='best')
	# plt.show()
	#
	# for i in range(n):
	# 	plt.figure()
	# 	plt.pcolor(coeff[:,:,i])
	# 	plt.colorbar()

	plt.show()

elif False:

	# LINES TO CALCULATE THE PARAMETERS FOR 2ms INTEGRATION TIME

	path='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'

	temperature=[temperature7,temperature8]
	files=[files7,files8]
	int=2	#ms
	nmax=5

	coleval.builf_poly_coeff(temperature,files,int,path,nmax)

elif False:

	#THESE ARE THE COMMANDS TO ANALYSE MULTIPLE SET OF DATA CO EVALUATE IF THE PARAMETERS HAVE BEEN CALCULATED RIGHT
	# FOR THE CASE 2ms AS INTEGRATION TIME

	path='/home/ffederic/work/irvb/Calibration_May10_2018_1nd_set_2.5ms'

	temperature=[temperature10]
	files=[files10]
	int=2.5	#ms
	nmax=3

	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)

	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	for n in range(2,nmax+1):
		coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
		errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
		for j in range(shapex):
			for k in range(shapey):
				# x=np.array(meancounttot[:,j,k])
				# xerr=np.array(meancountstdtot[:,j,k])
				# yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
				# score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
				# if ((j == 0) & ((k==134) | (k==135) | (k==136) | (k==137))):
				# 	plt.figure(k)
				# 	plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
				# 	plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='poly grade='+str(n)+' R2='+str(np.around(score[n-2,j,k],decimals=3)))
				# 	plt.legend()
				# 	plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
				# 	plt.legend(loc='best')
				# 	plt.xlabel('Counts [au]')
				# 	plt.ylabel('Temperature [°C]')
				# if ((j == 0) & ((k==134) | (k==135) | (k==136) | (k==137))):
				if ((j == 5) & ((k==160) or (k==80) or (k==240))):
					x=np.array(meancounttot[:,j,k])
					xerr=np.array(meancountstdtot[:,j,k])
					yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
					score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
					plt.figure(k)
					plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
					plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='poly grade='+str(n)+' R2='+str(np.around(score[n-2,j,k],decimals=3)))
					plt.legend()
					plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
					plt.legend(loc='best')
					plt.xlabel('Counts [au]')
					plt.ylabel('Temperature [°C]')

		# print('for a polinomial of degree '+str(n)+' the R^2 score is '+str(np.sum(score[n-2])))

	# plt.errorbar(np.concatenate(temperature),np.concatenate(y),yerr=np.concatenate(yerr),label='dataset',fmt='o')
	# plt.plot(np.concatenate(temperature),coleval.polygen(n)(np.concatenate(temperature),*temp1),'+')
	# plt.legend()
	# plt.legend(loc='best')
	# plt.show()
	#
	# for i in range(n):
	# 	plt.figure()
	# 	plt.pcolor(coeff[:,:,i])
	# 	plt.colorbar()

	plt.show()

elif False:

	#THIS AREA IS TO READ A SINGLE SELECTED FRAME AND CONVERT IT IN TEMPERATURE

	# degree of polynomial of choice
	n=3
	# type of integration time in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'
	# folder to read
	pathfiles='/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000057'
	#filestype
	# type='stat.npy'
	type='csv'

	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)
	data=coleval.read_csv2(pathfiles,filenames)
	# data=np.load(os.path.join(pathfiles,filenames))
	# temp=np.zeros(((1,1)+np.shape(data[1])))
	# temp[0,0,:,:]=data[0]
	# data=temp
	datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

	plt.figure()
	plt.title('Frame 50 in '+pathfiles)
	plt.pcolor(datatemp[0,50])
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame 50 in '+pathfiles)
	plt.pcolor(errdatatemp[0,50])
	plt.colorbar().set_label('Temp error [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	plt.figure()
	plt.title('Frame 50 in '+pathfiles)
	plt.pcolor(data[0,50])
	plt.colorbar().set_label('Counts')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	plt.figure()
	plt.pcolor(errparams[:,:,0])
	plt.title('Frame 50 in ')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Coefficient standard deviation [K]')
	plt.figure()
	plt.pcolor(errparams[:,:,1])
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Coefficient standard deviation [K/Counts]')
	plt.figure()
	plt.pcolor(errparams[:,:,2])
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Coefficient standard deviation [K/Counts^2]')
	plt.show()

	plt.figure()
	plt.imshow(params[:,:,0],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.figure()
	plt.imshow(params[:,:,1],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.figure()
	plt.imshow(params[:,:,2],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.show()

elif False:

	# THIS SECTION IS TO ANALYSE THE MEAN DATA OF A SET OF MEASUREMENTS, THEN I CAN PLOT/TRANSFORM ONE OF THAT FRAMES

	# degree of polynomial of choice
	n=3
	# type of integration time in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param'
	# folder to read

	# GROUP OF FOLDER ALREADY PROCESSED TO READ
	files=files6

	datatot=np.zeros((1,len(files),256,320))
	errdatatot=np.zeros((1,len(files),256,320))
	temptot=np.zeros((1,len(files),256,320))
	errtempatot=np.zeros((1,len(files),256,320))
	index=0
	for pathfiles in files:

		datastat=coleval.collect_stat(pathfiles)
		temp1=np.zeros(((1,1)+np.shape(datastat[1])))
		temp1[0,0,:,:]=datastat[0]
		data=temp1

		temp2=np.zeros(((1,1)+np.shape(datastat[1])))
		temp2[0,0,:,:]=datastat[1]
		errdata=temp2

		datatot[0,index,:,:]=data
		errdatatot[0,index,:,:]=errdata

		params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams,errdata=errdata)
		temptot[0,index,:,:]=datatemp
		errtempatot[0,index,:,:]=errdatatemp
		index+=1

		# pathfiles='/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000001'
		# #filestype
		# type='stat.npy'
		# # type='csv'
		# print('pathfiles',pathfiles)
		# print('files',files)
		# if pathfiles==files2[0]:
		# 	filenames=coleval.all_file_names(pathfiles,type)
		# 	data=np.load(os.path.join(pathfiles,filenames))
		# 	datatot=np.zeros((1,len(files))+np.shape(data[0])[-2:])
		#
		#
		# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		# filenames=coleval.all_file_names(pathfiles,type)
		# # data=coleval.read_csv(pathfiles,filenames)
		# data=np.load(os.path.join(pathfiles,filenames))
		# temp=np.zeros(((1,1)+np.shape(data[1])))
		# temp[0,0,:,:]=data[0]
		# data=temp
		# datatemp,errdatatemp=coleval.count_to_temp_poly(data,params,errparams)
		#
		# datatot[0,index,:,:]=datatemp
		# index+=1

	pathfiles='/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000057'

	plt.figure()
	plt.title('Mean Frame in '+pathfiles)
	plt.pcolor(temptot[0,8])
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Mean Frame in '+pathfiles)
	plt.pcolor(errtempatot[0,8])
	plt.colorbar().set_label('Temp error [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	# plt.show()

	plt.figure()
	plt.title('Mean Frame in '+pathfiles)
	plt.pcolor(datatot[0,8])
	plt.colorbar().set_label('Counts')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Mean Frame in '+pathfiles)
	plt.pcolor(errdatatot[0,8])
	plt.colorbar().set_label('Counts deviation')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()


	ani=coleval.movie_from_data(temptot,50,1,'Horizontal axis [pixles]','Vertical axis [pixles]','Temp [°C]')
	ani.save('Mean frame Temp in files6'+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()



	pixelmean=6	#number of pixel I do the mean of
	frame=temptot[0,8]
	# shapeorig=np.shape(frame)
	# shapeaver=(np.ceil(np.divide(shapeorig,pixelmean))).astype(int)
	# frameaver=np.zeros(shapeaver)
	# for i in range(shapeaver[0]):
	# 	if ((i+1)*pixelmean)>shapeorig[0]:
	# 		indexi=shapeorig[0]-1
	# 	else:
	# 		indexi=(i+1)*pixelmean
	# 	for j in range(shapeaver[1]):
	# 		if ((j+1)*pixelmean)>shapeorig[1]:
	# 			indexj=shapeorig[1]-1
	# 		else:
	# 			indexj=(j+1)*pixelmean
	# 		flat=np.ravel(frame[i*pixelmean:int(indexi),j*pixelmean:int(indexj)])
	# 		# if len(flat)>2:
	# 		# 	flat=np.delete(flat,np.argmax(flat))
	# 		# 	flat=np.delete(flat,np.argmin(flat))
	# 		frameaver[i,j]=np.mean(flat)
	# 		# print(i*pixelmean,indexi,j*pixelmean,indexj,flat)
	frameaver=coleval.average_frame(frame,pixelmean,extremedelete=True)

	plt.figure()
	plt.title('Mean Frame in '+pathfiles+' averaged each '+str(pixelmean)+'x'+str(pixelmean)+' pixels')
	plt.pcolor(frameaver)
	plt.colorbar().set_label('Temp')
	plt.xlabel('Horizontal averaged axis [pixles]')
	plt.ylabel('Vertical averaged axis [pixles]')
	plt.show()

elif False:

	#THIS BIT IS TO CONVERT IN TEMPERATURE THE LASER VIDEO AND WORK WITH IT

	# degree of polynomial of choice
	n=3
	# type of integration time in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'
	# folder to read
	pathfiles='/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000001'
	# framerate of the IR camera in Hz
	framerate=50
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	# data=coleval.read_csv2(pathfiles,filenames)
	data=np.load(os.path.join(pathfiles,filenames))
	# temp=np.zeros(((1,1)+np.shape(data[1])))
	# temp[0,0,:,:]=data[0]
	# data=temp
	datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

	frame=163
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles)
	plt.imshow(data[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	plt.imshow(datatemp[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	plt.imshow(errdatatemp[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp error [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	ani=coleval.movie_from_data(datatemp,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Temp [°C]')
	# ani.save(os.path.join(pathfiles,filenames+'_Temp'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()

#
# THE LINE OF CODE (OUT OF PYTHON) TO MERGE THE COUNTS AND TEMPERATURE VIDEO IS (FOR SOME REASON IT WANTS THE OUTPUT IN .avi)
#
# ffmpeg -i irvb_full-000001.mp4 -i irvb_full-000001.npy_Temp.mp4 -filter_complex hstack -c:v ffv1 output.avi
#

	# foillx=32
	# foilrx=288
	# foildw=29
	# foilup=226
	#
	# datatempcrop=datatemp[:,:,foildw:foilup,foillx:foilrx]
	# errdatatempcrop=errdatatemp[:,:,foildw:foilup,foillx:foilrx]

	# frame=163
	# plt.figure()
	# plt.title('Frame '+str(frame)+' in '+pathfiles)
	# plt.pcolor(datatempcrop[0,frame])
	# plt.colorbar().set_label('Temp [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# # plt.figure()
	# # plt.title('Frame '+str(frame)+' in '+pathfiles)
	# # plt.pcolor(errdatatempcrop[0,frame])
	# # plt.colorbar().set_label('Temp error [°C]')
	# # plt.xlabel('Horizontal axis [pixles]')
	# # plt.ylabel('Vertical axis [pixles]')
	# plt.show()
	#
	# ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Temp [°C]')
	# plt.show()



	#I WILL ROTATE A FRAME TO SEE IF I CAN COME BACK TO THE ORIGINARL SITUATION
	frame=163
	rotangle=0 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	testorig=datatemp[0,frame]
	testrot=rotate(testorig,foilrotdeg,axes=(-1,-2))
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	plt.show()

	# HERE I RESTORE THE SITUATION

	rotangle=180.25 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[159,127]
	# foilcenter=[0,0]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=256
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	# foilhorizwpixel=20
	# foilvertwpixel=20
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	# irhoriz=np.shape(datatemp)[-1]
	# irvert=np.shape(datatemp)[-2]
	# alpha=np.arccos((foilhorizwpixel/2)/r)
	# foilx=np.add(foilcenter[0],[r*np.sin(-np.pi/2-alpha-foilrot),r*np.sin(-np.pi/2+alpha-foilrot),r*np.sin(np.pi/2-alpha-foilrot),r*np.sin(np.pi/2+alpha-foilrot),r*np.sin(-np.pi/2-alpha-foilrot)])
	# foily=np.add(foilcenter[1],[r*np.cos(-np.pi/2-alpha-foilrot),r*np.cos(-np.pi/2+alpha-foilrot),r*np.cos(np.pi/2-alpha-foilrot),r*np.cos(np.pi/2+alpha-foilrot),r*np.cos(-np.pi/2-alpha-foilrot)])
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)
	plt.figure()
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.grid()
	plt.show()

	# plt.figure()
	# plt.title('Sample frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+'pixel, foil rot '+str(rotangle)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	# plt.imshow(testrot,'rainbow',origin='lower')
	# plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	# plt.colorbar().set_label('Temp [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=10)
	# plt.plot(foilx,foily,'k')
	# plt.show()


	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	plt.colorbar().set_label('Temp [°C]')
	plt.plot(foilxint,foilyint,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.figure()
	testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	#plt.title('Sample frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+', foil rot '+str(rotangle)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	plt.imshow(testrotback,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig)) #this set the color limits
	# if rotangle>0:
	#	#plt.plot(foilcenter[0]*np.cos(foilrot)+foilcenter[1]*np.sin(foilrot),(irhoriz-foilcenter[0])*np.sin(foilrot)+foilcenter[1]*np.cos(foilrot),'o',markersize=20)
	#	foilcenterrot=[foilcenter[0]*np.cos(foilrot)+foilcenter[1]*np.sin(foilrot),(irhoriz-foilcenter[0])*np.sin(foilrot)+foilcenter[1]*np.cos(foilrot)]
	# else:
	#	#plt.plot(irvert*np.sin(-foilrot)+foilcenter[0]*np.cos(-foilrot)-foilcenter[1]*np.sin(-foilrot),foilcenter[0]*np.sin(-foilrot)+foilcenter[1]*np.cos(-foilrot),'o',markersize=20)
	#	foilcenterrot=[irvert*np.sin(-foilrot)+foilcenter[0]*np.cos(-foilrot)-foilcenter[1]*np.sin(-foilrot),foilcenter[0]*np.sin(-foilrot)+foilcenter[1]*np.cos(-foilrot)]
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)
	plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	plt.plot(foilxrot,foilyrot,'r')
	plt.title('Frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.colorbar().set_label('Temp [°C]')
	plt.show()


	# foillx=int(round(foilcenterrot[0]-foilhorizwpixel/2+0.1))
	# foilrx=int(round(foilcenterrot[0]+foilhorizwpixel/2+0.1)+1)
	# foilhorizwpixel=foilrx-foillx
	# foildw=int(round(foilcenterrot[1]-foilvertwpixel/2+0.1))
	# foilup=int(round(foilcenterrot[1]+foilvertwpixel/2+0.1)+1)
	# foilvertwpixel=foilup-foildw

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
	datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
	errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
	errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]
	plt.figure()
	plt.imshow(datatempcrop[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.title('Only foil in frame '+str(frame)+' in '+pathfiles+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()



	# FOIL PROPERTY ADJUSTMENT

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	plt.figure()
	plt.title('Foil emissivity sacled to camera pixels')
	plt.imshow(foilemissivityscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Emissivity [adimensional]')

	plt.figure()
	plt.title('Foil thickness sacled to camera pixels')
	plt.imshow(foilthicknessscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Thickness [micrometer]')
	plt.show()


	basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)
	dt=1/framerate
	dx=foilhorizw/(foilhorizwpixel-1)
	dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
	d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
	d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
	d2Tdxy=np.add(d2Tdx2,d2Tdy2)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
	# T04=np.power(np.multiply(zeroC+basetemp,np.ones((foilvertwpixel-2,foilhorizwpixel-2))),4)
	T04=np.power(np.add(zeroC,basetemp),4)


	# power=[]
	BBrad=[]
	diffusion=[]
	timevariation=[]
	ktf=np.multiply(conductivityscaled,foilthicknessscaled)
	for i in range(len(datatempcrop[:,0,0,0])):
		# power.append([])
		BBrad.append([])
		diffusion.append([])
		timevariation.append([])
		for j in range(len(datatempcrop[0,1:-1,0,0])):
			BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
			BBrad[i].append(BBradtemp)
			diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
			diffusion[i].append(diffusiontemp)
			timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
			timevariation[i].append(timevariationtemp)
			# powertemp=np.add(np.add(diffusiontemp,timevariationtemp),BBradtemp)
			# power[i].append(powertemp)
	# power=np.array(power)
	BBrad=np.array(BBrad)
	diffusion=np.array(diffusion)
	timevariation=np.array(timevariation)


	# ani=coleval.movie_from_data(power,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Power [W/m2]')
	# #ani.save(os.path.join(pathfiles,filenames+'_Power'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# plt.show()

	# powerreferenceframe=power[0,160]
	BBradreferenceframe=BBrad[0,160]
	diffusionreferenceframe=diffusion[0,160]
	timevariationreferenceframe=timevariation[0,160]

	#lines removed 08/05/2018 on suggestion of Matthew Reinke because they should not be necessary given the calibration we did.
	# powernoback=np.add(power,np.negative(powerreferenceframe))
	# BBradnoback=np.add(BBrad,np.negative(BBradreferenceframe))
	# diffusionnoback=np.add(diffusion,np.negative(diffusionreferenceframe))
	# timevariationnoback=np.add(timevariation,np.negative(timevariationreferenceframe))

	# powernoback=np.add(power,0)
	BBradnoback=np.add(BBrad,0)
	diffusionnoback=np.add(diffusion,0)
	timevariationnoback=np.add(timevariation,0)


	powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)


	# THIS LINES ARE ONLY TO SAVE A VIDEO OF ALL THE COMPONENTS OF THE ABSORBED POWER
	# ani=coleval.movie_from_data(powernoback,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Power [W/m2]')
	# ani.save(os.path.join(pathfiles,filenames+'_Power'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(BBradnoback,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Black Body Power [W/m2]')
	# ani.save(os.path.join(pathfiles,filenames+'_BBrad'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(diffusionnoback,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Diffusion Power [W/m2]')
	# ani.save(os.path.join(pathfiles,filenames+'_diffusion'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(timevariationnoback,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Time Derivative Power [W/m2]')
	# ani.save(os.path.join(pathfiles,filenames+'_timevariation'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# plt.show()

	totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
	totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
	totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
	totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)

	plt.plot(totalpower,label='totalpower')
	plt.plot(totalBBrad,label='totalBBrad')
	plt.plot(totaldiffusion,label='totaldiffusion')
	plt.plot(totaltimevariation,label='totaltimevariation')
	plt.title('Sum of the power over all the foil')
	plt.xlabel('Frame number')
	plt.ylabel('Power sum [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()

	powerzoom=7
	laserspot=[63,80]
	maxdw=laserspot[0]-powerzoom
	maxup=laserspot[0]+powerzoom+1
	maxlx=laserspot[1]-powerzoom
	maxdx=laserspot[1]+powerzoom+1
	plt.figure()
	plt.imshow(powernoback[0,0,maxdw:maxup,maxlx:maxdx],'rainbow',origin='lower')
	plt.colorbar().set_label('Power density [W/m^2]')
	plt.title('Area over which I zoom to calculate the sum of the power \n in '+str(laserspot)+' +/- '+str(powerzoom))
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
	BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
	diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
	timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]

	totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
	totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
	totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
	totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)

	plt.plot(totalpowercrop,label='totalpowercrop')
	plt.plot(totalBBradcrop,label='totalBBradcrop')
	plt.plot(totaldiffusioncrop,label='totaldiffusioncrop')
	plt.plot(totaltimevariationcrop,label='totaltimevariationcrop')
	plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom))
	plt.xlabel('Frame number')
	plt.ylabel('Power sum [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()


	# COMPARISON BETWEEN ALL THE LASER TESTS TO VERIFY IF THE POWER WE MEASURE HAS THE SAME BEHAVIOUR

	# THIS BIT IN THE BEGINNING IS TO HAVE THE REFERENCE FRAME FOR THE BLACK BODY RADIATION

	# degree of polynomial of choice
	n=3
	# type of integration time in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'
	# folder to read
	pathfiles='/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000001'
	# framerate of the IR camera in Hz
	framerate=50
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

	frame=163
	rotangle=180.25 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	testorig=datatemp[0,frame]
	testrot=rotate(testorig,foilrotdeg,axes=(-1,-2))
	foilcenter=[159,127]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=256
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)


	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)


	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
	datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
	errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
	errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]

	# FOIL PROPERTY ADJUSTMENT

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)


	maxpower=[]
	for pathfiles in laser1:

		filenames=coleval.all_file_names(pathfiles,type)[0]
		data=np.load(os.path.join(pathfiles,filenames))
		datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

		datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
		datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
		errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
		errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]


		# basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)
		dt=1/framerate
		dx=foilhorizw/(foilhorizwpixel-1)
		dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
		d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
		d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
		d2Tdxy=np.add(d2Tdx2,d2Tdy2)
		negd2Tdxy=np.multiply(-1,d2Tdxy)
		T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
		# T04=np.power(np.multiply(zeroC+basetemp,np.ones((foilvertwpixel-2,foilhorizwpixel-2))),4)
		T04=np.power(np.add(zeroC,basetemp),4)


		# power=[]
		BBrad=[]
		diffusion=[]
		timevariation=[]
		ktf=np.multiply(conductivityscaled,foilthicknessscaled)
		for i in range(len(datatempcrop[:,0,0,0])):
			# power.append([])
			BBrad.append([])
			diffusion.append([])
			timevariation.append([])
			for j in range(len(datatempcrop[0,1:-1,0,0])):
				BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
				BBrad[i].append(BBradtemp)
				diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
				diffusion[i].append(diffusiontemp)
				timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
				timevariation[i].append(timevariationtemp)
				# powertemp=np.add(np.add(diffusiontemp,timevariationtemp),BBradtemp)
				# power[i].append(powertemp)
		# power=np.array(power)
		BBrad=np.array(BBrad)
		diffusion=np.array(diffusion)
		timevariation=np.array(timevariation)

		# powernoback=np.add(power,0)
		BBradnoback=np.add(BBrad,0)
		diffusionnoback=np.add(diffusion,0)
		timevariationnoback=np.add(timevariation,0)

		powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)


		totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
		totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
		totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
		totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)

		# plt.plot(totalpower,label='totalpower')
		# plt.plot(totalBBrad,label='totalBBrad')
		# plt.plot(totaldiffusion,label='totaldiffusion')
		# plt.plot(totaltimevariation,label='totaltimevariation')
		# plt.title('Sum of the power over all the foil')
		# plt.xlabel('Frame number')
		# plt.ylabel('Power sum [W]')
		# plt.legend()
		# plt.legend(loc='best')
		# plt.show()

		powerzoom=7
		laserspot=[63,80]
		maxdw=laserspot[0]-powerzoom
		maxup=laserspot[0]+powerzoom+1
		maxlx=laserspot[1]-powerzoom
		maxdx=laserspot[1]+powerzoom+1
		# plt.figure()
		# plt.imshow(powernoback[0,0,maxdw:maxup,maxlx:maxdx],'rainbow',origin='lower')
		# plt.colorbar().set_label('Power density [W/m^2]')
		# plt.title('Area over which I zoom to calculate the sum of the power \n in '+str(laserspot)+' +/- '+str(powerzoom))
		# plt.xlabel('Horizontal axis [pixles]')
		# plt.ylabel('Vertical axis [pixles]')
		# plt.show()

		powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
		BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
		diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
		timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]

		totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
		totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
		totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
		totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)

		plt.plot(totalpowercrop,label='totalpowercrop')
		plt.plot(totalBBradcrop,label='totalBBradcrop')
		plt.plot(totaldiffusioncrop,label='totaldiffusioncrop')
		plt.plot(totaltimevariationcrop,label='totaltimevariationcrop')
		plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom))
		plt.xlabel('Frame number')
		plt.ylabel('Power sum [W]')
		plt.legend()
		plt.legend(loc='best')
		plt.show()

		maxpower.append(max(totalpowercrop))

	comparepower=np.interp(voltlaser1,reflaserfvoltage,reflaserpower)
	plt.plot(maxpower,'k+',markersize=15,label='maximum of power sum from foil measurement')
	plt.plot(np.multiply(2.7,maxpower),'r+',markersize=15,label='maximum of power sum from foil measurement x 2.7')
	plt.plot(comparepower,'-',label='power from laser calibration')
	plt.title('Power of the first series of laser measurements')
	plt.xlabel('Index of the measurement')
	plt.ylabel('Power [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()




	# THIS LINES ARE TO DO THE SAME EVALUATION BUT AVERAGING OVER SOME PIXELS STARTING FROM THE TEMPERATURE
	# UP TO 12/06/2018 THIS WAS LEFT BEHIND

	datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
	errdatatempcrop=errdatatemp[:,:,foildw:foilup,foillx:foilrx]

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))


	basetemp=np.mean(datatempcrop[0,160,:,:])
	dt=1/framerate
	dx=foilhorizw/(foilhorizwpixel-1)
	dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
	d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
	d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
	d2Tdxy=np.add(d2Tdx2,d2Tdy2)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
	T04=np.power(np.multiply(zeroC+basetemp,np.ones((foilvertwpixel-2,foilhorizwpixel-2))),4)




	pixelmean=2
	# with warnings.catch_warnings():
	frames=BBradnoback
	shapeorig=np.shape(frames)
	nframes=shapeorig[1]
	framesaver=[]
	# framesaver.append([])
	framesaver.append([None]*nframes)
	framesaver=np.array(framesaver)
	for i in range(nframes):
		print("i",i)
		framesaver[0,i]=coleval.average_frame(frames[0,i],pixelmean,extremedelete)
	# framesaver=np.array(framesaver)
	# plt.imshow(framesaver[0,1],origin='lower')
	# plt.show()
	ani=coleval.movie_from_data(framesaver,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Power [W/m2]')
	#ani.save(os.path.join(pathfiles,filenames+'_Power'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()

	#
	# totalpower=[]
	# for frame in powernoback[0]:
	# 	powtemp=np.sum(frame)*dx**2
	# 	totalpower.append(powtemp)
	# plt.plot(totalpower)
	# plt.show()
	#
	# frame=power[0,2]
	# frameaver1=coleval.average_frame(frame,pixelmean,extremedelete=False)
	# frame=power[0,160]
	# frameaver2=coleval.average_frame(frame,pixelmean,extremedelete=False)
	# np.sum(frameaver1-frameaver2)*dx**2
	# plt.imshow(frameaver1-frameaver2,origin='lower')
	# plt.colorbar().set_label('Power [W/m2]')
	# plt.show()

	fig, axs = plt.subplots()
	axs.imshow(foilemissivity2)
	plt.show()




	# 12/06/2018 I WANT TO DO THE SAME PROCEDURE TO FIND THE MAXIMUM POWER ON THE FOIL FOR THE SECOND SET OF MEASUREMENTS
	# THIS WAS TAKEN WITH 2MS INTEGRATION TIME. i DON'T HAVE A GOOD FITTING FOR THE PARAMETERS WITHOUT WINDOW.
	# I WILL USE THE CONVERSION FACTORS WITH THE WINDOW. THE DIFFERENCE IS NOT MASSIVE AND I DO IT ONLY TO RECOGNISE
	# AN EXPECTED LASER POWER/MEASURED POWER CORRELATION

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles='/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000001'
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

	ani=coleval.movie_from_data(datatemp,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Temp [°C]')
	# ani.save(os.path.join(pathfiles,filenames+'_Temp'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()

	frame=200
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles)
	plt.imshow(data[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	plt.imshow(datatemp[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	plt.imshow(errdatatemp[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp error [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()


	testorig=datatemp[0,frame]
	testrot=testorig
	rotangle=0.55 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[161,121]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=256
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)
	plt.figure()
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.grid()
	plt.show()

	# plt.figure()
	# plt.title('Sample frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+'pixel, foil rot '+str(rotangle)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	# plt.imshow(testrot,'rainbow',origin='lower')
	# plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	# plt.colorbar().set_label('Temp [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=10)
	# plt.plot(foilx,foily,'k')
	# plt.show()


	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	plt.colorbar().set_label('Temp [°C]')
	plt.plot(foilxint,foilyint,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.figure()
	testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	plt.imshow(testrotback,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig)) #this set the color limits
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)
	plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	plt.plot(foilxrot,foilyrot,'r')
	plt.title('Frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.colorbar().set_label('Temp [°C]')
	plt.show()

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
	datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
	errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
	errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]
	plt.figure()
	plt.imshow(datatempcrop[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.title('Only foil in frame '+str(frame)+' in '+pathfiles+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()



	# FOIL PROPERTY ADJUSTMENT

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	plt.figure()
	plt.title('Foil emissivity sacled to camera pixels')
	plt.imshow(foilemissivityscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Emissivity [adimensional]')

	plt.figure()
	plt.title('Foil thickness sacled to camera pixels')
	plt.imshow(foilthicknessscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Thickness [micrometer]')
	plt.show()


	basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)
	dt=1/framerate
	dx=foilhorizw/(foilhorizwpixel-1)
	dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
	d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
	d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
	d2Tdxy=np.add(d2Tdx2,d2Tdy2)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
	T04=np.power(np.add(zeroC,basetemp),4)


	BBrad=[]
	diffusion=[]
	timevariation=[]
	ktf=np.multiply(conductivityscaled,foilthicknessscaled)
	for i in range(len(datatempcrop[:,0,0,0])):
		BBrad.append([])
		diffusion.append([])
		timevariation.append([])
		for j in range(len(datatempcrop[0,1:-1,0,0])):
			BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
			BBrad[i].append(BBradtemp)
			diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
			diffusion[i].append(diffusiontemp)
			timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
			timevariation[i].append(timevariationtemp)
	BBrad=np.array(BBrad)
	diffusion=np.array(diffusion)
	timevariation=np.array(timevariation)


	# ani=coleval.movie_from_data(power,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Power [W/m2]')
	# #ani.save(os.path.join(pathfiles,filenames+'_Power'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# plt.show()

	BBradnoback=np.add(BBrad,0)
	diffusionnoback=np.add(diffusion,0)
	timevariationnoback=np.add(timevariation,0)

	powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)

	totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
	totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
	totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
	totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)

	plt.plot(totalpower,label='totalpower')
	plt.plot(totalBBrad,label='totalBBrad')
	plt.plot(totaldiffusion,label='totaldiffusion')
	plt.plot(totaltimevariation,label='totaltimevariation')
	plt.title('Sum of the power over all the foil')
	plt.xlabel('Frame number')
	plt.ylabel('Power sum [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()

	searchframe=1500
	plt.figure()
	plt.title('Frame '+str(searchframe)+' in '+pathfiles)
	plt.imshow(BBradnoback[0,searchframe],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Power [W]')
	plt.show()


	powerzoom=7
	laserspot=[133,171]
	maxdw=laserspot[0]-powerzoom
	maxup=laserspot[0]+powerzoom+1
	maxlx=laserspot[1]-powerzoom
	maxdx=laserspot[1]+powerzoom+1
	plt.figure()
	plt.imshow(powernoback[0,0,maxdw:maxup,maxlx:maxdx],'rainbow',origin='lower')
	plt.colorbar().set_label('Power density [W/m^2]')
	plt.title('Area over which I zoom to calculate the sum of the power \n in '+str(laserspot)+' +/- '+str(powerzoom))
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
	BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
	diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
	timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]

	totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
	totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
	totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
	totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)

	plt.plot(totalpowercrop,label='totalpowercrop')
	plt.plot(totalBBradcrop,label='totalBBradcrop')
	plt.plot(totaldiffusioncrop,label='totaldiffusioncrop')
	plt.plot(totaltimevariationcrop,label='totaltimevariationcrop')
	plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom))
	plt.xlabel('Frame number')
	plt.ylabel('Power sum [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()


	# COMPARISON BETWEEN ALL THE LASER TESTS TO VERIFY IF THE POWER WE MEASURE HAS THE SAME BEHAVIOUR

	# THIS BIT IN THE BEGINNING IS TO HAVE THE REFERENCE FRAME FOR THE BLACK BODY RADIATION

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/2ms383Hz/average'
	# folder to read
	pathfiles='/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000001'
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=2
	#filestype
	type='npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)


	frame=200
	testorig=datatemp[0,frame]
	testrot=testorig
	rotangle=0.55 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[161,121]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=256
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)


	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)


	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
	datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
	errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
	errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]

	# FOIL PROPERTY ADJUSTMENT

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	frame=200
	basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)


	maxpower=[]
	for pathfiles in laser2:

		filenames=coleval.all_file_names(pathfiles,type)[0]
		data=np.load(os.path.join(pathfiles,filenames))
		datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

		datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
		datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
		errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
		errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]


		# basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)
		dt=1/framerate
		dx=foilhorizw/(foilhorizwpixel-1)
		dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
		d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
		d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
		d2Tdxy=np.add(d2Tdx2,d2Tdy2)
		negd2Tdxy=np.multiply(-1,d2Tdxy)
		T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
		T04=np.power(np.add(zeroC,basetemp),4)


		BBrad=[]
		diffusion=[]
		timevariation=[]
		ktf=np.multiply(conductivityscaled,foilthicknessscaled)
		for i in range(len(datatempcrop[:,0,0,0])):
			BBrad.append([])
			diffusion.append([])
			timevariation.append([])
			for j in range(len(datatempcrop[0,1:-1,0,0])):
				BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
				BBrad[i].append(BBradtemp)
				diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
				diffusion[i].append(diffusiontemp)
				timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
				timevariation[i].append(timevariationtemp)
		BBrad=np.array(BBrad)
		diffusion=np.array(diffusion)
		timevariation=np.array(timevariation)

		BBradnoback=np.add(BBrad,0)
		diffusionnoback=np.add(diffusion,0)
		timevariationnoback=np.add(timevariation,0)

		powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)


		totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
		totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
		totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
		totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)

		# plt.plot(totalpower,label='totalpower')
		# plt.plot(totalBBrad,label='totalBBrad')
		# plt.plot(totaldiffusion,label='totaldiffusion')
		# plt.plot(totaltimevariation,label='totaltimevariation')
		# plt.title('Sum of the power over all the foil')
		# plt.xlabel('Frame number')
		# plt.ylabel('Power sum [W]')
		# plt.legend()
		# plt.legend(loc='best')
		# plt.show()

		powerzoom=7
		laserspot=[133,171]
		maxdw=laserspot[0]-powerzoom
		maxup=laserspot[0]+powerzoom+1
		maxlx=laserspot[1]-powerzoom
		maxdx=laserspot[1]+powerzoom+1
		# plt.figure()
		# plt.imshow(powernoback[0,0,maxdw:maxup,maxlx:maxdx],'rainbow',origin='lower')
		# plt.colorbar().set_label('Power density [W/m^2]')
		# plt.title('Area over which I zoom to calculate the sum of the power \n in '+str(laserspot)+' +/- '+str(powerzoom))
		# plt.xlabel('Horizontal axis [pixles]')
		# plt.ylabel('Vertical axis [pixles]')
		# plt.show()

		powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
		BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
		diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
		timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]

		totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
		totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
		totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
		totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)

		plt.plot(totalpowercrop,label='totalpowercrop')
		plt.plot(totalBBradcrop,label='totalBBradcrop')
		plt.plot(totaldiffusioncrop,label='totaldiffusioncrop')
		plt.plot(totaltimevariationcrop,label='totaltimevariationcrop')
		plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom))
		plt.xlabel('Frame number')
		plt.ylabel('Power sum [W]')
		plt.legend()
		plt.legend(loc='best')
		plt.show()

		maxpower.append(max(totalpowercrop))

	comparepower=np.interp(voltlaser2,reflaserfvoltage,reflaserpower)
	plt.plot(maxpower,'k+',markersize=15,label='maximum of power sum from foil measurement')
	plt.plot(np.multiply(2.5,maxpower),'r+',markersize=15,label='maximum of power sum from foil measurement x 2.5')
	plt.plot(comparepower,'-',label='power from laser calibration')
	plt.title('Power of the second series of laser measurements')
	plt.xlabel('Index of the measurement')
	plt.ylabel('Power [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()


	# 26/07/2018
	# I do the same game again for the measurements taken in July.
	# I consider alltogether the measurements with focused and partially defocused laser
	# right over the pinhole out of aberration


	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
		pathfiles=laser20[0]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='.npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]

	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:128,128:]=datashort
	data=np.load(os.path.join(pathfiles,filenames))
	datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)


	ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=3000,extvmax=3200)
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()




	frame=1300
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles)
	plt.imshow(data[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	plt.imshow(datatemp[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	plt.imshow(errdatatemp[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp error [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()


	testorig=datatemp[0,frame]
	testrot=testorig
	rotangle=-1.5 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[160,133]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=240
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)
	plt.figure()
	plt.imshow(testrot,'rainbow',origin='lower')
	# plt.imshow(testrot,'rainbow',vmin=26., vmax=27.,origin='lower')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.grid()
	plt.show()

	# plt.figure()
	# plt.title('Sample frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+'pixel, foil rot '+str(rotangle)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	# plt.imshow(testrot,'rainbow',origin='lower')
	# plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	# plt.colorbar().set_label('Temp [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=10)
	# plt.plot(foilx,foily,'k')
	# plt.show()


	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	plt.colorbar().set_label('Temp [°C]')
	plt.plot(foilxint,foilyint,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.figure()
	testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	plt.imshow(testrotback,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testorig), vmax=np.max(testorig)) #this set the color limits
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)
	plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	plt.plot(foilxrot,foilyrot,'r')
	plt.title('Frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.colorbar().set_label('Temp [°C]')
	plt.show()

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
	datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
	errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
	errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]
	plt.figure()
	plt.imshow(datatempcrop[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.title('Only foil in frame '+str(frame)+' in '+pathfiles+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_foil_temp'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=26.5,extvmax=27.2)
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_foil_temp_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()

	# FOIL PROPERTY ADJUSTMENT

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	# foilemissivityscaled=1*np.ones((foilvertwpixel,foilhorizwpixel))
	# foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel,foilhorizwpixel))
	# conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel,foilhorizwpixel))
	# reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel,foilhorizwpixel))

	plt.figure()
	plt.title('Foil emissivity sacled to camera pixels')
	plt.imshow(foilemissivityscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Emissivity [adimensional]')

	plt.figure()
	plt.title('Foil thickness sacled to camera pixels')
	plt.imshow(1000000*foilthicknessscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Thickness [micrometer]')
	plt.show()


	# New way of calculating the background
	type='_reference.npy'
	filenames=coleval.all_file_names(pathfiles,type)[0]
	basecounts=np.load(os.path.join(pathfiles,filenames))
	basedatatemp,baseerrdatatemp=coleval.count_to_temp_poly2([[basecounts]],params,errparams)
	basedatatemprot=rotate(basedatatemp,foilrotdeg,axes=(-1,-2))
	basedatatempcrop=basedatatemprot[:,:,foildw:foilup,foillx:foilrx]
	basetemp=np.mean(basedatatempcrop[0,:,1:-1,1:-1],axis=0)

	# basetemp=np.mean(datatempcrop[0,frame-7:frame+7,1:-1,1:-1],axis=0)
	dt=1/framerate
	dx=foilhorizw/(foilhorizwpixel-1)
	dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
	d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
	d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
	d2Tdxy=np.add(d2Tdx2,d2Tdy2)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
	T04=np.power(np.add(zeroC,basetemp),4)


	BBrad=[]
	diffusion=[]
	timevariation=[]
	ktf=np.multiply(conductivityscaled,foilthicknessscaled)
	for i in range(len(datatempcrop[:,0,0,0])):
		BBrad.append([])
		diffusion.append([])
		timevariation.append([])
		for j in range(len(datatempcrop[0,1:-1,0,0])):
			BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
			BBrad[i].append(BBradtemp)
			diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
			diffusion[i].append(diffusiontemp)
			timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
			timevariation[i].append(timevariationtemp)
	BBrad=np.array(BBrad)
	diffusion=np.array(diffusion)
	timevariation=np.array(timevariation)

	BBradnoback=np.add(BBrad,0)
	diffusionnoback=np.add(diffusion,0)
	timevariationnoback=np.add(timevariation,0)

	powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)

	ani=coleval.movie_from_data(powernoback,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Power [W/m2]')
	# ani.save(os.path.join(pathfiles,filenames+'_Power'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()
	ani=coleval.movie_from_data(powernoback,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Power [W/m2]',extvmin=0,extvmax=800)
	plt.show()

	totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
	totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
	totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
	totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)

	plt.plot(totalpower,label='totalpower')
	plt.plot(totalBBrad,label='totalBBrad')
	plt.plot(totaldiffusion,label='totaldiffusion')
	plt.plot(totaltimevariation,label='totaltimevariation')
	plt.title('Sum of the power over all the foil')
	plt.xlabel('Frame number')
	plt.ylabel('Power sum [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.grid()
	plt.show()

	searchframe=280
	plt.figure()
	plt.title('Frame '+str(searchframe)+' in '+pathfiles)
	plt.imshow(powernoback[0,searchframe],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Power [W/m2]')
	plt.show()


	powerzoom=5
	laserspot=[41,172]
	maxdw=laserspot[0]-powerzoom
	maxup=laserspot[0]+powerzoom+1
	maxlx=laserspot[1]-powerzoom
	maxdx=laserspot[1]+powerzoom+1
	plt.figure()
	plt.imshow(powernoback[0,searchframe,maxdw:maxup,maxlx:maxdx],'rainbow',origin='lower')
	plt.colorbar().set_label('Power density [W/m^2]')
	plt.title('Area over which I zoom to calculate the sum of the power \n in '+str(laserspot)+' +/- '+str(powerzoom))
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
	BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
	diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
	timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]

	totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
	totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
	totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
	totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)

	powermean=np.mean(totalpowercrop)
	sample = totalpowercrop[(totalpowercrop>powermean)]
	guess=[sample.max()]
	test4,test3=curve_fit(costant,np.linspace(1,len(sample),len(sample)),sample, p0=guess, maxfev=100000000)


	plt.plot(totalpowercrop,label='totalpowercrop')
	plt.plot(totalBBradcrop,label='totalBBradcrop')
	plt.plot(totaldiffusioncrop,label='totaldiffusioncrop')
	plt.plot(totaltimevariationcrop,label='totaltimevariationcrop')
	plt.plot(costant(np.linspace(1,len(totalpowercrop),len(totalpowercrop)),*test4),label='sampled power')
	plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom)+' max='+str(int(test4*100000)/100)+'mW, std='+str(int(np.sqrt(test3)*10000000)/10000)+'mW')
	plt.xlabel('Frame number')
	plt.ylabel('Power sum [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.grid()
	plt.show()






	# COMPARISON BETWEEN ALL THE LASER TESTS TO VERIFY IF THE POWER WE MEASURE HAS THE SAME BEHAVIOUR
	#
	# 03/08/2018 CHANGED!!!  THIS NOW IS JUST THE PREPROCESSING OF THE DATA JUST CONVERTED IN npy
	# TO GET - ONLY - THE TEMPERATURES
	#
	# THIS BIT IN THE BEGINNING IS TO HAVE THE REFERENCE FRAME FOR THE BLACK BODY RADIATION


	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=laser10[0]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))

	frame=np.shape(data)[1]//2
	base_frame_range=np.shape(data)[1]//2
	testorig=data[0,frame]
	testrot=testorig
	rotangle=-1.5 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[160,133]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=240
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)


	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)


	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	# FOIL PROPERTY ADJUSTMENT

	# foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	# foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	# conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	# reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	foilemissivityscaled=1*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
	foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
	conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
	reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))

	basecounts=data


	maxpower=[]
	files=[laser13,laser14]
	files=coleval.flatten_full(files)
	for pathfiles in files:

		filenames=coleval.all_file_names(pathfiles,type)[0]
		if (pathfiles in laser12):
			framerate=994
			datashort=np.load(os.path.join(pathfiles,filenames))
			data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
			data[:,:,64:96,:]=datashort
			type_of_experiment='low duty cycle partially defocused'
			poscentred=[[15,80],[40,75],[80,85]]
		elif (pathfiles in [vacuum1[1],vacuum1[3]]):
			framerate=994
			datashort=np.load(os.path.join(pathfiles,filenames))
			data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
			data[:,:,64:96,:]=datashort
			type_of_experiment='low duty cycle partially defocused'
			poscentred=[[15,80],[40,75],[80,85]]
		elif (pathfiles in laser11):
			framerate=383
			data=np.load(os.path.join(pathfiles,filenames))
			type_of_experiment='partially defocused'
			poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
		else:
			framerate=383
			data=np.load(os.path.join(pathfiles,filenames))
			type_of_experiment='focused'
			poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]



		# Added 01/08/2018 to account for variations in the background counts over time
		data_mean_difference=[]
		data_mean_difference_std=[]
		# poscentred=[[70,70],[160,133],[250,200]]
		# poscentred=[[60,12],[170,12],[290,12]]
		for pos in poscentred:
			for a in [5,10]:
				temp1=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
				temp1std=np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
				temp2=np.mean(basecounts[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
				temp2std=np.std(np.mean(basecounts[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
				data_mean_difference.append(temp1-temp2)
				data_mean_difference_std.append(temp1std+temp2std)
		data_mean_difference=np.array(data_mean_difference)
		data_mean_difference_std=np.array(data_mean_difference_std)
		guess=[1]
		base_counts_correction,temp2=curve_fit(costant,np.linspace(1,len(data_mean_difference),len(data_mean_difference)),data_mean_difference, sigma=data_mean_difference_std,p0=guess, maxfev=100000000)
		print('background correction = '+str(int(base_counts_correction*1000)/1000))

		datatemp,errdatatemp=coleval.count_to_temp_poly2(basecounts[:,frame-base_frame_range//2:frame+base_frame_range//2]+base_counts_correction,params,errparams)
		datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
		datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
		# errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
		# errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]
		basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)



		datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)


		# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.close()
		# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=5900,extvmax=6020)
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.close()

		datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
		datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
		errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
		errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]

		np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_temperature_temperror_basetemp'),datatempcrop=datatempcrop,errdatatempcrop=errdatatempcrop,basetemp=basetemp)

		# ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_foil_temp'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.close()
		# ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=26.5,extvmax=27.2)
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_foil_temp_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.close()

		# basetemp=np.mean(datatempcrop[0,frame-5:frame+5,1:-1,1:-1],axis=0)
		# dt=1/framerate
		# dx=foilhorizw/(foilhorizwpixel-1)
		# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
		# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
		# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
		# d2Tdxy=np.add(d2Tdx2,d2Tdy2)
		# negd2Tdxy=np.multiply(-1,d2Tdxy)
		# T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
		# T04=np.power(np.add(zeroC,basetemp),4)
		#
		#
		# BBrad=[]
		# diffusion=[]
		# timevariation=[]
		# ktf=np.multiply(conductivityscaled,foilthicknessscaled)
		# for i in range(len(datatempcrop[:,0,0,0])):
		# 	BBrad.append([])
		# 	diffusion.append([])
		# 	timevariation.append([])
		# 	for j in range(len(datatempcrop[0,1:-1,0,0])):
		# 		BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
		# 		BBrad[i].append(BBradtemp)
		# 		diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
		# 		diffusion[i].append(diffusiontemp)
		# 		timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
		# 		timevariation[i].append(timevariationtemp)
		# BBrad=np.array(BBrad)
		# diffusion=np.array(diffusion)
		# timevariation=np.array(timevariation)
		#
		# BBradnoback=np.add(BBrad,0)
		# diffusionnoback=np.add(diffusion,0)
		# timevariationnoback=np.add(timevariation,0)
		#
		# powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)
		#
		#
		# totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
		# totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
		# totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
		# totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)
		#
		# # plt.plot(totalpower,label='totalpower')
		# # plt.plot(totalBBrad,label='totalBBrad')
		# # plt.plot(totaldiffusion,label='totaldiffusion')
		# # plt.plot(totaltimevariation,label='totaltimevariation')
		# # plt.title('Sum of the power over all the foil')
		# # plt.xlabel('Frame number')
		# # plt.ylabel('Power sum [W]')
		# # plt.legend()
		# # plt.legend(loc='best')
		# # plt.savefig(os.path.join(pathfiles,filenames[:-4]+'power_sum_all_foil'+'.eps'))
		# # plt.grid()
		# # plt.close()
		#
		# # store_data=[powernoback,BBradnoback,diffusionnoback,timevariationnoback]
		#
		# for addition in [-1,0,1,2]:
		# 	if ((pathfiles in laser11) or (pathfiles in laser12)):
		# 		# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_temperature'),powernoback=powernoback)
		# 		powerzoom=4+addition
		# 		laserspot=[41,172]
		# 	elif (pathfiles in laser13):
		# 		# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
		# 		powerzoom=2+addition
		# 		laserspot=[26,196]
		# 	elif (pathfiles in laser14):
		# 		# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
		# 		powerzoom=2+addition
		# 		laserspot=[27,156]
		# 	else:
		# 		# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total-BB-diff-time'),powernoback=powernoback,BBradnoback=BBradnoback,diffusionnoback=diffusionnoback,timevariationnoback=timevariationnoback)
		# 		# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
		# 		powerzoom=2+addition
		# 		laserspot=[41,172]
		#
		# 	# powerzoom=2
		# 	# laserspot=[41,172]
		# 	maxdw=laserspot[0]-powerzoom
		# 	maxup=laserspot[0]+powerzoom+1
		# 	maxlx=laserspot[1]-powerzoom
		# 	maxdx=laserspot[1]+powerzoom+1
		# 	# plt.figure()
		# 	# plt.imshow(powernoback[0,0,maxdw:maxup,maxlx:maxdx],'rainbow',origin='lower')
		# 	# plt.colorbar().set_label('Power density [W/m^2]')
		# 	# plt.title('Area over which I zoom to calculate the sum of the power \n in '+str(laserspot)+' +/- '+str(powerzoom))
		# 	# plt.xlabel('Horizontal axis [pixles]')
		# 	# plt.ylabel('Vertical axis [pixles]')
		# 	# plt.show()
		#
		# 	powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
		# 	BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
		# 	diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
		# 	timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]
		#
		# 	totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
		# 	totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
		# 	totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
		# 	totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)
		#
		# 	test2=np.sort(totalpowercrop)
		# 	test4=test2[-30]
		#
		# 	plt.plot(totalpowercrop,label='totalpowercrop')
		# 	plt.plot(totalBBradcrop,label='totalBBradcrop')
		# 	plt.plot(totaldiffusioncrop,label='totaldiffusioncrop')
		# 	plt.plot(totaltimevariationcrop,label='totaltimevariationcrop')
		# 	plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom)+' max='+str(int(test4*100000)/100)+'mW')
		# 	plt.xlabel('Frame number')
		# 	plt.ylabel('Power sum [W]')
		# 	plt.legend()
		# 	plt.legend(loc='best')
		# 	plt.grid()
		# 	plt.savefig(os.path.join(pathfiles,filenames[:-4]+'flat_propr_power_sum_zoom_'+str(powerzoom)+'.eps'))
		# 	plt.close()

	# 	maxpower.append(test4)
	#
	# comparepower=np.interp(voltlaser10,reflaserfvoltage1,reflaserpower1)
	# plt.plot(maxpower,'k+',markersize=15,label='maximum of power sum from foil measurement')
	# # plt.plot(np.multiply(2.5,maxpower),'r+',markersize=15,label='maximum of power sum from foil measurement x 2.5')
	# plt.plot(comparepower,'-',label='power from laser calibration')
	# plt.title('Power of the '+type_of_experiment+' series of laser measurements')
	# plt.xlabel('Index of the measurement')
	# plt.ylabel('Power [W]')
	# plt.legend()
	# plt.legend(loc='best')
	# plt.savefig(pathfiles+'/power_comparison'+'.eps')
	# plt.show()




	# PRE - PROCESSING 2 12/08/2018 CREATED TO DEAL WITH THE EXTRAPOLATION OF THE BACKGROUND FROM THE TIMESTAMP
	# OF THE MEASURE INSTEAD OF THE CHANGE OF BACKGROUND IN DIFFERENT POSITIONS.

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=vacuum2[20]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='_stat.npy'
	# type='csv'

	fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))[0]

	# frame=np.shape(data)[1]//2
	# base_frame_range=np.shape(data)[1]//2
	testrot=data
	rotangle=-1.5 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[160,133]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=240
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)


	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)


	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	# FOIL PROPERTY ADJUSTMENT

	# foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	# foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	# conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	# reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	# foilemissivityscaled=1*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
	# foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
	# conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
	# reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))

	basecounts=data



	files=[laser15,laser16,laser17,laser18,laser19]
	files=coleval.flatten_full(files)
	for pathfiles in files:
		framerate, data, type_of_experiment, poscentred, pathparams, inttime = laser_shot_library(pathfiles)
		# type='.npy'
		# filenames=coleval.all_file_names(pathfiles,type)[0]
		# if (pathfiles in laser12):
		# 	framerate=994
		# 	datashort=np.load(os.path.join(pathfiles,filenames))
		# 	data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	data[:,:,64:96,:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	poscentred=[[15,80],[40,75],[80,85]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		# 	inttime=1
		# elif (pathfiles in coleval.flatten_full([laser15,laser16,laser21,laser23])):
		# 	framerate=994
		# 	datashort=np.load(os.path.join(pathfiles,filenames))
		# 	data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	data[:,:,64:128,:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	poscentred=[[15,80],[40,75],[80,85]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		# 	inttime=1
		# elif (pathfiles in coleval.flatten_full([laser24,laser26,laser27,laser28,laser29,laser31])):
		# 	framerate=1974
		# 	datashort=np.load(os.path.join(pathfiles,filenames))
		# 	data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	data[:,:,64:128,128:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	poscentred=[[15,80],[40,75],[80,85]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/0.5ms383Hz/average'
		# 	inttime=0.5
		# elif (pathfiles in [vacuum1[1],vacuum1[3]]):
		# 	framerate=994
		# 	datashort=np.load(os.path.join(pathfiles,filenames))
		# 	data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	data[:,:,64:96,:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	poscentred=[[15,80],[40,75],[80,85]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		# 	inttime=1
		# elif (pathfiles in laser11):
		# 	framerate=383
		# 	data=np.load(os.path.join(pathfiles,filenames))
		# 	type_of_experiment='partially defocused'
		# 	poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		# 	inttime=1
		# elif (pathfiles in coleval.flatten_full([laser25,laser30])):
		# 	framerate=383
		# 	data=np.load(os.path.join(pathfiles,filenames))
		# 	type_of_experiment='focused'
		# 	poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/2ms383Hz/average'
		# 	inttime=2
		# else:
		# 	framerate=383
		# 	data=np.load(os.path.join(pathfiles,filenames))
		# 	type_of_experiment='focused'
		# 	poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
		# 	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		# 	inttime=1


			params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
			errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))



		if (pathfiles in coleval.flatten_full([laser12])):
			# Added 01/08/2018 to account for variations in the background counts over time

			base_counts_correction = track_change_from_baseframe(data, poscentred, basecounts)

			# data_mean_difference=[]
			# data_mean_difference_std=[]
			# # poscentred=[[70,70],[160,133],[250,200]]
			# # poscentred=[[60,12],[170,12],[290,12]]
			# for pos in poscentred:
			# 	for a in [5,10]:
			# 		temp1=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
			# 		temp1std=np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
			# 		temp2=np.mean(basecounts[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
			# 		temp2std=np.std(np.mean(basecounts[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
			# 		data_mean_difference.append(temp1-temp2)
			# 		data_mean_difference_std.append(temp1std+temp2std)
			# data_mean_difference=np.array(data_mean_difference)
			# data_mean_difference_std=np.array(data_mean_difference_std)
			# guess=[1]
			# base_counts_correction,temp2=curve_fit(costant,np.linspace(1,len(data_mean_difference),len(data_mean_difference)),data_mean_difference, sigma=data_mean_difference_std,p0=guess, maxfev=100000000)
			# print('background correction = '+str(int(base_counts_correction*1000)/1000))

			datatemp,errdatatemp=coleval.count_to_temp_poly2(basecounts[:,frame-base_frame_range//2:frame+base_frame_range//2]+base_counts_correction,params,errparams)
			datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
			datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
			basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)
		elif (pathfiles in coleval.flatten_full([laser10[8:]])):
			# Added 19/08/2018 to use the best data available for building the reference
			type='_reference.npy'
			filenames_basecounts=coleval.all_file_names(laser10[7],type)[0]
			basecounts=np.load(os.path.join(laser10[7],filenames_basecounts))
			filenames=filenames[:-4]+'_reference'+filenames[-4:]

			base_counts_correction = track_change_from_baseframe(data, poscentred, basecounts)

			# data_mean_difference=[]
			# data_mean_difference_std=[]
			# # poscentred=[[70,70],[160,133],[250,200]]
			# # poscentred=[[60,12],[170,12],[290,12]]
			# for pos in poscentred:
			# 	for a in [5,10]:
			# 		temp1=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
			# 		temp1std=np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
			# 		temp2=np.mean(basecounts[pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))
			# 		temp2std=0
			# 		data_mean_difference.append(temp1-temp2)
			# 		data_mean_difference_std.append(temp1std+temp2std)
			# data_mean_difference=np.array(data_mean_difference)
			# data_mean_difference_std=np.array(data_mean_difference_std)
			# guess=[1]
			# base_counts_correction,temp2=curve_fit(costant,np.linspace(1,len(data_mean_difference),len(data_mean_difference)),data_mean_difference, sigma=data_mean_difference_std,p0=guess, maxfev=100000000)
			# print('background correction = '+str(int(base_counts_correction*1000)/1000))

			datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]]+base_counts_correction,params,errparams)
			datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
			datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
			basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)
		elif (pathfiles in coleval.flatten_full([laser11[8:]])):
			# Added 19/08/2018 to use the best data available for building the reference
			type='_reference.npy'
			filenames_basecounts=coleval.all_file_names(laser11[7],type)[0]
			basecounts=np.load(os.path.join(laser11[7],filenames_basecounts))
			filenames=filenames[:-4]+'_reference'+filenames[-4:]

			base_counts_correction = track_change_from_baseframe(data, poscentred, basecounts)

			# data_mean_difference=[]
			# data_mean_difference_std=[]
			# # poscentred=[[70,70],[160,133],[250,200]]
			# # poscentred=[[60,12],[170,12],[290,12]]
			# for pos in poscentred:
			# 	for a in [5,10]:
			# 		temp1=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
			# 		temp1std=np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
			# 		temp2=np.mean(basecounts[pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))
			# 		temp2std=0
			# 		data_mean_difference.append(temp1-temp2)
			# 		data_mean_difference_std.append(temp1std+temp2std)
			# data_mean_difference=np.array(data_mean_difference)
			# data_mean_difference_std=np.array(data_mean_difference_std)
			# guess=[1]
			# base_counts_correction,temp2=curve_fit(costant,np.linspace(1,len(data_mean_difference),len(data_mean_difference)),data_mean_difference, sigma=data_mean_difference_std,p0=guess, maxfev=100000000)
			# print('background correction = '+str(int(base_counts_correction*1000)/1000))

			datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]]+base_counts_correction,params,errparams)
			datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
			datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
			basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)

		elif (pathfiles in coleval.flatten_full([laser15,laser16,laser21,laser23])):
			type='_reference.npy'
			filenames=coleval.all_file_names(pathfiles,type)[0]
			basecounts_short=np.load(os.path.join(pathfiles,filenames))
			basecounts=np.multiply(6000,np.ones((256,320)))
			basecounts[64:128,:]=basecounts_short
			print('Used the reference in '+filenames)
			datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]],params,errparams)
			datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
			datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
			basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)
		elif (pathfiles in coleval.flatten_full([laser24,laser26,laser27,laser28,laser29,laser31])):
			type='_reference.npy'
			filenames=coleval.all_file_names(pathfiles,type)[0]
			basecounts_short=np.load(os.path.join(pathfiles,filenames))
			basecounts=np.multiply(3000,np.ones((256,320)))
			basecounts[64:128,128:]=basecounts_short
			print('Used the reference in '+filenames)
			datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]],params,errparams)
			datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
			datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
			basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)
		else:
			type='_reference.npy'
			filenames=coleval.all_file_names(pathfiles,type)[0]
			basecounts=np.load(os.path.join(pathfiles,filenames))
			print('Used the reference in '+filenames)
			datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]],params,errparams)
			datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
			datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
			basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)





		datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)


		# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.close()
		# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=5900,extvmax=6020)
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.close()

		datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
		datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
		errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
		errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]

		np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_temperature_temperror_basetemp'),datatempcrop=datatempcrop,errdatatempcrop=errdatatempcrop,basetemp=basetemp)




	# POST PROCESS

	# LOAD GEOMETRY

	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=vacuum2[20]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='_stat.npy'
	# type='csv'

	# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))[0]

	# frame=np.shape(data)[1]//2
	# base_frame_range=np.shape(data)[1]//2
	testrot=data
	rotangle=-1.5 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[160,133]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=240
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)


	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)


	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	if False:
		foilemissivityscaled_orig=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
		foilthicknessscaled_orig=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
		conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
		reciprdiffusivityscaled_orig=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
		flat_properties=False
		diffusivity_mult_range=[0.1,0.2,0.4,0.6,0.8,1]
		emissivity_mult_range=[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6]
		thickness_mult_range=[1,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]
	else:
		foilemissivityscaled_orig=1*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
		foilthicknessscaled_orig=(2.5/1000000)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
		conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
		reciprdiffusivityscaled_orig=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
		flat_properties=True
		diffusivity_mult_range=[0.1,0.2,0.4,0.6,0.8,1]
		emissivity_mult_range=[0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2]
		thickness_mult_range=[0.4,0.6,0.8,1,1.2,1.4,1.6,1.8]

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=laser10[13]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='_foil_temperature_temperror_basetemp.npz'
	# type='csv'

	maxpower=[]
	files=[laser20[0]]
	control_frequency=[freqlaser20[0]]
	control_dutycycle=[dutylaser20[0]]

	files=coleval.flatten_full(files)
	control_frequency_flat=coleval.flatten_full(control_frequency)
	control_ducycycle_flat=coleval.flatten_full(control_dutycycle)

	for index,pathfiles in enumerate(files):

		framerate, data_to_trash, type_of_experiment, poscentred_to_trash, pathparams_to_trash, inttime_to_trash = laser_shot_library(pathfiles)
		#
		# if (pathfiles in laser12):
		# 	framerate=994
		# 	# datashort=np.load(os.path.join(pathfiles,filenames))
		# 	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	# data[:,:,64:96,:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	# poscentred=[[15,80],[40,75],[80,85]]
		# elif (pathfiles in [vacuum1[1],vacuum1[3]]):
		# 	framerate=994
		# 	# datashort=np.load(os.path.join(pathfiles,filenames))
		# 	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	# data[:,:,64:96,:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	# poscentred=[[15,80],[40,75],[80,85]]
		# elif (pathfiles in laser11):
		# 	framerate=383
		# 	# data=np.load(os.path.join(pathfiles,filenames))
		# 	type_of_experiment='partially defocused'
		# 	# poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
		# elif (pathfiles in coleval.flatten_full([laser15,laser16])):
		# 	framerate=994
		# 	# datashort=np.load(os.path.join(pathfiles,filenames))
		# 	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	# data[:,:,64:96,:]=datashort
		# 	type_of_experiment='low duty cycle partially defocused'
		# 	# poscentred=[[15,80],[40,75],[80,85]]
		# elif (pathfiles in coleval.flatten_full([laser21,laser23])):
		# 	framerate=994
		# 	# datashort=np.load(os.path.join(pathfiles,filenames))
		# 	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	# data[:,:,64:96,:]=datashort
		# 	type_of_experiment='fully defocused'
		# 	# poscentred=[[15,80],[40,75],[80,85]]
		# elif (pathfiles in coleval.flatten_full([laser24,laser26,laser27,laser28,laser29,laser31])):
		# 	framerate=1976
		# 	# datashort=np.load(os.path.join(pathfiles,filenames))
		# 	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# 	# data[:,:,64:96,:]=datashort
		# 	type_of_experiment='fully defocused'
		# 	# poscentred=[[15,80],[40,75],[80,85]]
		# else:
		# 	framerate=383
		# 	# data=np.load(os.path.join(pathfiles,filenames))
		# 	type_of_experiment='focused'
		# 	# poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]



		filenames=coleval.all_file_names(pathfiles,type)[0]
		datatempcrop=np.load(os.path.join(pathfiles,filenames))['datatempcrop']
		errdatatempcrop=np.load(os.path.join(pathfiles,filenames))['errdatatempcrop']
		basetemp=np.load(os.path.join(pathfiles,filenames))['basetemp']


		dt=1/framerate
		dx=foilhorizw/(foilhorizwpixel-1)
		dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
		d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
		d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
		d2Tdxy=np.add(d2Tdx2,d2Tdy2)
		negd2Tdxy=np.multiply(-1,d2Tdxy)
		T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
		T04=np.power(np.add(zeroC,basetemp),4)

		maxpower4=[]
		for diffusivity_mult in diffusivity_mult_range:
			maxpower3=[]
			for emissivity_mult in emissivity_mult_range:
				maxpower2=[]
				for thickness_mult in thickness_mult_range:
					reciprdiffusivityscaled=np.multiply(1/diffusivity_mult,reciprdiffusivityscaled_orig)
					foilemissivityscaled=np.multiply(emissivity_mult,foilemissivityscaled_orig)
					foilthicknessscaled=np.multiply(thickness_mult,foilthicknessscaled_orig)

					BBrad=[]
					diffusion=[]
					timevariation=[]
					ktf=np.multiply(conductivityscaled,foilthicknessscaled)
					for i in range(len(datatempcrop[:,0,0,0])):
						BBrad.append([])
						diffusion.append([])
						timevariation.append([])
						for j in range(len(datatempcrop[0,1:-1,0,0])):
							BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
							BBrad[i].append(BBradtemp)
							diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
							diffusion[i].append(diffusiontemp)
							timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
							timevariation[i].append(timevariationtemp)
					BBrad=np.array(BBrad)
					diffusion=np.array(diffusion)
					timevariation=np.array(timevariation)

					BBradnoback=np.add(BBrad,0)
					diffusionnoback=np.add(diffusion,0)
					timevariationnoback=np.add(timevariation,0)

					powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)


					totalpower=np.multiply(np.sum(powernoback[0],axis=(-1,-2)),dx**2)
					totalBBrad=np.multiply(np.sum(BBradnoback[0],axis=(-1,-2)),dx**2)
					totaldiffusion=np.multiply(np.sum(diffusionnoback[0],axis=(-1,-2)),dx**2)
					totaltimevariation=np.multiply(np.sum(timevariationnoback[0],axis=(-1,-2)),dx**2)
					#
					# plt.plot(totalpower,label='totalpower')
					# plt.plot(totalBBrad,label='totalBBrad')
					# plt.plot(totaldiffusion,label='totaldiffusion')
					# plt.plot(totaltimevariation,label='totaltimevariation')
					# plt.title('Sum of the power over all the foil')
					# plt.xlabel('Frame number')
					# plt.ylabel('Power sum [W]')
					# plt.legend()
					# plt.legend(loc='best')
					# # plt.savefig(os.path.join(pathfiles,filenames[:-4]+'power_sum_all_foil'+'.eps'))
					# plt.grid()
					# plt.show()

					maxpower1=[]
					for addition in [-1,0,1,2]:
						# if ((pathfiles in laser11) or (pathfiles in laser12)):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_temperature'),powernoback=powernoback)
						# 	powerzoom=5+addition
						# 	laserspot=[41,172]
						# elif (pathfiles in laser13):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=3+addition
						# 	laserspot=[26,196]
						# elif (pathfiles in laser14):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=3+addition
						# 	laserspot=[27,156]
						# elif (pathfiles in laser15):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=2+addition
						# 	laserspot=[43,188]
						# elif (pathfiles in laser16):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=3+addition
						# 	laserspot=[41,186]
						# elif (pathfiles in laser17):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=2+addition
						# 	laserspot=[21,216]
						# elif (pathfiles in laser18):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=1+addition
						# 	laserspot=[22,159]
						# elif (pathfiles in laser19):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=1+addition
						# 	laserspot=[54,187]
						# elif (pathfiles in laser20):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=6+addition
						# 	laserspot=[42,180]
						# elif (pathfiles in laser21):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=5+addition
						# 	laserspot=[42,180]
						# elif (pathfiles in coleval.flatten_full([laser22,laser23,laser24,laser25,lase26,laser27])):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=1+addition
						# 	laserspot=[42,189]
						# elif (pathfiles in coleval.flatten_full([laser28,laser29])):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=3+addition
						# 	laserspot=[43,188]
						# elif (pathfiles in coleval.flatten_full([laser30,laser31])):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=6+addition
						# 	laserspot=[42,190]
						# elif (pathfiles in coleval.flatten_full([laser32])):
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=2+addition
						# 	laserspot=[49,214]
						# else:
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total-BB-diff-time'),powernoback=powernoback,BBradnoback=BBradnoback,diffusionnoback=diffusionnoback,timevariationnoback=timevariationnoback)
						# 	# np.savez_compressed(os.path.join(pathfiles,filenames[:-4]+'_foil_power_density_total'),powernoback=powernoback)
						# 	powerzoom=3+addition
						# 	laserspot=[41,172]

						powerzoom, laserspot = laser_shot_location_library(pathfiles)


						# powerzoom=2
						# laserspot=[41,172]
						maxdw=laserspot[0]-powerzoom
						maxup=laserspot[0]+powerzoom+1
						maxlx=laserspot[1]-powerzoom
						maxdx=laserspot[1]+powerzoom+1


						powernobackcrop=powernoback[:,:,maxdw:maxup,maxlx:maxdx]
						BBradnobackcrop=BBradnoback[:,:,maxdw:maxup,maxlx:maxdx]
						diffusionnobackcrop=diffusionnoback[:,:,maxdw:maxup,maxlx:maxdx]
						timevariationnobackcrop=timevariationnoback[:,:,maxdw:maxup,maxlx:maxdx]

						totalpowercrop=np.multiply(np.sum(powernobackcrop[0],axis=(-1,-2)),dx**2)
						totalBBradcrop=np.multiply(np.sum(BBradnobackcrop[0],axis=(-1,-2)),dx**2)
						totaldiffusioncrop=np.multiply(np.sum(diffusionnobackcrop[0],axis=(-1,-2)),dx**2)
						totaltimevariationcrop=np.multiply(np.sum(timevariationnobackcrop[0],axis=(-1,-2)),dx**2)


						# 2018/08/16 modified because i can make a better use of the mobile average, knowing the pulse frequency and duty cycle
						flat_time_high=control_ducycycle_flat[index]/control_frequency_flat[index]
						flat_frames_high=int(flat_time_high*framerate)
						flat_time_low=(1-control_ducycycle_flat[index])/control_frequency_flat[index]
						flat_frames_low=int(flat_time_low*framerate)

						if flat_frames_high>2:
							flat_frames_high=flat_frames_high-1
							moving_average_high=np.convolve(totalpowercrop, np.ones((flat_frames_high,))/flat_frames_high, mode='full')
							sample_high=[]
							for i in range(1,len(moving_average_high)-2):
								border_left=np.array([i+1-int(flat_frames_high//1.1),0])
								border_left=(border_left[border_left>=0]).max()
								border_right=np.array([i+1+int(flat_frames_low//1.1),len(moving_average_high)-1])
								border_right=(border_right[border_right<=len(moving_average_high)]).min()
								if ((moving_average_high[i+1]>(moving_average_high[border_left:i+1]).max()) & (moving_average_high[i+1]>=(moving_average_high[i+1:border_right]).max())):
									sample_high.append(moving_average_high[i+1])
							sample_high=np.array(sample_high)

							moving_average_low=np.convolve(totalpowercrop, np.ones((flat_frames_low,))/flat_frames_low, mode='full')
							sample_low = moving_average_low[1:-1][((moving_average_low[1:-1]-moving_average_low[0:-2]<0) & (moving_average_low[1:-1]-moving_average_low[2:]<0))]

							sample_high_max=sample_high.max()
							sample_low_min=sample_low.min()
							sample_high = sample_high[(sample_high>(sample_high_max-sample_low_min)/2)]
							sample_low = sample_low[(sample_low<(sample_high_max-sample_low_min)/2)]

							if len(sample_high)>3:
								test4=np.mean(sample_high)
								test3=np.std(sample_high)
								sample=sample_high
							else:
								max_power_index=(np.ndarray.tolist(moving_average_high)).index(sample_high.max())
								test4=moving_average_high[max_power_index]
								test3=np.std(totalpowercrop[max(0,max_power_index-flat_frames_high):min(max_power_index+1,len(totalpowercrop)-1)])
								sample=np.array([test4-test3,test4+test3])
						else:
							# powermean=np.mean(totalpowercrop)
							sample = totalpowercrop[1:-1][((totalpowercrop[1:-1]-totalpowercrop[0:-2]>0) & (totalpowercrop[1:-1]-totalpowercrop[2:]>0))]
							sample = sample[(sample>sample.max()/2)]
							test4=np.mean(sample)
							test3=np.std(sample)
							# guess=[sample.max()]
										# 	# test4,test3=curve_fit(costant,np.linspace(1,len(sample),len(sample)),sample, p0=guess, maxfev=100000000)



						# this section was deleted 2018/08/16 because I noticed the ensamble average should be good if i do it for monimum 3 frames
						# if (pathfiles in coleval.flatten_full([laser12,laser15,laser16])):
						# 	# powermean=np.mean(totalpowercrop)
						# 	sample = totalpowercrop[1:-1][((totalpowercrop[1:-1]-totalpowercrop[0:-2]>0) & (totalpowercrop[1:-1]-totalpowercrop[2:]>0))]
						# 	sample = sample[(sample>totalpowercrop.max()/2)]
						# 	test4=np.mean(sample)
						# 	test3=np.std(sample)
						# 	# guess=[sample.max()]
						# 	# test4,test3=curve_fit(costant,np.linspace(1,len(sample),len(sample)),sample, p0=guess, maxfev=100000000)
						#
						# elif (pathfiles in coleval.flatten_full([laser10[:7],laser11[:7],laser13,laser14,laser17[:6],laser18[:6],laser19[:6]])):
						#
						# 	moving_average=np.convolve(totalpowercrop, np.ones((flat_frames,))/flat_frames, mode='full')
						# 	max_power_index=moving_average.argmax()
						# 	test4=moving_average[max_power_index]
						# 	test3=np.std(totalpowercrop[max_power_index-flat_frames:max_power_index+1])
						# 	sample=np.array([test4-test3,test4+test3])
						#
						# else:
						# 	coleval.find_nearest_index()
						# 	powermean=np.mean(totalpowercrop)
						# 	sample = totalpowercrop[(totalpowercrop>powermean)]
						# 	test4=np.mean(sample)
						# 	test3=np.std(sample)
						# 	# guess=[sample.max()]
						# 	# test4,test3=curve_fit(costant,np.linspace(1,len(sample),len(sample)),sample, p0=guess, maxfev=100000000)


						# test=[]
						# for i in range(len(totalpowercrop)-4):
						# 	test.append(totalpowercrop[i:i+4])
						# test2=np.sort(test)
						# test3=test2[:,1]
						# test4=max(test3)

						# test2=np.sort(totalpowercrop)
						# test4=test2[-30]

						x_axis=(1/framerate )*np.linspace(1,len(totalpowercrop),len(totalpowercrop))
						plt.figure(figsize=(20,10))
						plt.plot(x_axis,totalpowercrop,label='totalpowercrop')
						plt.plot(x_axis,totalBBradcrop,label='totalBBradcrop')
						plt.plot(x_axis,totaldiffusioncrop,label='totaldiffusioncrop')
						plt.plot(x_axis,totaltimevariationcrop,label='totaltimevariationcrop')
						plt.plot(x_axis,costant(np.linspace(1,len(totalpowercrop),len(totalpowercrop)),*[test4]),label='sampled power')
						plt.plot(x_axis,costant(np.linspace(1,len(totalpowercrop),len(totalpowercrop)),*[sample.min()]),'k--',linewidth=0.5)
						plt.plot(x_axis,costant(np.linspace(1,len(totalpowercrop),len(totalpowercrop)),*[sample.max()]),'k--',label='used for sampling',linewidth=0.5)
						plt.title('Sum of the power over the area \n in '+str(laserspot)+' +/- '+str(powerzoom)+' max='+str(int(test4*100000)/100)+'mW, std='+str(int(test3*10000000)/10000)+'mW')
						plt.xlabel('Time [s]')
						plt.ylabel('Power sum [W]')
						plt.legend()
						plt.legend(loc='best')
						plt.grid()
						if not flat_properties:
							plt.savefig(os.path.join(pathfiles,filenames[:-len(type)]+'_diffusivity_x_'+str(diffusivity_mult)+'_-emissivity_x_'+str(emissivity_mult)+'_-thickness_x_'+str(thickness_mult)+'_-power_sum_zoom_window'+str(powerzoom)+'.eps'))
							plt.close()
						else:
							plt.savefig(os.path.join(pathfiles,filenames[:-len(type)]+'_flat_properties_diffusivity_x_'+str(diffusivity_mult)+'_-emissivity_x_'+str(emissivity_mult)+'_-thickness_x_'+str(thickness_mult)+'_-power_sum_zoom_window'+str(powerzoom)+'.eps'))
							plt.close()


						maxpower1.append([test4,test3])
					maxpower2.append(maxpower1)
				maxpower3.append(maxpower2)
			maxpower4.append(maxpower3)
		maxpower4=np.array(maxpower4)
		if flat_properties:
			np.savez_compressed(os.path.join(pathfiles,filenames[:-len(type)]+'_flat_properties_diffusivity_emissivity_thickness_window_scan'),maxpower4=maxpower4)
		else:
			np.savez_compressed(os.path.join(pathfiles,filenames[:-len(type)]+'_diffusivity_emissivity_thickness_window_scan'),maxpower4=maxpower4)




	# POST PROCESSING 2
	# I WANT TO FIND THE RIGHT PROPERTIES OF THE FOIL

	type='emissivity_thickness_window_scan.npz'

	files_measure=[laser10[:6],laser11[:6],laser13[:6],laser14[:6]]
	scan_collection=[]
	for files in files_measure:
		files=coleval.flatten_full(files)
		scan_collection_temp=[]
		for pathfiles in files:
			filenames=coleval.all_file_names(pathfiles,type)[0]
			maxpower=np.load(os.path.join(pathfiles,filenames))['maxpower4']
			scan_collection.append(maxpower)
		# scan_collection_temp=np.array(scan_collection_temp)
		# scan_collection.append(scan_collection_temp)
	scan_collection=np.array(scan_collection)


	comparepower=np.interp(voltlaser10[:6],reflaserfvoltage1[:6],reflaserpower1[:6])
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,0,1],'y*',label='emissivity x 0.8, thickness x 0.8')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,1,1],'r*',label='emissivity x 0.8, thickness x 1')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,3,1],'m*',label='emissivity x 0.8, thickness x 1.4')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,5,1],'c*',label='emissivity x 0.8, thickness x 1.8')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,0,1],'yx',label='emissivity x 1, thickness x 0.8')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,1,1],'rx',label='emissivity x 1, thickness x 1')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,3,1],'mx',label='emissivity x 1, thickness x 1.4')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,5,1],'cx',label='emissivity x 1, thickness x 1.8')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,0,1],'y+',label='emissivity x 1.2, thickness x 0.8')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,1,1],'r+',label='emissivity x 1.2, thickness x 1')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,3,1],'m+',label='emissivity x 1.2, thickness x 1.4')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,5,1],'c+',label='emissivity x 1.2, thickness x 1.8')
	plt.plot(comparepower,comparepower,'k--',label='expected power')
	plt.title('Power scan')
	plt.xlabel('Expected power [W]')
	plt.ylabel('Maximum of power sum from foil measurement [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.grid()
	plt.show()


	comparepower=np.interp(voltlaser10[:6],reflaserfvoltage1[:6],reflaserpower1[:6])
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,0,1],'y*',label='emissivity x 0.8, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,1,1],'r*',label='emissivity x 0.8, thickness x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,3,1],'m*',label='emissivity x 0.8, thickness x 1.4')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,5,1],'r*',label='emissivity x 0.8, thickness x 1.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,0,1],'yx',label='emissivity x 1, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,1,1],'rx',label='emissivity x 1, thickness x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,3,1],'mx',label='emissivity x 1, thickness x 1.4')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,5,1],'cx',label='emissivity x 1, thickness x 1.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,0,1],'y+',label='emissivity x 1.2, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,1,1],'r+',label='emissivity x 1.2, thickness x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,3,1],'m+',label='emissivity x 1.2, thickness x 1.4')
	plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,5,1],'m+',label='emissivity x 1.2, thickness x 1.8')
	plt.plot(comparepower,comparepower,'k--',label='expected power')
	plt.title('Power scan')
	plt.xlabel('Expected power [W]')
	plt.ylabel('Maximum of power sum from foil measurement [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.grid()
	plt.show()



	type='emissivity_thickness_window_scan.npz'

	files_measure=[laser10[7:],laser11[7:]]
	scan_collection=[]
	for files in files_measure:
		files=coleval.flatten_full(files)
		scan_collection_temp=[]
		for pathfiles in files:
			filenames=coleval.all_file_names(pathfiles,type)[0]
			maxpower=np.load(os.path.join(pathfiles,filenames))['maxpower4']
			scan_collection.append(maxpower)
		# scan_collection_temp=np.array(scan_collection_temp)
		# scan_collection.append(scan_collection_temp)
	scan_collection=np.array(scan_collection)

	# comparepower=np.interp(voltlaser10[7:],reflaserfvoltage1,reflaserpower1)
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,0,1],'y*',label='emissivity x 0.8, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,1,1],'r*',label='emissivity x 0.8, thickness x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,0,3,1],'m*',label='emissivity x 0.8, thickness x 1.4')
	plt.plot(coleval.flatten_full([freqlaser10[7:],freqlaser10[7:]]),scan_collection[:,0,2,5,1],'r*',label='diffusivity x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,0,1],'yx',label='emissivity x 1, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,1,1],'rx',label='emissivity x 1, thickness x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,2,3,1],'mx',label='emissivity x 1, thickness x 1.4')
	plt.plot(coleval.flatten_full([freqlaser10[7:],freqlaser10[7:]]),scan_collection[:,2,2,5,1],'cx',label='diffusivity x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,0,1],'y+',label='emissivity x 1.2, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,1,1],'r+',label='emissivity x 1.2, thickness x 1')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),scan_collection[:,2,4,3,1],'m+',label='emissivity x 1.2, thickness x 1.4')
	plt.plot(coleval.flatten_full([freqlaser10[7:],freqlaser10[7:]]),scan_collection[:,4,2,5,1],'m+',label='diffusivity x 1.2')
	plt.plot(freqlaser10[7:],comparepower,'k--',label='expected power')
	plt.title('Frequency scan')
	plt.xlabel('frequency [Hz]')
	plt.ylabel('Maximum of power sum from foil measurement [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.grid()
	plt.show()

	# SECOND ITERATION

	type='emissivity_thickness_window_scan.npz'

	files_measure=[laser10[:6],laser11[:6],laser13[:6],laser14[:6]]
	scan_collection=[]
	for files in files_measure:
		files=coleval.flatten_full(files)
		scan_collection_temp=[]
		for pathfiles in files:
			filenames=coleval.all_file_names(pathfiles,type)[0]
			maxpower=np.load(os.path.join(pathfiles,filenames))['maxpower4']
			scan_collection.append(maxpower)
		# scan_collection_temp=np.array(scan_collection_temp)
		# scan_collection.append(scan_collection_temp)
	scan_collection=np.array(scan_collection)
	comparepower=np.interp(voltlaser10[:6],reflaserfvoltage1[:6],reflaserpower1[:6])

	test=0
	diff=0
	for diffusivity_mult in [0.8,0.9,1,1.1,1.2]:
		emi=0
		for emissivity_mult in [0.8,0.9,1,1.1,1.2]:
			thick=0
			for thickness_mult in [0.8,1,1.2,1.4,1.6,1.8]:
				add=0
				for addition in [-1,0,1]:
					array=scan_collection[:,diff,emi,thick,add]
					x=coleval.flatten_full([comparepower,comparepower,comparepower,comparepower])
					array=np.array([array for _,array in sorted(zip(x,array))])
					x=np.sort(x)
					test1=coleval.rsquared(x,array)
					if test1>test:
						print('we have a winner: R2='+str(test1)+' diffusivity x '+str(diffusivity_mult)+' emissivity x '+str(emissivity_mult)+' thickness x '+str(thickness_mult)+' zoom window addition ='+str(addition))
						test=test1
						winner=array
					add+=1
				thick+=1
			emi+=1
		diff+=1

	plt.plot(x,winner,'m*',label='diffusivity x 1.2, emissivity x 0.8, thickness x 0.8')
	# plt.plot(coleval.flatten_full([comparepower,comparepower,comparepower,comparepower]),winner,'r*',label='diffusivity x 1.2, emissivity x 0.8, thickness x 0.8')
	plt.plot(comparepower,comparepower,'k--',label='expected power')
	plt.title('Power scan')
	plt.xlabel('Expected power [W]')
	plt.ylabel('Maximum of power sum from foil measurement [W]')
	plt.legend()
	plt.legend(loc='best')
	plt.grid()
	plt.show()




	# POST PROCESSING 3
	# 14/08/2018 I WANT TO FIND THE RIGHT PROPERTIES OF THE FOIL, BUT I WANT TO DO IT IN A MORE
	# RIGOROUS WAY RATHER THAN BY EYE

	type='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz'
	# type='_reference_diffusivity_emissivity_thickness_window_scan.npz'


	# files_measure=[laser15,laser16,laser17,laser18,laser19]
	# control_voltage=[voltlaser15,voltlaser16,voltlaser17,voltlaser18,voltlaser19]
	# files_measure=[laser10[:7],laser11[:7],laser17[:6],laser18[:6],laser19[:6]]
	# control_voltage=[voltlaser17[:7],voltlaser11[:7],voltlaser17[:6],voltlaser18[:6],voltlaser19[:6]]
	# files_measure=[laser17[:6],laser18[:6],laser19[:6]]
	# control_voltage=[voltlaser17[:6],voltlaser18[:6],voltlaser19[:6]]
	all_files=laser17
	all_control_voltage=voltlaser17
	all_control_freq=freqlaser17
	last_power_scan=6
	last_freq_scan=len(freqlaser17)

	files_measure_1=[all_files[:last_power_scan]]
	control_voltage_1=[all_control_voltage[:last_power_scan]]
	control_freq_1=[all_control_freq[:last_power_scan]]
	files_measure_2=[all_files[last_power_scan:last_freq_scan]]
	control_voltage_2=[all_control_voltage[last_power_scan:last_freq_scan]]
	control_freq_2=[all_control_freq[last_power_scan:last_freq_scan]]

	if type!='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		diffusivity_mult_range=[0.1,0.2,0.4,0.6,0.8,1]
		emissivity_mult_range=[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6]
		thickness_mult_range=[1,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]
	else:
		diffusivity_mult_range=[0.1,0.2,0.4,0.6,0.8,1]
		emissivity_mult_range=[0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2]
		thickness_mult_range=[0.4,0.6,0.8,1,1.2,1.4,1.6,1.8]

		diffusivity_index_INITIAL=5
		emissivity_index_INITIAL=0

	# DATA COLLECTION

	files_measure=files_measure_1
	control_voltage=control_voltage_1
	control_voltage_flat=coleval.flatten_full(control_voltage)
	comparepower_flat_1=np.interp(control_voltage_flat,reflaserfvoltage1,reflaserpower1)
	# power_weigth=np.divide(0.000001,comparepower_flat)
	scan_collection=[]
	for files in files_measure:
		files=coleval.flatten_full(files)
		scan_collection_temp=[]
		for pathfiles in files:
			filenames=coleval.all_file_names(pathfiles,type)[0]
			maxpower=np.load(os.path.join(pathfiles,filenames))['maxpower4']
			scan_collection_temp.append(maxpower)
		scan_collection_temp=np.array(scan_collection_temp)
		window_scan_max=scan_collection_temp[:,:,:,:,:,0].argmax(axis=-1)
		window_scan_best=np.mean(coleval.flatten_full(window_scan_max))
		window_scan_index=int(round(window_scan_best+0.000000000000001))
		scan_collection_temp=scan_collection_temp[:,:,:,:,window_scan_index]
		print('The best window size for this set of experiments is '+str(window_scan_best))
		scan_collection.append(scan_collection_temp)
	scan_collection_1=coleval.flatten(np.array(scan_collection))

	files_measure=files_measure_2
	control_voltage=control_voltage_2
	control_voltage_flat=coleval.flatten_full(control_voltage)
	comparepower_flat_2=np.interp(control_voltage_flat,reflaserfvoltage1,reflaserpower1)
	# power_weigth=np.divide(0.000001,comparepower_flat)
	scan_collection=[]
	for files in files_measure:
		files=coleval.flatten_full(files)
		scan_collection_temp=[]
		for pathfiles in files:
			filenames=coleval.all_file_names(pathfiles,type)[0]
			maxpower=np.load(os.path.join(pathfiles,filenames))['maxpower4']
			scan_collection_temp.append(maxpower)
		scan_collection_temp=np.array(scan_collection_temp)
		window_scan_max=scan_collection_temp[:,:,:,:,:,0].argmax(axis=-1)
		window_scan_best=np.mean(coleval.flatten_full(window_scan_max))
		window_scan_index=int(round(window_scan_best+0.000000000000001))
		scan_collection_temp=scan_collection_temp[:,:,:,:,window_scan_index]
		print('The best window size for this set of experiments is '+str(window_scan_best))
		scan_collection.append(scan_collection_temp)
	scan_collection_2=coleval.flatten(np.array(scan_collection))

	# FIRST ITERATION with diffusivify fixed


	scan_collection=scan_collection_1
	comparepower_flat=comparepower_flat_1
	n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	fit_record1=[]
	for index_diff in [diffusivity_index_INITIAL]:
		fit_record2=[]
		for index_emis in [emissivity_index_INITIAL]:
			fit_record3=[]
			for index_thick in range(n_thick):
				scan=scan_collection[:,index_diff,index_emis,index_thick]
				power_scan=coleval.flatten_full(scan[:,0])
				power_scan_std=coleval.flatten_full(scan[:,1])
				temp1=np.mean(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))

				# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
				#
				# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
				# guess=[1]
				# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
				# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
				# if np.abs(temp1[1])>0.00005:
				# 	temp1[0]=0
				# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
				fit_record3.append(temp1)
			fit_record2.append(fit_record3)
		fit_record1.append(fit_record2)
	fit_record1=np.array(fit_record1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)


	thickness_index_FIRST=index_best[2]

	diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_INITIAL]
	emissivity_mult=(emissivity_mult_range)[emissivity_index_INITIAL]
	thickness_mult=(thickness_mult_range)[thickness_index_FIRST]
	if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used calibration data')
	if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used flat properties')

	print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	print('The mean relative quadratic error is '+str(fit_record1[0,0,thickness_index_FIRST]))

	plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index_INITIAL,emissivity_index_INITIAL,thickness_index_FIRST,0],fmt='o',yerr=scan_collection[:,diffusivity_index_INITIAL,emissivity_index_INITIAL,thickness_index_FIRST,1])
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	plt.plot(np.sort(comparepower_flat),np.sort(comparepower_flat),'--')
	# plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0]))
	plt.title('Power scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Expected power [W]')
	plt.ylabel('Measured power [W]')
	plt.show()

	# # FIRST ITERATION with diffusivify not fixed
	#
	# scan_collection=scan_collection_1
	# comparepower_flat=comparepower_flat_1
	# n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	# fit_record1=[]
	# for index_diff in range(n_diff):
	# 	fit_record2=[]
	# 	for index_emis in range(n_emis):
	# 		fit_record3=[]
	# 		for index_thick in range(n_thick):
	# 			scan=scan_collection[:,index_diff,index_emis,index_thick]
	# 			power_scan=coleval.flatten_full(scan[:,0])
	# 			power_scan_std=coleval.flatten_full(scan[:,1])
	# 			guess=[1]
	# 			temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
	#
	# 			# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
	# 			# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
	# 			#
	# 			# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
	# 			# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
	# 			# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
	# 			# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
	# 			# if np.abs(temp1[1])>0.00005:
	# 			# 	temp1[0]=0
	# 			# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
	# 			fit_record3.append(temp1)
	# 		fit_record2.append(fit_record3)
	# 	fit_record1.append(fit_record2)
	# fit_record1=np.array(fit_record1)
	# # index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)
	# diffusivity_index=index_best[0]
	# emissivity_index=index_best[1]
	# thickness_index=index_best[2]
	#
	# diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_INITIAL]
	# emissivity_mult=(emissivity_mult_range)[emissivity_index]
	# thickness_mult=(thickness_mult_range)[thickness_index]
	# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
	# 	print('I used calibration data')
	# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
	# 	print('I used flat properties')
	#
	# print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	# print('The sum of relative quadratic error is '+str(fit_record1[diffusivity_index,emissivity_index,thickness_index]))
	#
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],fmt='o',yerr=scan_collection[:,diffusivity_index,emissivity_index,thickness_index,1])
	# # plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	# plt.plot(np.sort(comparepower_flat),np.sort(comparepower_flat),'--')
	# # plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0]))
	# plt.show()

	# SECOND ITERATION FOR diffusivity_index

	scan_collection=scan_collection_2
	comparepower_flat=comparepower_flat_2
	n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	fit_record1=[]
	for index_diff in range(n_diff):
		fit_record2=[]
		for index_emis in [emissivity_index_INITIAL]:
			fit_record3=[]
			for index_thick in [thickness_index_FIRST]:
				scan=scan_collection[:,index_diff,index_emis,index_thick]
				power_scan=coleval.flatten_full(scan[:,0])
				power_scan_std=coleval.flatten_full(scan[:,1])
				guess=[1]
				temp1=np.mean(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))

				# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
				#
				# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
				# guess=[1]
				# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
				# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
				# if np.abs(temp1[1])>0.00005:
				# 	temp1[0]=0
				# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
				fit_record3.append(temp1)
			fit_record2.append(fit_record3)
		fit_record1.append(fit_record2)
	fit_record1=np.array(fit_record1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)

	diffusivity_index_SECOND=index_best[0]

	diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_SECOND]
	emissivity_mult=(emissivity_mult_range)[emissivity_index_INITIAL]
	thickness_mult=(thickness_mult_range)[thickness_index_FIRST]
	if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used calibration data')
	if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used flat properties')

	print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	print('The mean relative quadratic error is '+str(fit_record1[diffusivity_index_SECOND,0,0]))

	plt.errorbar(coleval.flatten_full(control_freq_2),scan_collection[:,diffusivity_index_SECOND,emissivity_index_INITIAL,thickness_index_FIRST,0],fmt='o',yerr=scan_collection[:,diffusivity_index_SECOND,emissivity_index_INITIAL,thickness_index_FIRST,1])
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	plt.plot(coleval.flatten_full(control_freq_2),comparepower_flat,'--')
	# plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0]))
	plt.title('Frequency scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Laser frequency [Hz]')
	plt.ylabel('Measured power [mW]')
	plt.show()

	# THIRD ITERATION FOR ESTABLISH emissivity_index AND thickness_index

	scan_collection=scan_collection_1
	comparepower_flat=comparepower_flat_1
	n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	fit_record1=[]
	for index_diff in [diffusivity_index_SECOND]:
		fit_record2=[]
		for index_emis in [emissivity_index_INITIAL]:
			fit_record3=[]
			for index_thick in range(n_thick):
				scan=scan_collection[:,index_diff,index_emis,index_thick]
				power_scan=coleval.flatten_full(scan[:,0])
				power_scan_std=coleval.flatten_full(scan[:,1])
				guess=[1]
				temp1=np.mean(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))

				# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
				#
				# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
				# guess=[1]
				# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
				# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
				# if np.abs(temp1[1])>0.00005:
				# 	temp1[0]=0
				# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
				fit_record3.append(temp1)
			fit_record2.append(fit_record3)
		fit_record1.append(fit_record2)
	fit_record1=np.array(fit_record1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)


	thickness_index_SECOND=index_best[2]

	diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_SECOND]
	emissivity_mult=(emissivity_mult_range)[emissivity_index_INITIAL]
	thickness_mult=(thickness_mult_range)[thickness_index_SECOND]
	if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used calibration data')
	if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used flat properties')

	print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	print('The mean relative quadratic error is '+str(fit_record1[0,0,thickness_index_SECOND]))

	plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index_SECOND,emissivity_index_INITIAL,thickness_index_SECOND,0],fmt='o',yerr=scan_collection[:,diffusivity_index_SECOND,emissivity_index_INITIAL,thickness_index_SECOND,1])
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	plt.plot(np.sort(comparepower_flat),np.sort(comparepower_flat),'--')
	# plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0]))
	plt.title('Power scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Expected power [W]')
	plt.ylabel('Measured power [W]')
	plt.show()

	# FOURTH ITERATION FOR diffusivity_index

	scan_collection=scan_collection_2
	comparepower_flat=comparepower_flat_2
	n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	fit_record1=[]
	for index_diff in range(n_diff):
		fit_record2=[]
		for index_emis in [emissivity_index_INITIAL]:
			fit_record3=[]
			for index_thick in [thickness_index_SECOND]:
				scan=scan_collection[:,index_diff,index_emis,index_thick]
				power_scan=coleval.flatten_full(scan[:,0])
				power_scan_std=coleval.flatten_full(scan[:,1])
				guess=[1]
				temp1=np.mean(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))

				# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
				#
				# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
				# guess=[1]
				# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
				# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
				# if np.abs(temp1[1])>0.00005:
				# 	temp1[0]=0
				# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
				fit_record3.append(temp1)
			fit_record2.append(fit_record3)
		fit_record1.append(fit_record2)
	fit_record1=np.array(fit_record1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)

	diffusivity_index_THIRD=index_best[0]

	diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_THIRD]
	emissivity_mult=(emissivity_mult_range)[emissivity_index_INITIAL]
	thickness_mult=(thickness_mult_range)[thickness_index_SECOND]
	if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used calibration data')
	if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used flat properties')

	print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	print('The mean relative quadratic error is '+str(fit_record1[diffusivity_index_SECOND,0,0]))

	plt.errorbar(coleval.flatten_full(control_freq_2),scan_collection[:,diffusivity_index_THIRD,emissivity_index_INITIAL,thickness_index_SECOND,0],fmt='o',yerr=scan_collection[:,diffusivity_index_SECOND,emissivity_index_INITIAL,thickness_index_SECOND,1])
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	plt.plot(coleval.flatten_full(control_freq_2),comparepower_flat,'--')
	# plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0])
	plt.title('Frequency scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Laser frequency [Hz]')
	plt.ylabel('Measured power [mW]')
	plt.show()

	# FIFTH ITERATION FOR ESTABLISH emissivity_index AND thickness_index

	scan_collection=scan_collection_1
	comparepower_flat=comparepower_flat_1
	n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	fit_record1=[]
	for index_diff in [diffusivity_index_THIRD]:
		fit_record2=[]
		for index_emis in [emissivity_index_INITIAL]:
			fit_record3=[]
			for index_thick in range(n_thick):
				scan=scan_collection[:,index_diff,index_emis,index_thick]
				power_scan=coleval.flatten_full(scan[:,0])
				power_scan_std=coleval.flatten_full(scan[:,1])
				guess=[1]
				temp1=np.mean(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))

				# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
				#
				# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
				# guess=[1]
				# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
				# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
				# if np.abs(temp1[1])>0.00005:
				# 	temp1[0]=0
				# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
				fit_record3.append(temp1)
			fit_record2.append(fit_record3)
		fit_record1.append(fit_record2)
	fit_record1=np.array(fit_record1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)


	thickness_index_THIRD=index_best[2]

	diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_THIRD]
	emissivity_mult=(emissivity_mult_range)[emissivity_index_INITIAL]
	thickness_mult=(thickness_mult_range)[thickness_index_THIRD]
	if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used calibration data')
	if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used flat properties')

	print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	print('The mean relative quadratic error is '+str(fit_record1[0,0,thickness_index_THIRD]))

	plt.errorbar(comparepower_flat_1,scan_collection_1[:,diffusivity_index_THIRD,emissivity_index_INITIAL,thickness_index_THIRD,0],fmt='o',yerr=scan_collection_1[:,diffusivity_index_THIRD,emissivity_index_INITIAL,thickness_index_THIRD,1])
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	plt.plot(np.sort(comparepower_flat_1),np.sort(comparepower_flat_1),'--')
	plt.title('Power scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Expected power [W]')
	plt.ylabel('Measured power [W]')
	# plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0]))

	plt.figure()
	plt.errorbar(coleval.flatten_full(control_freq_2),scan_collection_2[:,diffusivity_index_THIRD,emissivity_index_INITIAL,thickness_index_THIRD,0],fmt='o',yerr=scan_collection_2[:,diffusivity_index_THIRD,emissivity_index_INITIAL,thickness_index_THIRD,1])
	plt.plot(coleval.flatten_full(control_freq_2),comparepower_flat_2,'--')
	plt.title('Frequency scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Laser frequency [Hz]')
	plt.ylabel('Measured power [mW]')
	plt.show()

	# DEFOCUSED LASER EVALUATION

	type='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz'
	# type='_reference_diffusivity_emissivity_thickness_window_scan.npz'


	all_files=[laser33[1:5]]
	all_control_voltage=[voltlaser33[1:5]]
	all_control_freq=[freqlaser33[1:5]]

	# diffusivity_index_THIRD=1
	# thickness_index_THIRD=4

	files_measure_3=[all_files]
	control_voltage_3=[all_control_voltage]
	control_freq_3=[all_control_freq]

	if type!='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		diffusivity_mult_range=[0.1,0.2,0.4,0.6,0.8,1]
		emissivity_mult_range=[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6]
		thickness_mult_range=[1,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]
	else:
		diffusivity_mult_range=[0.1,0.2,0.4,0.6,0.8,1]
		emissivity_mult_range=[0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2]
		thickness_mult_range=[0.4,0.6,0.8,1,1.2,1.4,1.6,1.8]



	# DATA COLLECTION

	files_measure=files_measure_3
	control_voltage=control_voltage_3
	control_voltage_flat=coleval.flatten_full(control_voltage)
	comparepower_flat_3=np.multiply(0.9/4.14,np.interp(control_voltage_flat,reflaserfvoltage1,reflaserpower1))
	# power_weigth=np.divide(0.000001,comparepower_flat)
	scan_collection=[]
	for files in files_measure:
		files=coleval.flatten_full(files)
		scan_collection_temp=[]
		for pathfiles in files:
			filenames=coleval.all_file_names(pathfiles,type)[0]
			maxpower=np.load(os.path.join(pathfiles,filenames))['maxpower4']
			scan_collection_temp.append(maxpower)
		scan_collection_temp=np.array(scan_collection_temp)
		window_scan_max=scan_collection_temp[:,:,:,:,:,0].argmax(axis=-1)
		window_scan_best=np.mean(coleval.flatten_full(window_scan_max))
		window_scan_index=int(round(window_scan_best+0.000000000000001))
		scan_collection_temp=scan_collection_temp[:,:,:,:,window_scan_index]
		print('The best window size for this set of experiments is '+str(window_scan_best))
		scan_collection.append(scan_collection_temp)
	scan_collection_3=coleval.flatten(np.array(scan_collection))


	# LAST ITERATION FOR ESTABLISH emissivity_index

	scan_collection=scan_collection_3
	comparepower_flat=comparepower_flat_3
	n_diff,n_emis,n_thick,trash=np.shape(scan_collection)[1:]
	fit_record1=[]
	for index_diff in [diffusivity_index_THIRD]:
		fit_record2=[]
		for index_emis in range(n_emis):
			fit_record3=[]
			for index_thick in [thickness_index_THIRD]:
				scan=scan_collection[:,index_diff,index_emis,index_thick]
				power_scan=coleval.flatten_full(scan[:,0])
				power_scan_std=coleval.flatten_full(scan[:,1])
				guess=[1]
				temp1=np.mean(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))

				# if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),np.power(comparepower_flat,2)))
				#
				# if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
				# 	temp1=np.sum(np.divide(np.power(np.add(comparepower_flat,-power_scan),2),1))
				# guess=[1]
				# temp1,temp2=curve_fit(line_throug_zero, comparepower_flat, power_scan, p0=guess, sigma=power_weigth, maxfev=100000000)
				# temp1,temp2=fit1,errfit1=curve_fit(line, comparepower_flat, power_scan, p0=guess, sigma=power_scan_std, maxfev=100000000)
				# if np.abs(temp1[1])>0.00005:
				# 	temp1[0]=0
				# fit_record3.append([temp1,np.sqrt([temp2[0,0],temp2[1,1]])])
				fit_record3.append(temp1)
			fit_record2.append(fit_record3)
		fit_record1.append(fit_record2)
	fit_record1=np.array(fit_record1)
	# index_best=coleval.find_nearest_index(fit_record1[:,:,:,0,0],1)
	index_best=coleval.find_nearest_index(fit_record1[:,:,:],0)

	emissivity_index_FIRST=index_best[1]


	diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_THIRD]
	emissivity_mult=(emissivity_mult_range)[emissivity_index_FIRST]
	thickness_mult=(thickness_mult_range)[thickness_index_THIRD]
	if type=='_reference_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used calibration data')
	if type=='_reference_flat_properties_diffusivity_emissivity_thickness_window_scan.npz':
		print('I used flat properties')

	print('The best parameters are calibration values multiplied by: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	print('The mean relative quadratic error is '+str(fit_record1[0,emissivity_index_FIRST,0]))

	plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index_THIRD,emissivity_index_FIRST,thickness_index_THIRD,0],fmt='o',yerr=scan_collection[:,diffusivity_index_THIRD,emissivity_index_FIRST,thickness_index_THIRD,1])
	plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index_THIRD,3,thickness_index_THIRD,0],fmt='o',yerr=scan_collection[:,diffusivity_index_THIRD,3,thickness_index_THIRD,1])
	plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index_THIRD,6,thickness_index_THIRD,0],fmt='o',yerr=scan_collection[:,diffusivity_index_THIRD,6,thickness_index_THIRD,1])
	# plt.errorbar(comparepower_flat,scan_collection[:,diffusivity_index,emissivity_index,thickness_index,0],yerr=power_weigth)
	plt.plot(np.sort(comparepower_flat),np.sort(comparepower_flat),'--')
	# plt.plot(np.sort(comparepower_flat),line_throug_zero(np.sort(comparepower_flat),*fit_record1[diffusivity_index,emissivity_index,thickness_index,0]))
	plt.title('Power scan with: diffusivity '+str(diffusivity_mult)+' , emissivity '+str(emissivity_mult)+' , thickness '+str(thickness_mult))
	plt.xlabel('Expected power [W]')
	plt.ylabel('Measured power [W]')
	plt.show()



	# # LINES TO CHECK THE FREQUENCY SCAN
	# diffusivity_mult=(diffusivity_mult_range)[diffusivity_index_THIRD]
	# thickness_mult=(thickness_mult_range)[thickness_index_THIRD]
	#
	# plt.figure()
	# for emissivity_SCAN in [2,4,6]:
	# 	plt.errorbar(coleval.flatten_full(control_freq_3),scan_collection_3[:,diffusivity_index_THIRD,emissivity_SCAN,thickness_index_THIRD,0],fmt='o',yerr=scan_collection_3[:,diffusivity_index_THIRD,emissivity_SCAN,thickness_index_THIRD,1],label='Emissivity '+str(emissivity_mult_range[emissivity_SCAN]))
	# plt.plot(coleval.flatten_full(control_freq_3),comparepower_flat_3,'--')
	# plt.title('Frequency scan with: diffusivity '+str(diffusivity_mult)+' , thickness '+str(thickness_mult))
	# plt.xlabel('Laser frequency [Hz]')
	# plt.ylabel('Measured power [mW]')
	# plt.legend(loc='best')
	# plt.grid()
	# plt.show()







	# I WANT TO SEE WHICH POINTS I'M EXAMINING

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=vacuum2[19]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='_stat.npy'
	# type='csv'

	# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]

	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:128,:]=datashort
	data=np.load(os.path.join(pathfiles,filenames))[0]
	# datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)


	# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# # ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=5900,extvmax=6020)
	# # ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# plt.show()


	# frame=500
	plt.figure()
	plt.title('Reference frame of '+pathfiles)
	plt.imshow(data,'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	# plt.figure()
	# plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	# plt.imshow(datatemp[0,frame],'rainbow',origin='lower')
	# plt.colorbar().set_label('Temp [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.figure()
	# plt.title('Frame '+str(frame)+' in '+pathfiles+' and coefficients \n'+fullpathparams,size=8)
	# plt.imshow(errdatatemp[0,frame],'rainbow',origin='lower')
	# plt.colorbar().set_label('Temp error [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	plt.show()


	testorig=data
	testrot=testorig
	rotangle=-1.5 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[160,133]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=240
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype(int)
	foilyint=(np.rint(foily)).astype(int)
	plt.figure()
	plt.imshow(testrot,'rainbow',origin='lower')
	# plt.imshow(testrot,'rainbow',vmin=26., vmax=27.,origin='lower')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.grid()
	plt.show()

	# plt.figure()
	# plt.title('Sample frame '+str(frame)+' in '+pathfiles+'\n foil center '+str(foilcenter)+'pixel, foil rot '+str(rotangle)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	# plt.imshow(testrot,'rainbow',origin='lower')
	# plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	# plt.colorbar().set_label('Temp [°C]')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=10)
	# plt.plot(foilx,foily,'k')
	# plt.show()


	# plt.figure()
	# plt.title('Reference frame of '+pathfiles+'\n foil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	# plt.imshow(testrot,'rainbow',origin='lower')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.clim(vmin=np.min(testorig), vmax=np.max(testorig))
	# plt.colorbar().set_label('Temp [°C]')
	# plt.plot(foilxint,foilyint,'r')
	# plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	# plt.figure()
	# testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	# plt.imshow(testrotback,'rainbow',origin='lower')
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.clim(vmin=np.min(testorig), vmax=np.max(testorig)) #this set the color limits
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype(int)
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype(int)
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype(int)
	# plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	# plt.plot(foilxrot,foilyrot,'r')
	# plt.title('Reference frame of '+pathfiles+'\n foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	# plt.colorbar().set_label('Temp [°C]')
	# plt.show()

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	datatemprot=rotate(data,foilrotdeg,axes=(-1,-2))
	datatempcrop=datatemprot[foildw+1:foilup-1,foillx+1:foilrx-1]
	# errdatatemprot=rotate(errdatatemp,foilrotdeg,axes=(-1,-2))
	# errdatatempcrop=errdatatemprot[:,:,foildw:foilup,foillx:foilrx]
	plt.figure()
	plt.imshow(datatempcrop,'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.title('Only foil reference frame of '+pathfiles+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')

	plt.plot(172,41,'+',mew=2,markersize=20,label='laser10 ps+fs focused, FLAT=x0.4, x0.4, x1, laser11 ps+fs part defocused, FLAT=x0.4, x0.8, x1')
	plt.plot(196,26,'x',mew=2,markersize=10,label='laser13 ps focused')
	plt.plot(156,27,'x',mew=2,markersize=10,label='laser14 ps focused')
	plt.plot(188,43,'x',mew=2,markersize=10,label='laser15 ELMs focused')
	plt.plot(186,41,'x',mew=2,markersize=10,label='laser16 ELMs part defocused')
	plt.plot(216,21,'+',mew=2,markersize=20,label='laser17 ps+fs focused, FLAT=x0.2, x0.6, x1')
	plt.plot(159,22,'+',mew=2,markersize=20,label='laser18 ps+fs focused, FLAT=x0.4, x1.2, x0.8')
	plt.plot(187,54,'+',mew=2,markersize=20,label='laser19 ps+fs focused, FLAT=x0.4, x1.2, x0.8')
	plt.plot(180,42,'x',mew=2,markersize=10,label='laser20 fs fully defocused, laser21 fs fully defocused (high FR)')
	plt.legend(loc='best')
	# plt.show()

	# ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_foil_temp'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(datatempcrop,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=26.5,extvmax=27.2)
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_foil_temp_limited'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# plt.show()

	# FOIL PROPERTY ADJUSTMENT

	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))

	# foilemissivityscaled=1*np.ones((foilvertwpixel,foilhorizwpixel))
	# foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel,foilhorizwpixel))
	# conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel,foilhorizwpixel))
	# reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel,foilhorizwpixel))

	plt.figure()
	plt.title('Foil emissivity sacled to camera pixels')
	plt.imshow(foilemissivityscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Emissivity [adimensional]')
	plt.plot(172,41,'+',mew=2,markersize=20,label='laser10 ps+fs focused, FLAT=x0.4, x0.4, x1, laser11 ps+fs part defocused, FLAT=x0.4, x0.8, x1')
	plt.plot(196,26,'x',mew=2,markersize=10,label='laser13 ps focused')
	plt.plot(156,27,'x',mew=2,markersize=10,label='laser14 ps focused')
	plt.plot(188,43,'x',mew=2,markersize=10,label='laser15 ELMs focused')
	plt.plot(186,41,'x',mew=2,markersize=10,label='laser16 ELMs part defocused')
	plt.plot(216,21,'+',mew=2,markersize=20,label='laser17 ps+fs focused, FLAT=x0.2, x0.6, x1')
	plt.plot(159,22,'+',mew=2,markersize=20,label='laser18 ps+fs focused, FLAT=x0.4, x1.2, x0.8')
	plt.plot(187,54,'+',mew=2,markersize=20,label='laser19 ps+fs focused, FLAT=x0.4, x1.2, x0.8')
	plt.plot(180,42,'x',mew=2,markersize=10,label='laser20 fs fully defocused, laser21 fs fully defocused (high FR)')
	plt.legend(loc='best')



	plt.figure()
	plt.title('Foil thickness sacled to camera pixels')
	plt.imshow(foilthicknessscaled,'rainbow',origin='lower')
	plt.xlabel('Foil reference axis [pixles]')
	plt.ylabel('Foil reference axis [pixles]')
	plt.colorbar().set_label('Thickness [micrometer]')
	plt.plot(172,41,'+',mew=2,markersize=20,label='laser10 ps+fs focused, FLAT=x0.4, x0.4, x1, laser11 ps+fs part defocused, FLAT=x0.4, x0.8, x1')
	plt.plot(196,26,'x',mew=2,markersize=10,label='laser13 ps focused')
	plt.plot(156,27,'x',mew=2,markersize=10,label='laser14 ps focused')
	plt.plot(188,43,'x',mew=2,markersize=10,label='laser15 ELMs focused')
	plt.plot(186,41,'x',mew=2,markersize=10,label='laser16 ELMs part defocused')
	plt.plot(216,21,'+',mew=2,markersize=20,label='laser17 ps+fs focused, FLAT=x0.2, x0.6, x1')
	plt.plot(159,22,'+',mew=2,markersize=20,label='laser18 ps+fs focused, FLAT=x0.4, x1.2, x0.8')
	plt.plot(187,54,'+',mew=2,markersize=20,label='laser19 ps+fs focused, FLAT=x0.4, x1.2, x0.8')
	plt.plot(180,42,'x',mew=2,markersize=10,label='laser20 fs fully defocused, laser21 fs fully defocused (high FR)')
	plt.legend(loc='best')

	plt.show()















	# EXTRA ANALISYS 1: FREQUENCY OF OSCILLATION GIVEN BY THE CAMERA 30/07/2018

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=laser18[-1]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	# datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

	# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# plt.show()


	datasample=[]
	poscentred=[[15,80],[40,75],[80,85],[70,200],[160,133],[250,200]]
	# poscentred=[[70,70],[160,133],[250,200]]
	# poscentred=[[60,12],[170,12],[290,12]]
	plt.figure(figsize=(20,10))
	for pos in poscentred:
		for a in [10]:
			datasample.append(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
			plt.plot(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)),label='average counts in [H,V] '+str(pos)+' +/- '+str(a))
	plt.legend(loc='best')
	plt.grid()
	plt.title('Averaged counts in different locations in '+pathfiles)
	plt.xlabel('Frames')
	plt.ylabel('Counts [au]')
	plt.grid()
	plt.savefig('/home/ffederic/work/TOPRINT/laser200.eps')
	plt.close()


	frame=10
	plt.figure(figsize=(20,10))
	plt.title('Frame '+str(frame)+' in '+pathfiles)
	plt.imshow(data[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	for pos in poscentred:
		plt.plot(pos[0],pos[1],'k+',markersize=20)
	plt.show()

	def fit_sin(x, *params):
		import numpy as np
		c1 = params[0]
		c2 = params[1]
		c3 = params[2]
		c4 = params[3]
		c5 = params[4]
		return c1+c5*x+c2 * np.sin(x/c3+c4)


	plt.figure(figsize=(20,10))
	framerate=50
	maxstep=99
	minstep=0
	steps=maxstep-minstep
	guess=[5999.5,1.7,6,1,1]
	# poscentred=[[70,70],[70,200],[160,133],[250,200],[250,70]]
	color=['b','r','m','y','g','c']
	index=0
	for pos in poscentred:
		for a in [10]:
			datasample=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))
			plt.plot(datasample,color[index],label='averaged counts in [H,V] '+str(pos)+' +/- '+str(a))
			# index+=1
			temp1,temp2=curve_fit(fit_sin,  np.linspace(minstep,maxstep-1,steps),datasample[minstep:maxstep], p0=guess, maxfev=100000000)
			plt.plot(np.linspace(minstep+1,maxstep,steps),fit_sin(np.linspace(minstep+1,maxstep,steps),*temp1),color[index]+'--',label='fitting in [H,V] '+str(pos)+' +/- '+str(a)+' with amplitude and frequency '+str(int(abs(temp1[1])*100)/100)+'[au], '+str(int(100*framerate/(abs(temp1[2])*2*np.pi))/100)+'Hz',linewidth=3)
			index+=1
	plt.legend(loc='best')
	plt.grid()
	plt.title('Averaged counts in different locations fitted with a sinusoidal curve to extract frequency and amplitude  \n from '+pathfiles)
	plt.xlabel('Frames')
	plt.ylabel('Counts [au]')
	plt.show()




	# EXTRA ANALISYS 2: CHANGE IN BACKGROUND THROUGH THE MEASUREMENTS 31/07/2018

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=laser10[1]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	meancounts=[]
	poscentred=[[15,80],[40,75],[80,85],[70,200],[160,133],[250,200]]

	files=[laser10,laser11,laser12,vacuum1[0:2],laser13,laser14,vacuum1[2:]]
	files=coleval.flatten_full(files)
	for pathfiles in files:

		# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
		# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		filenames=coleval.all_file_names(pathfiles,type)[0]
		if (pathfiles in laser12):
			framerate=994
			datashort=np.load(os.path.join(pathfiles,filenames))
			data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
			data[:,:,64:96,:]=datashort
			type_of_experiment='low duty cycle partially defocused'
		elif (pathfiles in [vacuum1[1],vacuum1[3]]):
			framerate=994
			datashort=np.load(os.path.join(pathfiles,filenames))
			data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
			data[:,:,64:96,:]=datashort
			type_of_experiment='low duty cycle partially defocused'
		elif (pathfiles in laser11):
			framerate=383
			data=np.load(os.path.join(pathfiles,filenames))
			type_of_experiment='partially defocused'
		else:
			framerate=383
			data=np.load(os.path.join(pathfiles,filenames))
			type_of_experiment='focused'
		# datatemp,errdatatemp=coleval.count_to_temp_poly2(data,params,errparams)

		# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
		# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.show()


		datasmean=[]
		datastd=[]
		# poscentred=[[70,70],[160,133],[250,200]]
		# poscentred=[[60,12],[170,12],[290,12]]
		for pos in poscentred:
			for a in [5,10]:
				datasmean.append(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3)))
				datastd.append(np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))))
				# plt.plot(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)),label='average counts in [H,V] '+str(pos)+' +/- '+str(a))
		meancounts.append([datasmean,datastd])
	meancounts=np.array(meancounts)

	plt.figure(figsize=(20,10))
	index=0
	for pos in poscentred:
		for a in [5,10]:
			plt.errorbar(np.linspace(1,len(meancounts[:,0,index]),len(meancounts[:,0,index])),meancounts[:,0,index],yerr=meancounts[:,1,index],label='Variation of background in [H,V] '+str(pos)+' +/- '+str(a))
			index+=1
	plt.legend(loc='best')
	plt.grid()
	plt.title('Mean within the experiment of the averaged counts \n in different locations through all the measurements')
	plt.xlabel('Measurement number (1-37=laser/Jul25_2018/irvb_full-000001 to 37, \n 38-39=vacuum_chamber_testing/Aug02_2018/irvb_full-000001-2, 40-43=laser/Aug02_2018/irvb_full-000001 to 14, \n 44-45=vacuum_chamber_testing/Aug02_2018/irvb_full-000003-4)')
	plt.ylabel('Counts [au]')
	# plt.xticks(range(len(files)-1), files,rotation=60)
	plt.savefig('/home/ffederic/work/TOPRINT/laser18_1.eps')
	plt.close()
	# plt.show()

	frame=10
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles)
	plt.imshow(data[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	for pos in poscentred:
		plt.plot(pos[0],pos[1],'k+',markersize=20)
	plt.savefig('/home/ffederic/work/TOPRINT/laser19.eps')
	plt.close()






	# EXTRA ANALISYS 3: CHANGE IN BACKGROUND THROUGH THE MEASUREMENTS 2 ,with backgroud MEASUREMENTS
	# taken each 5 laser samples. is it really different from adapting the background with reference points?

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=laser17[1]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='npy'
	# type='csv'

	timestamp=[]
	meancounts=[]
	poscentred=[[15,80],[40,75],[80,85],[70,200],[160,133],[250,200]]

	# files=[vacuum2[19],laser17[5],laser17[5],laser17[6],laser17[6],laser17[7],laser17[7],laser17[8],laser17[8],laser17[9],laser17[9],vacuum2[20]]
	files=[vacuum2[18:30]]
	files=coleval.flatten_full(files)
	index=0
	for pathfiles in files:

		# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
		# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))



		# if (index in [0,1,3,5,7,9,11]):
		# 	type='npy'
		# 	filenames=coleval.all_file_names(pathfiles,type)[0]
		# 	framerate=383
		# 	data=np.load(os.path.join(pathfiles,filenames))
		# 	data=np.mean(data[0],axis=(0))
		# elif (index in [2,4,6,8,10]):
		# 	type='_reference.npy'
		# 	filenames=coleval.all_file_names(pathfiles,type)[0]
		# 	framerate=383
		# 	data=np.load(os.path.join(pathfiles,filenames))

		type='_stat.npy'
		filenames=coleval.all_file_names(pathfiles,type)[0]
		framerate=383
		data=np.array(np.load(os.path.join(pathfiles,filenames))[0])

		type='_timestamp.npy'
		filenames=coleval.all_file_names(pathfiles,type)[0]
		timestamp.append(np.load(os.path.join(pathfiles,filenames))[2])


		datasmean=[]
		datastd=[]
		for pos in poscentred:
			for a in [10]:
				datasmean.append(np.mean(data[pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
		meancounts.append(datasmean)
		index+=1
	meancounts=np.array(meancounts)
	timestamp=np.array(timestamp)

	plt.figure(figsize=(20,20))
	index=0
	for pos in poscentred:
		for a in [10]:
			plt.plot(timestamp,meancounts[:,index],label='Background from dedicated measures in [H,V] '+str(pos)+' +/- '+str(a))
			plt.plot(timestamp,meancounts[:,index],'o',markersize=10)
			index+=1

	timestamp=[]
	meancounts=[]
	files=[laser17,laser18,laser19]
	files=coleval.flatten_full(files)
	index=0
	for pathfiles in files:

		type='npy'
		filenames=coleval.all_file_names(pathfiles,type)[0]
		data=np.load(os.path.join(pathfiles,filenames))
		data=np.mean(data[0],axis=(0))


		type='_timestamp.npy'
		filenames=coleval.all_file_names(pathfiles,type)[0]
		timestamp.append(np.load(os.path.join(pathfiles,filenames))[2])


		datasmean=[]
		datastd=[]
		for pos in poscentred:
			for a in [10]:
				datasmean.append(np.mean(data[pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
		meancounts.append(datasmean)
		index+=1
	meancounts=np.array(meancounts)
	timestamp=np.array(timestamp)


	index=0
	for pos in poscentred:
		for a in [10]:
			plt.plot(timestamp,meancounts[:,index],'+',label='Background from laser measures in [H,V] '+str(pos)+' +/- '+str(a),markersize=10)
			index+=1





	plt.legend(loc='best')
	plt.grid()
	plt.title('Mean within the experiment of the averaged counts \n in different locations through all the measurements')
	plt.xlabel('Time from timestamp [s]')
	plt.ylabel('Counts [au]')
	# plt.xticks(range(len(files)-1), files,rotation=60)
	plt.savefig('/home/ffederic/work/TOPRINT/laser203.eps')
	plt.close()
	# plt.show()

	frame=10
	plt.figure()
	plt.title('Frame '+str(frame)+' in '+pathfiles)
	plt.imshow(data[0,frame],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	for pos in poscentred:
		plt.plot(pos[0],pos[1],'k+',markersize=20)
	plt.savefig('/home/ffederic/work/TOPRINT/laser19.eps')
	plt.close()




	# 2018/10/08 extra to just visualize a frame in temperature

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles=files14[1]
	NUCtemperature=temperature14[1]
	# framerate of the IR camera in Hz
	framerate=383
	# integration time of the camera in ms
	inttime=1
	#filestype
	type='_stat.npy'
	# type='csv'

	# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
	# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	filenames=coleval.all_file_names(pathfiles,type)[0]

	# datashort=np.load(os.path.join(pathfiles,filenames))
	# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
	# data[:,:,64:128,:]=datashort
	data=np.load(os.path.join(pathfiles,filenames))[0]
	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	datatemp=coleval.count_to_temp_poly2([[data]],params,errparams)[0][0,0]



	plt.figure()
	plt.title('Averaged counts in '+pathfiles+' \n NUC plate temperature '+str(NUCtemperature))
	plt.imshow(data,'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Averaged temperature in '+pathfiles+' \n NUC plate temperature '+str(NUCtemperature))
	plt.imshow(datatemp,'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()


	# 2018/10/08 extra to visualize a all the frames of a calibration series of measutements

	# degree of polynomial of choice
	n=3
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/numcoeff3/average'
	# folder to read

	alldata=[]
	alldatatemp=[]
	allNUCtemperature=[]
	for pathfiles,NUCtemperature in np.transpose([files16,temperature16]):
		# pathfiles=files14[1]
		# NUCtemperature=temperature14[1]
		# framerate of the IR camera in Hz
		framerate=383
		# integration time of the camera in ms
		inttime=1
		#filestype
		type='_stat.npy'
		# type='csv'

		# fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
		# params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		# errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		filenames=coleval.all_file_names(pathfiles,type)[0]

		# datashort=np.load(os.path.join(pathfiles,filenames))
		# data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
		# data[:,:,64:128,:]=datashort
		data=np.load(os.path.join(pathfiles,filenames))[0]
		params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
		datatemp=coleval.count_to_temp_poly2([[data]],params,errparams)[0][0,0]
		alldata.append(data)
		alldatatemp.append(datatemp)
		allNUCtemperature.append(float(NUCtemperature))


	alldata=np.array(alldata)
	alldatatemp=np.array(alldatatemp)
	allNUCtemperature=np.array(allNUCtemperature)
	# ani=coleval.movie_from_data(np.array([alldatatemp]),framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]')
	# ani.save(os.path.join(pathfiles,filenames[:-4]+'_full_counts'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	# ani=coleval.movie_from_data(data,framerate,inttime,'Horizontal axis [pixles]','Vertical axis [pixles]','Counts [au]',extvmin=3000,extvmax=3200)
	# ani.save(os.path.join('/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/numcoeff4/average','nice video3'+'.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
	#plt.show()

	index=15
	plt.figure()
	plt.title('Averaged counts in '+pathfiles+' \n NUC plate temperature '+str(allNUCtemperature[index])+'°C, framerate '+str(framerate)+'Hz, int. time '+str(inttime)+'ms, coefficients from \n'+ pathparams)
	plt.imshow(alldata[index],'rainbow',origin='lower')
	plt.colorbar().set_label('Counts [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.figure()
	plt.title('Averaged temperature in '+pathfiles+' \n NUC plate temperature '+str(allNUCtemperature[index])+'°C, framerate '+str(framerate)+'Hz, int. time '+str(inttime)+'ms, coefficients from \n'+ pathparams)
	plt.imshow(alldatatemp[index],'rainbow',origin='lower')
	plt.colorbar().set_label('Temp [°C]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.show()

	#2018/10/08 This is to calculate the coefficients with the new functions

	fileshot=np.array([files14[4:]])
	temperaturehot=np.array([temperature14[4:]])
	filescold=np.array([files20[0:]])
	temperaturecold=np.array([temperature20[0:]])

	fileshot=np.array([files16[2:],files14[5:]])
	temperaturehot=np.array([temperature16[2:],temperature14[5:]])
	filescold=np.array([files18[4:],files20[2:]])
	temperaturecold=np.array([temperature18[4:],temperature20[2:]])

	pathparam='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters'
	coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,1,383,pathparam,n)
	coleval.build_average_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,1,383,pathparam,n)






elif False:

	# tHIS LINES ARE CREATED AFTER 10/05/2018 TO ANALYSE THE NEW DATA OF THE CAMERA CALIBRATION.
	# SPECIFICALLY WE WANT TO ANALYSE THE ABORRATION THAT LOOKS LIKE SOME KIND OF REFLECTION CLEARLY VISIBLE ON THE IR CAMERA IMAGES


	for file in files10:
		coleval.collect_subfolderfits_limited(file,0,99)
		coleval.evaluate_back(file)

	# LINES TO CALCULATE THE PARAMETERS FOR 2ms INTEGRATION TIME

	path='/home/ffederic/work/irvb/Calibration_Jul14_2018_set3_1ms'

	temperature=[temperature13]
	files=[files13]
	int=1	#ms
	nmax=3

	coleval.builf_poly_coeff(temperature,files,int,path,nmax)

	n=3
	# integration time of the camera in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/Calibration_May10_2018_set2_'+str(inttime)+'ms'
	# framerate of the IR camera in Hz
	framerate=383
	#filestype
	type='npy'
	# type='csv'

	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))

	plt.figure()
	plt.imshow(params[:,:,0])
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.figure()
	plt.pcolor(params[:,:,1])
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.figure()
	plt.pcolor(params[:,:,2])
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.show()


	# SEARCH FOR SATURATION

	# file to read
	pathfiles='/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064'
	#filestype
	type='npy'
	# type='csv'

	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	plt.imshow(data[0,0,50:,75:],origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts')
	plt.show()
	np.max(data[0,0])

	# FITTING EVALUATION

	#THESE ARE THE COMMANDS TO ANALYSE MULTIPLE SET OF DATA TO EVALUATE IF THE PARAMETERS HAVE BEEN CALCULATED RIGHT
	# FOR THE CASE 2ms AS INTEGRATION TIME

	path='/home/ffederic/work/irvb/Calibration_Jul14_2018_set3_1ms'
	temperature=[temperature13]
	files=[files13]
	int=1	#ms
	nmax=3

	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)

	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	for n in range(2,nmax+1):
		coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
		errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
		for j in range(shapex):
			for k in range(shapey):
				if ((j == 160) & ((k==128) or (k==40) or (k==168))):
					x=np.array(meancounttot[:,j,k])
					xerr=np.array(meancountstdtot[:,j,k])
					yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
					score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
					plt.figure(k*j)
					plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
					plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='poly grade='+str(n)+' R2='+str(np.around(score[n-2,j,k],decimals=3)))
					plt.legend()
					plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
					plt.legend(loc='best')
					plt.xlabel('Counts [au]')
					plt.ylabel('Temperature [°C]')

	plt.show()

	# COMPARISON OF THE FEATURE EFFECT ON PARAMETERS FOR ITS TWO POSITIONS

	n=3
	# integration time of the camera in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/Calibration_May10_2018_set1_'+str(inttime)+'ms'
	# framerate of the IR camera in Hz
	framerate=383
	#filestype
	type='npy'
	# type='csv'


	params_1=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos1=[174,134]
	pathfiles='/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000008'
	type='npy'
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	figure=1
	plt.figure(figure)
	plt.imshow(data[0,0],origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts')
	plt.plot(featurepos1[0],featurepos1[1],'+r',markersize=20)
	plt.title('Position of the feature case 1')

	n=3
	# integration time of the camera in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/Calibration_May10_2018_set2_'+str(inttime)+'ms'
	# framerate of the IR camera in Hz
	framerate=383
	#filestype
	type='npy'
	# type='csv'

	params_2=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos2=[175,205]
	pathfiles='/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064'
	type='npy'
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	figure+=1
	plt.figure(figure)
	plt.imshow(data[0,0],origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts')
	plt.plot(featurepos2[0],featurepos2[1],'+r',markersize=20)
	plt.title('Position of the feature case 2')


	max=np.max((np.max(params_1[:,:,0]),np.max(params_2[:,:,0])))
	min=np.min((np.min(params_1[:,:,0]),np.min(params_2[:,:,0])))
	figure+=1
	plt.figure(figure)
	plt.pcolor(params_1[:,:,0],vmin=min, vmax=max)
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.title('C0 for case 1')
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.pcolor(params_2[:,:,0],vmin=min, vmax=max)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.title('C0 for case 2')
	plt.plot((featurepos2[0],featurepos2[0]),(0,len(params_2[:,0,0])),'k')
	plt.plot((0,len(params_2[0,:,0])),(featurepos2[1],featurepos2[1]),'k')


	max=np.max((np.max(params_1[:,:,1]),np.max(params_2[:,:,1])))
	min=np.min((np.min(params_1[:,:,1]),np.min(params_2[:,:,1])))
	figure+=1
	plt.figure(figure)
	plt.pcolor(params_1[:,:,1],vmin=min, vmax=max)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.title('C1 for case 1')
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.pcolor(params_2[:,:,1],vmin=min, vmax=max)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.title('C1 for case 2')
	plt.plot((featurepos2[0],featurepos2[0]),(0,len(params_2[:,0,0])),'k')
	plt.plot((0,len(params_2[0,:,0])),(featurepos2[1],featurepos2[1]),'k')


	max=np.max((np.max(params_1[:,:,2]),np.max(params_2[:,:,2])))
	min=np.min((np.min(params_1[:,:,2]),np.min(params_2[:,:,2])))
	figure+=1
	plt.figure(figure)
	plt.pcolor(params_1[:,:,2],vmin=min, vmax=max)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.title('C2 for case 1')
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.pcolor(params_2[:,:,2],vmin=min, vmax=max)
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.title('C2 for case 2')
	plt.plot((featurepos2[0],featurepos2[0]),(0,len(params_2[:,0,0])),'k')
	plt.plot((0,len(params_2[0,:,0])),(featurepos2[1],featurepos2[1]),'k')


	C0horiz1=params_1[featurepos1[1],:,0]
	C1horiz1=params_1[featurepos1[1],:,1]
	C2horiz1=params_1[featurepos1[1],:,2]
	C0vert1=params_1[:,featurepos1[0],0]
	C1vert1=params_1[:,featurepos1[0],1]
	C2vert1=params_1[:,featurepos1[0],2]

	C0horiz2=params_2[featurepos2[1],:,0]
	C1horiz2=params_2[featurepos2[1],:,1]
	C2horiz2=params_2[featurepos2[1],:,2]
	C0vert2=params_2[:,featurepos2[0],0]
	C1vert2=params_2[:,featurepos2[0],1]
	C2vert2=params_2[:,featurepos2[0],2]

	shape=np.shape(params_1[:,:,0])
	vertaxis1=np.linspace(0,shape[0]-1,shape[0])
	vertaxis1=np.add(vertaxis1,-featurepos1[1])
	horizaxis1=np.linspace(0,shape[1]-1,shape[1])
	horizaxis1=np.add(horizaxis1,-featurepos1[0])

	vertaxis2=np.linspace(0,shape[0]-1,shape[0])
	vertaxis2=np.add(vertaxis2,-featurepos2[1])
	horizaxis2=np.linspace(0,shape[1]-1,shape[1])
	horizaxis2=np.add(horizaxis2,-featurepos2[0])

	figure+=1
	plt.figure(figure)
	plt.plot(horizaxis1,C0horiz1,label='C0 horizontal case 1')
	plt.plot(horizaxis2,C0horiz2,label='C0 horizontal case 2')
	max=np.max((np.max(C0horiz1),np.max(C0horiz2)))
	min=np.min((np.min(C0horiz1),np.min(C0horiz2)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C0 horizontal dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	plt.plot(horizaxis1,C1horiz1,label='C1 horizontal case 1')
	plt.plot(horizaxis2,C1horiz2,label='C1 horizontal case 2')
	max=np.max((np.max(C1horiz1),np.max(C1horiz2)))
	min=np.min((np.min(C1horiz1),np.min(C1horiz2)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Linear coefficient [K/Counts]')
	plt.title('Comparison of C1 horizontal dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	plt.plot(horizaxis1,C2horiz1,label='C2 horizontal case 1')
	plt.plot(horizaxis2,C2horiz2,label='C2 horizontal case 2')
	max=np.max((np.max(C2horiz1),np.max(C2horiz2)))
	min=np.min((np.min(C2horiz1),np.min(C2horiz2)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Quadratic coefficient [K/Counts^2]')
	plt.title('Comparison of C2 horizontal dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	plt.plot(vertaxis1,C0vert1,label='C0 vertical case 1')
	plt.plot(vertaxis2,C0vert2,label='C0 vertical case 2')
	max=np.max((np.max(C0vert1),np.max(C0vert2)))
	min=np.min((np.min(C0vert1),np.min(C0vert2)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Vertical axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C0 vertical dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	plt.plot(vertaxis1,C1vert1,label='C1 vertical case 1')
	plt.plot(vertaxis2,C1vert2,label='C1 vertical case 2')
	max=np.max((np.max(C1vert1),np.max(C1vert2)))
	min=np.min((np.min(C1vert1),np.min(C1vert2)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Vertical axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C1 vertical dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	plt.plot(vertaxis1,C2vert1,label='C2 vertical case 1')
	plt.plot(vertaxis2,C2vert2,label='C2 vertical case 2')
	max=np.max((np.max(C2vert1),np.max(C2vert2)))
	min=np.min((np.min(C2vert1),np.min(C2vert2)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Vertical axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C2 vertical dependance around feature')
	plt.legend()
	plt.legend(loc='best')


	plt.show()

	# 14/06/2018
	# THE INTERPOLATION WE DID HERE IS VERY CRUDE, VERY LITTLE DATA POINTS INCLUDED AND ONLY ONE HOT>ROOM RAMP PER FEATURE POSITION
	# I WANT TO COMPARE THE SITUATION WITH AND WITNOUT WINDOW, WHERE I HAVE MUCH MORE DATA OF

	n=3
	# integration time of the camera in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-06-10_multiple_search_for_parameters/1ms50Hz/averageREDhhhh-c'
	pathparams1='No window, averaged parameters'
	# framerate of the IR camera in Hz
	framerate=383
	#filestype
	type='npy'
	# type='csv'


	params_1=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_1=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos1=[177,128]
	pathfiles='/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000018'
	type='npy'
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	figure=1
	plt.figure(figure)
	plt.imshow(data[0,0],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts')
	plt.plot(featurepos1[0],featurepos1[1],'+r',markersize=80)
	plt.title('Position of the feature , '+pathparams1)

	n=3
	# integration time of the camera in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	pathparams2='Window, central aberration, averaged parameters'
	# framerate of the IR camera in Hz
	framerate=50
	#filestype
	type='npy'
	# type='csv'

	params_2=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_2=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos2=[177,128]
	pathfiles='/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000011'
	type='npy'
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	figure+=1
	plt.figure(figure)
	plt.imshow(data[0,0],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts')
	plt.plot(featurepos2[0],featurepos2[1],'+r',markersize=80)
	plt.title('Position of the feature , '+pathparams2+' \n aberration in '+str(featurepos2))

	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/1-1'
	pathparams4='Window, central aberration, 1-1 parameters'
	params_4=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_4=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos4=[177,128]

	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/1-2'
	pathparams5='Window, central aberration, 1-2 parameters'
	params_5=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_5=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos5=[177,128]

	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/2-1'
	pathparams6='Window, central aberration, 2-1 parameters'
	params_6=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_6=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos6=[177,128]

	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/2-2'
	pathparams7='Window, central aberration, 2-2 parameters'
	params_7=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_7=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos7=[177,128]



	n=3
	# integration time of the camera in ms
	inttime=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/Calibration_Jul14_2018_set3_1ms'
	pathparams3='Window, off-center aberration, only 1 hot>room'
	# framerate of the IR camera in Hz
	framerate=50
	#filestype
	type='npy'
	# type='csv'

	params_3=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	errparams_3=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
	featurepos3=[177,211]
	pathfiles='/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000042'
	type='npy'
	filenames=coleval.all_file_names(pathfiles,type)[0]
	data=np.load(os.path.join(pathfiles,filenames))
	figure+=1
	plt.figure(figure)
	plt.imshow(data[0,0],'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts')
	plt.plot(featurepos3[0],featurepos3[1],'+r',markersize=80)
	plt.title('Position of the feature , '+pathparams3+' \n aberration in '+str(featurepos3))
	plt.show()



	max=np.max((np.max(params_1[:,:,0]),np.max(params_2[:,:,0]),np.max(params_3[:,:,0])))
	min=np.min((np.min(params_1[:,:,0]),np.min(params_2[:,:,0]),np.max(params_3[:,:,0])))
	figure+=1
	plt.figure(figure)
	plt.imshow(params_1[:,:,0],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.title('C0 , '+pathparams1)
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.imshow(params_2[:,:,0],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.title('C0 , '+pathparams2)
	plt.plot((featurepos2[0],featurepos2[0]),(0,len(params_2[:,0,0])),'k')
	plt.plot((0,len(params_2[0,:,0])),(featurepos2[1],featurepos2[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.imshow(params_3[:,:,0],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Constant coefficient [K]')
	plt.title('C0 , '+pathparams3)
	plt.plot((featurepos3[0],featurepos3[0]),(0,len(params_3[:,0,0])),'k')
	plt.plot((0,len(params_3[0,:,0])),(featurepos3[1],featurepos3[1]),'k')


	max=np.max((np.max(params_1[:,:,1]),np.max(params_2[:,:,1])))
	min=np.min((np.min(params_1[:,:,1]),np.min(params_2[:,:,1])))
	figure+=1
	plt.figure(figure)
	plt.imshow(params_1[:,:,1],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.title('C1 , '+pathparams1)
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.imshow(params_2[:,:,1],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.title('C1 , '+pathparams2)
	plt.plot((featurepos2[0],featurepos2[0]),(0,len(params_2[:,0,0])),'k')
	plt.plot((0,len(params_2[0,:,0])),(featurepos2[1],featurepos2[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.imshow(params_3[:,:,1],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Linear coefficient [K/Counts]')
	plt.title('C1 , '+pathparams3)
	plt.plot((featurepos3[0],featurepos3[0]),(0,len(params_3[:,0,0])),'k')
	plt.plot((0,len(params_3[0,:,0])),(featurepos3[1],featurepos3[1]),'k')


	max=np.max((np.max(params_1[:,:,2]),np.max(params_2[:,:,2])))
	min=np.min((np.min(params_1[:,:,2]),np.min(params_2[:,:,2])))
	figure+=1
	plt.figure(figure)
	plt.imshow(params_1[:,:,2],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.title('C2 , '+pathparams1)
	plt.plot((featurepos1[0],featurepos1[0]),(0,len(params_1[:,0,0])),'k')
	plt.plot((0,len(params_1[0,:,0])),(featurepos1[1],featurepos1[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.imshow(params_2[:,:,2],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.title('C2 , '+pathparams2)
	plt.plot((featurepos2[0],featurepos2[0]),(0,len(params_2[:,0,0])),'k')
	plt.plot((0,len(params_2[0,:,0])),(featurepos2[1],featurepos2[1]),'k')
	figure+=1
	plt.figure(figure)
	plt.imshow(params_3[:,:,2],'rainbow',vmin=min, vmax=max,origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
	plt.title('C2 , '+pathparams3)
	plt.plot((featurepos3[0],featurepos3[0]),(0,len(params_3[:,0,0])),'k')
	plt.plot((0,len(params_3[0,:,0])),(featurepos3[1],featurepos3[1]),'k')


	C0horiz1=params_1[featurepos1[1],:,0]
	errC0horiz1=errparams_1[featurepos1[1],:,0]
	C1horiz1=params_1[featurepos1[1],:,1]
	errC1horiz1=errparams_1[featurepos1[1],:,1]
	C2horiz1=params_1[featurepos1[1],:,2]
	errC2horiz1=errparams_1[featurepos1[1],:,2]
	C0vert1=params_1[:,featurepos1[0],0]
	errC0vert1=errparams_1[:,featurepos1[0],0]
	C1vert1=params_1[:,featurepos1[0],1]
	errC1vert1=errparams_1[:,featurepos1[0],1]
	C2vert1=params_1[:,featurepos1[0],2]
	errC2vert1=errparams_1[:,featurepos1[0],2]

	C0horiz2=params_2[featurepos2[1],:,0]
	errC0horiz2=errparams_2[featurepos2[1],:,0]
	C1horiz2=params_2[featurepos2[1],:,1]
	errC1horiz2=errparams_2[featurepos2[1],:,1]
	C2horiz2=params_2[featurepos2[1],:,2]
	errC2horiz2=errparams_2[featurepos2[1],:,2]
	C0vert2=params_2[:,featurepos2[0],0]
	errC0vert2=errparams_2[:,featurepos2[0],0]
	C1vert2=params_2[:,featurepos2[0],1]
	errC1vert2=errparams_2[:,featurepos2[0],1]
	C2vert2=params_2[:,featurepos2[0],2]
	errC2vert2=errparams_2[:,featurepos2[0],2]

	C0horiz3=params_3[featurepos3[1],:,0]
	errC0horiz3=errparams_3[featurepos3[1],:,0]
	C1horiz3=params_3[featurepos3[1],:,1]
	errC1horiz3=errparams_3[featurepos3[1],:,1]
	C2horiz3=params_3[featurepos3[1],:,2]
	errC2horiz3=errparams_3[featurepos3[1],:,2]
	C0vert3=params_3[:,featurepos3[0],0]
	errC0vert3=errparams_3[:,featurepos3[0],0]
	C1vert3=params_3[:,featurepos3[0],1]
	errC1vert3=errparams_3[:,featurepos3[0],1]
	C2vert3=params_3[:,featurepos3[0],2]
	errC2vert3=errparams_3[:,featurepos3[0],2]

	C0horiz4=params_4[featurepos4[1],:,0]
	errC0horiz4=errparams_4[featurepos4[1],:,0]
	C1horiz4=params_4[featurepos4[1],:,1]
	errC1horiz4=errparams_4[featurepos4[1],:,1]
	C2horiz4=params_4[featurepos4[1],:,2]
	errC2horiz4=errparams_4[featurepos4[1],:,2]
	C0vert4=params_4[:,featurepos4[0],0]
	errC0vert4=errparams_4[:,featurepos4[0],0]
	C1vert4=params_4[:,featurepos4[0],1]
	errC1vert4=errparams_4[:,featurepos4[0],1]
	C2vert4=params_4[:,featurepos4[0],2]
	errC2vert4=errparams_4[:,featurepos4[0],2]

	C0horiz5=params_5[featurepos5[1],:,0]
	errC0horiz5=errparams_5[featurepos5[1],:,0]
	C1horiz5=params_5[featurepos5[1],:,1]
	errC1horiz5=errparams_5[featurepos5[1],:,1]
	C2horiz5=params_5[featurepos5[1],:,2]
	errC2horiz5=errparams_5[featurepos5[1],:,2]
	C0vert5=params_5[:,featurepos5[0],0]
	errC0vert5=errparams_5[:,featurepos5[0],0]
	C1vert5=params_5[:,featurepos5[0],1]
	errC1vert5=errparams_5[:,featurepos5[0],1]
	C2vert5=params_5[:,featurepos5[0],2]
	errC2vert5=errparams_5[:,featurepos5[0],2]

	C0horiz6=params_6[featurepos6[1],:,0]
	errC0horiz6=errparams_6[featurepos6[1],:,0]
	C1horiz6=params_6[featurepos6[1],:,1]
	errC1horiz6=errparams_6[featurepos6[1],:,1]
	C2horiz6=params_6[featurepos6[1],:,2]
	errC2horiz6=errparams_6[featurepos6[1],:,2]
	C0vert6=params_6[:,featurepos6[0],0]
	errC0vert6=errparams_6[:,featurepos6[0],0]
	C1vert6=params_6[:,featurepos6[0],1]
	errC1vert6=errparams_6[:,featurepos6[0],1]
	C2vert6=params_6[:,featurepos6[0],2]
	errC2vert6=errparams_6[:,featurepos6[0],2]

	C0horiz7=params_7[featurepos7[1],:,0]
	errC0horiz7=errparams_7[featurepos7[1],:,0]
	C1horiz7=params_7[featurepos7[1],:,1]
	errC1horiz7=errparams_7[featurepos7[1],:,1]
	C2horiz7=params_7[featurepos7[1],:,2]
	errC2horiz7=errparams_7[featurepos7[1],:,2]
	C0vert7=params_7[:,featurepos7[0],0]
	errC0vert7=errparams_7[:,featurepos7[0],0]
	C1vert7=params_7[:,featurepos7[0],1]
	errC1vert7=errparams_7[:,featurepos7[0],1]
	C2vert7=params_7[:,featurepos7[0],2]
	errC2vert7=errparams_7[:,featurepos7[0],2]

	shape=np.shape(params_1[:,:,0])
	vertaxis1=np.linspace(0,shape[0]-1,shape[0])
	vertaxis1=np.add(vertaxis1,-featurepos1[1])
	horizaxis1=np.linspace(0,shape[1]-1,shape[1])
	horizaxis1=np.add(horizaxis1,-featurepos1[0])

	vertaxis2=np.linspace(0,shape[0]-1,shape[0])
	vertaxis2=np.add(vertaxis2,-featurepos2[1])
	horizaxis2=np.linspace(0,shape[1]-1,shape[1])
	horizaxis2=np.add(horizaxis2,-featurepos2[0])

	vertaxis3=np.linspace(0,shape[0]-1,shape[0])
	vertaxis3=np.add(vertaxis3,-featurepos3[1])
	horizaxis3=np.linspace(0,shape[1]-1,shape[1])
	horizaxis3=np.add(horizaxis3,-featurepos3[0])

	vertaxis4=np.linspace(0,shape[0]-1,shape[0])
	vertaxis4=np.add(vertaxis4,-featurepos4[1])
	horizaxis4=np.linspace(0,shape[1]-1,shape[1])
	horizaxis4=np.add(horizaxis4,-featurepos4[0])

	vertaxis5=np.linspace(0,shape[0]-1,shape[0])
	vertaxis5=np.add(vertaxis5,-featurepos5[1])
	horizaxis5=np.linspace(0,shape[1]-1,shape[1])
	horizaxis5=np.add(horizaxis5,-featurepos5[0])

	vertaxis6=np.linspace(0,shape[0]-1,shape[0])
	vertaxis6=np.add(vertaxis6,-featurepos6[1])
	horizaxis6=np.linspace(0,shape[1]-1,shape[1])
	horizaxis6=np.add(horizaxis6,-featurepos6[0])

	vertaxis7=np.linspace(0,shape[0]-1,shape[0])
	vertaxis7=np.add(vertaxis7,-featurepos7[1])
	horizaxis7=np.linspace(0,shape[1]-1,shape[1])
	horizaxis7=np.add(horizaxis7,-featurepos7[0])



	# C0horiz4=params_2[featurepos2[1]+25,:,0]
	# C0horiz5=params_2[featurepos2[1]+50,:,0]
	# C0horiz6=params_2[featurepos2[1]-25,:,0]
	# C0horiz7=params_2[featurepos2[1]-50,:,0]
	figure+=1
	plt.figure(figure)
	# plt.errorbar(horizaxis1,C0horiz1,yerr=errC0horiz1,fmt='b',label='C0 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
	# plt.errorbar(horizaxis2,C0horiz2,yerr=errC0horiz2,fmt='r',label='C0 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
	# plt.errorbar(horizaxis3,C0horiz3,yerr=errC0horiz3,fmt='y',label='C0 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
	plt.plot(horizaxis1,C0horiz1,'b',label='C0 horizontal \n '+pathparams1)
	# guess=[1,1]
	# fit1,errfit1=curve_fit(coleval.polygen3(2), horizaxis2, C0horiz2, p0=guess, sigma=errC0horiz2, maxfev=100000000)
	# plt.plot(horizaxis2,coleval.polygen3(2)(horizaxis2,*fit1),'k')
	# plt.plot(horizaxis2,np.add(-18,(C0horiz2-coleval.polygen3(2)(horizaxis2,*fit1))),label='1')
	plt.plot(horizaxis2,C0horiz2,'r',label='C0 horizontal \n '+pathparams2)
	# guess=[1,1]
	# fit2,errfit2=curve_fit(coleval.polygen3(2), horizaxis3, C0horiz3, p0=guess, sigma=errC0horiz3, maxfev=100000000)
	# plt.plot(horizaxis3,coleval.polygen3(2)(horizaxis3,*fit2),'k')
	# plt.plot(horizaxis3,np.add(-18,(C0horiz3-coleval.polygen3(2)(horizaxis3,*fit2))),label='2')
	plt.plot(horizaxis3,C0horiz3,'y',label='C0 horizontal \n '+pathparams3)
	plt.plot(horizaxis4,C0horiz4,label='C0 horizontal \n '+pathparams4)
	plt.plot(horizaxis5,C0horiz5,label='C0 horizontal \n '+pathparams5)
	plt.plot(horizaxis6,C0horiz6,label='C0 horizontal \n '+pathparams6)
	plt.plot(horizaxis7,C0horiz7,label='C0 horizontal \n '+pathparams7)
	max=np.max((np.max(C0horiz1),np.max(C0horiz2),np.max(C0horiz3)))
	min=np.min((np.min(C0horiz1),np.min(C0horiz2),np.max(C0horiz3)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C0 horizontal dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	# plt.errorbar(horizaxis1,C1horiz1,yerr=errC1horiz1,fmt='b',label='C1 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
	# plt.errorbar(horizaxis2,C1horiz2,yerr=errC1horiz2,fmt='r',label='C1 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
	# plt.errorbar(horizaxis3,C1horiz3,yerr=errC1horiz3,fmt='y',label='C1 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
	plt.plot(horizaxis1,C1horiz1,'b',label='C1 horizontal \n '+pathparams1)
	# mean=np.mean(C1horiz1)
	# guess=[1,1,1,1]
	# fit1,errfit1=curve_fit(coleval.polygen3(2), horizaxis2, C1horiz2, p0=guess, sigma=errC1horiz2, maxfev=100000000)
	# plt.plot(horizaxis2,coleval.polygen3(2)(horizaxis2,*fit1),'k')
	# plt.plot(horizaxis2,np.add(mean,(C1horiz2-coleval.polygen3(2)(horizaxis2,*fit1))),label='1')
	plt.plot(horizaxis2,C1horiz2,'r',label='C1 horizontal \n '+pathparams2)
	# guess=[1,1,1,1]
	# fit2,errfit2=curve_fit(coleval.polygen3(2), horizaxis3, C1horiz3, p0=guess, sigma=errC1horiz3, maxfev=100000000)
	# plt.plot(horizaxis3,coleval.polygen3(2)(horizaxis3,*fit2),'k')
	# plt.plot(horizaxis3,np.add(mean,(C1horiz3-coleval.polygen3(2)(horizaxis3,*fit2))),label='2')
	plt.plot(horizaxis3,C1horiz3,'y',label='C1 horizontal \n '+pathparams3)
	plt.plot(horizaxis4,C1horiz4,label='C1 horizontal \n '+pathparams4)
	plt.plot(horizaxis5,C1horiz5,label='C1 horizontal \n '+pathparams5)
	plt.plot(horizaxis6,C1horiz6,label='C1 horizontal \n '+pathparams6)
	plt.plot(horizaxis7,C1horiz7,label='C1 horizontal \n '+pathparams7)
	max=np.max((np.max(C1horiz1),np.max(C1horiz2),np.max(C1horiz3)))
	min=np.min((np.min(C1horiz1),np.min(C1horiz2),np.min(C1horiz3)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Linear coefficient [K/Counts]')
	plt.title('Comparison of C1 horizontal dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure+=1
	plt.figure(figure)
	# plt.errorbar(horizaxis1,C2horiz1,yerr=errC2horiz1,fmt='b',label='C2 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
	# plt.errorbar(horizaxis2,C2horiz2,yerr=errC2horiz2,fmt='r',label='C2 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
	# plt.errorbar(horizaxis3,C2horiz3,yerr=errC2horiz3,fmt='y',label='C2 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
	plt.plot(horizaxis1,C2horiz1,'b',label='C2 horizontal \n '+pathparams1)
	plt.plot(horizaxis2,C2horiz2,'r',label='C2 horizontal \n '+pathparams2)
	plt.plot(horizaxis3,C2horiz3,'y',label='C2 horizontal \n '+pathparams3)
	plt.plot(horizaxis4,C2horiz4,label='C2 horizontal \n '+pathparams4)
	plt.plot(horizaxis5,C2horiz5,label='C2 horizontal \n '+pathparams5)
	plt.plot(horizaxis6,C2horiz6,label='C2 horizontal \n '+pathparams6)
	plt.plot(horizaxis7,C2horiz7,label='C2 horizontal \n '+pathparams7)
	max=np.max((np.max(C2horiz1),np.max(C2horiz2),np.max(C2horiz3)))
	min=np.min((np.min(C2horiz1),np.min(C2horiz2),np.max(C2horiz3)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Quadratic coefficient [K/Counts^2]')
	plt.title('Comparison of C2 horizontal dependance around feature')
	plt.legend()
	plt.legend(loc='best')

	figure-=2
	plt.figure(figure)
	# plt.errorbar(vertaxis1,C0vert1,yerr=errC0vert1,fmt='b--',label='C0 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
	# plt.errorbar(vertaxis2,C0vert2,yerr=errC0vert2,fmt='r--',label='C0 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
	# plt.errorbar(vertaxis3,C0vert3,yerr=errC0vert3,fmt='y--',label='C0 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
	plt.plot(vertaxis1,C0vert1,'b--',label='C0 vertical \n '+pathparams1)
	plt.plot(vertaxis2,C0vert2,'r--',label='C0 vertical \n '+pathparams2)
	plt.plot(vertaxis3,C0vert3,'y--',label='C0 vertical \n '+pathparams3)
	max=np.max((np.max(C0vert1),np.max(C0vert2),np.max(C0vert3)))
	min=np.min((np.min(C0vert1),np.min(C0vert2),np.max(C0vert3)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Vertical axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C0 dependance around centre of aberration')
	plt.legend()
	plt.legend(loc='best', prop={'size': 10})

	figure+=1
	plt.figure(figure)
	# plt.errorbar(vertaxis1,C1vert1,yerr=errC1vert1,fmt='b--',label='C1 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
	# plt.errorbar(vertaxis2,C1vert2,yerr=errC1vert2,fmt='r--',label='C1 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
	# plt.errorbar(vertaxis3,C1vert3,yerr=errC1vert3,fmt='y--',label='C1 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
	plt.plot(vertaxis1,C1vert1,'b--',label='C1 vertical \n '+pathparams1)
	plt.plot(vertaxis2,C1vert2,'r--',label='C1 vertical \n '+pathparams2)
	plt.plot(vertaxis3,C1vert3,'y--',label='C1 vertical \n '+pathparams3)
	max=np.max((np.max(C1vert1),np.max(C1vert2),np.max(C1vert3)))
	min=np.min((np.min(C1vert1),np.min(C1vert2),np.max(C1vert3)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Vertical axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C1 dependance around centre of aberration')
	plt.legend()
	plt.legend(loc='best', prop={'size': 10})

	figure+=1
	plt.figure(figure)
	# plt.errorbar(vertaxis1,C2vert1,yerr=errC2vert1,fmt='b--',label='C2 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
	# plt.errorbar(vertaxis2,C2vert2,yerr=errC2vert2,fmt='r--',label='C2 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
	# plt.errorbar(vertaxis3,C2vert3,yerr=errC2vert3,fmt='y--',label='C2 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
	plt.plot(vertaxis1,C2vert1,'b--',label='C2 vertical \n '+pathparams1)
	plt.plot(vertaxis2,C2vert2,'r--',label='C2 vertical \n '+pathparams2)
	plt.plot(vertaxis3,C2vert3,'y--',label='C2 vertical \n '+pathparams3)
	max=np.max((np.max(C2vert1),np.max(C2vert2),np.min(C2vert3)))
	min=np.min((np.min(C2vert1),np.min(C2vert2),np.min(C2vert3)))
	plt.plot((0,0),(min,max),'k')
	plt.xlabel('Vertical axis [pixles]')
	plt.ylabel('Constant coefficient [K]')
	plt.title('Comparison of C2 dependance around centre of aberration')
	plt.legend()
	plt.legend(loc='best', prop={'size': 10})

	plt.show()


	# 19/06/2018 LET'S TEST THIS
	# FROM LABBOOK 1 PAGE 96, IT SHOULD BE POSSIBLE FROM ONE CALIBRATION WITH WINDOW TO OBTAIN ANOTHER
	# WITH THE ABERRATION IN A DIFFERENT POSITION.
	# IN ORDER TO GET THE COEFFICIENTS FOR THE SECOND CONFIGURATION I SHOULD EXPAND THE COEFFICIENTS OF THE FIRST
	# IN A WAY THAN WHEN I RESTRICT THEM I CAN COME BACK TO HAVE ENOUGH COEFFICIENTS FOR THE WHOLE PICTURE.
	# I DON'T WANT TO DO IT NOW, SO INSTEAD I CUT MY PARAMETARS ONLY TO THE PART OF THE PICTURE THAT IS COMMON

	feature_right=np.min([np.shape(params_2)[1]-featurepos2[0],np.shape(params_3)[1]-featurepos3[0]])
	feature_left=np.min([featurepos2[0],featurepos3[0]])
	feature_top=np.min([np.shape(params_2)[0]-featurepos2[1],np.shape(params_3)[0]-featurepos3[1]])
	feature_bottom=np.min([featurepos2[1],featurepos3[1]])
	maxv=feature_top+feature_bottom
	maxh=feature_right+feature_left

	params_2_crop=params_2[featurepos2[1]-feature_bottom:featurepos2[1]+feature_top,featurepos2[0]-feature_left:featurepos2[0]+feature_right]
	params_3_crop=params_3[featurepos3[1]-feature_bottom:featurepos3[1]+feature_top,featurepos3[0]-feature_left:featurepos3[0]+feature_right]


	temperature=[temperature14,temperature16,temperature18,temperature20]
	files=[files14,files16,files18,files20]
	while np.shape(temperature[0])!=():
		temperature=coleval.flatten(temperature)
		files=coleval.flatten(files)

	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	meancounttot=meancounttot[:,featurepos2[1]-feature_bottom:featurepos2[1]+feature_top,featurepos2[0]-feature_left:featurepos2[0]+feature_right]


	test=coleval.d2dx2(params_2_crop[:,:,0],1,0)
	test=np.add(test,np.multiply(meancounttot[:,1:-1],coleval.d2dx2(params_2_crop[:,:,1],1,0)))
	test=np.add(test,np.multiply(np.power(meancounttot[:,1:-1],2),coleval.d2dx2(params_2_crop[:,:,2],1,0)))
	test2=np.multiply(2,coleval.ddx(params_2_crop[:,:,1],1,0))
	test2=np.add(test2,np.multiply(4,np.multiply(meancounttot[:,1:-1],coleval.ddx(params_2_crop[:,:,2],1,0))))
	test2=np.add(test2,np.multiply(2,np.multiply(params_2_crop[1:-1,:,2],coleval.ddx(meancounttot,1,1))))
	test=np.add(test,np.multiply(coleval.ddx(meancounttot,1,1),test2))
	test2=np.add(params_2_crop[1:-1,:,1],np.multiply(2,np.multiply(meancounttot[:,1:-1],params_2_crop[1:-1,:,2])))
	test=np.add(test,np.multiply(coleval.d2dx2(meancounttot,1,1),test2))

	temperature=[temperature13]
	files=[files13]
	while np.shape(temperature[0])!=():
		temperature=coleval.flatten(temperature)
		files=coleval.flatten(files)

	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	meancounttot=meancounttot[:,featurepos3[1]-feature_bottom:featurepos3[1]+feature_top,featurepos3[0]-feature_left:featurepos3[0]+feature_right]

	test=coleval.d2dx2(params_2_crop[:,:,0],1,0)
	test=np.add(test,np.multiply(meancounttot[:,1:-1],coleval.d2dx2(params_2_crop[:,:,1],1,0)))
	test=np.add(test,np.multiply(np.power(meancounttot[:,1:-1],2),coleval.d2dx2(params_2_crop[:,:,2],1,0)))
	test2=np.multiply(2,coleval.ddx(params_2_crop[:,:,1],1,0))
	test2=np.add(test2,np.multiply(4,np.multiply(meancounttot[:,1:-1],coleval.ddx(params_2_crop[:,:,2],1,0))))
	test2=np.add(test2,np.multiply(2,np.multiply(params_2_crop[1:-1,:,2],coleval.ddx(meancounttot,1,1))))
	test=np.add(test,np.multiply(coleval.ddx(meancounttot,1,1),test2))
	test2=np.add(params_2_crop[1:-1,:,1],np.multiply(2,np.multiply(meancounttot[:,1:-1],params_2_crop[1:-1,:,2])))
	test=np.add(test,np.multiply(coleval.d2dx2(meancounttot,1,1),test2))

	counts=coleval.flatten_full(meancounttot[:,1:-1])
	d2Cdx2=coleval.flatten_full(coleval.d2dx2(meancounttot,1,1))
	dCdx=coleval.flatten_full(coleval.ddx(meancounttot,1,1))
	d2Tdx2=coleval.flatten_full(test)
	def residuals(coeffs, d2Tdx2, d2Cdx2, dCdx, counts):
		Da1=coeffs[0]
		Da2=coeffs[1]
		return np.add(-d2Tdx2,np.add(np.multiply(Da1,d2Cdx2),np.multiply(2*Da2,np.add(np.power(dCdx,2),np.multiply(counts,d2Cdx2)))))

	guess=np.array([-0.015,0.0000001])
	x1,x2,x3,x4,x5=leastsq(residuals, guess, args=(d2Tdx2,d2Cdx2,dCdx,counts),full_output=1,epsfcn=0.000000001)

	s_sq=(residuals(x1, d2Tdx2,d2Cdx2,dCdx,counts)**2).sum()/(len(d2Tdx2)-len(guess))
	pcov=x2*s_sq
	pcov=np.sqrt(np.diag(pcov))

	params_2_crop[:,:,1]=params_2_crop[:,:,1]+x1[0]
	params_2_crop[:,:,2]=params_2_crop[:,:,2]+x1[1]


	# meancounttotcrop=meancounttot[:,1:-1]
	# while np.shape(meancounttot[0])!=():
	# 	meancounttot=coleval.flatten(meancounttot)
	guess=[1,1]
	fit,temp=curve_fit(temporary, meancounttot, np.zeros(np.shape(meancounttot[:,1:-1])),p0=guess, maxfev=100000000)


	def temporary(counts, *params):

		da1=params[0]
		da2=params[1]

		out=np.multiply(da1,coleval.d2dx2(counts,1,1))
		out2=np.multiply(2*da2,np.add(np.power(coleval.ddx(counts,1,1),2),np.multiply(counts[:,1:-1],coleval.d2dx2(counts,1,1))))
		out=np.add(out,out2)
		return out


	# I want to learn how to use scipy.optimize.leastsq

	temperature=[temperature16]
	files=[files16]
	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		Ifiles=np.concatenate(files)
	Imeancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
	    meancounttot.append(np.array(data[0]))
	    meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)

	data=meancounttot[:,0,0]
	from scipy.optimize import leastsq

	def residuals(coeffs, y, x):
		return coleval.polygen3(3)(x,*coeffs)-y

	guess=np.array([1,1,1])
	x1,x2,x3,x4,x5=leastsq(residuals, guess, args=(temperature,data),full_output=1,epsfcn=0.0001)

	s_sq=(residuals(x1, temperature, data)**2).sum()/(len(temperature)-len(guess))
	pcov=x2*s_sq
	pcov=np.sqrt(np.diag(pcov))
	guess,temp2=curve_fit(coleval.polygen3(3), data, temperature, p0=guess, maxfev=100000000)
	temp2=np.sqrt(np.diag(temp2))
	# they match!!!


elif False:

	# 14/05/2018 LINES TO CALCULATE FROM MULTIPLE HOT>ROOM AND COLD >ROOM CYCLES THE COEFFICIENTS FOR ALL THE POSSIBLE COMBINATIONS

	inttime=['1ms383Hz','2ms383Hz'];inttimedigits=3
	# inttime=['0.5ms383Hz','1.5ms383Hz'];inttimedigits=5

	fileshot=np.array([[files14,files16],[files15,files17]])
	# fileshot=np.array([[files24,files28],[files25,files29]])
	maxfileshot=0
	for file in fileshot:
		maxfileshot=max(maxfileshot,len(file))
	temperaturehot=np.array([[temperature14,temperature16],[temperature15,temperature17]])
	# temperaturehot=np.array([[temperature24,temperature28],[temperature25,temperature29]])
	filescold=np.array([[files18,files20],[files19,files21]])
	# filescold=np.array([[files26,files30],[files27,files31]])
	maxfilescold=0
	for file in filescold:
		maxfilescold=max(maxfilescold,len(file))
	temperaturecold=np.array([[temperature18,temperature20],[temperature19,temperature21]])
	# temperaturecold=np.array([[temperature26,temperature30],[temperature27,temperature31]])

	# WHERE TO PUT ALL THE PARAMETERS
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters'

	# THIS COMPUTE THE 1-1, 1-2, 2-1, 2-2 PARAMETERS
	for k in range(len(inttime)):
		for i in range(len(fileshot[k])):
			for j in range(len(filescold[k])):
				path=pathparams+'/'+inttime[k]+'/'+str(i+1)+'-'+str(j+1)
				if not os.path.exists(path):
					os.makedirs(path)
				temperature=[temperaturehot[k,i],temperaturecold[k,j]]
				files=[fileshot[k,i],filescold[k,j]]
				int=(inttime[k][0:inttimedigits])	#ms
				nmax=3
				coleval.builf_poly_coeff(temperature,files,int,path,nmax)

	# THIS MAKES AND SAVE THE AVERAGE
	n=3
	first=True
	for k in range(len(inttime)):
		for i in range(len(fileshot[k])):
			for j in range(len(filescold[k])):
				path=pathparams+'/'+inttime[k]+'/'+str(i+1)+'-'+str(j+1)
				int=(inttime[k][0:inttimedigits])
				params=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				if first==True:
					shape=np.shape(params)
					shape=np.concatenate(((len(inttime),maxfileshot,maxfilescold),shape))
					parameters=np.zeros(shape)
					first=False
				parameters[k,i,j]=params

	meanparameters=np.mean(parameters,axis=(1,2))
	stdparameters=np.std(parameters,axis=(1,2))

	for k in range(len(inttime)):
		int=(inttime[k][0:inttimedigits])
		path=pathparams+'/'+inttime[k]+'/average'
		if not os.path.exists(path):
			os.makedirs(path)
		np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)),meanparameters[k])
		np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)),stdparameters[k])


	# THIS CALCULATE THE GLOBAL FITTING FOR COMPARISON

	for intindex in range(2):
		path=pathparams+'/'+inttime[intindex]+'/global'
		temperature=[temperaturehot[intindex],temperaturecold[intindex]]
		files=[fileshot[intindex],filescold[intindex]]
		nmax=3
		while np.shape(temperature[0])!=():
			temperature=np.concatenate(temperature)
			files=np.concatenate(files)
		if not os.path.exists(path):
			os.makedirs(path)
		nmax=3
		int=(inttime[intindex][0:inttimedigits-2])	#ms
		coleval.builf_poly_coeff(temperature,files,int,path,nmax)



	# SINGLE GROUP OF PARAMETERS EAVLUATION
	for intindex in range(2):
		for hot in range(len(fileshot[intindex])):
			for cold in range(len(filescold[intindex])):
				path=pathparams+'/'+inttime[intindex]+'/'+str(hot+1)+'-'+str(cold+1)
				temperature=[temperaturehot[intindex,hot],temperaturecold[intindex,cold]]
				files=[fileshot[intindex,hot],filescold[intindex,cold]]
				int=(inttime[intindex][0:inttimedigits])	#ms
				nmax=3

				while np.shape(temperature[0])!=():
					temperature=np.concatenate(temperature)
					files=np.concatenate(files)

				meancounttot=[]
				meancountstdtot=[]
				for file in files:
					data=coleval.collect_stat(file)
					meancounttot.append(np.array(data[0]))
					meancountstdtot.append(np.array(data[1]))

				meancounttot=np.array(meancounttot)
				meancountstdtot=np.array(meancountstdtot)

				shapex=np.shape(meancounttot[0])[0]
				shapey=np.shape(meancounttot[0])[1]
				score=np.zeros((nmax-1,shapex,shapey))


				for n in range(2,nmax+1):
					coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
					errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
					for j in range(shapex):
						for k in range(shapey):
							if ((j == 128) & ((k==160) or (k==40) or (k==168))):
								x=np.array(meancounttot[:,j,k])
								xerr=np.array(meancountstdtot[:,j,k])
								yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
								score[n-2,j,k]=coleval.rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
								plt.figure(k*j*(intindex+1))
								plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
								plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+'params '+str(inttime[intindex])+' '+str(hot)+'-'+str(cold))
								plt.legend()
								plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
								plt.legend(loc='best')
								plt.xlabel('Counts [au]')
								plt.ylabel('Temperature [°C]')

	plt.show()



	# LINES TO CALCULATE THE PARAMETERS FOR 1ms INTEGRATION TIME AND 994Hz

	path='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms994Hz'

	temperature=[temperature22,temperature23]
	files=[files22,files23]
	int=1	#ms
	nmax=3
	coleval.builf_poly_coeff(temperature,files,int,path,nmax)


	#THESE ARE THE COMMANDS TO ANALYSE MULTIPLE SET OF DATA TO EVALUATE IF THE PARAMETERS HAVE BEEN CALCULATED RIGHT
	# FOR THE CASE 1ms AS INTEGRATION TIME


	n=3
	# integration time of the camera in ms
	int=1
	# folder of the parameters path
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms994Hz'
	# framerate of the IR camera in Hz
	framerate=994
	#filestype
	type='npy'
	# type='csv'

	params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))
	errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms.npy'))

	paramspartial=np.zeros((256,320,3))
	errparamspartial=np.zeros((256,320,3))
	paramspartial[95:159,:,:]=params
	errparamspartial[95:159,:,:]=errparams

	# LINES TO VISUALIZE THE PARAMETERS FROM THE PARTIAL FRAME
	# # file to read
	# pathfiles=files22[20]
	# #filestype
	# type='npy'
	# # type='csv'
	#
	# filenames=coleval.all_file_names(pathfiles,type)[0]
	# data=meancounttot
	# plt.pcolor(data[10,:,:],vmin=7600, vmax=7000)
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.colorbar().set_label('Counts')
	# plt.show()


	inttime=['1ms383Hz','2ms383Hz'];inttimedigits=3
	fileshot=np.array([[files14,files16],[files15,files17]])
	maxfileshot=0
	for file in fileshot:
		maxfileshot=np.max((maxfileshot,len(file)))
	temperaturehot=np.array([[temperature14,temperature16],[temperature15,temperature17]])
	filescold=np.array([[files18,files20],[files19,files21]])
	maxfilescold=0
	for file in filescold:
		maxfilescold=np.max((maxfilescold,len(file)))
	temperaturecold=np.array([[temperature18,temperature20],[temperature19,temperature21]])




	path='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms994Hz'
	temperature=[temperature22,temperature23]
	files=[files22,files23]
	int='1ms'	#ms
	nmax=3
	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)

	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=coleval.collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	temp1=np.array(meancounttot)
	temp2=np.array(meancountstdtot)

	meancounttot=np.zeros((len(files),256,320))
	meancountstdtot=np.zeros((len(files),256,320))
	meancounttot[:,95:159,:]=temp1
	meancountstdtot[:,95:159,:]=temp2

	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	for n in range(3,nmax+1):
		coeff=paramspartial
		errcoeff=errparamspartial
		for j in range(shapex):
			for k in range(shapey):
				if (((j == 128) or (j == 115) or (j == 100)) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
					x=np.array(meancounttot[:,j,k])
					xerr=np.array(meancountstdtot[:,j,k])
					yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
					score[n-2,j,k]=coleval.rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
					plt.figure(k*j*1)
					plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
					xerr=np.array([xerr for _,xerr in sorted(zip(x,xerr))])
					# xerr=np.zeros(np.shape(x))
					x=np.sort(x)
					# plt.errorbar(x,(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),yerr=((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x))),fmt='c--',linewidth=1.5,label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+'params 1ms994Hz')
					plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+'params 1ms994Hz')
					plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
					plt.xlabel('Counts [au]')
					plt.ylabel('Temperature [°C]')
					plt.legend()
					plt.legend(loc='best')



	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters'
	for intindex in range(1):
		for hot in range(len(fileshot[intindex])):
			for cold in range(len(filescold[intindex])):
				path=pathparams+'/'+inttime[intindex]+'/'+str(hot+1)+'-'+str(cold+1)
				temperature=[temperaturehot[intindex,hot],temperaturecold[intindex,cold]]
				files=[fileshot[intindex,hot],filescold[intindex,cold]]
				nmax=3
				int='1ms'	#ms

				while np.shape(temperature[0])!=():
					temperature=np.concatenate(temperature)
					files=np.concatenate(files)

				meancounttot=[]
				meancountstdtot=[]
				for file in files:
					data=coleval.collect_stat(file)
					meancounttot.append(np.array(data[0]))
					meancountstdtot.append(np.array(data[1]))

				meancounttot=np.array(meancounttot)
				meancountstdtot=np.array(meancountstdtot)

				shapex=np.shape(meancounttot[0])[0]
				shapey=np.shape(meancounttot[0])[1]
				score=np.zeros((nmax-1,shapex,shapey))


				for n in range(3,nmax+1):
					coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
					errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
					for j in range(shapex):
						for k in range(shapey):
							if (((j == 128) or (j == 115) or (j == 100)) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
								x=np.array(meancounttot[:,j,k])
								xerr=np.array(meancountstdtot[:,j,k])
								yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
								score[n-2,j,k]=coleval.rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
								plt.figure(k*j*(intindex+1))
								plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
								plt.plot(np.sort(x),coleval.polygen3(n)(np.sort(x),*coeff[j,k,:]),label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+'params '+str(inttime[intindex])+' '+str(hot)+'-'+str(cold))
								plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
								plt.xlabel('Counts [au]')
								plt.ylabel('Temperature [°C]')
								plt.legend()
								plt.legend(loc='best')


	temperature=[]
	files=[]
	pathparams='/home/ffederic/work/preliminary_calibration_data/count_to_temp_param_reduced'
	for intindex in range(1):
		for hot in range(len(fileshot[intindex])):
			for cold in range(len(filescold[intindex])):
				path=pathparams
				temperature.append([temperaturehot[intindex,hot],temperaturecold[intindex,cold]])
				files.append([fileshot[intindex,hot],filescold[intindex,cold]])
		nmax=3
		int=(inttime[intindex][0:inttimedigits])	#ms

		while np.shape(temperature[0])!=():
			temperature=np.concatenate(temperature)
			files=np.concatenate(files)

		meancounttot=[]
		meancountstdtot=[]
		for file in files:
			data=coleval.collect_stat(file)
			meancounttot.append(np.array(data[0]))
			meancountstdtot.append(np.array(data[1]))

		meancounttot=np.array(meancounttot)
		meancountstdtot=np.array(meancountstdtot)

		shapex=np.shape(meancounttot[0])[0]
		shapey=np.shape(meancounttot[0])[1]
		score=np.zeros((nmax-1,shapex,shapey))


		for n in range(3,nmax+1):
			coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
			errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
			for j in range(shapex):
				for k in range(shapey):
					if ((j == 128) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
						x=np.array(meancounttot[:,j,k])
						xerr=np.array(meancountstdtot[:,j,k])
						yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
						score[n-2,j,k]=coleval.rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
						plt.figure(k*j*(intindex+1))
						plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
						xerr=np.array([xerr for _,xerr in sorted(zip(x,xerr))])
						x=np.sort(x)
						plt.errorbar(x,(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),yerr=((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x))),fmt='c--',linewidth=1.5,label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+' without window and global fitting')
						plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
						plt.xlabel('Counts [au]')
						plt.ylabel('Temperature [°C]')
						plt.legend()
						plt.legend(loc='best')


	first=0
	aproacheslabel=[' with window and global fitting',' with window and averaged parameters',' with window and averaged parameters REDh-cc',' with window and averaged parameters REDhh-c']
	aproaches=['global','average','averageREDh-cc','averageREDhh-c']
	for which in range(len(aproaches)):

		temperature=[]
		files=[]
		pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters'
		for intindex in range(1):
			for hot in range(len(fileshot[intindex])):
				for cold in range(len(filescold[intindex])):
					path=pathparams+'/'+inttime[intindex]+'/'+aproaches[which]
					temperature.append([temperaturehot[intindex,hot],temperaturecold[intindex,cold]])
					files.append([fileshot[intindex,hot],filescold[intindex,cold]])
			nmax=3
			int=(inttime[intindex][0:inttimedigits])	#ms

			while np.shape(temperature[0])!=():
				temperature=np.concatenate(temperature)
				files=np.concatenate(files)

			meancounttot=[]
			meancountstdtot=[]
			for file in files:
				data=coleval.collect_stat(file)
				meancounttot.append(np.array(data[0]))
				meancountstdtot.append(np.array(data[1]))

			meancounttot=np.array(meancounttot)
			meancountstdtot=np.array(meancountstdtot)

			shapex=np.shape(meancounttot[0])[0]
			shapey=np.shape(meancounttot[0])[1]
			score=np.zeros((nmax-1,shapex,shapey))


			for n in range(3,nmax+1):
				coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				for j in range(shapex):
					for k in range(shapey):
						if ((j == 128) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
							x=np.array(meancounttot[:,j,k])
							xerr=np.array(meancountstdtot[:,j,k])
							yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
							score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
							plt.figure(k*j*(intindex+1))
							if first<3:
								plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
							xerr=np.array([xerr for _,xerr in sorted(zip(x,xerr))])
							x=np.sort(x)
							plt.errorbar(x,(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),yerr=((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x))),fmt='--',label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+aproacheslabel[which])
							plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j))
							plt.xlabel('Counts [au]')
							plt.ylabel('Temperature [°C]')
							plt.legend()
							plt.legend(loc='best')
							first+=1




	plt.xlabel('Counts [au]')
	plt.ylabel('Temperature [°C]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()

	# ISSUE: THE ERROR ASSOCIATED WITH THE AVERAGED PARAMETERS IS WORST THAM THE ERROR OF THE GLOBAL FITTING!!

	# THIS MAKES AND SAVE THE AVERAGE

	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters'
	n=3
	first=True
	maxfileshot=2
	maxfilescold=1
	for k in range(1):
		for i in range(maxfileshot):
			for j in range(maxfilescold):
				path=pathparams+'/'+inttime[k]+'/'+str(i+1)+'-'+str(j+1)
				int=(inttime[k][0:inttimedigits])
				params=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				if first==True:
					shape=np.shape(params)
					shape=np.concatenate(((len(inttime),maxfileshot,maxfilescold),shape))
					parameters=np.zeros(shape)
					first=False
				parameters[k,i,j]=params

	meanparameters=np.mean(parameters,axis=(1,2))
	stdparameters=np.std(parameters,axis=(1,2))

	for k in range(len(inttime)):
		int=(inttime[k][0:inttimedigits])
		path=pathparams+'/'+inttime[k]+'/averageREDhh-c'
		if not os.path.exists(path):
			os.makedirs(path)
		np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)),meanparameters[k])
		np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)),stdparameters[k])



	# COMPARE THE FITTINGS FOR ONE PIXEL

	temperature=[]
	files=[]
	pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters'
	for intindex in range(1):
		for hot in range(2):
			for cold in range(2):
				temperature.append([temperaturehot[intindex,hot],temperaturecold[intindex,cold]])
				files.append([fileshot[intindex,hot],filescold[intindex,cold]])
		nmax=3
		int=(inttime[intindex][0:inttimedigits])	#ms

		while np.shape(temperature[0])!=():
			temperature=np.concatenate(temperature)
			files=np.concatenate(files)

		meancounttot=[]
		meancountstdtot=[]
		for file in files:
			data=coleval.collect_stat(file)
			meancounttot.append(np.array(data[0]))
			meancountstdtot.append(np.array(data[1]))

		meancounttot=np.array(meancounttot)
		meancountstdtot=np.array(meancountstdtot)

		shapex=np.shape(meancounttot[0])[0]
		shapey=np.shape(meancounttot[0])[1]
		score=np.zeros((nmax-1,shapex,shapey))

	aproaches=['global','average','averageREDh-cc','averageREDhh-c']
	for which in aproaches:
		for intindex in range(1):
			path=pathparams+'/'+inttime[intindex]+'/'+which

			for n in range(3,nmax+1):
				coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				for j in range(shapex):
					for k in range(shapey):
						if ((j == 128) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
							x=np.array(meancounttot[:,j,k])
							xerr=np.array(meancountstdtot[:,j,k])
							yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2
							score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
							plt.figure(k*j*(intindex+1))
							#plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
							xerr=np.array([xerr for _,xerr in sorted(zip(x,xerr))])
							x=np.sort(x)
							#plt.errorbar(x,(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),yerr=((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x))),fmt='r--',linewidth=1.5,label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+' with window and global fitting')
							plt.plot((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),np.divide((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x)),(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x))),label=which)
							plt.title('Comparison of temperature relative error for pixel horiz='+str(k)+' vert='+str(j))
							plt.xlabel('Temperature [°C]')
							plt.ylabel('Temperature error/temperature')
							plt.legend()
							plt.legend(loc='best')
	plt.show()


	for which in aproaches:
		meancounts=np.mean(meancounttot,axis=(-1,-2))
		for intindex in range(1):
			path=pathparams+'/'+inttime[intindex]+'/'+which

			for n in range(3,nmax+1):
				coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				temp=np.divide((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[1]),(coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[0]))
				temp=np.mean(temp,axis=(0,-1,-2))
				errperc=temp
				errperc=np.array([errperc for _,errperc in sorted(zip(meancounts,errperc))])

		temp=np.sort(np.mean((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[0]),axis=(0,-1,-2)))
		plt.plot(temp,errperc,label=which)
		plt.title('Mean of relative temperature error across all pixels')
		plt.xlabel('Temperature [°C]')
		plt.ylabel('Temperature error/temperature')
		plt.legend()
		plt.legend(loc='best')
	plt.show()

	# 13/60/2018 I WANT TO SEE NOT THE COMPONENTS OF THE ERROR, I DO IT ONLY FOR THE AVERAGED COEFFICIENTS

	which='average'
	meancounts=np.mean(meancounttot,axis=(-1,-2))
	intindex=0
	path=pathparams+'/'+inttime[intindex]+'/'+which
	n=3
	coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
	errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
	errtemp=(coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[1])
	Ea0=np.mean(np.divide(errcoeff[:,:,0],errtemp),axis=(0,-1,-2))
	Ea0=np.array([Ea0 for _,Ea0 in sorted(zip(meancounts,Ea0))])
	Ea1=np.mean(np.divide(np.multiply(meancounttot,errcoeff[:,:,1]),errtemp),axis=(0,-1,-2))
	Ea1=np.array([Ea1 for _,Ea1 in sorted(zip(meancounts,Ea1))])
	Ea2=np.mean(np.divide(np.multiply(np.power(meancounttot,2),errcoeff[:,:,2]),errtemp),axis=(0,-1,-2))
	Ea2=np.array([Ea2 for _,Ea2 in sorted(zip(meancounts,Ea2))])
	Ec=np.mean(np.divide(np.sqrt(np.add(np.power(np.multiply(coeff[:,:,1],meancountstdtot),2),np.power(np.multiply(np.multiply(np.multiply(coeff[:,:,2],meancountstdtot),2),meancounttot),2))),errtemp),axis=(0,-1,-2))
	Ec=np.array([Ec for _,Ec in sorted(zip(meancounts,Ec))])

	temp=np.sort(np.mean((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[0]),axis=(0,-1,-2)))
	plt.plot(temp,Ea0,'r',label='Ea0/Et')
	plt.plot(temp,Ea1,'b',label='Ea1/Et')
	plt.plot(temp,Ea2,'k',label='Ea2/Et')
	plt.plot(temp,Ec,'y',label='Ec/Et')
	plt.title('Mean of the error per component relative to the total error across all pixels parameters from \n'+path,size=10)
	plt.xlabel('Temperature [°C]')
	plt.ylabel('Partial temperature error/ total temperature error')
	plt.legend()
	plt.legend(loc='best')
	plt.show()

	# ISSUE. IT IS REALLY LIKE THAT! THE ERROR ASSOCIATED WITH THE AVERAGE IS HIGH AS THE GLOBAL ONE!
	# THE MOST OF THE VARIABILITY IS IN THE HOT>ROOM PROCESS. WHAT IF I TAKE MULTIPLE OF THAT?
	# I CAN TAKE THE ORIGINAL DATA WITHOUT WINDOW AND CHECK IT OUT (I DID MANY HOT>ROOM TESTS)

	inttime=['1ms50Hz'];inttimedigits=3
	fileshot=np.array([[files2,files3,files4,files5]])
	maxfileshot=0
	for file in fileshot:
		maxfileshot=max(maxfileshot,len(file))
	temperaturehot=np.array([[temperature2,temperature3,temperature4,temperature5]])
	filescold=np.array([[files6]])
	maxfilescold=0
	for file in filescold:
		maxfilescold=max(maxfilescold,len(file))
	temperaturecold=np.array([[temperature6]])

	# WHERE TO PUT ALL THE PARAMETERS
	pathparams='/home/ffederic/work/irvb/2018-06-10_multiple_search_for_parameters'

	# THIS COMPUTE THE 1-1, 1-2, 2-1, 2-2 PARAMETERS
	for k in range(len(inttime)):
		for i in range(len(fileshot[k])):
			for j in range(len(filescold[k])):
				path=pathparams+'/'+inttime[k]+'/'+str(i+1)+'-'+str(j+1)
				if not os.path.exists(path):
					os.makedirs(path)
				temperature=[temperaturehot[k,i],temperaturecold[k,j]]
				files=[fileshot[k,i],filescold[k,j]]
				int=(inttime[k][0:inttimedigits-2])	#ms
				nmax=3
				coleval.builf_poly_coeff(temperature,files,int,path,nmax)


	# THIS MAKES AND SAVE THE AVERAGE
	n=3
	first=True
	for k in range(len(inttime)):
		for i in range(len(fileshot[k])):
			for j in range(len(filescold[k])):
				path=pathparams+'/'+inttime[k]+'/'+str(i+1)+'-'+str(j+1)
				int=(inttime[k][0:inttimedigits])
				params=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				if first==True:
					shape=np.shape(params)
					shape=np.concatenate(((len(inttime),maxfileshot,maxfilescold),shape))
					parameters=np.zeros(shape)
					first=False
				parameters[k,i,j]=params

	for i in range(1,len(fileshot[k])):
		meanparameters=np.mean(parameters[:,0:(i+1)],axis=(1,2))
		stdparameters=np.std(parameters[:,0:(i+1)],axis=(1,2))

		for k in range(len(inttime)):
			int=(inttime[k][0:inttimedigits])
			path=pathparams+'/'+inttime[k]+'/averageRED'+'h'*(i+1)+'-c'
			if not os.path.exists(path):
				os.makedirs(path)
			np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)),meanparameters[k])
			np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)),stdparameters[k])

	# THIS CALCULATE THE GLOBAL FITTING FOR COMPARISON

	for intindex in range(1):
		path=pathparams+'/'+inttime[intindex]+'/global'
		temperature=[temperaturehot[intindex],temperaturecold[intindex]]
		files=[fileshot[intindex],filescold[intindex]]
		nmax=3
		while np.shape(temperature[0])!=():
			temperature=coleval.flatten(temperature)
			files=coleval.flatten(files)
		if not os.path.exists(path):
			os.makedirs(path)
		nmax=3
		int=(inttime[intindex][0:inttimedigits-2])	#ms
		coleval.builf_poly_coeff(temperature,files,int,path,nmax)



	# COMPARE THE FITTINGS FOR ONE PIXEL

	index=0
	color=['m','c','y','b','r','k','g','m']
	aproacheslabel=[' with window and global fitting',' with window and parameters averageREDhh-c',' with window and parameters averageREDhhh-c',' with window and parameters averageREDhhhh-c',' 1-1',' 2-1',' 3-1',' 4-1']
	aproaches=['global','averageREDhh-c','averageREDhhh-c','averageREDhhhh-c','1-1','2-1','3-1','4-1']
	for which in range(len(aproaches)):

		temperature=[]
		files=[]
		pathparams='/home/ffederic/work/irvb/2018-06-10_multiple_search_for_parameters'
		for intindex in range(1):
			for hot in range(len(fileshot[intindex])):
				for cold in range(len(filescold[intindex])):
					path=pathparams+'/'+inttime[intindex]+'/'+aproaches[which]
					temperature.append([temperaturehot[intindex,hot],temperaturecold[intindex,cold]])
					files.append([fileshot[intindex,hot],filescold[intindex,cold]])
			nmax=3
			int='1ms'	#ms

			while np.shape(temperature[0])!=():
				temperature=coleval.flatten(temperature)
				files=coleval.flatten(files)

			meancounttot=[]
			meancountstdtot=[]
			for file in files:
				data=coleval.collect_stat(file)
				meancounttot.append(np.array(data[0]))
				meancountstdtot.append(np.array(data[1]))

			meancounttot=np.array(meancounttot)
			meancountstdtot=np.array(meancountstdtot)

			shapex=np.shape(meancounttot[0])[0]
			shapey=np.shape(meancounttot[0])[1]
			score=np.zeros((nmax-1,shapex,shapey))


			for n in range(3,nmax+1):
				coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				for j in range(shapex):
					for k in range(shapey):
						if ((j == 128) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
							x=np.array(meancounttot[:,j,k])
							xerr=np.array(meancountstdtot[:,j,k])
							yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2 #this is to consider the error in the counts (standard deviation)
							yerr=np.zeros(np.shape(yerr)) #I delete the error on counts. to see only the error due to parameters
							score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
							plt.figure(k*j*(intindex+1))
							plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
							xerr=np.array([xerr for _,xerr in sorted(zip(x,xerr))])
							x=np.sort(x)
							plt.errorbar(x,(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),yerr=((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x))),fmt=color[index]+'--',label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+aproacheslabel[which])
							plt.title('Comparison for pixel horiz='+str(k)+' vert='+str(j)+' without window')
							plt.xlabel('Counts [au]')
							plt.ylabel('Temperature [°C]')
							plt.legend()
							plt.legend(loc='best')
		index+=1



	plt.xlabel('Counts [au]')
	plt.ylabel('Temperature [°C]')
	plt.legend()
	plt.legend(loc='best')
	plt.show()



	temperature=[]
	files=[]
	pathparams='/home/ffederic/work/irvb/2018-06-10_multiple_search_for_parameters'
	for intindex in range(1):
		for hot in range(4):
			for cold in range(1):
				temperature.append([temperaturehot[intindex,hot],temperaturecold[intindex,cold]])
				files.append([fileshot[intindex,hot],filescold[intindex,cold]])
		nmax=3
		int='1ms'	#ms

		while np.shape(temperature[0])!=():
			temperature=coleval.flatten(temperature)
			files=coleval.flatten(files)

		meancounttot=[]
		meancountstdtot=[]
		for file in files:
			data=coleval.collect_stat(file)
			meancounttot.append(np.array(data[0]))
			meancountstdtot.append(np.array(data[1]))

		meancounttot=np.array(meancounttot)
		meancountstdtot=np.array(meancountstdtot)

		shapex=np.shape(meancounttot[0])[0]
		shapey=np.shape(meancounttot[0])[1]
		score=np.zeros((nmax-1,shapex,shapey))

	aproaches=['global','averageREDhh-c','averageREDhhh-c','averageREDhhhh-c','1-1','2-1','3-1','4-1']
	for which in aproaches:
		for intindex in range(1):
			path=pathparams+'/'+inttime[intindex]+'/'+which

			for n in range(3,nmax+1):
				coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				for j in range(shapex):
					for k in range(shapey):
						if ((j == 128) & ((k==160) or (k==40) or (k==240))): # j is horizontal, k is vertical
							x=np.array(meancounttot[:,j,k])
							xerr=np.array(meancountstdtot[:,j,k])
							yerr=(coleval.polygen3(n)((x+xerr),*coeff[j,k,:])-coleval.polygen3(n)((x-xerr),*coeff[j,k,:]))/2 #this is to consider the error in the counts (standard deviation)
							yerr=np.zeros(np.shape(yerr)) #I delete the error on counts. to see only the error due to parameters
							score[n-2,j,k]=rsquared(temperature,coleval.polygen3(n)(x,*coeff[j,k,:]))
							plt.figure(k*j*(intindex+1))
							#plt.errorbar(x,temperature,yerr=yerr,fmt='o',markersize=5)
							xerr=np.array([xerr for _,xerr in sorted(zip(x,xerr))])
							x=np.sort(x)
							#plt.errorbar(x,(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),yerr=((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x))),fmt='r--',linewidth=1.5,label='polynomial grade='+str(n-1)+' R2='+str(np.around(score[n-2,j,k],decimals=3))+' with window and global fitting')
							plt.plot((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x)),np.divide((coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[1]).reshape(len(x)),(coleval.count_to_temp_poly2(x.reshape(1,len(x),1),np.array([[coeff[j,k,:]]]),np.array([[errcoeff[j,k,:]]]),errdata=(xerr.reshape(1,len(x),1)))[0]).reshape(len(x))),label=which)
							plt.title('Comparison of temperature relative error for pixel horiz='+str(k)+' vert='+str(j)+' without window')
							plt.xlabel('Temperature [°C]')
							plt.ylabel('Temperature error/temperature')
							plt.legend()
							plt.legend(loc='best')
	plt.show()


	meancounts=np.mean(meancounttot,axis=(-1,-2))
	for which in aproaches:
		for intindex in range(1):
			path=pathparams+'/'+inttime[intindex]+'/'+which

			for n in range(3,nmax+1):
				coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
				#temp=np.divide((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[1]),(coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[0])) #Here I consider counts error
				temp=np.divide((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff)[1]),(coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff)[0])) #Here I don't consider counts error
				temp=np.mean(temp,axis=(0,-1,-2))
				errperc=temp
				errperc=np.array([errperc for _,errperc in sorted(zip(meancounts,errperc))])

		#temp=np.sort(np.mean((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[0]),axis=(0,-1,-2))) #Here I consider counts error
		temp=np.sort(np.mean((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff)[0]),axis=(0,-1,-2))) #Here I don't consider counts error
		plt.plot(temp,errperc,label=which)
		plt.title('Mean of relative temperature error across all pixels without window')
		plt.xlabel('Temperature [°C]')
		plt.ylabel('Temperature error/temperature')
		plt.legend()
		plt.legend(loc='best')
	plt.show()

	which='averageREDhhhh-c'
	meancounts=np.mean(meancounttot,axis=(-1,-2))
	intindex=0
	path=pathparams+'/'+inttime[intindex]+'/'+which
	n=3
	coeff=np.load(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
	errcoeff=np.load(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'.npy'))
	errtemp=(coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[1])
	Ea0=np.mean(np.divide(errcoeff[:,:,0],errtemp),axis=(0,-1,-2))
	Ea0=np.array([Ea0 for _,Ea0 in sorted(zip(meancounts,Ea0))])
	Ea1=np.mean(np.divide(np.multiply(meancounttot,errcoeff[:,:,1]),errtemp),axis=(0,-1,-2))
	Ea1=np.array([Ea1 for _,Ea1 in sorted(zip(meancounts,Ea1))])
	Ea2=np.mean(np.divide(np.multiply(np.power(meancounttot,2),errcoeff[:,:,2]),errtemp),axis=(0,-1,-2))
	Ea2=np.array([Ea2 for _,Ea2 in sorted(zip(meancounts,Ea2))])
	Ec=np.mean(np.divide(np.sqrt(np.add(np.power(np.multiply(coeff[:,:,1],meancountstdtot),2),np.power(np.multiply(np.multiply(np.multiply(coeff[:,:,2],meancountstdtot),2),meancounttot),2))),errtemp),axis=(0,-1,-2))
	Ec=np.array([Ec for _,Ec in sorted(zip(meancounts,Ec))])

	temp=np.sort(np.mean((coleval.count_to_temp_poly2(np.array([meancounttot]),coeff,errcoeff,errdata=meancountstdtot)[0]),axis=(0,-1,-2)))
	plt.plot(temp,Ea0,'r',label='Ea0/Et')
	plt.plot(temp,Ea1,'b',label='Ea1/Et')
	plt.plot(temp,Ea2,'k',label='Ea2/Et')
	plt.plot(temp,Ec,'y',label='Ec/Et')
	plt.title('Mean of the error per component relative to the total error across all pixels parameters from \n'+path,size=10)
	plt.xlabel('Temperature [°C]')
	plt.ylabel('Partial temperature error/ total temperature error')
	plt.legend()
	plt.legend(loc='best')
	plt.show()


elif True:

	# 2018/07/02 this is to test the FEniCS package
	from IPython import embed; embed()


	import matplotlib.pyplot as plt

	from fenics import *

	# Create mesh and define function space
	mesh = UnitSquareMesh(8, 8)
	V = FunctionSpace(mesh, 'P', 1)

	# Define boundary condition
	u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

	def boundary(x, on_boundary):
		return on_boundary

	bc = DirichletBC(V, u_D, boundary)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Constant(-6.0)
	a = dot(grad(u), grad(v))*dx
	L = f*v*dx

	# Compute solution
	u = Function(V)
	solve(a == L, u, bc)

	plot(u)
	plot(mesh)
	plt.show()



elif False:

	# 2018/07/04 I read the SOLPS data about a detached plasma to see if I get enough radieated power to have more than 100W/m2 on the IRVB foil



	import sys
	# import copy
	# import functools
	# import math
	# import itertools
	import matplotlib.pyplot as plt
	# from matplotlib.path import Path
	# from matplotlib.cm import get_cmap
	# from matplotlib import colors
	import numpy as np
	import xarray as xr
	# import pandas as pd
	# from scipy import ndimage, interpolate, stats, linalg, integrate
	# import scipy.io as spio
	# from skimage import measure
	# import cython

	mastu_path = "/home/ffederic/work/SOLPS/seeding/seed_10"
	if not mastu_path in sys.path:
		sys.path.append(mastu_path)
	ds_puff8 = xr.open_dataset('/home/ffederic/work/SOLPS/seeding/seed_10/balance.nc', autoclose=True).load()

	grid_x = ds_puff8.crx.mean(dim='4')
	grid_y = ds_puff8.cry.mean(dim='4')
	impurity_radiation = ds_puff8.b2stel_she_bal.sum('ns')
	hydrogen_radiation = ds_puff8.eirene_mc_eael_she_bal.sum('nstra') - ds_puff8.eirene_mc_papl_sna_bal.isel(ns=1).sum('nstra') * 13.6 * 1.6e-19
	total_radiation = -hydrogen_radiation + impurity_radiation
	total_radiation_density = -np.divide(hydrogen_radiation + impurity_radiation,ds_puff8.vol)

	fig, ax = plt.subplots()
	# grid_x.plot.line(ax=ax, x='nx_plus2')
	# grid_y.plot.line(ax=ax, x='ny_plus2')
	# plt.pcolormesh(grid_x.values, grid_y.values, impurity_radiation.values)
	plt.pcolor(grid_x.values, grid_y.values, total_radiation_density.values,cmap='rainbow')
	ax.set_ylim(top=-0.5)
	ax.set_xlim(left=0.)
	plt.colorbar()

	x=np.linspace(0.55-0.075,0.55+0.075,10)
	y=-1.2+np.sqrt(0.08**2-(x-0.55)**2)
	y_=-1.2-np.sqrt(0.08**2-(x-0.55)**2)
	plt.plot(x,y,'k')
	plt.plot(x,y_,'k')
	plt.show()

elif False:

	#09/07/2018 I try here to evaluate the temperature on the foil for a given power on it

	from IPython import embed; embed()


	from fenics import *

	pathfiles='/home/ffederic/work/irvb/standoff_pinhole_best/so45_ph4_0.5MW_R0.55_Z-1.2_noback_foilpower.npy'
	data=np.load(pathfiles)

	data=rotate(data,-90,axes=(-1,-2))
	data=np.flip(data,0)
	data=np.flip(data,1)
	# plt.imshow(data,origin='lower')
	# plt.colorbar()
	# plt.show()

	ny,nx=np.shape(data)
	y,x=np.shape(foilemissivity)
	a=np.zeros((max(x,y),max(x,y)))
	a[0:y,0:x]=foilemissivity
	foilemissivityrotated=rotate(a,90,order=0)[0:x,0:y]
	foilemissivityscaled=resize(foilemissivityrotated,(ny,nx),order=0)
	a[0:y,0:x]=foilthickness
	foilthicknessrotated=rotate(a,90,order=0)[0:x,0:y]
	foilthicknessscaled=resize(foilthicknessrotated,(ny,nx),order=0)


	foilvert=0.09
	foilhoriz=foilvert*nx/ny
	dvert=foilvert/ny
	dhoriz=foilhoriz/nx
	# vertcoord=np.linspace(dvert/2,foilvert-dvert/2,num=ny)
	# horizcoord=np.linspace(dhoriz/2,foilhoriz-dhoriz/2,num=nx)
	# foilemissivityinterp = interp2d(horizcoord, vertcoord, foilemissivityrotated, kind='linear')

	# Create mesh and define function space
	mesh = RectangleMesh(Point(0, 0), Point(foilhoriz, foilvert), nx, ny)
	V = FunctionSpace(mesh, 'P', 1)
	V0 = FunctionSpace(mesh, 'DG', 0)   # Discontinuous element of degree 0
	# mesh = UnitSquareMesh(8, 8)
	# V = FunctionSpace(mesh, 'P', 1)

	# Define boundary condition
	u_D = Expression('300',degree=2)
	# u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

	def boundary_neumann(x, on_boundary):
		# return on_boundary
		tol=1e-10
		foilhoriz=0.07
		foilvert=0.09
		if on_boundary:
			if abs(x[0]-foilhoriz) < tol:
				if abs(x[1]) < foilvert/4:
					return True
			else:
				return False
		else:
			return False

	def boundary_dirichlet(x, on_boundary):
		return on_boundary



	bc = DirichletBC(V, u_D, boundary_dirichlet)

	# f = Expression(str(interpolation(x,y,data))'1 + x[0]*x[0] + 2*x[1]*x[1]', degree=0)

	E  = Function(V0) #emissivity
	tf  = Function(V0) #thickness
	P  = Function(V0) #power source term


	pos=[]
	for cell in cells(mesh):

		cell_pos=cell.get_vertex_coordinates()
		pos.append(cell_pos)

	length=len(pos)
	pos=np.array(pos)
	pos=pos.reshape((length,3,2))
	pos=np.mean(pos,axis=1)
	poshor=(pos[:,0]//dhoriz).astype(int)
	posver=(pos[:,1]//dvert).astype(int)
	# plt.plot(pos[:,0],pos[:,1],'o',markersize=1)

	for cell in cells(mesh):
		index=cell.index()
		E.vector()[index] = foilemissivityscaled[posver[index],poshor[index]]
		tf.vector()[index] = foilthicknessscaled[posver[index],poshor[index]]
		P.vector()[index] = data[posver[index],poshor[index]]
	# plot(D)


	diffusivitymult=0.4
	emissivitymult=1
	thicknessmult=1

	alpha=2*E*sigmaSB*Ptthermaldiffusivity/(Ptthermalconductivity*tf)
	alphatest=2*1*sigmaSB*Ptthermaldiffusivity/(Ptthermalconductivity*2.5/1000000)*emissivitymult*diffusivitymult*thicknessmult
	beta=Ptthermaldiffusivity/(Ptthermalconductivity*tf)
	betatest=Ptthermaldiffusivity/(Ptthermalconductivity*2.5/1000000)*diffusivitymult*thicknessmult
	T0=Constant(300)
	T04=Constant(300**4)
	cell_area=Face(mesh, 2).area()
	g = Expression('0', degree=0)

	def pow4(T,T0):
		return T**4-T0**4


	#
	# this is for steady state
	#

	T = Function(V)
	v = TestFunction(V)

	# FT = 2*E*sigmaSB*pow4(T,T0)*v*dx + Ptthermalconductivity*tf*dot(grad(T), grad(v))*dx - P*v*dx  - g*v*ds
	# FT = alpha*pow4(T,T0)*v*dx + Ptthermaldiffusivity*dot(grad(T), grad(v))*dx - beta*P*v*dx  - g*v*ds
	FT = alphatest*pow4(T,T0)*v*dx + Ptthermaldiffusivity*dot(grad(T), grad(v))*dx - betatest*P*v*dx  - g*v*ds
	# Compute solution
	solve(FT == 0, T, bc,solver_parameters={"newton_solver":{"maximum_iterations":200}})

	X=np.linspace(0, foilhoriz, nx)
	Y=np.linspace(0, foilvert, ny)
	test=[]
	for y in Y:
		for x in X:
			test.append(T(Point(x,y)))
	test=np.array(test)
	test2=test.reshape((ny,nx));plt.imshow(test2,'rainbow',origin='lower');plt.colorbar().set_label('Temp [K]');plt.xlabel('Horizontal pixels');plt.ylabel('Vertical pixels');plt.show()


	#
	#this is for time dependent
	#

	T = Function(V)
	v = TestFunction(V)

	ttot=0.1 #seconds
	num_steps=100
	dt_initial=0.0001
	dt_expression=Expression('dt',degree=0,dt=dt_initial)
	T_n = project(T0, V)
	timestep_increase=1.5

	FT = T*v*dx - T_n*v*dx + dt_expression*alpha*pow4(T,T0)*v*dx + dt_expression*Ptthermaldiffusivity*dot(grad(T), grad(v))*dx - dt_expression*beta*P*v*dx - dt_expression*g*v*ds
	T_evolution=[]
	t=0
	time=[]
	time.append(t)

	X=np.linspace(0, foilhoriz, nx)
	Y=np.linspace(0, foilvert, ny)
	test=[]
	for y in Y:
		for x in X:
			test.append(T_n(Point(x,y)))
	test=np.array(test)
	test2=test.reshape((ny,nx))
	T_evolution.append(test2)

	index=1
	for n in range(num_steps):
		dt=index*dt_initial
		dt_expression.dt=dt
		t+=dt
		time.append(t)
		index+=timestep_increase
		print('timestep ',1+n,' : t=',t,' , dt=',dt)
		# Compute solution
		solve(FT == 0, T, bc,solver_parameters={"newton_solver":{"absolute_tolerance":1e-15},"newton_solver":{"relative_tolerance":1e-13},"newton_solver":{"maximum_iterations":200}})
		T_n.assign(T)
		test=[]
		for y in Y:
			for x in X:
				test.append(T(Point(x,y)))
		test=np.array(test)
		test2=test.reshape((ny,nx))
		T_evolution.append(test2)

	i=50;plt.imshow(T_evolution[i],origin='lower');plt.title('time = '+str(time[i]));plt.colorbar();plt.show()
	ani=coleval.movie_from_data(np.array([T_evolution]),50,1,'Horizontal axis [pixles]','Vertical axis [pixles]','Temp [°C]',timesteps=time)
	ani.save('/home/ffederic/work/irvb/thermal simulations/Laser/flat_propr_x'+str(diffusivitymult)+'x'+str(emissivitymult)+'x'+str(thicknessmult)+'_laser_diam'+str(laser_spot_size*1000)+'mm_pow'+str(total_laser_power*1000)+'mW'+'.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	np.savez_compressed('/home/ffederic/work/irvb/thermal simulations/Laser/flat_propr_x'+str(diffusivitymult)+'x'+str(emissivitymult)+'x'+str(thicknessmult)+'_laser_diam'+str(laser_spot_size*1000)+'mm_pow'+str(total_laser_power*1000)+'mW',T_evolution=T_evolution,time=time)
	plt.show()



	#
	# this is for a time dependent laser spot
	#

	T = Function(V)
	v = TestFunction(V)

	total_laser_power=0.004 #W
	laser_spot_size=0.002 #I consider the 1/e2 of the gaussian beam
	max_laser_intensity=2*total_laser_power/(np.pi*((laser_spot_size/2)**2))
	laser_spot_location=[0.02,0.05]
	P_laser=Expression('max_laser_intensity*exp(-2*(pow(x[0]-laser_spot_location_x,2)+pow(x[1]-laser_spot_location_y,2))/(pow(laser_spot_sigma,2)))',degree=3,max_laser_intensity=max_laser_intensity,laser_spot_location_x=laser_spot_location[0],laser_spot_location_y=laser_spot_location[1],laser_spot_sigma=laser_spot_size/(2))
	laser_frequency=0.2
	laser_duty_cycle=0.5
	on_off=Expression('(t*laser_frequency-int(t*laser_frequency))<=laser_duty_cycle ? 1 : 0',degree=0,t=0,laser_frequency=laser_frequency,laser_duty_cycle=laser_duty_cycle)


	ttot=10 #seconds
	num_steps=1000
	dt_initial=0.02
	dt_expression=Expression('dt',degree=0,dt=dt_initial)
	T_n = project(T0, V)
	timestep_increase=2

	FT = T*v*dx - T_n*v*dx + dt_expression*alpha*pow4(T,T0)*v*dx + dt_expression*Ptthermaldiffusivity*dot(grad(T), grad(v))*dx - dt_expression*beta*P_laser*v*dx - dt_expression*g*v*ds
	T_evolution=[]
	t=0
	time=[]
	time.append(t)

	X=np.linspace(0, foilhoriz, nx)
	Y=np.linspace(0, foilvert, ny)
	test=[]
	for y in Y:
		for x in X:
			test.append(T_n(Point(x,y)))
	test=np.array(test)
	test2=test.reshape((ny,nx))
	T_evolution.append(test2)


	solve(FT == 0, T, bc,solver_parameters={"newton_solver":{"absolute_tolerance":1e-15},"newton_solver":{"relative_tolerance":1e-13},"newton_solver":{"maximum_iterations":200}})




	index=1
	for n in range(num_steps):
		dt=dt_initial
		dt_expression.dt=dt
		t+=dt
		on_off.t=t
		time.append(t)
		index+=timestep_increase
		print('timestep ',1+n,' : t=',t,' , dt=',dt)
		# Compute solution
		solve(FT == 0, T, bc,solver_parameters={"newton_solver":{"absolute_tolerance":1e-15},"newton_solver":{"relative_tolerance":1e-13},"newton_solver":{"maximum_iterations":200}})
		T_n.assign(T)
		test=[]
		for y in Y:
			for x in X:
				test.append(T(Point(x,y)))
		test=np.array(test)
		test2=test.reshape((ny,nx))
		T_evolution.append(test2)

	# i=50;plt.imshow(T_evolution[i],origin='lower');plt.title('time = '+str(time[i]));plt.colorbar();plt.show()
	ani=coleval.movie_from_data(np.array([T_evolution]),50,1,'Horizontal axis [pixles]','Vertical axis [pixles]','Temp [°C]',timesteps=time)
	ani.save('/home/ffederic/work/irvb/thermal simulations/Laser/laser_diam'+str(laser_spot_size*1000)+'mm_pow'+str(total_laser_power*1000)+'mW'+'.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	np.savez_compressed('/home/ffederic/work/irvb/thermal simulations/Laser/laser_diam'+str(laser_spot_size*1000)+'mm_pow'+str(total_laser_power*1000)+'mW',T_evolution=T_evolution,time=time)
	plt.show()






	plot(T);plt.show()
	# plot(mesh);plt.show()

	temp=T.compute_vertex_values(mesh)


	tempsolution=solution.compute_vertex_values(mesh)
	error=np.divide(temp- tempsolution,tempsolution)

	plt.plot(error);plt.show()


	mesh_points=mesh.coordinates()
	x0=mesh_points[:,0]
	x1=mesh_points[:,1]
	# u_array2=np.array([T(Point(x,y)) for x,y in mesh_points])

	test=[]
	for x,y in mesh_points:
		test.append(T(Point(x,y)))
	test=np.array(test)

	plt.scatter(x0,x1,c=test,marker='o');plt.colorbar();plt.show()


	X=np.linspace(0, foilhoriz, nx)
	Y=np.linspace(0, foilvert, ny)
	test=[]
	for y in Y:
		for x in X:
			test.append(T(Point(x,y)))
	test=np.array(test)
	test2=test.reshape((ny,nx));plt.imshow(test2,origin='lower');plt.colorbar();plt.show()


	plt.plot(temp)
	plt.plot(test)
	plt.show()



	#
	# 2018/07/16 textbook example
	#


	mesh = RectangleMesh(Point(0, 0), Point(foilhoriz, foilvert), nx, ny)
	V = FunctionSpace(mesh, 'P', 1)
	V0 = FunctionSpace(mesh, 'DG', 0)   # Discontinuous element of degree 0

	# boundary_markers = FacetFunction('size_t', mesh)


	def boundary_D(x, on_boundary):
		tol = 1E-10
		foilhoriz=0.07
		# return on_boundary and ((near(x[0], 0, tol)) or (near(x[0], foilhoriz, tol)))

		if on_boundary:
			if abs(x[0]) < tol:
				return True
			elif abs(x[0]-foilhoriz) < tol:
				return True
			else:
				return False
		else:
			return False

	T1=Expression('300',degree=0)
	T2=Expression('320',degree=0)
	T_boundary = Expression('x[0]<(foilhoriz/2) ? T1 : T2 ',degree=0,foilhoriz=foilhoriz,T1=T1,T2=T2)
	bc = DirichletBC(V, T_boundary, boundary_D)

	# class BoundaryX0(SubDomain):
	# 	tol = 1E-14
	# 	def inside(self, x, on_boundary):
	# 		return on_boundary and near(x[0], 0, tol)
	# bx0 = BoundaryX0()
	# bx0.mark(boundary_markers, 0)
	#
	# class BoundaryX1(SubDomain):
	# 	tol = 1E-14
	# 	def inside(self, x, on_boundary):
	# 		return on_boundary and near(x[0], foilhoriz, tol)
	# bx0 = BoundaryX0()
	# bx0.mark(boundary_markers, 1)
	#
	# class BoundaryX2(SubDomain):
	# 	tol = 1E-14
	# 	def inside(self, x, on_boundary):
	# 		return on_boundary and near(x[1], 0, tol)
	# bx0 = BoundaryX0()
	# bx0.mark(boundary_markers, 2)
	#
	# class BoundaryX3(SubDomain):
	# 	tol = 1E-14
	# 	def inside(self, x, on_boundary):
	# 		return on_boundary and near(x[1], foilvert, tol)
	# bx0 = BoundaryX0()
	# bx0.mark(boundary_markers, 3)


	# T2 = Expression('320',degree=1)

	# boundary_conditions = {0: {'Neumann': 0}, 1: {'Neumann': 0}, 2: {'Dirichlet': T1}, 3: {'Dirichlet', T2}}
	#
	#
	# bcs = []
	# for i in boundary_conditions:
	# 	if 'Dirichlet' in boundary_conditions[i]:
	# 		bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'], boundary_markers, i)
	# 		bcs.append(bc)

	T = Function(V)
	v = TestFunction(V)
	power = Expression('500',degree=0)
	g = Expression('0', degree=0)
	T0=Constant(300)

	def pow4(T,T0):
		return T**4-T0**4

	# FT = -Ptthermalconductivity*(2.5/1000000)*dot(grad(T), grad(v))*dx + power*v*dx - g*v*ds
	FT = -Ptthermalconductivity*(2.5/1000000)*dot(grad(T), grad(v))*dx + power*v*dx - g*v*ds - 2*1*sigmaSB*pow4(T,T0)*v*dx

	solve(FT == 0, T, bc)

	temp=T.compute_vertex_values(mesh)

	phi= Expression('power*(pow(foilhoriz,2))/(2*Ptthermalconductivity*(2.5/1000000))',degree=0,foilhoriz=foilhoriz,Ptthermalconductivity=Ptthermalconductivity,power=power)
	solution=Expression('phi*(x[0]/foilhoriz-pow(x[0]/foilhoriz,2))+T2*x[0]/foilhoriz+T1*(1-x[0]/foilhoriz)',degree=2,phi=phi,foilhoriz=foilhoriz,T1=T1,T2=T2)
	tempsolution=solution.compute_vertex_values(mesh)
	error=np.divide(temp- tempsolution,tempsolution)
	plot(T);plt.show()
	plt.plot(error);plt.show()


	#
	# test2
	#


	mesh = RectangleMesh(Point(0, 0), Point(foilhoriz, foilvert), nx, ny)
	V = FunctionSpace(mesh, 'P', 1)
	V0 = FunctionSpace(mesh, 'DG', 0)   # Discontinuous element of degree 0
	# mesh = UnitSquareMesh(8, 8)
	# V = FunctionSpace(mesh, 'P', 1)

	# Define boundary condition
	u_D = Expression('300',degree=2)
	# u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

	def boundary_neumann(x, on_boundary):
		# return on_boundary
		tol=1e-10
		foilhoriz=0.07
		foilvert=0.09
		if on_boundary:
			if abs(x[0]-foilhoriz) < tol:
				if abs(x[1]) < foilvert/4:
					return True
			else:
				return False
		else:
			return False

	def boundary_dirichlet(x, on_boundary):
		return on_boundary

	##      Parameters for a test solution
	Texp=Expression('50*exp(-1000*(pow(x[0]-0.035,2)+pow(x[1]-0.02,2)))',degree=4)
	solution=Expression('300+Texp',degree=4,Texp=Texp)
	dsolution_dx=Expression('-1000*2*(x[0]-0.035)*Texp',degree=4,Texp=Texp)
	d2solution_dx2=Expression('dsolution_dx*(-1000*2*(x[0]-0.035))+Texp*(-1000*2)',degree=4,dsolution_dx=dsolution_dx,Texp=Texp)
	dsolution_dy=Expression('-1000*2*(x[1]-0.02)*Texp',degree=4,Texp=Texp)
	d2solution_dy2=Expression('dsolution_dy*(-1000*2*(x[1]-0.02))+Texp*(-1000*2)',degree=4,dsolution_dy=dsolution_dy,Texp=Texp)
	power_expression=Expression('-Ptthermalconductivity*tf*(d2solution_dx2+d2solution_dy2)+2*1*sigmaSB*(pow(solution,4)-pow(300,4))',degree=4,d2solution_dx2=d2solution_dx2,d2solution_dy2=d2solution_dy2,solution=solution,Ptthermalconductivity=Ptthermalconductivity,tf=2.5/1000000,sigmaSB=sigmaSB)
	bctest = DirichletBC(V, solution, boundary)
	Ptest = Function(V0) #power source term for the test solution


	pos=[]
	for cell in cells(mesh):

		cell_pos=cell.get_vertex_coordinates()
		pos.append(cell_pos)

	length=len(pos)
	pos=np.array(pos)
	pos=pos.reshape((length,3,2))
	pos=np.mean(pos,axis=1)
	poshor=(pos[:,0]//dhoriz).astype(int)
	posver=(pos[:,1]//dvert).astype(int)
	# plt.plot(pos[:,0],pos[:,1],'o',markersize=1)

	for cell in cells(mesh):
		index=cell.index()

		Ptest.vector()[index] = power_expression(Point(pos[index,0],pos[index,1]))
	# plot(D)



	alphatest=2*1*sigmaSB/(Ptthermalconductivity*2.5/1000000)
	betatest=1/(Ptthermalconductivity*2.5/1000000)
	T0=Constant(300)
	T04=Constant(300**4)
	cell_area=Face(mesh, 2).area()
	g = Expression('0', degree=0)

	def pow4(T,T0):
		return T**4-T0**4

	# Define variational problem
	# T = TrialFunction(V)
	T = Function(V)
	v = TestFunction(V)


	FT = -(Ptthermalconductivity*2.5/1000000)*dot(grad(T), grad(v))*dx -2*1*sigmaSB*pow4(T,T0)*v*dx +P*v*dx


	# Compute solution
	solve(FT == 0, T, bctest,solver_parameters={"newton_solver":{"relative_tolerance":1e-10},"newton_solver":{"maximum_iterations":200}})


	plot(T);plt.show()
	# plot(mesh);plt.show()

	temp=T.compute_vertex_values(mesh)
	tempsolution=solution.compute_vertex_values(mesh)
	error=np.divide(temp- tempsolution,tempsolution)

	plt.plot(error);plt.show()


	mesh_points=mesh.coordinates()
	x0=mesh_points[:,0]
	x1=mesh_points[:,1]
	# u_array2=np.array([T(Point(x,y)) for x,y in mesh_points])


	X=np.linspace(0, foilhoriz, nx)
	Y=np.linspace(0, foilvert, ny)
	test=[]
	for y in Y:
		for x in X:
			test.append(T(Point(x,y)))
	test=np.array(test)
	test2=test.reshape((ny,nx));plt.imshow(test2,origin='lower');plt.colorbar();plt.show()



	# 2018/02/06 I check what happens after you shitch on the camera

	data_all=[]
	for path in vacuum8:
		filenames=coleval.all_file_names(path,'npy')[0]
		data=np.load(os.path.join(path,filenames))
		data_all.append(data[0])
	data_all = np.array(data_all)

	plt.plot(vacuumtime8,np.mean(data_all[:,0],axis=(1,2,3)))









	data_all = []
	for path in vacuum7:
		filenames = coleval.all_file_names(path, 'npy')[0]
		data = np.load(os.path.join(path, filenames))
		data_all.append(data[0])
	data_all = np.array(data_all)

	mean = []
	for data in data_all:
		mean.append(np.mean(data,axis=(0,1,2)))


	plt.plot(vacuumtime7, mean)










	from scipy import interpolate
	poscentred=[[250,80],[100,70],[70,200],[160,128],[250,200]]
	color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm','pink']
	# poscentred = [[160, 70],[80,70], [80, 128], [80, 200],[160,200], [250, 200],[250, 128],[250, 70],[160, 128]]
	index=np.linspace(0,len(data_all)-1,len(data_all),dtype=int)
	for i,pos in enumerate(poscentred):
		mean = []
		std=[]
		for data in data_all:
			mean.append(np.mean(data[:,pos[1],pos[0]], axis=(0)))
			std.append(2*np.std(data[:,pos[1],pos[0]],axis=0))
		plt.plot((np.array(vacuumtime8)-vacuumtime8[0]+5),mean,color[i],marker='+',label='pixel '+str(pos),linewidth=0.3)
		# plt.plot((np.array(vacuumtime8) - vacuumtime8[0] + 5), mean, '+',color=color[i], label='pixel ' + str(pos))
		# plt.plot((np.array(vacuumtime8)-vacuumtime8[0]+5)[:-1],mean[:-1],color[i],marker='+',label='pixel '+str(pos),linewidth=1)
		# plt.plot((np.array(vacuumtime8)-vacuumtime8[0]+5)[:-1],mean[:-1],color[i],marker='+')
		f = interpolate.interp1d(vacuumtime8[-2:], mean[-2:], fill_value='extrapolate')
		plt.plot((np.array(vacuumtime8)-vacuumtime8[0]+5),f(vacuumtime8),color=color[i],linestyle='--',linewidth=0.5)
		# plt.plot((np.array(vacuumtime8) - vacuumtime8[0] + 5)[:-1], f(vacuumtime8[:-1]), color=color[i], linestyle='--',linewidth=0.5)
		# plt.errorbar(vacuumtime8,mean,yerr=std,label='pixel '+str(pos))
		print(mean[-1]-mean[-2])

	plt.legend(loc='best')
	plt.xlabel('Time from camera switch on [min]')
	plt.ylabel('Counts [au]')
	plt.savefig('gna2.eps')
	plt.close()


	plt.imshow(data_all[-1,0],origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.colorbar().set_label('Counts [au]')

	for pos in poscentred:
		plt.plot(pos[0],pos[1],'+',markersize=10 ,label='pixel '+str(pos))

	plt.legend(loc='best')
	plt.savefig('gna3.eps')
	plt.close()

