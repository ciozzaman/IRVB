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
import collect_and_eval as coleval
import scipy.stats
import os,sys
import matplotlib.animation as animation
import warnings
from scipy.interpolate import RectBivariateSpline,interp2d
import concurrent.futures


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

def costant(x, *params):
	import numpy as np

	c = params[0]
	len=np.ones((np.shape(x)))
	return c*len

# def square_wave(x, *params):
# 	import numpy as np
#
# 	A = params[0]
# 	phase = params[2]
# 	freq = params[1]
# 	onoff=(((np.array(x)*freq+phase) % 1)>=0.5)
# 	return A*onoff
#
# def square_wave(x, *params):
# 	from scipy import signal
# 	import numpy as np
#
# 	A = params[0]
# 	phase = params[1]
# 	freq = params[2]
#
# 	return A*signal.square(2*np.pi*x*freq+phase)



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
temperature10=[39.6,37.4,36.8,36.0,35.1,34.1,32.7,31.5,30.2,29.6,29.5,29.0,28.6,28.1,27.7,27.3,26.9,26.6,26.4,26.2,26.0]
#temperature10=[29.6,29.5,29.0,28.6,28.1,27.7,27.3,26.9,26.6,26.4,26.2,26.0]
files10=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000034']
#files10=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000034']

#integration time 1ms, frequency 383Hz
temperature11=[31.8,30.8,29.9,29.2,28.7,28.3,27.9,27.5,27.2,26.8,26.5,26.3,26.1,25.9]
files11=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000035']

#integration time 2ms, frequency 383Hz
temperature12=[41.3,40.1,39.1,38.1,37.1,36.4,35.7,34.9,33.8,33.3,32.5,31.8,30.9,30.1,29.5,28.8,28.1,27.6,27.1,26.5,26.1]
#temperature12=[30.1,29.5,28.8,28.1,27.6,27.1,26.5,26.1]
files12=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000081']
#files12=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000081']

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
voltlaser15=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser15=[10,50,100,10,50,100,10,50,100]
dutylaser15=[0.02,0.02,0.02,0.05,0.05,0.05,0.1,0.1,0.1]


# Laser experiments 08/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) partially defocused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
laser16=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000030']
voltlaser16=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser16=[10,50,100,10,50,100,10,50,100]
dutylaser16=[0.02,0.02,0.02,0.05,0.05,0.05,0.1,0.1,0.1]


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
vacuum4=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000004*2','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000029','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000030','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000031','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000032','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000033','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000034','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000035','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000036','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000037','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000038','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000039','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000040','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000041','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000042','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000043','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000044','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000045']
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


# files=[vacuum5]
# files=coleval.flatten_full(files)
# for file in files:
# 	try:
# 		coleval.collect_subfolderfits(file)
# 		coleval.evaluate_back(file)
# 	except:
# 		print('error in '+file)
# 	coleval.save_timestamp(file)


# file_ref='/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018'
# files=[laser25[0]]
# files=coleval.flatten_full(files)
# # files=[laser32[3:]]
# files=coleval.flatten_full(files)
# for file in files:
# 	try:
# 		coleval.collect_subfolderfits(file)
# 	except:
# 		print('error in '+file)
# 	# #
# 	coleval.save_timestamp(file)
# 	# try:
# 	coleval.search_background_timestamp(file,file_ref)
# 	# except:
# 	# 	print('error in '+file)


# def run(file_from_outside):
# 	file_ref='/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018'
# 	coleval.search_background_timestamp(file_from_outside,file_ref)
# 	coleval.collect_subfolderfits(file)



def run(file_from_outside):
	try:
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

		file_from_outside=[file_from_outside]
		for pathfiles in file_from_outside:
			type='.npy'
			filenames=coleval.all_file_names(pathfiles,type)[0]
			if (pathfiles in laser12):
				framerate=994
				datashort=np.load(os.path.join(pathfiles,filenames))
				data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
				data[:,:,64:96,:]=datashort
				type_of_experiment='low duty cycle partially defocused'
				poscentred=[[15,80],[40,75],[80,85]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
				inttime=1
			elif (pathfiles in coleval.flatten_full([laser15,laser16,laser21,laser23])):
				framerate=994
				datashort=np.load(os.path.join(pathfiles,filenames))
				data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
				data[:,:,64:128,:]=datashort
				type_of_experiment='low duty cycle partially defocused'
				poscentred=[[15,80],[40,75],[80,85]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
				inttime=1
			elif (pathfiles in coleval.flatten_full([laser24,laser26,laser27,laser28,laser29,laser31])):
				framerate=1974
				datashort=np.load(os.path.join(pathfiles,filenames))
				data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
				data[:,:,64:128,128:]=datashort
				type_of_experiment='low duty cycle partially defocused'
				poscentred=[[15,80],[40,75],[80,85]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/0.5ms383Hz/average'
				inttime=0.5
			elif (pathfiles in [vacuum1[1],vacuum1[3]]):
				framerate=994
				datashort=np.load(os.path.join(pathfiles,filenames))
				data=np.multiply(6000,np.ones((1,np.shape(datashort)[1],256,320)))
				data[:,:,64:96,:]=datashort
				type_of_experiment='low duty cycle partially defocused'
				poscentred=[[15,80],[40,75],[80,85]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
				inttime=1
			elif (pathfiles in laser11):
				framerate=383
				data=np.load(os.path.join(pathfiles,filenames))
				type_of_experiment='partially defocused'
				poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
				inttime=1
			elif (pathfiles in coleval.flatten_full([laser25,laser30])):
				framerate=383
				data=np.load(os.path.join(pathfiles,filenames))
				type_of_experiment='focused'
				poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/2ms383Hz/average'
				inttime=2
			else:
				framerate=383
				data=np.load(os.path.join(pathfiles,filenames))
				type_of_experiment='focused'
				poscentred=[[15,80],[80,80],[70,200],[160,133],[250,200]]
				pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
				inttime=1


				params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
				errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))



			if (pathfiles in pathfiles in coleval.flatten_full([laser12])):
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
				basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)
			elif (pathfiles in pathfiles in coleval.flatten_full([laser10[8:]])):
				# Added 19/08/2018 to use the best data available for building the reference
				type='_reference.npy'
				filenames_basecounts=coleval.all_file_names(laser10[7],type)[0]
				basecounts=np.load(os.path.join(laser10[7],filenames_basecounts))
				filenames=filenames[:-4]+'_reference'+filenames[-4:]
				data_mean_difference=[]
				data_mean_difference_std=[]
				# poscentred=[[70,70],[160,133],[250,200]]
				# poscentred=[[60,12],[170,12],[290,12]]
				for pos in poscentred:
					for a in [5,10]:
						temp1=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
						temp1std=np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
						temp2=np.mean(basecounts[pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))
						temp2std=0
						data_mean_difference.append(temp1-temp2)
						data_mean_difference_std.append(temp1std+temp2std)
				data_mean_difference=np.array(data_mean_difference)
				data_mean_difference_std=np.array(data_mean_difference_std)
				guess=[1]
				base_counts_correction,temp2=curve_fit(costant,np.linspace(1,len(data_mean_difference),len(data_mean_difference)),data_mean_difference, sigma=data_mean_difference_std,p0=guess, maxfev=100000000)
				print('background correction = '+str(int(base_counts_correction*1000)/1000))

				datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]]+base_counts_correction,params,errparams)
				datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
				datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
				basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)
			elif (pathfiles in pathfiles in coleval.flatten_full([laser11[8:]])):
				# Added 19/08/2018 to use the best data available for building the reference
				type='_reference.npy'
				filenames_basecounts=coleval.all_file_names(laser11[7],type)[0]
				basecounts=np.load(os.path.join(laser11[7],filenames_basecounts))
				filenames=filenames[:-4]+'_reference'+filenames[-4:]
				data_mean_difference=[]
				data_mean_difference_std=[]
				# poscentred=[[70,70],[160,133],[250,200]]
				# poscentred=[[60,12],[170,12],[290,12]]
				for pos in poscentred:
					for a in [5,10]:
						temp1=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2,-3))
						temp1std=np.std(np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2)))
						temp2=np.mean(basecounts[pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))
						temp2std=0
						data_mean_difference.append(temp1-temp2)
						data_mean_difference_std.append(temp1std+temp2std)
				data_mean_difference=np.array(data_mean_difference)
				data_mean_difference_std=np.array(data_mean_difference_std)
				guess=[1]
				base_counts_correction,temp2=curve_fit(costant,np.linspace(1,len(data_mean_difference),len(data_mean_difference)),data_mean_difference, sigma=data_mean_difference_std,p0=guess, maxfev=100000000)
				print('background correction = '+str(int(base_counts_correction*1000)/1000))

				datatemp,errdatatemp=coleval.count_to_temp_poly2([[basecounts]]+base_counts_correction,params,errparams)
				datatemprot=rotate(datatemp,foilrotdeg,axes=(-1,-2))
				datatempcrop=datatemprot[:,:,foildw:foilup,foillx:foilrx]
				basetemp=np.mean(datatempcrop[0,:,1:-1,1:-1],axis=0)

			elif (pathfiles in pathfiles in coleval.flatten_full([laser15,laser16,laser21,laser23])):
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
			elif (pathfiles in pathfiles in coleval.flatten_full([laser24,laser26,laser27,laser28,laser29,laser31])):
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

		return 'done'

	except:
		print('error in '+file)
		return 'error'

files=[laser24[4:15],laser25,laser26,laser27,laser28,laser29]
files=coleval.flatten_full(files)


with concurrent.futures.ProcessPoolExecutor() as executor:
	for file_,output in zip(files,executor.map(run, files)):
		print(file_+' '+output)
