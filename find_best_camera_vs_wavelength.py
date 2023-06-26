import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame
from functions.GaussFitData import doLateralfit_time_tependent
import collections

import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks, peak_prominences as get_proms
from multiprocessing import Pool,cpu_count
number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))

from pycine.color import color_pipeline, resize
from pycine.raw import read_frames
from pycine.file import read_header
from functions.fabio_add import all_file_names,find_index_of_file




if False:	# main purpose, show image contrast with wavelength
	function_of_reflection = lambda x: np.logical_or(x<4,x>5)

	fig, ax = plt.subplots(3,1,figsize=(5, 10),sharex=True)
	ax[0].grid()
	ax[1].semilogy()
	ax[1].axhline(y=16000,linestyle='--',color='k')
	ax[1].grid()
	ax[2].semilogy()
	# ax[2].axhline(y=5,linestyle='--',color='k')
	# ax[2].axhline(y=300,linestyle='--',color='k')
	ax[0].axvline(x=1.5,linestyle='--',color='k')
	ax[0].axvline(x=3,linestyle='--',color='k')
	ax[0].axvline(x=4,linestyle='--',color='k')
	ax[2].axvline(x=1.5,linestyle='--',color='k')
	ax[2].axvline(x=3,linestyle='--',color='k')
	ax[2].axvline(x=4,linestyle='--',color='k')
	ax[1].axvline(x=1.5,linestyle='--',color='k')
	ax[1].axvline(x=3,linestyle='--',color='k')
	ax[1].axvline(x=4,linestyle='--',color='k')
	ax[2].axhline(y=300/5,linestyle='--',color='k')
	ax[2].grid()
	min_temp = 20
	delta_wave = 1.6
	delta_t = 1
	ref_int_time = 0.002
	int_time = 0.001
	f_number_old = 3
	f_number_new = 3
	old_res_max = 320
	new_res_max = 320
	old_freq = 383
	new_freq = 885
	temp = []
	temp1 = []
	temp2 = []
	wave_range = np.arange(1,14,0.5)
	for start in wave_range:
		x = np.arange(start,start+delta_wave,0.0001)*1e-6
		x0=((f_number_old/f_number_new)**2)*1*int_time*(1*9.74e-05*3/0.002*3.42E-14 * np.trapz(function_of_reflection(x)*2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp)))-1)),x=x) + 3.42E-14 * np.trapz(2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp)))-1)),x=x))
		x1=((f_number_old/f_number_new)**2)*1*int_time*(1*9.74e-05*3/0.002*3.42E-14 * np.trapz(function_of_reflection(x)*2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp)))-1)),x=x) + 3.42E-14 * np.trapz(2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp+delta_t)))-1)),x=x))
		# x0=1*int_time*(1*1430/0.002 + 3.42E-14 * np.trapz(2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp)))-1)),x=x))
		# x1=1*int_time*(1*1430/0.002 + 3.42E-14 * np.trapz(2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp+delta_t)))-1)),x=x))
		# x0 += 1500
		# x1 += 1500
		# print(x1-x0)
		# print((x1-x0)/x0 * 100)
		noise = 2#(max(coleval.estimate_counts_std(x0),5)**2+max(coleval.estimate_counts_std(x1),5)**2)**0.5	# noise with no binning
		noise /= ((min(1/int_time,new_freq)/old_freq)**0.5)
		noise /= ((new_res_max/old_res_max))
		temp.append((x1-x0)/x0 * 100)
		temp1.append(x1)
		temp2.append((x1-x0)*11000/(max(x0,11000))/noise)
	ax[0].plot(wave_range,temp,label='min_t=%.3g, d_w=%.3g, d_t=%.3g, int=%.3g' %(min_temp,delta_wave,delta_t,int_time))
	ax[1].plot(wave_range,temp1,label='min_t=%.3g, d_w=%.3g, d_t=%.3g, int=%.3g' %(min_temp,delta_wave,delta_t,int_time))
	ax[2].plot(wave_range,temp2,label='min_t=%.3g, d_w=%.3g, d_t=%.3g, int=%.3g' %(min_temp,delta_wave,delta_t,int_time))
	ax[1].set_ylim(bottom=1000,top=30000)
	ax[2].set_ylim(bottom=10)

	ax[0].legend(loc='best', fontsize='xx-small')





##	Here I put the code for examining simple samples and check the performance of different camera models.
# FLIR

# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/IRVB1 2023-05-31 test'
# filenames = ['IRVB-test-000003.ats','IRVB-test-000005.ats','IRVB-test-000007.ats','test_2.ptw','IRVB-test-000004.ats','IRVB-test-000006.ats','test_1.ptw']
#
# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-06-07'
# filenames = ['0-SSD .ats','2-SSD .ats','7-SSD .ats','SSD -13.ats','SSD -19.ats','SSD -24.ats','SSD -5.ats','10-SSD .ats','3a-SSD .ats','8-SSD .ats','SSD -14.ats','SSD -1.ats','SSD -25.ats','SSD -6.ats','11-SSD .ats','3-SSD .ats','9-SSD .ats','SSD -15.ats','SSD -20.ats','SSD -26.ats','SSD -7.ats','12-SSD .ats','4-SSD .ats','SSD -10.ats','SSD -16.ats','SSD -21.ats','SSD -2.ats','SSD -8.ats','13-SSD .ats','5-SSD .ats','SSD -11.ats','SSD -17.ats','SSD -22.ats','SSD -3.ats','SSD -9.ats','1-SSD .ats','6-SSD .ats','SSD -12.ats','SSD -18.ats','SSD -23.ats','SSD -4.ats']

path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-05-31'
# filename = 'Rec-0009.ats'
filenames = ['1-SSD.ats','Rec-0011.ats','Rec-0019.ats','Rec-0027.ats','Rec-0035.ats','Untitled-1 copy.ats','2-SSD.ats','Rec-0012.ats','Rec-0020.ats','Rec-0028.ats','Rec-0036.ats','Untitled-2.ats','Rec-0006.ats','Rec-0013.ats','Rec-0021.ats','Rec-0029.ats','Rec-0037.ats','Untitled-3.ats','Rec-0007.ats','Rec-0014.ats','Rec-0022.ats','Rec-0030.ats','SSD-1.ats','Untitled-4.ats','Rec-0008.ats','Rec-0015.ats','Rec-0023.ats','Rec-0031.ats','SSD-2.ats','Rec-0009.ats','Rec-0016.ats','Rec-0024.ats','Rec-0032.ats','SSD-3.ats','Rec-0009.npz','Rec-0017.ats','Rec-0025.ats','Rec-0033.ats','SSD-4.ats','Rec-0010.ats','Rec-0018.ats','Rec-0026.ats','Rec-0034.ats','Untitled-1.ats']
# path = '/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018'
# filename = 'irvb_sample-000002.ats'

for filename in filenames:
	laser_to_analyse = path +'/' + filename
	try:
		laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
		laser_dict.allow_pickle=True
		laser_digitizer_ID = laser_dict['uniques_digitizer_ID']
		# if laser_to_analyse[-4:]=='.ptw':
		# 	test = laser_dict['discarded_frames']
	except:
		print('missing '+laser_to_analyse[:-4]+'.npz'+' file. rigenerated')
		if laser_to_analyse[-4:]=='.ats':
			full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse)
		else:
			full_saved_file_dict = coleval.ptw_to_dict(laser_to_analyse,max_time_s = 30)
		np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
		# laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
		# laser_dict.allow_pickle=True
# full_saved_file_dict = dict(laser_dict)
