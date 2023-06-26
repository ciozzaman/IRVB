# 17/05/23	read Cyd cowley results on the stability of the thermal front

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


import netCDF4
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']


path = '/home/ffederic/work/analysis_scripts/from_Cyd_inner_leg_front_stability'
shots = [45468, 45469, 45470, 45473]

shot_index = 3
f = netCDF4.Dataset(path+'/'+'DLS_output_'+str(shots[shot_index])+'.nc')
f_out = netCDF4.Dataset(path+'/'+'DLS_output_'+str(shots[shot_index])+'outer.nc')

time = f['Time'][:].data
sFrontPar = f['sFrontPar'][:].data
sFrontPol = f['sFrontPol'][:].data
C = f['C'][:].data

time_out = f_out['Time'][:].data
sFrontPar_out = f_out['sFrontPar'][:].data
sFrontPol_out = f_out['sFrontPol'][:].data
C_out = f_out['C'][:].data

name = 'IRVB-MASTU_shot-'+str(shots[shot_index])+'.ptw'
path = '/home/ffederic/work/irvb/MAST-U/'
shot_list = get_data(path+'shot_list2.ods')

temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
for i in range(1,len(shot_list['Sheet1'])):
	if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
		date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
		break
i_day,day = 0,str(date.date())
laser_to_analyse=path+day+'/'+name

pass_number = 0
full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
full_saved_file_dict_FAST.allow_pickle=True
full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)

nu_cowley = full_saved_file_dict_FAST['multi_instrument'].all()['nu_cowley']
grid_resolution = 2	# cm
inverted_dict = full_saved_file_dict_FAST['first_pass'].all()['inverted_dict']
time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']

plt.figure(figsize=(10,6))
plt.title('Detachment front stability')
plt.xlabel(r'$\hat{L}_f$  [au]')
plt.ylabel(r'$C_1$ [au]')
plt.grid()
for i_,t in enumerate([0.386,0.511,0.605]):
	i_t = np.abs(time-t).argmin()
	plt.plot(sFrontPol[i_t]/np.max(sFrontPol[i_t]),C[i_t],color=color[i_],linestyle='-',label='%.3gms' %(t*1000))
	i_t = np.abs(time_out-t).argmin()
	plt.plot(sFrontPol_out[i_t]/np.max(sFrontPol_out[i_t]),C_out[i_t],color=color[i_],linestyle='--')
plt.legend(loc='best', fontsize='x-small')
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/analysis_scripts/from_Cyd_inner_leg_front_stability'+'/' +str(shots[shot_index]) +'_front_stability' +'.eps', bbox_inches='tight')
plt.close('all')



#
