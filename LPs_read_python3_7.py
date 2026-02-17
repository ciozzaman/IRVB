import numpy as np
from scipy import misc
import scipy.signal
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy.optimize import curve_fit
from scipy.optimize import newton_krylov	# added 2018-11-13 to replace Fenics for heat transfer simulations



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm	# added 2018-11-17 to allow logarithmic scale plots
matplotlib.rcParams.update({'font.size': 15})	# added 2020-05-15 to have a larger font by default
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import math
import statistics as s
import csv
import scipy.stats
import os,sys
import xarray as xr	# added 2018-11-05 to allow importing data from SOLPS

os.chdir("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval")
import warnings
from scipy.interpolate import RectBivariateSpline,interp2d,interp1d
import peakutils
import pickle
import collections
import collect_and_eval as coleval
import copy as cp
from scipy.interpolate import interp1d,splrep,splev,bisplrep,bisplev,griddata
from scipy.signal import medfilt,find_peaks, peak_prominences as get_proms
from scipy.ndimage import convolve,median_filter,generic_filter
from uncertainties import ufloat,unumpy
from scipy.special import hyp1f1
from scipy.linalg import svd
from uncertainties import ufloat,unumpy,correlated_values
from uncertainties.unumpy import exp,nominal_values,std_devs,sqrt
import time as tm
from datetime import datetime
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))

# to read the amount of gas supplied to the plasma
from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe, probe_array, compare_shots
from mastu_exhaust_analysis.calc_ne_bar import calc_ne_bar
from mastu_exhaust_analysis.calc_w_dot import calc_w_dot
from mastu_exhaust_analysis.calc_pohm import calc_pohm
from pyexcel_ods import get_data

# LPs_dict=np.load('/home/ffederic/work/irvb/MAST-U/2025-08-08/IRVB-MASTU_shot-52493_temp_LPs_dict'+'.npz')
print('Temp source: '+sys.argv[1]+'.npz')
LPs_dict=np.load(sys.argv[1]+'.npz')
LPs_dict.allow_pickle=True
shotnumber = int(LPs_dict['shotnumber'])
LP_file_path = str(LPs_dict['LP_file_path'])
LP_file_type = str(LPs_dict['LP_file_type'])
time_full_binned_crop = LPs_dict['time_full_binned_crop']
badLPs_V0 = LPs_dict['badLPs_V0']
efit_reconstruction = LPs_dict['efit_reconstruction'].item()

LPs_dict = dict([])


from mast.geom.geomTileSurfaceUtils import get_nearest_s_coordinates_mastu,get_s_coords_tables_mastu	# added 19/02/2025
import pyuda
from mastu_exhaust_analysis.eich_gui import prepare_data_for_Eich_fit_fn_time,Eich_fit_fn_time
from mastu_exhaust_analysis.flux_expansion import calc_fpol
fpol=calc_fpol(shotnumber,max_s=20e-2,number_steps=30)
lambda_q_compression = (fpol['fpol_lower']/np.sin(fpol['poloidal_angle_lower']))[:,0]
lambda_q_compression_lower_interp = interp1d(fpol['time'],lambda_q_compression,fill_value="extrapolate",bounds_error=False)
lambda_q_compression = (fpol['fpol_upper']/np.sin(fpol['poloidal_angle_upper']))[:,0]
lambda_q_compression_upper_interp = interp1d(fpol['time'],lambda_q_compression,fill_value="extrapolate",bounds_error=False)

# to go from wall coord to s
client=pyuda.Client()
limiter=client.geometry("/limiter/efit",50000, no_cal=False)
limiter_r=limiter.data.R
limiter_z=limiter.data.Z
s_lookup=get_s_coords_tables_mastu(limiter_r, limiter_z, ds=1e-4, debug_plot=False)
coleval.reset_connection(client)
del client

try:
	# Eich_fit_fn_time works only in python 3.7 because of the function iminuit.Struct
	if sys.version_info.major==3 and sys.version_info.minor==7:
		lambda_q_determination = True
	else:
		print('LPs can be real properly only with Pithon 3.7, and it was not used here, so this step is skipped')
		lambda_q_determination = False
		# sblu=sgne	# I want an error to occour, as LPs can be real properly only with Pithon 3.7
	# try:
	# 	fdir = coleval.uda_transfer(shotnumber,'elp',extra_path='/0'+str(shotnumber)[:2])
	# 	lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path = os.path.split(fdir)[0])
	# 	os.remove(fdir)
	# except:
	# 	lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path=LP_file_path)
	try:	# considering that Peter just (12/02/2025) renwed the analisys code and is saving the files locally, I look at that first now
		lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path=LP_file_path)
	except Exception as e:
		logging.exception('with error (LP section): ' + str(e))
		fdir = coleval.uda_transfer(shotnumber,'elp',extra_path='/0'+str(shotnumber)[:2])
		lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path = os.path.split(fdir)[0])
		os.remove(fdir)
	if True:	# use ot Peter Ryan function to combine data within time slice and to filter when strike point is on a dead channel
		temp = np.logical_and(time_full_binned_crop>output_contour1['time'][0][0].min(),time_full_binned_crop<output_contour1['time'][0][0].max())
		trange = tuple(map(list, np.array([time_full_binned_crop[temp] - np.mean(np.diff(time_full_binned_crop))/2,time_full_binned_crop[temp] + np.mean(np.diff(time_full_binned_crop))/2]).T))
		# LP_lambdaq_time = np.mean(trange,axis=1)
		try:
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T1','T2','T3','T4','T5'],show=False)#,log_plot_data=True)
			temp = output_contour1['y'][0][0]
			temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
			temp[np.isnan(temp)] = 0
			if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
				for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
					if probe_name in badLPs_V0:
						# print(probe_name)
						temp[:,i_] = 0
			# s10_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[5,1]),axis=0)	# threshold for broken probes
			if shotnumber<46191:
				s10_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
			else:	# 2025/10/24 after MU01 if a probe is bad then it is left in the data as blank, no need of an external file
				s10_lower_good_probes = np.ones((np.shape(temp)[1])).astype(bool)
			s10_lower_s = output_contour1['s'][0][0]
			s10_lower_r = output_contour1['R'][0][0]
			s10_lower_z = output_contour1['Z'][0][0]
			if False:
				plt.figure()
				plt.imshow(temp, extent=[output_contour1['R'][0][0].min(),output_contour1['R'][0][0].max(),output_contour1['time'][0][0].max(),output_contour1['time'][0][0].min()])
				plt.gca().set_aspect(0.5)
			output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=10, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
			s10_lower_jsat = []
			s10_lower_jsat_sigma = []
			s10_lower_jsat_r = []
			s10_lower_jsat_s = []
			for i in range(len(output)):
				s10_lower_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
				s10_lower_jsat_sigma.append(output[i]['y_error'][0][0])
				s10_lower_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
				s10_lower_jsat_s.append(output[i]['s'][0][0])
			if lambda_q_determination:	# section to calculate OMP lambda_q
				output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=10, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
				x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
				if False:
					tag_cycle='elp'
					shot_cycle = shotnumber
					lp_data = LangmuirProbe(shot=None,toroidal_tilt=True,filename=LP_file_path+'/'+tag_cycle+'0'+str(shot_cycle)+'.nc')
				fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

				lambda_q_compression = lambda_q_compression_lower_interp(time_all_finite)
				lambda_q_10_lower = fit_store['lambda_q']/lambda_q_compression
				if not len(lambda_q_10_lower)==0:	# this means that there was no data to fit
					lambda_q_10_lower = lambda_q_10_lower[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_10_lower_sigma = fit_store['lambda_q_err']/lambda_q_compression
					lambda_q_10_lower_sigma = lambda_q_10_lower_sigma[fit_store['acceptable_fit'].astype(bool)]
					time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_10_lower = [lambda_q_10_lower[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
					lambda_q_10_lower_sigma = [lambda_q_10_lower_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
				else:
					lambda_q_10_lower = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_10_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				if False:
					# plt.figure()
					# for i in range(len(Eich)):
					# 	try:
					# 		i = i*3
					# 		gna, = plt.plot(Eich[i]['R'][0][0][0],Eich[i]['y'][0][0][0],'--',label=r'$\lambda_q =$ %.3gmm, time=%.3gms' %(Eich[i]['lambda_q'][0][0][0]*1000,np.nanmean(Eich[i]['time'][0][0][0])*1000))
					# 		# plt.plot(output[i]['R'][0][0],output[i]['y'][0][0],'--',color=gna.get_color())
					# 		plt.errorbar(output[i]['R'][0][0],output[i]['y'][0][0],yerr=output[i]['y_error'][0][0],color=gna.get_color())
					# 	except:
					# 		pass
					# plt.legend()
					from mastu_exhaust_analysis.eich_gui import prepare_data_for_Eich_fit_fn_time,Eich_fit_fn_time,Eich_fit_GUI
					fit_store_edit, start_limits_store_edit=Eich_fit_GUI(time_all_finite,x_all_finite,y_all_finite,y_err_all_finite,fit_store,start_limits_store)
		except:
			print('s10_lower failed')
			s10_lower_good_probes = np.zeros((10)).astype(bool)
			s10_lower_s = np.zeros((10))
			s10_lower_r = np.zeros((10))
			s10_lower_jsat = np.zeros((len(trange),10))
			s10_lower_jsat_sigma = np.zeros((len(trange),10))
			s10_lower_jsat_r = np.zeros((len(trange),10))
			s10_lower_jsat_s = np.zeros((len(trange),10))
			lambda_q_10_lower = np.ones((len(time_full_binned_crop)))*np.nan
			lambda_q_10_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		plt.close('all')

		try:
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=4, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)#,log_plot_data=True)
			# output_contour1=lp_data.plot_tile_profile(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=4, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=True)
			# output,Eich=compare_shots(filepath='/common/uda-scratch/pryan/',shot=48561,bin_x_step=1e-3,bad_probes=None,trange=[0.5,0.55],divertor='lower', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False)

			temp = output_contour1['y'][0][0]
			temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
			temp[np.isnan(temp)] = 0
			if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
				for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
					if probe_name in badLPs_V0:
						temp[:,i_] = 0
			# s4_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
			if shotnumber<46191:
				s4_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
			else:	# 2025/10/24 after MU01 if a probe is bad then it is left in the data as blank, no need of an external file
				s4_lower_good_probes = np.ones((np.shape(temp)[1])).astype(bool)
			s4_lower_s = output_contour1['s'][0][0]
			s4_lower_r = output_contour1['R'][0][0]
			s4_lower_z = output_contour1['Z'][0][0]
			output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=4, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
			s4_lower_jsat = []
			s4_lower_jsat_sigma = []
			s4_lower_jsat_r = []
			s4_lower_jsat_s = []
			for i in range(len(output)):
				s4_lower_jsat.append(output[i]['y'][0][0])	# here there are only small probes
				s4_lower_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only small probes
				s4_lower_jsat_r.append(output[i]['R'][0][0])	# here there are only small probes
				s4_lower_jsat_s.append(output[i]['s'][0][0])
			if lambda_q_determination:	# section to calculate OMP lambda_q
				output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=4, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
				x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
				fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

				lambda_q_compression = lambda_q_compression_lower_interp(time_all_finite)
				lambda_q_4_lower = fit_store['lambda_q']/lambda_q_compression
				if not len(lambda_q_4_lower)==0:	# this means that there was no data to fit
					lambda_q_4_lower = lambda_q_4_lower[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_4_lower_sigma = fit_store['lambda_q_err']/lambda_q_compression
					lambda_q_4_lower_sigma = lambda_q_4_lower_sigma[fit_store['acceptable_fit'].astype(bool)]
					time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_4_lower = [lambda_q_4_lower[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
					lambda_q_4_lower_sigma = [lambda_q_4_lower_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
				else:
					lambda_q_4_lower = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_4_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		except:
			print('s4_lower failed')
			s4_lower_good_probes = np.zeros((10)).astype(bool)
			s4_lower_s = np.zeros((10))
			s4_lower_r = np.zeros((10))
			s4_lower_z = np.zeros((10))
			s4_lower_jsat = np.zeros((len(trange),10))
			s4_lower_jsat_sigma = np.zeros((len(trange),10))
			s4_lower_jsat_r = np.zeros((len(trange),10))
			s4_lower_jsat_s = np.zeros((len(trange),10))
			lambda_q_4_lower = np.ones((len(time_full_binned_crop)))*np.nan
			lambda_q_4_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		plt.close('all')

		try:
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=4, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
			temp = output_contour1['y'][0][0]
			temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
			temp[np.isnan(temp)] = 0
			if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
				for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
					if probe_name in badLPs_V0:
						temp[:,i_] = 0
						# print(i_)
			# s4_upper_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
			if shotnumber<46191:
				s4_upper_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
			else:	# 2025/10/24 after MU01 if a probe is bad then it is left in the data as blank, no need of an external file
				s4_upper_good_probes = np.ones((np.shape(temp)[1])).astype(bool)
			s4_upper_s = output_contour1['s'][0][0]
			s4_upper_r = output_contour1['R'][0][0]
			s4_upper_z = output_contour1['Z'][0][0]
			output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=4, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
			s4_upper_jsat = []
			s4_upper_jsat_sigma = []
			s4_upper_jsat_r = []
			s4_upper_jsat_s = []
			for i in range(len(output)):
				s4_upper_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
				s4_upper_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only standard probes
				s4_upper_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
				s4_upper_jsat_s.append(output[i]['s'][0][0])
			if lambda_q_determination:	# section to calculate OMP lambda_q
				output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=4, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
				x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
				fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

				lambda_q_compression = lambda_q_compression_upper_interp(time_all_finite)
				lambda_q_4_upper = fit_store['lambda_q']/lambda_q_compression
				if not len(lambda_q_4_upper)==0:	# this means that there was no data to fit
					lambda_q_4_upper = lambda_q_4_upper[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_4_upper_sigma = fit_store['lambda_q_err']/lambda_q_compression
					lambda_q_4_upper_sigma = lambda_q_4_upper_sigma[fit_store['acceptable_fit'].astype(bool)]
					time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_4_upper = [lambda_q_4_upper[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
					lambda_q_4_upper_sigma = [lambda_q_4_upper_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
				else:
					lambda_q_4_upper = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_4_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		except:
			print('s4_upper failed')
			s4_upper_good_probes = np.zeros((10)).astype(bool)
			s4_upper_s = np.zeros((10))
			s4_upper_r = np.zeros((10))
			s4_upper_jsat = np.zeros((len(trange),10))
			s4_upper_jsat_sigma = np.zeros((len(trange),10))
			s4_upper_jsat_r = np.zeros((len(trange),10))
			s4_upper_jsat_s = np.zeros((len(trange),10))
			lambda_q_4_upper = np.ones((len(time_full_binned_crop)))*np.nan
			lambda_q_4_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		plt.close('all')

		try:
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4'],show=False)
			temp = output_contour1['y'][0][0]
			temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
			temp[np.isnan(temp)] = 0
			if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
				for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
					if probe_name in badLPs_V0:
						temp[:,i_] = 0
			# s10_upper_std_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
			# s10_upper_std_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
			if shotnumber<46191:
				s10_upper_std_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
			else:	# 2025/10/24 after MU01 if a probe is bad then it is left in the data as blank, no need of an external file
				s10_upper_std_good_probes = np.ones((np.shape(temp)[1])).astype(bool)
			s10_upper_std_s = output_contour1['s'][0][0]
			s10_upper_std_r = output_contour1['R'][0][0]
			s10_upper_std_z = output_contour1['Z'][0][0]
			output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-4,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4'],time_combine=True,show=False,Eich_fit=False)
			s10_upper_std_jsat = []
			s10_upper_std_jsat_sigma = []
			s10_upper_std_jsat_r = []
			s10_upper_std_jsat_s = []
			for i in range(len(output)):
				s10_upper_std_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
				s10_upper_std_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only standard probes
				s10_upper_std_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
				s10_upper_std_jsat_s.append(output[i]['s'][0][0])
			if lambda_q_determination:	# section to calculate OMP lambda_q
				output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4'],time_combine=True,show=False,Eich_fit=False)
				x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
				fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

				lambda_q_compression = lambda_q_compression_upper_interp(time_all_finite)
				lambda_q_10_upper = fit_store['lambda_q']/lambda_q_compression
				if not len(lambda_q_10_upper)==0:	# this means that there was no data to fit
					lambda_q_10_upper = lambda_q_10_upper[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_10_upper_sigma = fit_store['lambda_q_err']/lambda_q_compression
					lambda_q_10_upper_sigma = lambda_q_10_upper_sigma[fit_store['acceptable_fit'].astype(bool)]
					time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
					lambda_q_10_upper = [lambda_q_10_upper[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
					lambda_q_10_upper_sigma = [lambda_q_10_upper_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
				else:
					lambda_q_10_upper = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_10_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		except:
			print('s10_upper_std failed')
			s10_upper_std_good_probes = np.zeros((10)).astype(bool)
			s10_upper_std_s = np.zeros((10))
			s10_upper_std_r =np.zeros((10))
			s10_upper_std_jsat = np.zeros((len(trange),10))
			s10_upper_std_jsat_sigma = np.zeros((len(trange),10))
			s10_upper_std_jsat_r = np.zeros((len(trange),10))
			s10_upper_std_jsat_s = np.zeros((len(trange),10))
			lambda_q_10_upper = np.ones((len(time_full_binned_crop)))*np.nan
			lambda_q_10_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
		plt.close('all')

		try:
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['T5'],show=False)
			temp = output_contour1['y'][0][0]
			temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
			temp[np.isnan(temp)] = 0
			if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
				for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
					if probe_name in badLPs_V0:
						temp[:,i_] = 0
			# s10_upper_large_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
			# s10_upper_large_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
			if shotnumber<46191:
				s10_upper_large_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
			else:	# 2025/10/24 after MU01 if a probe is bad then it is left in the data as blank, no need of an external file
				s10_upper_large_good_probes = np.ones((np.shape(temp)[1])).astype(bool)
			s10_upper_large_s = output_contour1['s'][0][0]
			s10_upper_large_r = output_contour1['R'][0][0]
			s10_upper_large_z = output_contour1['Z'][0][0]
			output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='s',tiles=['T5'],time_combine=True,show=False,Eich_fit=False)
			s10_upper_large_jsat = []
			s10_upper_large_jsat_sigma = []
			s10_upper_large_jsat_r = []
			s10_upper_large_jsat_s = []
			for i in range(len(output)):
				s10_upper_large_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
				s10_upper_large_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only standard probes
				s10_upper_large_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
				s10_upper_large_jsat_s.append(output[i]['s'][0][0])
		except:
			print('s10_upper_large')
			s10_upper_large_good_probes = np.zeros((10)).astype(bool)
			s10_upper_large_s = np.zeros((10))
			s10_upper_large_r = np.zeros((10))
			s10_upper_large_jsat = np.zeros((len(trange),10))
			s10_upper_large_jsat_sigma = np.zeros((len(trange),10))
			s10_upper_large_jsat_r = np.zeros((len(trange),10))
			s10_upper_large_jsat_s = np.zeros((len(trange),10))
		plt.close('all')

		if False:	# plot of good vs bad probes
			fig, ax = plt.subplots( 2,1,figsize=(10, 8), squeeze=False,sharex=True)

			ax[0,0].plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			ax[0,0].plot(s10_lower_r[np.logical_not(s10_lower_good_probes)],s10_lower_z[np.logical_not(s10_lower_good_probes)],'xr',fillstyle='none')
			ax[0,0].plot(s4_lower_r[np.logical_not(s4_lower_good_probes)],s4_lower_z[np.logical_not(s4_lower_good_probes)],'xr',fillstyle='none')
			ax[0,0].plot(s4_upper_r[np.logical_not(s4_upper_good_probes)],s4_upper_z[np.logical_not(s4_upper_good_probes)],'xr',fillstyle='none')
			ax[0,0].plot(s10_upper_std_r[np.logical_not(s10_upper_std_good_probes)],s10_upper_std_z[np.logical_not(s10_upper_std_good_probes)],'xr',fillstyle='none')
			ax[0,0].plot(s10_upper_large_r[np.logical_not(s10_upper_large_good_probes)],s10_upper_large_z[np.logical_not(s10_upper_large_good_probes)],'xr',fillstyle='none')
			ax[0,0].plot(s10_lower_r[s10_lower_good_probes],s10_lower_z[s10_lower_good_probes],'og',fillstyle='none')
			ax[0,0].plot(s4_lower_r[s4_lower_good_probes],s4_lower_z[s4_lower_good_probes],'og',fillstyle='none')
			ax[0,0].plot(s4_upper_r[s4_upper_good_probes],s4_upper_z[s4_upper_good_probes],'og',fillstyle='none')
			ax[0,0].plot(s10_upper_std_r[s10_upper_std_good_probes],s10_upper_std_z[s10_upper_std_good_probes],'og',fillstyle='none')
			ax[0,0].plot(s10_upper_large_r[s10_upper_large_good_probes],s10_upper_large_z[s10_upper_large_good_probes],'og',fillstyle='none')
			ax[0,0].set_ylim(bottom=1.1,top=2.2)
			ax[0,0].set_xlim(right=1.8)
			ax[0,0].set_aspect('equal', adjustable='box')

			ax[1,0].plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			ax[1,0].plot(s10_lower_r[np.logical_not(s10_lower_good_probes)],s10_lower_z[np.logical_not(s10_lower_good_probes)],'xr',fillstyle='none')
			ax[1,0].plot(s4_lower_r[np.logical_not(s4_lower_good_probes)],s4_lower_z[np.logical_not(s4_lower_good_probes)],'xr',fillstyle='none')
			ax[1,0].plot(s4_upper_r[np.logical_not(s4_upper_good_probes)],s4_upper_z[np.logical_not(s4_upper_good_probes)],'xr',fillstyle='none')
			ax[1,0].plot(s10_upper_std_r[np.logical_not(s10_upper_std_good_probes)],s10_upper_std_z[np.logical_not(s10_upper_std_good_probes)],'xr',fillstyle='none')
			ax[1,0].plot(s10_upper_large_r[np.logical_not(s10_upper_large_good_probes)],s10_upper_large_z[np.logical_not(s10_upper_large_good_probes)],'xr',fillstyle='none')
			ax[1,0].plot(s10_lower_r[s10_lower_good_probes],s10_lower_z[s10_lower_good_probes],'og',fillstyle='none')
			ax[1,0].plot(s4_lower_r[s4_lower_good_probes],s4_lower_z[s4_lower_good_probes],'og',fillstyle='none')
			ax[1,0].plot(s4_upper_r[s4_upper_good_probes],s4_upper_z[s4_upper_good_probes],'og',fillstyle='none')
			ax[1,0].plot(s10_upper_std_r[s10_upper_std_good_probes],s10_upper_std_z[s10_upper_std_good_probes],'og',fillstyle='none')
			ax[1,0].plot(s10_upper_large_r[s10_upper_large_good_probes],s10_upper_large_z[s10_upper_large_good_probes],'og',fillstyle='none')
			ax[1,0].set_ylim(top=-1.1,bottom=-2.2)
			ax[1,0].set_xlim(right=1.8)
			ax[1,0].set_aspect('equal', adjustable='box')
			ax[0,0].set_ylabel('top Z [m]')
			ax[1,0].set_ylabel('bottom Z [m]')
			ax[1,0].set_xlabel('R [m]')
			plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/working_vs_bad_LPs.png', bbox_inches='tight')
			plt.close()
		else:
			pass


		closeness_limit_to_dead_channels = 0.01	# m
		closeness_limit_for_good_channels = np.median(np.abs(np.diff(s10_lower_s).tolist()+np.diff(s4_lower_s).tolist()+np.diff(s4_upper_s).tolist()+np.diff(s10_upper_std_s).tolist()+np.diff(s10_upper_large_s).tolist()))*2	# *5
		closeness_limit_for_good_channels = 0.02	# m	# let's fix it to a sinble parameter
		print('closeness_limit_for_good_channels '+str(closeness_limit_for_good_channels))
		jsat_upper_inner_small_max = []
		jsat_upper_outer_small_max = []
		jsat_upper_inner_mid_max = []
		jsat_upper_outer_mid_max = []
		jsat_upper_inner_large_max = []	# T5
		jsat_upper_outer_large_max = []	# T5
		jsat_lower_inner_small_max = []	#central column
		jsat_lower_outer_small_max = []	#central column
		jsat_lower_inner_mid_max = []
		jsat_lower_outer_mid_max = []
		jsat_lower_inner_large_max = []
		jsat_lower_outer_large_max = []

		jsat_upper_inner_small_max_sigma = []
		jsat_upper_outer_small_max_sigma = []
		jsat_upper_inner_mid_max_sigma = []
		jsat_upper_outer_mid_max_sigma = []
		jsat_upper_inner_large_max_sigma = []	# T5
		jsat_upper_outer_large_max_sigma = []	# T5
		jsat_lower_inner_small_max_sigma = []	#central column
		jsat_lower_outer_small_max_sigma = []	#central column
		jsat_lower_inner_mid_max_sigma = []
		jsat_lower_outer_mid_max_sigma = []
		jsat_lower_inner_large_max_sigma = []
		jsat_lower_outer_large_max_sigma = []

		jsat_upper_inner_small_integrated = []
		jsat_upper_outer_small_integrated = []
		jsat_upper_inner_mid_integrated = []
		jsat_upper_outer_mid_integrated = []
		jsat_upper_inner_large_integrated = []	# T5
		jsat_upper_outer_large_integrated = []	# T5
		jsat_lower_inner_small_integrated = []	#central column
		jsat_lower_outer_small_integrated = []	#central column
		jsat_lower_inner_mid_integrated = []
		jsat_lower_outer_mid_integrated = []
		jsat_lower_inner_large_integrated = []
		jsat_lower_outer_large_integrated = []

		jsat_upper_inner_small_integrated_sigma = []
		jsat_upper_outer_small_integrated_sigma = []
		jsat_upper_inner_mid_integrated_sigma = []
		jsat_upper_outer_mid_integrated_sigma = []
		jsat_upper_inner_large_integrated_sigma = []	# T5
		jsat_upper_outer_large_integrated_sigma = []	# T5
		jsat_lower_inner_small_integrated_sigma = []	#central column
		jsat_lower_outer_small_integrated_sigma = []	#central column
		jsat_lower_inner_mid_integrated_sigma = []
		jsat_lower_outer_mid_integrated_sigma = []
		jsat_lower_inner_large_integrated_sigma = []
		jsat_lower_outer_large_integrated_sigma = []

		# preselecting only good probes
		s10_lower_jsat = np.array(s10_lower_jsat)[:,s10_lower_good_probes]
		s10_lower_jsat_sigma = np.array(s10_lower_jsat_sigma)[:,s10_lower_good_probes]
		s10_lower_jsat_r = np.array(s10_lower_jsat_r)[:,s10_lower_good_probes]
		s10_lower_jsat_s = np.array(s10_lower_jsat_s)[:,s10_lower_good_probes]
		s4_lower_jsat = np.array(s4_lower_jsat)[:,s4_lower_good_probes]
		s4_lower_jsat_sigma = np.array(s4_lower_jsat_sigma)[:,s4_lower_good_probes]
		s4_lower_jsat_r = np.array(s4_lower_jsat_r)[:,s4_lower_good_probes]
		s4_lower_jsat_s = np.array(s4_lower_jsat_s)[:,s4_lower_good_probes]
		s4_upper_jsat = np.array(s4_upper_jsat)[:,s4_upper_good_probes]
		s4_upper_jsat_sigma = np.array(s4_upper_jsat_sigma)[:,s4_upper_good_probes]
		s4_upper_jsat_r = np.array(s4_upper_jsat_r)[:,s4_upper_good_probes]
		s4_upper_jsat_s = np.array(s4_upper_jsat_s)[:,s4_upper_good_probes]
		s10_upper_std_jsat = np.array(s10_upper_std_jsat)[:,s10_upper_std_good_probes]
		s10_upper_std_jsat_sigma = np.array(s10_upper_std_jsat_sigma)[:,s10_upper_std_good_probes]
		s10_upper_std_jsat_r = np.array(s10_upper_std_jsat_r)[:,s10_upper_std_good_probes]
		s10_upper_std_jsat_s = np.array(s10_upper_std_jsat_s)[:,s10_upper_std_good_probes]
		s10_upper_large_jsat = np.array(s10_upper_large_jsat)[:,s10_upper_large_good_probes]
		s10_upper_large_jsat_sigma = np.array(s10_upper_large_jsat_sigma)[:,s10_upper_large_good_probes]
		s10_upper_large_jsat_r = np.array(s10_upper_large_jsat_r)[:,s10_upper_large_good_probes]
		s10_upper_large_jsat_s = np.array(s10_upper_large_jsat_s)[:,s10_upper_large_good_probes]

		# skipped = 0
		for time in time_full_binned_crop:
			if time<np.min(output[0]['time']) or time>np.max(output[-1]['time']):
				jsat_upper_inner_small_max.append(np.nan)
				jsat_upper_outer_small_max.append(np.nan)
				jsat_upper_inner_small_integrated.append(np.nan)
				jsat_upper_outer_small_integrated.append(np.nan)
				jsat_upper_inner_mid_max.append(np.nan)
				jsat_upper_inner_mid_integrated.append(np.nan)
				jsat_upper_outer_mid_max.append(np.nan)
				jsat_upper_outer_mid_integrated.append(np.nan)
				jsat_upper_inner_large_max.append(np.nan)
				jsat_upper_inner_large_integrated.append(np.nan)
				jsat_upper_outer_large_max.append(np.nan)
				jsat_upper_outer_large_integrated.append(np.nan)
				jsat_lower_inner_small_max.append(np.nan)
				jsat_lower_inner_small_integrated.append(np.nan)
				jsat_lower_outer_small_max.append(np.nan)
				jsat_lower_outer_small_integrated.append(np.nan)
				jsat_lower_inner_mid_max.append(np.nan)
				jsat_lower_inner_mid_integrated.append(np.nan)
				jsat_lower_outer_mid_max.append(np.nan)
				jsat_lower_outer_mid_integrated.append(np.nan)
				jsat_lower_inner_large_max.append(np.nan)
				jsat_lower_outer_large_max.append(np.nan)
				jsat_lower_inner_large_integrated.append(np.nan)
				jsat_lower_outer_large_integrated.append(np.nan)

				jsat_upper_inner_small_max_sigma.append(np.nan)
				jsat_upper_outer_small_max_sigma.append(np.nan)
				jsat_upper_inner_small_integrated_sigma.append(np.nan)
				jsat_upper_outer_small_integrated_sigma.append(np.nan)
				jsat_upper_inner_mid_max_sigma.append(np.nan)
				jsat_upper_inner_mid_integrated_sigma.append(np.nan)
				jsat_upper_outer_mid_max_sigma.append(np.nan)
				jsat_upper_outer_mid_integrated_sigma.append(np.nan)
				jsat_upper_inner_large_max_sigma.append(np.nan)
				jsat_upper_inner_large_integrated_sigma.append(np.nan)
				jsat_upper_outer_large_max_sigma.append(np.nan)
				jsat_upper_outer_large_integrated_sigma.append(np.nan)
				jsat_lower_inner_small_max_sigma.append(np.nan)
				jsat_lower_inner_small_integrated_sigma.append(np.nan)
				jsat_lower_outer_small_max_sigma.append(np.nan)
				jsat_lower_outer_small_integrated_sigma.append(np.nan)
				jsat_lower_inner_mid_max_sigma.append(np.nan)
				jsat_lower_inner_mid_integrated_sigma.append(np.nan)
				jsat_lower_outer_mid_max_sigma.append(np.nan)
				jsat_lower_outer_mid_integrated_sigma.append(np.nan)
				jsat_lower_inner_large_max_sigma.append(np.nan)
				jsat_lower_outer_large_max_sigma.append(np.nan)
				jsat_lower_inner_large_integrated_sigma.append(np.nan)
				jsat_lower_outer_large_integrated_sigma.append(np.nan)
				# skipped += 1
				continue

			i_time = np.abs(time-np.mean(trange,axis=1)).argmin()

			i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
			inner_strike_point = [np.min(efit_reconstruction.strikepointR[i_efit_time][:2]),0,0]
			inner_strike_point[1] = -efit_reconstruction.strikepointZ[i_efit_time][np.abs(efit_reconstruction.strikepointR[i_efit_time]-inner_strike_point[0]).argmin()]
			temp = get_nearest_s_coordinates_mastu([inner_strike_point[0]],[inner_strike_point[1]],s_lookup, tol=5e-1)
			inner_strike_point[2] = np.abs(temp[0][0])	# this is all info for the upper divertor [R,Z,s]
			outer_strike_point = [np.max(efit_reconstruction.strikepointR[i_efit_time][:2]),0,0]
			outer_strike_point[1] = -efit_reconstruction.strikepointZ[i_efit_time][np.abs(efit_reconstruction.strikepointR[i_efit_time]-outer_strike_point[0]).argmin()]
			temp = get_nearest_s_coordinates_mastu([outer_strike_point[0]],[outer_strike_point[1]],s_lookup, tol=5e-1)
			outer_strike_point[2] = np.abs(temp[0][0])	# this is all info for the upper divertor [R,Z,s]
			mid_strike_point = [np.mean(efit_reconstruction.strikepointR[i_efit_time][:2]),-np.mean(efit_reconstruction.strikepointZ[i_efit_time][:2]),0]
			temp = get_nearest_s_coordinates_mastu([mid_strike_point[0]],[mid_strike_point[1]],s_lookup, tol=5e-1)
			mid_strike_point[2] = np.abs(temp[0][0])	# this is all info for the upper divertor [R,Z,s]

			# upper divertor
			# 19/02/2025 I split the second condition so that I need close good plobes on both sides of the strike point to be able to say anything
			temp = s4_upper_s-inner_strike_point[2]
			s4_upper_inner_test = np.sum(np.logical_not(s4_upper_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_upper_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_upper_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s10_upper_std_s-inner_strike_point[2]
			s10_upper_std_inner_test = np.sum(np.logical_not(s10_upper_std_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s10_upper_large_s-inner_strike_point[2]
			s10_upper_large_inner_test = np.sum(np.logical_not(s10_upper_large_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s4_upper_s-outer_strike_point[2]
			s4_upper_outer_test = np.sum(np.logical_not(s4_upper_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_upper_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_upper_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s10_upper_std_s-outer_strike_point[2]
			s10_upper_std_outer_test = np.sum(np.logical_not(s10_upper_std_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s10_upper_large_s-outer_strike_point[2]
			s10_upper_large_outer_test = np.sum(np.logical_not(s10_upper_large_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0

			jsat_upper_inner_small_max.append(0)
			jsat_upper_inner_small_max_sigma.append(0)
			jsat_upper_outer_small_max.append(0)
			jsat_upper_outer_small_max_sigma.append(0)
			jsat_upper_inner_small_integrated.append(0)
			jsat_upper_inner_small_integrated_sigma.append(0)
			jsat_upper_outer_small_integrated.append(0)
			jsat_upper_outer_small_integrated_sigma.append(0)

			temp = [np.nan]
			if s4_upper_inner_test:
				temp = np.concatenate([[0],temp,s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]])
			if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp = np.concatenate([[0],temp,s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s4_upper_inner_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]])
			if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]])
			jsat_upper_inner_mid_max.append(np.nanmax(temp))
			jsat_upper_inner_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = [np.nan]
			temp_sigma_extra = 0
			if s4_upper_inner_test:
				temp.append( np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]) )
			if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp.append( np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]) )
			if np.nanmax(temp) - np.nanmin(temp)<np.nanmean(temp)/2:	# step added to see if averaging among the 2 sectors rather than taking the max matters
				temp_sigma_extra = np.nanmax(temp) - np.nanmin(temp)
				temp =np.nanmean(temp)
			else:
				temp = np.nan
			if not(s4_upper_inner_test) and not(s10_upper_std_inner_test):
				temp = np.nan
			temp_sigma = 0
			if s4_upper_inner_test:
				temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]])**2,x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]])])
			if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]])**2,x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]])])
			if not(s4_upper_inner_test) and not(s10_upper_std_inner_test):
				temp_sigma = np.nan
			jsat_upper_inner_mid_integrated.append(temp)
			jsat_upper_inner_mid_integrated_sigma.append(max(temp_sigma**0.5,temp_sigma_extra))

			temp = [np.nan]
			if s4_upper_outer_test:
				temp = np.concatenate([[0],temp,s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]])
			if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp = np.concatenate([[0],temp,s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s4_upper_outer_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]])
			if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]])
			jsat_upper_outer_mid_max.append(np.nanmax(temp))
			jsat_upper_outer_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = [np.nan]
			temp_sigma_extra = 0
			if s4_upper_outer_test:
				temp.append( np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]) )
			if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp.append( np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]) )
			if np.nanmax(temp) - np.nanmin(temp)<np.nanmean(temp)/2:	# step added to see if averaging among the 2 sectors rather than taking the max matters
				temp_sigma_extra = np.nanmax(temp) - np.nanmin(temp)
				temp =np.nanmean(temp)
			else:
				temp = np.nan
			if not(s4_upper_outer_test) and not(s10_upper_std_outer_test):
				temp = np.nan
			temp_sigma = 0
			if s4_upper_outer_test:
				temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]])**2,x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]])])
			if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
				temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]])**2,x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]])])
			if not(s4_upper_outer_test) and not(s10_upper_std_outer_test):
				temp_sigma = np.nan
			jsat_upper_outer_mid_integrated.append(temp)
			jsat_upper_outer_mid_integrated_sigma.append(max(temp_sigma**0.5,temp_sigma_extra))

			temp = [np.nan]
			if s10_upper_large_inner_test:
				temp = np.concatenate([[0],temp,s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s10_upper_large_inner_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]])
			jsat_upper_inner_large_max.append(np.nanmax(temp))
			jsat_upper_inner_large_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = 0
			if s10_upper_large_inner_test:
				temp += np.trapz(s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])
			else:
				temp = np.nan
			temp_sigma = 0
			if s10_upper_large_inner_test:
				temp_sigma += coleval.np_trapz_error((s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])**2,x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])
			else:
				temp_sigma = np.nan
			jsat_upper_inner_large_integrated.append(temp)
			jsat_upper_inner_large_integrated_sigma.append(temp_sigma**0.5)

			temp = [np.nan]
			if s10_upper_large_outer_test:
				temp = np.concatenate([[0],temp,s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s10_upper_large_outer_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]])
			jsat_upper_outer_large_max.append(np.nanmax(temp))
			jsat_upper_outer_large_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = 0
			if s10_upper_large_outer_test:
				temp += np.trapz(s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])
			else:
				temp = np.nan
			temp_sigma = 0
			if s10_upper_large_outer_test:
				temp_sigma += coleval.np_trapz_error((s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])**2,x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])
			else:
				temp_sigma = np.nan
			jsat_upper_outer_large_integrated.append(temp)
			jsat_upper_outer_large_integrated_sigma.append(temp_sigma**0.5)


			# lower divertor
			# I add the second part to exluce the case there are NO good channel close to the strike point, diving a bit of slack allowing twice the normal distance between probes
			# 19/02/2025 I split the second condition so that I need close good plobes on both sides of the strike point to be able to say anything
			temp = s4_lower_s-(-inner_strike_point[2])
			s4_lower_inner_test = np.sum(np.logical_not(s4_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s10_lower_s-(-inner_strike_point[2])
			s10_lower_inner_test = np.sum(np.logical_not(s10_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s4_lower_s-(-outer_strike_point[2])
			s4_lower_outer_test = np.sum(np.logical_not(s4_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
			temp = s10_lower_s-(-outer_strike_point[2])
			s10_lower_outer_test = np.sum(np.logical_not(s10_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0

			temp = [np.nan]
			if s4_lower_inner_test:
				temp = np.concatenate([[0],temp,s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s4_lower_inner_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]])
			jsat_lower_inner_small_max.append(np.nanmax(temp))
			jsat_lower_inner_small_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = 0
			if s4_lower_inner_test:
				temp += np.trapz(s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]],x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])
			else:
				temp = np.nan
			temp_sigma = 0
			if s4_lower_inner_test:
				temp_sigma += coleval.np_trapz_error((s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])**2,x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])
			else:
				temp_sigma = np.nan
			jsat_lower_inner_small_integrated.append(-temp)
			jsat_lower_inner_small_integrated_sigma.append(temp_sigma**0.5)

			temp = [np.nan]
			if s4_lower_outer_test:
				temp = np.concatenate([[0],temp,s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s4_lower_outer_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]])
			jsat_lower_outer_small_max.append(np.nanmax(temp))
			jsat_lower_outer_small_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = 0
			if s4_lower_outer_test:
				temp += np.trapz(s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]],x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])
			else:
				temp = np.nan
			temp_sigma = 0
			if s4_lower_outer_test:
				temp_sigma += coleval.np_trapz_error((s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])**2,x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])
			else:
				temp_sigma = np.nan
			jsat_lower_outer_small_integrated.append(-temp)
			jsat_lower_outer_small_integrated_sigma.append(temp_sigma**0.5)

			temp = [np.nan]
			if s10_lower_inner_test:
				temp = np.concatenate([[0],temp,s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s10_lower_inner_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]])
			jsat_lower_inner_mid_max.append(np.nanmax(temp))
			jsat_lower_inner_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = 0
			if s10_lower_inner_test:
				temp += np.trapz(s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]],x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])
			else:
				temp = np.nan
			temp_sigma = 0
			if s10_lower_inner_test:
				temp_sigma += coleval.np_trapz_error((s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])**2,x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])
			else:
				temp_sigma = np.nan
			jsat_lower_inner_mid_integrated.append(-temp)
			jsat_lower_inner_mid_integrated_sigma.append(temp_sigma**0.5)

			temp = [np.nan]
			if s10_lower_outer_test:
				temp = np.concatenate([[0],temp,s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]])
			temp_sigma = [np.nan]
			if s10_lower_outer_test:
				temp_sigma = np.concatenate([[0],temp_sigma,s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]])
			jsat_lower_outer_mid_max.append(np.nanmax(temp))
			jsat_lower_outer_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

			temp = 0
			if s10_lower_outer_test:
				temp += np.trapz(s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]],x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])
			else:
				temp = np.nan
			temp_sigma = 0
			if s10_lower_outer_test:
				temp_sigma += coleval.np_trapz_error((s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])**2,x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])
			else:
				temp_sigma = np.nan
			jsat_lower_outer_mid_integrated.append(-temp)
			jsat_lower_outer_mid_integrated_sigma.append(temp_sigma**0.5)

			jsat_lower_inner_large_max.append(0)
			jsat_lower_inner_large_max_sigma.append(0)
			jsat_lower_outer_large_max.append(0)
			jsat_lower_outer_large_max_sigma.append(0)
			jsat_lower_inner_large_integrated.append(0)
			jsat_lower_inner_large_integrated_sigma.append(0)
			jsat_lower_outer_large_integrated.append(0)
			jsat_lower_outer_large_integrated_sigma.append(0)

		jsat_time = cp.deepcopy(time_full_binned_crop)
		jsat_read = True

		jsat_lower_inner_small_integrated = np.array(jsat_lower_inner_small_integrated)
		LPs_dict['jsat_lower_inner_small_integrated'] = jsat_lower_inner_small_integrated
		jsat_lower_outer_small_integrated = np.array(jsat_lower_outer_small_integrated)
		LPs_dict['jsat_lower_outer_small_integrated'] = jsat_lower_outer_small_integrated
		jsat_lower_inner_mid_integrated = np.array(jsat_lower_inner_mid_integrated)
		LPs_dict['jsat_lower_inner_mid_integrated'] = jsat_lower_inner_mid_integrated
		jsat_lower_outer_mid_integrated = np.array(jsat_lower_outer_mid_integrated)
		LPs_dict['jsat_lower_outer_mid_integrated'] = jsat_lower_outer_mid_integrated
		jsat_lower_inner_large_integrated = np.array(jsat_lower_inner_large_integrated)
		LPs_dict['jsat_lower_inner_large_integrated'] = jsat_lower_inner_large_integrated
		jsat_lower_outer_large_integrated = np.array(jsat_lower_outer_large_integrated)
		LPs_dict['jsat_lower_outer_large_integrated'] = jsat_lower_outer_large_integrated
		jsat_upper_inner_small_integrated = np.array(jsat_upper_inner_small_integrated)
		LPs_dict['jsat_upper_inner_small_integrated'] = jsat_upper_inner_small_integrated
		jsat_upper_outer_small_integrated = np.array(jsat_upper_outer_small_integrated)
		LPs_dict['jsat_upper_outer_small_integrated'] = jsat_upper_outer_small_integrated
		jsat_upper_inner_mid_integrated = np.array(jsat_upper_inner_mid_integrated)
		LPs_dict['jsat_upper_inner_mid_integrated'] = jsat_upper_inner_mid_integrated
		jsat_upper_outer_mid_integrated = np.array(jsat_upper_outer_mid_integrated)
		LPs_dict['jsat_upper_outer_mid_integrated'] = jsat_upper_outer_mid_integrated
		jsat_upper_inner_large_integrated = np.array(jsat_upper_inner_large_integrated)
		LPs_dict['jsat_upper_inner_large_integrated'] = jsat_upper_inner_large_integrated
		jsat_upper_outer_large_integrated = np.array(jsat_upper_outer_large_integrated)
		LPs_dict['jsat_upper_outer_large_integrated'] = jsat_upper_outer_large_integrated

		jsat_lower_inner_small_integrated_sigma = np.array(jsat_lower_inner_small_integrated_sigma)
		LPs_dict['jsat_lower_inner_small_integrated_sigma'] = jsat_lower_inner_small_integrated_sigma
		jsat_lower_outer_small_integrated_sigma = np.array(jsat_lower_outer_small_integrated_sigma)
		LPs_dict['jsat_lower_outer_small_integrated_sigma'] = jsat_lower_outer_small_integrated_sigma
		jsat_lower_inner_mid_integrated_sigma = np.array(jsat_lower_inner_mid_integrated_sigma)
		LPs_dict['jsat_lower_inner_mid_integrated_sigma'] = jsat_lower_inner_mid_integrated_sigma
		jsat_lower_outer_mid_integrated_sigma = np.array(jsat_lower_outer_mid_integrated_sigma)
		LPs_dict['jsat_lower_outer_mid_integrated_sigma'] = jsat_lower_outer_mid_integrated_sigma
		jsat_lower_inner_large_integrated_sigma = np.array(jsat_lower_inner_large_integrated_sigma)
		LPs_dict['jsat_lower_inner_large_integrated_sigma'] = jsat_lower_inner_large_integrated_sigma
		jsat_lower_outer_large_integrated_sigma = np.array(jsat_lower_outer_large_integrated_sigma)
		LPs_dict['jsat_lower_outer_large_integrated_sigma'] = jsat_lower_outer_large_integrated_sigma
		jsat_upper_inner_small_integrated_sigma = np.array(jsat_upper_inner_small_integrated_sigma)
		LPs_dict['jsat_upper_inner_small_integrated_sigma'] = jsat_upper_inner_small_integrated_sigma
		jsat_upper_outer_small_integrated_sigma = np.array(jsat_upper_outer_small_integrated_sigma)
		LPs_dict['jsat_upper_outer_small_integrated_sigma'] = jsat_upper_outer_small_integrated_sigma
		jsat_upper_inner_mid_integrated_sigma = np.array(jsat_upper_inner_mid_integrated_sigma)
		LPs_dict['jsat_upper_inner_mid_integrated_sigma'] = jsat_upper_inner_mid_integrated_sigma
		jsat_upper_outer_mid_integrated_sigma = np.array(jsat_upper_outer_mid_integrated_sigma)
		LPs_dict['jsat_upper_outer_mid_integrated_sigma'] = jsat_upper_outer_mid_integrated_sigma
		jsat_upper_inner_large_integrated_sigma = np.array(jsat_upper_inner_large_integrated_sigma)
		LPs_dict['jsat_upper_inner_large_integrated_sigma'] = jsat_upper_inner_large_integrated_sigma
		jsat_upper_outer_large_integrated_sigma = np.array(jsat_upper_outer_large_integrated_sigma)
		LPs_dict['jsat_upper_outer_large_integrated_sigma'] = jsat_upper_outer_large_integrated_sigma

		jsat_lower_inner_small_max_sigma = np.array(jsat_lower_inner_small_max_sigma)
		LPs_dict['jsat_lower_inner_small_max_sigma'] = jsat_lower_inner_small_max_sigma
		jsat_lower_outer_small_max_sigma = np.array(jsat_lower_outer_small_max_sigma)
		LPs_dict['jsat_lower_outer_small_max_sigma'] = jsat_lower_outer_small_max_sigma
		jsat_lower_inner_mid_max_sigma = np.array(jsat_lower_inner_mid_max_sigma)
		LPs_dict['jsat_lower_inner_mid_max_sigma'] = jsat_lower_inner_mid_max_sigma
		jsat_lower_outer_mid_max_sigma = np.array(jsat_lower_outer_mid_max_sigma)
		LPs_dict['jsat_lower_outer_mid_max_sigma'] = jsat_lower_outer_mid_max_sigma
		jsat_lower_inner_large_max_sigma = np.array(jsat_lower_inner_large_max_sigma)
		LPs_dict['jsat_lower_inner_large_max_sigma'] = jsat_lower_inner_large_max_sigma
		jsat_lower_outer_large_max_sigma = np.array(jsat_lower_outer_large_max_sigma)
		LPs_dict['jsat_lower_outer_large_max_sigma'] = jsat_lower_outer_large_max_sigma
		jsat_upper_inner_small_max_sigma = np.array(jsat_upper_inner_small_max_sigma)
		LPs_dict['jsat_upper_inner_small_max_sigma'] = jsat_upper_inner_small_max_sigma
		jsat_upper_outer_small_max_sigma = np.array(jsat_upper_outer_small_max_sigma)
		LPs_dict['jsat_upper_outer_small_max_sigma'] = jsat_upper_outer_small_max_sigma
		jsat_upper_inner_mid_max_sigma = np.array(jsat_upper_inner_mid_max_sigma)
		LPs_dict['jsat_upper_inner_mid_max_sigma'] = jsat_upper_inner_mid_max_sigma
		jsat_upper_outer_mid_max_sigma = np.array(jsat_upper_outer_mid_max_sigma)
		LPs_dict['jsat_upper_outer_mid_max_sigma'] = jsat_upper_outer_mid_max_sigma
		jsat_upper_inner_large_max_sigma = np.array(jsat_upper_inner_large_max_sigma)
		LPs_dict['jsat_upper_inner_large_max_sigma'] = jsat_upper_inner_large_max_sigma
		jsat_upper_outer_large_max_sigma = np.array(jsat_upper_outer_large_max_sigma)
		LPs_dict['jsat_upper_outer_large_max_sigma'] = jsat_upper_outer_large_max_sigma
	else:	# old method without Peter Ryan functions

		for probe_size,probe_type in [[9.69975e-06,'small'],[1.35553e-05,'mid'],[4.56942e-05,'large']]:
			try:
				data_s10 = lp_data.s10_lower_data.jsat[:,lp_data.s10_lower_data.physical_area==probe_size]
			except:
				data_s10 = lp_data.s10_lower_data.jsat_tile[:,lp_data.s10_lower_data.physical_area==probe_size]
			time_orig_s10 = lp_data.s10_lower_data.time
			s_orig_s10 = lp_data.s10_lower_data.s[lp_data.s10_lower_data.physical_area==probe_size]
			r_orig_s10 = lp_data.s10_lower_data.r[lp_data.s10_lower_data.physical_area==probe_size]
			tiles_covered_s10 = lp_data.s10_lower_data.tiles_covered
			try:
				data_s4 = lp_data.s4_lower_data.jsat[:,lp_data.s4_lower_data.physical_area==probe_size]
			except:
				data_s4 = lp_data.s4_lower_data.jsat_tile[:,lp_data.s4_lower_data.physical_area==probe_size]
			time_orig_s4 = lp_data.s4_lower_data.time
			s_orig_s4 = lp_data.s4_lower_data.s[lp_data.s4_lower_data.physical_area==probe_size]
			r_orig_s4 = lp_data.s4_lower_data.r[lp_data.s4_lower_data.physical_area==probe_size]
			coordinate = np.concatenate([r_orig_s4,r_orig_s10])
			jsat = np.concatenate([data_s4.T,data_s10.T]).T
			tiles_covered_s4 = lp_data.s10_lower_data.tiles_covered
			inner_max = []
			outer_max = []
			for i_time,time in enumerate(time_orig_s10):
				i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
				mid_strike_point = np.mean(efit_reconstruction.strikepointR[i_efit_time][:2])
				inner_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate<=mid_strike_point]])))
				outer_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate>mid_strike_point]])))
			if probe_type=='small':
				jsat_lower_inner_small_max = np.array(inner_max)	#central column
				jsat_lower_outer_small_max = np.array(outer_max)	#central column
			elif probe_type=='mid':
				jsat_lower_inner_mid_max = np.array(inner_max)
				jsat_lower_outer_mid_max = np.array(outer_max)
			elif probe_type=='large':
				jsat_lower_inner_large_max = np.array(inner_max)
				jsat_lower_outer_large_max = np.array(outer_max)


		for probe_size,probe_type in [[9.69975e-06,'small'],[1.35553e-05,'mid'],[4.56942e-05,'large']]:
			try:
				data_s10 = lp_data.s10_upper_data.jsat[:,lp_data.s10_upper_data.physical_area==probe_size]
			except:
				data_s10 = lp_data.s10_upper_data.jsat_tile[:,lp_data.s10_upper_data.physical_area==probe_size]
			time_orig_s10 = lp_data.s10_upper_data.time
			s_orig_s10 = lp_data.s10_upper_data.s[lp_data.s10_upper_data.physical_area==probe_size]
			r_orig_s10 = lp_data.s10_upper_data.r[lp_data.s10_upper_data.physical_area==probe_size]
			tiles_covered_s10 = lp_data.s10_upper_data.tiles_covered
			try:
				data_s4 = lp_data.s4_upper_data.jsat[:,lp_data.s4_upper_data.physical_area==probe_size]
			except:
				data_s4 = lp_data.s4_upper_data.jsat_tile[:,lp_data.s4_upper_data.physical_area==probe_size]
			time_orig_s4 = lp_data.s4_upper_data.time
			s_orig_s4 = lp_data.s4_upper_data.s[lp_data.s4_upper_data.physical_area==probe_size]
			r_orig_s4 = lp_data.s4_upper_data.r[lp_data.s4_upper_data.physical_area==probe_size]
			coordinate = np.concatenate([r_orig_s4,r_orig_s10])
			jsat = np.concatenate([data_s4.T,data_s10.T]).T
			tiles_covered_s4 = lp_data.s10_upper_data.tiles_covered
			inner_max = []
			outer_max = []
			for i_time,time in enumerate(time_orig_s10):
				i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
				mid_strike_point = np.mean(efit_reconstruction.strikepointR[i_efit_time][:2])
				inner_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate<=mid_strike_point]])))
				outer_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate>mid_strike_point]])))
			if probe_type=='small':
				jsat_upper_inner_small_max = np.array(inner_max)
				jsat_upper_outer_small_max = np.array(outer_max)
			elif probe_type=='mid':
				jsat_upper_inner_mid_max = np.array(inner_max)
				jsat_upper_outer_mid_max = np.array(outer_max)
			elif probe_type=='large':
				jsat_upper_inner_large_max = np.array(inner_max)	# T5
				jsat_upper_outer_large_max = np.array(outer_max)	# T5
		# time,r = np.meshgrid(time_orig_s4,coordinate)
		# plt.figure(figsize=(10,10))
		# plt.pcolor(r,time,jsat.T,norm=LogNorm())#,vmin=np.nanmax(data)*1e-3)
		# plt.colorbar().set_label('jsat')
		# for __i in range(np.shape(efit_reconstruction.strikepointR)[1]):
		# 	plt.plot(efit_reconstruction.strikepointR[:,__i],efit_reconstruction.time,'--r')
		# plt.grid()
		# plt.xlim(left=0.5)
		# plt.ylabel('time [s]')
		# plt.xlabel('R [m]')
		# # plt.title(str(shot)+'\n'+str(tiles_covered))
		# plt.pause(0.01)
		jsat_time = cp.deepcopy(time_orig_s10)
		jsat_read = True

	LPs_dict['jsat_time'] = jsat_time
	jsat_lower_inner_small_max = np.array(jsat_lower_inner_small_max)
	LPs_dict['jsat_lower_inner_small_max'] = jsat_lower_inner_small_max
	jsat_lower_outer_small_max = np.array(jsat_lower_outer_small_max)
	LPs_dict['jsat_lower_outer_small_max'] = jsat_lower_outer_small_max
	jsat_lower_inner_mid_max = np.array(jsat_lower_inner_mid_max)
	LPs_dict['jsat_lower_inner_mid_max'] = jsat_lower_inner_mid_max
	jsat_lower_outer_mid_max = np.array(jsat_lower_outer_mid_max)
	LPs_dict['jsat_lower_outer_mid_max'] = jsat_lower_outer_mid_max
	jsat_lower_inner_large_max = np.array(jsat_lower_inner_large_max)
	LPs_dict['jsat_lower_inner_large_max'] = jsat_lower_inner_large_max
	jsat_lower_outer_large_max = np.array(jsat_lower_outer_large_max)
	LPs_dict['jsat_lower_outer_large_max'] = jsat_lower_outer_large_max
	jsat_upper_inner_small_max = np.array(jsat_upper_inner_small_max)
	LPs_dict['jsat_upper_inner_small_max'] = jsat_upper_inner_small_max
	jsat_upper_outer_small_max = np.array(jsat_upper_outer_small_max)
	LPs_dict['jsat_upper_outer_small_max'] = jsat_upper_outer_small_max
	jsat_upper_inner_mid_max = np.array(jsat_upper_inner_mid_max)
	LPs_dict['jsat_upper_inner_mid_max'] = jsat_upper_inner_mid_max
	jsat_upper_outer_mid_max = np.array(jsat_upper_outer_mid_max)
	LPs_dict['jsat_upper_outer_mid_max'] = jsat_upper_outer_mid_max
	jsat_upper_inner_large_max = np.array(jsat_upper_inner_large_max)
	LPs_dict['jsat_upper_inner_large_max'] = jsat_upper_inner_large_max
	jsat_upper_outer_large_max = np.array(jsat_upper_outer_large_max)
	LPs_dict['jsat_upper_outer_large_max'] = jsat_upper_outer_large_max
	if lambda_q_determination:
		# LPs_dict['LP_lambdaq_time'] = np.array(LP_lambdaq_time)
		LPs_dict['lambda_q_10_lower'] = np.array(lambda_q_10_lower)
		LPs_dict['lambda_q_10_lower_sigma'] = np.array(lambda_q_10_lower_sigma)
		LPs_dict['lambda_q_4_lower'] = np.array(lambda_q_4_lower)
		LPs_dict['lambda_q_4_lower_sigma'] = np.array(lambda_q_4_lower_sigma)
		LPs_dict['lambda_q_4_upper'] = np.array(lambda_q_4_upper)
		LPs_dict['lambda_q_4_upper_sigma'] = np.array(lambda_q_4_upper_sigma)
		LPs_dict['lambda_q_10_upper'] = np.array(lambda_q_10_upper)
		LPs_dict['lambda_q_10_upper_sigma'] = np.array(lambda_q_10_upper_sigma)
	print('marker LP done')
except Exception as e:
	logging.exception('with error (LP section): ' + str(e))
	jsat_read = False
	lambda_q_determination = False
	print('LP skipped')


LPs_dict['jsat_read'] = jsat_read
LPs_dict['lambda_q_determination'] = lambda_q_determination
np.savez_compressed(sys.argv[1],**LPs_dict)






#######
