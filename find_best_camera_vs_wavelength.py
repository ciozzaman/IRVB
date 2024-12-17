import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())


# import matplotlib.pyplot as plt
#import .functions
# os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
# from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
# from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
# from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace
# from functions.GetSpectrumGeometry import getGeom
# from functions.SpectralFit import doSpecFit_single_frame
# from functions.GaussFitData import doLateralfit_time_tependent
# import collections

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
# from pycine.color import color_pipeline, resize
# from pycine.raw import read_frames
# from pycine.file import read_header
# from functions.fabio_add import all_file_names,find_index_of_file




if False:	# main purpose, show image contrast with wavelength
	function_of_reflection = lambda x: np.logical_or(x<4,x>5)	# reflections are prevented onlt in the window of 4 to 5nm
	quantuum_efficiency = lambda x: 0.3*(x<3) + 0.9*np.logical_and(x>3,x<5) + 0.5*np.logical_and(x>5,x<14) + 0.3*(x>14)	# super approximate variation in quantuum efficiency. see the FLIR documentation if you want something better

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
	reflection_temperature = 20
	min_temp = 40
	delta_wave = 3	# I consider a sensor of that measures the wavelengths from start to start+delta_wave
	delta_t = 1
	ref_int_time = 0.001
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
		# counts at the low temperature
		x0=((f_number_old/f_number_new)**2)*1*int_time*(
		1*9.74e-05*3/0.002*3.42E-14 * np.trapz(quantuum_efficiency(x)*function_of_reflection(x)* 	# component from reflections
		2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+reflection_temperature)))-1)),x=x)	# this has to be some photons/wavelength correlation, you can check wikipedia for this
		 + 3.42E-14 * np.trapz(quantuum_efficiency(x)*2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp)))-1)),x=x))	# 3.42E-14 is the proportionality coefficient from the temperature calibration
		 # counts at the high temperature
		x1=((f_number_old/f_number_new)**2)*1*int_time*(
		1*9.74e-05*3/0.002*3.42E-14 * np.trapz(quantuum_efficiency(x)*function_of_reflection(x)*2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+reflection_temperature)))-1)),x=x) 	# component from reflections
		+ 3.42E-14 * np.trapz(quantuum_efficiency(x)*2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp+delta_t)))-1)),x=x))	# 3.42E-14 is the proportionality coefficient from the temperature calibration
		# x0=1*int_time*(1*1430/0.002 + 3.42E-14 * np.trapz(2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp)))-1)),x=x))
		# x1=1*int_time*(1*1430/0.002 + 3.42E-14 * np.trapz(2*3.14* 299792458 /(x**4 * (exp(6.62607015E-34*299792458/(x*1.380649E-23*(273+min_temp+delta_t)))-1)),x=x))
		# x0 += 1500
		# x1 += 1500
		# print(x1-x0)
		# print((x1-x0)/x0 * 100)
		noise = 2#(max(coleval.estimate_counts_std(x0),5)**2+max(coleval.estimate_counts_std(x1),5)**2)**0.5	# noise with no binning
		noise /= ((min(1/int_time,new_freq)/old_freq)**0.5)
		noise /= ((new_res_max/old_res_max))	# I assume the additional pixels available scale proportionally in both dimensions, so i have (1/x**2)**0.5 = 1/x reduction of noise, with x=new res / old res
		temp.append((x1-x0)/x0 * 100)	# this must be the % of counts increase, basically the contrast
		temp1.append(x1)	# these are the max counts. it has to be lower than the max allowed by the number of bits, usually 14, therefore 2**14=16384. if that is too high the integration time has to decrease accordingly
		temp2.append((x1-x0)*11000/(max(x0,11000))/noise)	# this is the contrast divided by the noise, with the noise reduced based on increased spatial and temporal resolution. this is scaled to 11000 counts. this is to account that the int time has to be reduced if x0> some value to avoid saturation. here it is scaled so that x0<11000
	ax[0].plot(wave_range,temp,label='min_t=%.3g, d_w=%.3g, d_t=%.3g, int=%.3g' %(min_temp,delta_wave,delta_t,int_time))
	ax[1].plot(wave_range,temp1,label='min_t=%.3g, d_w=%.3g, d_t=%.3g, int=%.3g' %(min_temp,delta_wave,delta_t,int_time))
	ax[2].plot(wave_range,temp2,label='min_t=%.3g, d_w=%.3g, d_t=%.3g, int=%.3g' %(min_temp,delta_wave,delta_t,int_time))
	ax[1].set_ylim(bottom=1000,top=30000)
	ax[2].set_ylim(bottom=10)
	ax[0].set_ylabel('contrast %')
	ax[1].set_ylabel('counts max')
	ax[2].set_ylabel('contrast / noise scaled')
	ax[2].set_xlabel('start of the wavelength range [nm]')

	ax[0].legend(loc='best', fontsize='xx-small')
else:
	pass




##	Here I put the code for examining simple samples and check the performance of different camera models.
# FLIR

if False:
	# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-06-07'
	# filenames = ['0-SSD ','2-SSD ','7-SSD ','SSD -13','SSD -19','SSD -24','SSD -5','10-SSD ','3a-SSD ','8-SSD ','SSD -14','SSD -1','SSD -25','SSD -6','11-SSD ','3-SSD ','9-SSD ','SSD -15','SSD -20','SSD -26','SSD -7','12-SSD ','4-SSD ','SSD -10','SSD -16','SSD -21','SSD -2','SSD -8','13-SSD ','5-SSD ','SSD -11','SSD -17','SSD -22','SSD -3','SSD -9','1-SSD ','6-SSD ','SSD -12','SSD -18','SSD -23','SSD -4']

	# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-05-31'
	# filenames = ['1-SSD','Rec-0011','Rec-0007','Rec-0008','Rec-0009','Rec-0010','Rec-0014','Rec-0015','Rec-0016','Rec-0017','Rec-0018','Rec-0022','Rec-0023','Rec-0024','Rec-0025','Rec-0026','Rec-0019','Rec-0027','Rec-0035','Untitled-1 copy','2-SSD','Rec-0012','Rec-0020','Rec-0028','Rec-0036','Untitled-2','Untitled-3','Untitled-4','Rec-0006','Rec-0013','Rec-0021','Rec-0029','Rec-0037','Untitled-3','Rec-0007','Rec-0014','Rec-0022','Rec-0030','SSD-1','Untitled-4','Rec-0008','Rec-0015','Rec-0023','Rec-0031','SSD-2','Rec-0009','Rec-0016','Rec-0024','Rec-0032','SSD-3','Rec-0009.npz','Rec-0017','Rec-0025','Rec-0033','SSD-4','Rec-0010','Rec-0018','Rec-0026','Rec-0034']

	# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/Infratec 2023-06-12'
	# filenames = ['UKAEA_0002','UKAEA_0005','UKAEA_0008','UKAEA_0011','UKAEA_0014','UKAEA_0001','UKAEA_0003','UKAEA_0006','UKAEA_0009','UKAEA_0012','UKAEA_0015','UKAEA_0004','UKAEA_0007','UKAEA_0010','UKAEA_0013','UKAEA_0016']

	# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/Infratec 2023-07-12/IIR8300hs/converted'
	# filenames = ['1000Hz_450,0µs_ImageIR8300hs','1000Hz_600,0µs_ImageIR8300hs','1000Hz_700,0µs_ImageIR8300hs','guidance.txt','1000Hz_450,0µs_ImageIR8300hs_Hand','1000Hz_600,0µs_ImageIR8300hs_Hand','1000Hz_700,0µs_ImageIR8300hs_Hand','1000Hz_450,0µs_3000frames']

	# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/Infratec 2023-07-12/IIR9400hs/converted'
	# filenames = ['600Hz_1050,0µs_ImageIR9400hs','600Hz_1300,0µs_ImageIR9400hs','600Hz_800,0µs_ImageIR9400hs','600Hz_1050,0µs_ImageIR9400hs_Hand','600Hz_1300,0µs_ImageIR9400hs_Hand','600Hz_800,0µs_ImageIR9400hs_Hand']

	# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/IRVB1_2023-05-31_test'
	# filenames = ['IRVB-test-000003','IRVB-test-000004','IRVB-test-000005','IRVB-test-000006','IRVB-test-000007']

	path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-08-25_X6981'
	filenames = ['100frames','100nohand','NUC','unNUCed','very_long_record_1']


	# for filename in filenames:
	for filename in np.flip(filenames,axis=0):
		laser_to_analyse = path +'/' + filename
		try:
			try:
				laser_dict = np.load(laser_to_analyse+'.npz')
				laser_dict.allow_pickle=True
				laser_digitizer_ID = laser_dict['uniques_digitizer_ID']
				if laser_dict['FrameRate']==0:
					bla=sga	# I want an errer to continue
				print(laser_to_analyse+' OK')
				# if laser_to_analyse[-4:]=='.ptw':
				# 	test = laser_dict['discarded_frames']
			except Exception as e:
				print('Error '+str(e))
				print('missing '+laser_to_analyse+'.npz'+' file. rigenerated')
				full_saved_file_dict = coleval.read_IR_file(laser_to_analyse,force_regeneration=True)
				# np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				# laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
				# laser_dict.allow_pickle=True
				if False:
					laser_dict = np.load(laser_to_analyse+'.npz')
					laser_dict.allow_pickle=True
					laser_dict = dict(laser_dict)
					data = laser_dict['data'] + laser_dict['data_median']
					FrameRate = laser_dict['FrameRate']
					IntegrationTime = laser_dict['IntegrationTime']


					filtered_data = median_filter(data[:,:,:],size=[1,5,5])
					dead = np.mean(data-filtered_data,axis=0)>100

					noise = np.median(np.std(data[:,np.logical_not(dead)],axis=0))
					noise_dead_pix = np.median(np.std(data[:],axis=0))

					plt.figure()
					plt.title(laser_to_analyse+'\nFrameRate=%.4gHz, IntegrationTime=%.4gms\n noise=%.4g noise dead=%.4g' %(FrameRate,IntegrationTime,noise,noise_dead_pix))
					# plt.imshow(data_1[10]-data[10])#[250:,:400])
					plt.imshow(np.median(filtered_data,axis=0),'rainbow')#[250:,:400])
					plt.colorbar()
					plt.savefig(laser_to_analyse+'_average_FrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime)+'.eps', bbox_inches='tight')
					plt.close()

					spectra_orig=np.fft.fft(np.mean(filtered_data[:,10:11,10:11],axis=(-1,-2)))
					# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
					# spectra_orig=np.fft.fft(data[:,50,50])
					magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
					# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
					freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
					magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
					freq = np.sort(freq)

					plt.figure(figsize=(12,7))
					plt.title(laser_to_analyse+'\nFrameRate=%.4gHz, IntegrationTime=%.4gms\n noise=%.4g noise dead=%.4g' %(FrameRate,IntegrationTime,noise,noise_dead_pix))
					plt.plot(freq,magnitude)
					plt.semilogy()
					plt.grid()
					plt.xlim(left=-10)#,right=100)
					plt.savefig(laser_to_analyse+'_spectra_FrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime)+'.eps', bbox_inches='tight')
					plt.close()

					plt.figure(figsize=(12,7))
					plt.title(laser_to_analyse+'\nFrameRate=%.4gHz, IntegrationTime=%.4gms\n noise=%.4g noise dead=%.4g' %(FrameRate,IntegrationTime,noise,noise_dead_pix))
					try:
						plt.plot(filtered_data[:,300,300]-np.median(filtered_data[:,300,300]))
					except:
						pass
					try:
						plt.plot(filtered_data[:,100,100]-np.median(filtered_data[:,100,100]))
					except:
						pass
					try:
						plt.plot(filtered_data[:,100,500]-np.median(filtered_data[:,100,500]))
					except:
						pass
					try:
						plt.plot(filtered_data[:,400,100]-np.median(filtered_data[:,400,100]))
					except:
						pass
					try:
						plt.plot(filtered_data[:,400,400]-np.median(filtered_data[:,400,400]))
					except:
						pass
					plt.savefig(laser_to_analyse+'_look_for_jumps_FrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime)+'.eps', bbox_inches='tight')
					plt.close()

		except Exception as e:
			print('Error '+str(e))
			print('missing '+laser_to_analyse+' file not found or some other problem')

	# full_saved_file_dict = dict(laser_dict)
	exit()
else:
	pass

#cSecond process

path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-06-07'
filenames = ['0-SSD ','2-SSD ','7-SSD ','SSD -13','SSD -19','SSD -24','SSD -5','10-SSD ','3a-SSD ','8-SSD ','SSD -14','SSD -1','SSD -25','SSD -6','11-SSD ','3-SSD ','9-SSD ','SSD -15','SSD -20','SSD -26','SSD -7','12-SSD ','4-SSD ','SSD -10','SSD -16','SSD -21','SSD -2','SSD -8','13-SSD ','5-SSD ','SSD -11','SSD -17','SSD -22','SSD -3','SSD -9','1-SSD ','6-SSD ','SSD -12','SSD -18','SSD -23','SSD -4']
filename = '7-SSD '
# filename = 'SSD -22'

# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-05-31'
# filenames = ['1-SSD','Rec-0011','Rec-0007','Rec-0008','Rec-0009','Rec-0010','Rec-0014','Rec-0015','Rec-0016','Rec-0017','Rec-0018','Rec-0022','Rec-0023','Rec-0024','Rec-0025','Rec-0026','Rec-0019','Rec-0027','Rec-0035','Untitled-1 copy','2-SSD','Rec-0012','Rec-0020','Rec-0028','Rec-0036','Untitled-2','Untitled-3','Untitled-4','Rec-0006','Rec-0013','Rec-0021','Rec-0029','Rec-0037','Untitled-3','Rec-0007','Rec-0014','Rec-0022','Rec-0030','Untitled-4','Rec-0008','Rec-0015','Rec-0023','Rec-0031','Rec-0009','Rec-0016','Rec-0024','Rec-0032','Rec-0009.npz','Rec-0017','Rec-0025','Rec-0033','Rec-0010','Rec-0018','Rec-0026','Rec-0034']
# # filename = 'SSD -4'
# filename = 'Rec-0011'

# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/IRVB1_2023-05-31_test'
# filenames = ['IRVB-test-000003','IRVB-test-000004','IRVB-test-000005','IRVB-test-000006','IRVB-test-000007']
# filename = 'IRVB-test-000007'

# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/Infratec 2023-07-12/IIR8300hs/converted'
# filenames = ['1000Hz_450,0µs_ImageIR8300hs','1000Hz_600,0µs_ImageIR8300hs','1000Hz_700,0µs_ImageIR8300hs','1000Hz_450,0µs_ImageIR8300hs_Hand','1000Hz_600,0µs_ImageIR8300hs_Hand','1000Hz_700,0µs_ImageIR8300hs_Hand','1000Hz_450,0µs_3000frames','1000Hz_700,0µs_3000frames']
# # filenames = ['1000Hz_450,0µs_ImageIR8300hs','1000Hz_600,0µs_ImageIR8300hs','1000Hz_700,0µs_ImageIR8300hs']
# filename = '1000Hz_600,0µs_ImageIR8300hs_Hand'
# frequency_all = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]


# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/Infratec 2023-07-12/IIR9400hs/converted'
# filenames = ['600Hz_1050,0µs_ImageIR9400hs','600Hz_1300,0µs_ImageIR9400hs','600Hz_800,0µs_ImageIR9400hs','600Hz_1050,0µs_ImageIR9400hs_Hand','600Hz_1300,0µs_ImageIR9400hs_Hand','600Hz_800,0µs_ImageIR9400hs_Hand']
# # filenames = ['600Hz_800,0µs_ImageIR9400hs','600Hz_1050,0µs_ImageIR9400hs','600Hz_1300,0µs_ImageIR9400hs']
# # filename = '600Hz_1050,0µs_ImageIR9400hs'
# frequency_all = [600,600,600,600,600,600]

# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/Infratec 2023-06-12'
# filenames = ['UKAEA_0002','UKAEA_0005','UKAEA_0008','UKAEA_0011','UKAEA_0014','UKAEA_0001','UKAEA_0003','UKAEA_0006','UKAEA_0009','UKAEA_0012','UKAEA_0015','UKAEA_0004','UKAEA_0007','UKAEA_0010','UKAEA_0013','UKAEA_0016']
# filename = 'UKAEA_0010'
# frequency_all = [150,150,150,150,150,150,150,150,467,600,600,600,600,600,600,600,600]

# path = '/home/ffederic/work/irvb/IRVB2 camera investigation/FLIR-2023-08-25_X6981'
# filenames = ['100frames','100nohand','NUC','unNUCed']

#
# for i_,filename in enumerate(filenames):
for i_,filename in enumerate(np.flip(filenames,axis=0)):
	try:
		laser_to_analyse = path +'/' + filename
		print(laser_to_analyse)
		laser_dict = np.load(laser_to_analyse+'.npz')
		laser_dict.allow_pickle=True
		laser_dict = dict(laser_dict)
		data = laser_dict['data'] + laser_dict['data_median']
		width = laser_dict['width']
		height = laser_dict['height']
		FrameRate = laser_dict['FrameRate']
		IntegrationTime = laser_dict['IntegrationTime']
		if FrameRate==0:
			print(laser_to_analyse)
			print('framerate externally provided')
			FrameRate = frequency_all[i_]
			# continue

		filtered_data = median_filter(data[:,:,:],size=[1,5,5])
		dead_pixels = np.mean(data-filtered_data,axis=0)>100

		dead_frame = np.zeros((len(data))).astype(bool)
		plt.figure(figsize=(12,7))
		plt.title(laser_to_analyse+'\nFrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime))
		try:
			temp_filtered_data = filtered_data[:,300,300]
			temp_filtered_data = temp_filtered_data-np.median(temp_filtered_data)
			dead_frame[1:] = np.logical_or(dead_frame[1:],np.abs(np.diff(temp_filtered_data)) > 100)
			plt.plot(temp_filtered_data)
		except:
			pass
		try:
			temp_filtered_data = filtered_data[:,100,100]
			temp_filtered_data = temp_filtered_data-np.median(temp_filtered_data)
			dead_frame[1:] = np.logical_or(dead_frame[1:],np.abs(np.diff(temp_filtered_data)) > 100)
			plt.plot(temp_filtered_data)
		except:
			pass
		try:
			temp_filtered_data = filtered_data[:,100,500]
			temp_filtered_data = temp_filtered_data-np.median(temp_filtered_data)
			dead_frame[1:] = np.logical_or(dead_frame[1:],np.abs(np.diff(temp_filtered_data)) > 100)
			plt.plot(temp_filtered_data)
		except:
			pass
		try:
			temp_filtered_data = filtered_data[:,400,100]
			temp_filtered_data = temp_filtered_data-np.median(temp_filtered_data)
			dead_frame[1:] = np.logical_or(dead_frame[1:],np.abs(np.diff(temp_filtered_data)) > 100)
			plt.plot(temp_filtered_data)
		except:
			pass
		try:
			temp_filtered_data = filtered_data[:,400,400]
			temp_filtered_data = temp_filtered_data-np.median(temp_filtered_data)
			dead_frame[1:] = np.logical_or(dead_frame[1:],np.abs(np.diff(temp_filtered_data)) > 100)
			plt.plot(temp_filtered_data)
		except:
			pass
		if np.sum(dead_frame)>0:
			for position in np.arange(len(dead_frame))[dead_frame]:
				plt.axvline(x=position,linestyle='--',color='k')
		plt.savefig(laser_to_analyse+'_look_for_jumps_FrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime)+'.eps', bbox_inches='tight')
		plt.close()

		filtered_data[dead_frame] = median_filter(filtered_data,size=[5,1,1])[dead_frame]
		data[dead_frame] = median_filter(data,size=[5,1,1])[dead_frame]

		noise = np.median(np.std(data[:,np.logical_not(dead_pixels)],axis=0))
		noise_dead_pix = np.median(np.std(data,axis=0))

		plt.figure()
		plt.title(laser_to_analyse+'\nFrameRate=%.4gHz, IntegrationTime=%.4gms\n noise=%.4g noise dead=%.4g' %(FrameRate,IntegrationTime,noise,noise_dead_pix))
		# plt.imshow(data_1[10]-data[10])#[250:,:400])
		plt.imshow(np.median(filtered_data,axis=0),'rainbow')#[250:,:400])
		plt.colorbar()
		plt.savefig(laser_to_analyse+'_average_FrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime)+'.eps', bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(12,7))
		plt.title(laser_to_analyse+'\nFrameRate=%.4gHz, IntegrationTime=%.4gms\n noise=%.4g noise dead=%.4g' %(FrameRate,IntegrationTime,noise,noise_dead_pix))
		spectra_orig=np.fft.fft(np.mean(filtered_data[:],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(data[:,50,50])
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
		freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		plt.plot(freq,magnitude,label='full')

		width_4 = int(width/4)
		height_4 = int(height/4)
		spectra_orig=np.fft.fft(np.mean(filtered_data[:,height_4:height_4*3,width_4:width_4*3],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(data[:,50,50])
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
		freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		plt.plot(freq,magnitude,'--',label='centre')

		spectra_orig=np.fft.fft(np.mean(filtered_data[:,:height_4*2,:width_4*2],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(data[:,50,50])
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
		freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		plt.plot(freq,magnitude,'--',label='lowlow')

		spectra_orig=np.fft.fft(np.mean(filtered_data[:,:height_4*2,width_4*2:],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(data[:,50,50])
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
		freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		plt.plot(freq,magnitude,'--',label='lowhigh')

		spectra_orig=np.fft.fft(np.mean(filtered_data[:,height_4*2:,:width_4*2],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(data[:,50,50])
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
		freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		plt.plot(freq,magnitude,'--',label='highlow')

		spectra_orig=np.fft.fft(np.mean(filtered_data[:,height_4*2:,width_4*2:],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
		# spectra_orig=np.fft.fft(data[:,50,50])
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
		freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
		magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
		freq = np.sort(freq)
		plt.plot(freq,magnitude,'--',label='highhigh')

		plt.legend(loc='best', fontsize='xx-small')
		plt.semilogy()
		plt.grid()
		plt.xlim(left=-10)#,right=100)
		plt.savefig(laser_to_analyse+'_spectra_FrameRate=%.4gHz, IntegrationTime=%.4gms' %(FrameRate,IntegrationTime)+'.eps', bbox_inches='tight')
		plt.close()
	except Exception as e:
		print('Error '+str(e))

exit()

plt.figure()
# plt.imshow(data_1[10]-data[10])#[250:,:400])
plt.imshow(data[50])#[250:,:400])
plt.colorbar()
# plt.pause(0.01)

plt.figure()
plt.imshow(median_filter(np.std(data[:,:,:],axis=0),size=[3,3]))
plt.colorbar()

plt.figure()
plt.plot(data[:,300,300])
plt.plot(data[:,100,100])
plt.plot(data[:,100,500])
plt.plot(data[:,400,100])
plt.plot(data[:,400,400])



spectra_orig=np.fft.fft(np.mean(median_filter(data[:,:,:],size=[1,3,3]),axis=(-1,-2)))
# spectra_orig=np.fft.fft(np.mean(data[:,50:60,50:60],axis=(-1,-2)))
# spectra_orig=np.fft.fft(data[:,50,50])
magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
# freq = np.fft.fftfreq(len(magnitude), d=1/1000)
freq = np.fft.fftfreq(len(magnitude), d=1/FrameRate)
magnitude = np.array([y for _, y in sorted(zip(freq, magnitude))])
freq = np.sort(freq)

plt.figure()
plt.title(laser_to_analyse)
plt.plot(freq,magnitude)
plt.semilogy()
plt.grid()
plt.xlim(left=-10)#,right=100)


# how big do I want the foil?


stand_off_length = 0.06	# m	# >= MU02
# stand_off_length = 0.045	# m	# = MU01
# stand_off_length = 0.075	# m	# MU?

pinhole_radious = 0.02

# MU01/2/3
pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z
# FROM MU04
# pinhole_offset = np.array([0.0198,-0.0198])	# toroidal direction parallel to the place surface, z

# pinhole_location = np.array([-1.04087,1.068856,-0.7198])	# x,y,z
pinhole_location = coleval.locate_pinhole(pinhole_offset=pinhole_offset)	# x,y,z

Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
# foil_size = [0.07,0.09]
foil_size = [0.13,0.13]
R_centre_column = 0.2608	# m
pinhole_relative_location = np.array(foil_size)/2 -pinhole_offset #+ 0.0198

cv0 = np.zeros((np.array(foil_size)*100*100).astype(int)).T

structure_point_location_on_foil = coleval.return_structure_point_location_on_foil(plane_equation=plane_equation,centre_of_foil=centre_of_foil,pinhole_offset=pinhole_offset,foil_size=foil_size)
fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil(plane_equation=plane_equation,pinhole_offset=pinhole_offset,centre_of_foil=centre_of_foil,foil_size=foil_size)

structure_alpha = 1

fig, ax = plt.subplots(figsize=(7,7))
# plt.figure(figsize=(7,7))
# for i in range(len(fueling_point_location_on_foil)):
# 	# plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
# 	# plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
# 	plt.plot(np.array(fueling_point_location_on_foil[i][:,0]),np.array(fueling_point_location_on_foil[i][:,1]),'+k',markersize=40,alpha=structure_alpha)
# 	plt.plot(np.array(fueling_point_location_on_foil[i][:,0]),np.array(fueling_point_location_on_foil[i][:,1]),'ok',markersize=5,alpha=structure_alpha)
for i in range(len(structure_point_location_on_foil)):
	# plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	plt.plot(np.array(structure_point_location_on_foil[i][:,0]),np.array(structure_point_location_on_foil[i][:,1]),'--k',alpha=structure_alpha)
plt.plot([0,foil_size[0],foil_size[0],0,0],[0,0,foil_size[1],foil_size[1],0],'k')
plt.plot(foil_size[0]/2+np.linspace(-0.115/2,0.115/2,100),foil_size[1]/2+((0.115/2)**2-(np.linspace(-0.115/2,0.115/2,100))**2)**0.5,'k')	# max diameter circle
plt.plot(foil_size[0]/2+np.linspace(-0.115/2,0.115/2,100),foil_size[1]/2-((0.115/2)**2-(np.linspace(-0.115/2,0.115/2,100))**2)**0.5,'k')	# max diameter circle
# MU01/2/3 foil size
# plt.plot([foil_size[0]/2-0.07/2,foil_size[0]/2+0.07/2,foil_size[0]/2+0.07/2,foil_size[0]/2-0.07/2,foil_size[0]/2-0.07/2],[foil_size[1]/2-0.09/2,foil_size[1]/2-0.09/2,foil_size[1]/2+0.09/2,foil_size[1]/2+0.09/2,foil_size[1]/2-0.09/2],'k')


try:
	gna = efit_reconstruction
except:
	name = 'IRVB-MASTU_shot-45371.ptw'
	path = '/home/ffederic/work/irvb/MAST-U/'
	i_day,day = 0,coleval.retrive_shot_date_and_time(name[-9:-4])[0]
	laser_to_analyse=path+day+'/'+name

	pass_number = 0
	full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
	full_saved_file_dict_FAST.allow_pickle=True
	full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
	try:
		full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
	except:
		full_saved_file_dict_FAST['multi_instrument'] = dict([])
	if pass_number==0:
		full_saved_file_dict_FAST['first_pass'] = full_saved_file_dict_FAST['first_pass'].all()
		inverted_dict = full_saved_file_dict_FAST['first_pass']['inverted_dict']
	else:
		full_saved_file_dict_FAST['second_pass'] = full_saved_file_dict_FAST['second_pass'].all()
		inverted_dict = full_saved_file_dict_FAST['second_pass']['inverted_dict']

	grid_resolution = 2	# cm
	time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
	inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
	inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
	binning_type = inverted_dict[str(grid_resolution)]['binning_type']

	EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
	efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
	inversion_R = inverted_dict[str(grid_resolution)]['geometry']['R']
	inversion_Z = inverted_dict[str(grid_resolution)]['geometry']['Z']


all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil,foil_size=foil_size)
plot1 = plt.plot(0,0,'-r', alpha=1)[0]

all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil,foil_size=foil_size)
plot2 = plt.plot(0,0,'--r', alpha=1)[0]

all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)

all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil,foil_size=foil_size)
plot3 = plt.plot(0,0,'xr',markersize=20, alpha=1)[0]
plot4 = []
for __i in range(len(all_time_strike_points_location_rot[0])):
	plot4.append(plt.plot(0,0,'-r', alpha=1)[0])

all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil,foil_size=foil_size)
plot5 = []
for __i in range(len(all_time_separatrix[0])):
	plot5.append(plt.plot(0,0,'--b', alpha=1)[0])


i_time = np.abs(0.4-efit_reconstruction.time).argmin()
if np.sum(np.isnan(all_time_x_point_location[i_time]))>=len(all_time_x_point_location[i_time]):	# means that all the points calculated are outside the foil
	plot1.set_data(([],[]))
else:
	plot1.set_data((all_time_x_point_location[i_time][:,0],all_time_x_point_location[i_time][:,1]))
# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
# 	plot2.set_data(([],[]))
# else:
plot2.set_data((all_time_mag_axis_location[i_time][:,0],all_time_mag_axis_location[i_time][:,1]))
# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
# 	plot3.set_data(([],[]))
# 	for __i in range(len(plot4)):
# 		plot4[__i].set_data(([],[]))
# else:
plot3.set_data((all_time_strike_points_location[i_time][:,0],all_time_strike_points_location[i_time][:,1]))
for __i in range(len(plot4)):
	plot4[__i].set_data((all_time_strike_points_location_rot[i_time][__i][:,0],all_time_strike_points_location_rot[i_time][__i][:,1]))
# if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
for __i in range(len(plot5)):
	plot5[__i].set_data((all_time_separatrix[i_time][__i][:,0],all_time_separatrix[i_time][__i][:,1]))

plt.gca().set_aspect('equal')
plt.xlim(left=0,right=foil_size[0])
plt.ylim(bottom=0,top=foil_size[1])

labels = [str(np.round(val,2)) for val in np.arange(7)*0.02-foil_size[0]/2]
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)


####
