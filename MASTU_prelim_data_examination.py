# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

# added to reat the .ptw
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
import pyradi.ryptw as ryptw

# degree of polynomial of choice
n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']
f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams):
	f.append(dirnames)
parameters_available = f[0]
parameters_available_int_time = []
parameters_available_framerate = []
for path in parameters_available:
	parameters_available_int_time.append(np.float(path[:path.find('ms')]))
	parameters_available_framerate.append(np.float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time = np.array(parameters_available_int_time)
parameters_available_framerate = np.array(parameters_available_framerate)


path = '/home/ffederic/work/irvb/MAST-U/'
to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25']
to_do = np.flip(to_do,axis=0)
# to_do = ['2021-06-03']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']

seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s


f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.append(dirnames)
days_available = f[0]
shot_available = []
for i_day,day in enumerate(to_do):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path+day+'/'):
		f.append(filenames)
	shot_available.append([])
	for name in f[0]:
		if name[-3:]=='ats' or name[-3:]=='ptw':
			shot_available[i_day].append(name)


every_pixel_independent = False
for i_day,day in enumerate(to_do):
# for i_day,day in enumerate(np.flip(to_do,axis=0)):
	# for name in shot_available[i_day]:
	for name in np.flip(shot_available[i_day],axis=0):
		laser_to_analyse=path+day+'/'+name
		print('starting ' + laser_to_analyse)

		try:
			laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
			a=laser_dict['IntegrationTime']
		except:
			print('missing '+laser_to_analyse[:-4]+'.npz'+' file. rigenerated')
			if laser_to_analyse[-4:]=='.ats':
				full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse)
			else:
				full_saved_file_dict = coleval.ptw_to_dict(laser_to_analyse)
			np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
			laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
		full_saved_file_dict = dict(laser_dict)

		if laser_to_analyse[-9:-4] in list(MASTU_shots_timing.keys()):
			start_time_of_pulse = MASTU_shots_timing[laser_to_analyse[-9:-4]]['pulse_start']	# s
		else:
			start_time_of_pulse = 2.5	# s

		try:
			data_median = full_saved_file_dict['data_median']
			del data_median
		except:
			print('shrinking '+laser_to_analyse)
			data = full_saved_file_dict['data']
			data_median = int(np.median(data))
			if np.abs(data-data_median).max()<2**8/2-1:
				data_minus_median = (data-data_median).astype(np.int8)
			elif np.abs(data-data_median).max()<2**16/2-1:
				data_minus_median = (data-data_median).astype(np.int16)
			elif np.abs(data-data_median).max()<2**32/2-1:
				data_minus_median = (data-data_median).astype(np.int32)
			full_saved_file_dict['data_median'] = data_median
			full_saved_file_dict['data'] = data_minus_median
			np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
			del data,data_median,data_minus_median

		try:

			laser_counts, laser_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
			time_of_experiment = full_saved_file_dict['time_of_measurement']
			laser_digitizer = full_saved_file_dict['digitizer_ID']
			SensorTemp_0 = full_saved_file_dict['SensorTemp_0']
			SensorTemp_3 = full_saved_file_dict['SensorTemp_3']
			DetectorTemp = full_saved_file_dict['DetectorTemp']
			time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)
			for i in range(len(laser_digitizer_ID)):
				if np.diff(time_of_experiment_digitizer_ID[i])[0]<0:
					laser_counts[i] = laser_counts[i][1:]
					SensorTemp_0 = SensorTemp_0[time_of_experiment_digitizer_ID[i][0] != time_of_experiment]
					SensorTemp_3 = SensorTemp_3[time_of_experiment_digitizer_ID[i][0] != time_of_experiment]
					DetectorTemp = DetectorTemp[time_of_experiment_digitizer_ID[i][0] != time_of_experiment]
					time_of_experiment_digitizer_ID[i] = time_of_experiment_digitizer_ID[i][1:]
			time_of_experiment = np.sort(np.concatenate(time_of_experiment_digitizer_ID))

			laser_framerate = 1e6/np.mean(np.sort(np.diff(time_of_experiment))[2:-2])
			laser_int_time = full_saved_file_dict['IntegrationTime']

			try:
				full_correction_coefficients = full_saved_file_dict['full_correction_coeff']
				full_correction_coefficients_present = True
			except:
				full_correction_coefficients_present = False
				full_correction_coefficients = np.zeros((2,*np.shape(laser_counts[0])[1:],6))*np.nan

			try:
				temp = full_saved_file_dict['aggregated_correction_coeff'][0][5]
				aggregated_correction_coefficients_present = True
				aggregated_correction_coefficients = full_saved_file_dict['aggregated_correction_coeff']
			except:
				aggregated_correction_coefficients_present = False
				aggregated_correction_coefficients = np.zeros((2,6))*np.nan

			fig, ax = plt.subplots( 4,1,figsize=(10, 24), squeeze=False, sharex=True)
			plot_index = 0
			horizontal_coord = np.arange(np.shape(laser_counts[0])[2])
			vertical_coord = np.arange(np.shape(laser_counts[0])[1])
			horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
			select = np.logical_or(np.logical_or(vertical_coord<30,vertical_coord>240),np.logical_or(horizontal_coord<30,horizontal_coord>290))
			startup_correction = []
			laser_counts_corrected = []
			temp=0
			exponential = lambda t,c1,c2,c3,c4,t0: -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))
			exponential_biased = lambda t,c1,c2,c3,c4,c5,t0: -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))+c5*(t-t0)
			external_clock_marker = False
			for i in range(len(laser_digitizer_ID)):
				reference = np.mean(laser_counts[i][:,select],axis=-1)
				if np.abs(reference[-1]-reference[10])>30:
					external_clock_marker = True
				laser_counts_corrected.append(laser_counts[i].astype(np.float))
				temp = max(temp,np.max(np.mean(laser_counts[i][10:],axis=(-1,-2))))
				time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
				bds = [[0,0,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]]
				guess=[100,1,0,0,time_of_experiment_digitizer_ID_seconds[10],0]
				select_time = np.logical_or(time_of_experiment_digitizer_ID_seconds<start_time_of_pulse,time_of_experiment_digitizer_ID_seconds>start_time_of_pulse+5)	# I avoid the period with the plasma
				select_time[:8]=False	# to avoid the very firsts frames that can have unreasonable counts
				ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,np.mean(laser_counts[i],axis=(-1,-2)),color=color[i],label='all foil mean DIG'+str(laser_digitizer_ID[i]))
				ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,reference,'--',color=color[i],label='out of foil area DIG'+str(laser_digitizer_ID[i]))
				if every_pixel_independent:	# I do the startup correction for every pixel independently
					for v in range(np.shape(laser_counts[0])[1]):
						for h in range(np.shape(laser_counts[0])[2]):
							if full_correction_coefficients_present:
								laser_counts_corrected[i][:,v,h] += -exponential_biased(time_of_experiment_digitizer_ID_seconds,*full_correction_coefficients[i,v,h])
							else:
								try:
									fit = curve_fit(exponential_biased, time_of_experiment_digitizer_ID_seconds[select_time], laser_counts[i][select_time,v,h]-np.mean(laser_counts[i][-int(1*laser_framerate/len(laser_digitizer_ID)):,v,h]), p0=guess,bounds=bds,maxfev=int(1e6))
									laser_counts_corrected[i][:,v,h] += -exponential_biased(time_of_experiment_digitizer_ID_seconds,*fit[0])
									full_correction_coefficients[i,v,h] = fit[0]
									guess = fit[0]
								except:
									print('%.3gv,%.3gh failed' %(v,h))
									continue
				else:
					fit = curve_fit(exponential_biased, time_of_experiment_digitizer_ID_seconds[select_time], reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]), p0=guess,bounds=bds,maxfev=int(1e6))
					aggregated_correction_coefficients[i] = fit[0]
					ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,exponential_biased(time_of_experiment_digitizer_ID_seconds,*fit[0])+np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),':',color=color[i],label='fit of out of foil area')
					# startup_correction.append(np.mean(reference[(time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6>12]) - reference)
					# startup_correction.append(-exponential((time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6,*fit[0]))
					laser_counts_corrected[i] = (laser_counts_corrected[i].astype(np.float).T -exponential_biased(time_of_experiment_digitizer_ID_seconds,*fit[0])).T
					ax[plot_index,0].axhline(y=np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),linestyle=':',color=color[i])
					ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,np.mean(laser_counts_corrected[i][:,select],axis=-1),'-.',color=color[i],label='CORRECTED out of foil area DIG'+str(laser_digitizer_ID[i]))
			ax[plot_index,0].set_ylabel('mean counts [au]')
			fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz\n' %(laser_int_time/1000,laser_framerate) + str(aggregated_correction_coefficients))
			ax[plot_index,0].set_ylim(top=temp)
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
			ax[plot_index,0].grid()
			ax[plot_index,0].legend(loc='best', fontsize='x-small')

			plot_index += 1
			ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse,SensorTemp_0)
			ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
			ax[plot_index,0].grid()
			plot_index += 1
			ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse,SensorTemp_3)
			ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
			ax[plot_index,0].grid()
			plot_index += 1
			ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse,DetectorTemp)
			ax[plot_index,0].set_ylabel('detector temp [K]\n(DetectorTemp)')
			ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
			ax[plot_index,0].grid()
			ax[plot_index,0].set_xlabel('time [s]')

			full_saved_file_dict['external_clock_marker'] = external_clock_marker	# True if the camera is thought as in external clock
			if every_pixel_independent:	# I do the startup correction for every pixel independently
				full_saved_file_dict['full_correction_coeff'] = full_correction_coefficients	# it will be saved later anyways, no need to do it here
				full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))+c5*(t-t0)'
				# if full_correction_coefficients_present:
				# 	if np.sum((full_saved_file_dict['full_correction_coeff']!=full_correction_coefficients))>0:
				# 		np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				# else:
				# 	full_saved_file_dict['full_correction_coeff'] = full_correction_coefficients
				# 	full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
				# 	np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
			else:
				full_saved_file_dict['aggregated_correction_coeff'] = aggregated_correction_coefficients	# it will be saved later anyways, no need to do it here
				full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))+c5*(t-t0)'
				# if aggregated_correction_coefficients_present:
				# 	if np.sum((full_saved_file_dict['aggregated_correction_coeff']!=aggregated_correction_coefficients))>1:
				# 		np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				# else:
				# 	full_saved_file_dict['aggregated_correction_coeff'] = aggregated_correction_coefficients
				# 	full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
				# 	np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
				plt.savefig(laser_to_analyse[:-4]+'_1.eps', bbox_inches='tight')
			plt.close('all')

			ROI_horizontal = [220,238,50,100]
			# ROI_horizontal = [35,45,68,73]
			ROI_vertical = [100,170,28,45]
			plt.figure(figsize=(10, 5))
			for i in range(len(laser_digitizer_ID)):
				# horizontal_displacement = np.diff((np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2).T/(np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2)[:,0])).T,axis=1).argmin(axis=1)
				horizontal_displacement = np.diff((np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2)),axis=1).argmin(axis=1)
				horizontal_displacement = horizontal_displacement-horizontal_displacement[10]
				# vertical_displacement = np.diff((np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1).T/(np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1)[:,0])).T,axis=1).argmax(axis=1)
				vertical_displacement = np.diff((np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1)),axis=1).argmax(axis=1)
				vertical_displacement = vertical_displacement-vertical_displacement[10]
				time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
				if i==0:
					plt.plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,horizontal_displacement,label='horizontal',color=color[0])
					plt.plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,vertical_displacement,label='vertical',color=color[1])
				else:
					plt.plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,horizontal_displacement,color=color[0])
					plt.plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,vertical_displacement,color=color[1])
			plt.xlabel('time [s]')
			plt.ylabel('pixels of displacement [au]')
			plt.title('Displacement of the image\nlooking for the effect of disruptions')
			plt.legend(loc='best', fontsize='x-small')
			plt.savefig(laser_to_analyse[:-4]+'_2.eps', bbox_inches='tight')
			plt.close('all')

			if every_pixel_independent:	# I do the startup correction for every pixel independently
				for i in range(len(laser_digitizer_ID)):
					fig, ax = plt.subplots( 5,1,figsize=(5, 13), squeeze=False, sharex=True)
					fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz' %(laser_int_time/1000,laser_framerate) + '\ndigitizer '+str(laser_digitizer_ID[i]))

					plot_index = 0
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,0],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c1 [au]')
					ax[plot_index,0].set_title('correction coefficient c1 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,1],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c2 [1/s]')
					ax[plot_index,0].set_title('correction coefficient c2 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,2],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c3 [1/s^2]')
					ax[plot_index,0].set_title('correction coefficient c3 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,3],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('c4 [1/s^2]')
					ax[plot_index,0].set_title('correction coefficient c4 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')

					plot_index += 1
					ax[plot_index,0].imshow(full_correction_coefficients[i,:,:,4],'rainbow',origin='lower')
					ax[plot_index,0].colorbar().set_label('t0 [s]')
					ax[plot_index,0].set_title('correction coefficient t0 in\n-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))')
					ax[plot_index,0].set_xlabel('Horizontal axis [pixles]')
					ax[plot_index,0].set_ylabel('Vertical axis [pixles]')
					plt.savefig(laser_to_analyse[:-4]+'_corr_coeff_dig'+str(laser_digitizer_ID[i])+'.eps', bbox_inches='tight')
					plt.close('all')


			# plt.figure(figsize=(8, 5))
			# plt.imshow(laser_counts[0][3],'rainbow',origin='lower')
			# plt.colorbar().set_label('counts [au]')
			# # plt.title('Only foil in frame '+str(0)+' in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
			# plt.xlabel('Horizontal axis [pixles]')
			# plt.ylabel('Vertical axis [pixles]')
			# plt.pause(0.01)

			try:
				test = full_saved_file_dict['only_pulse_data'].all()['laser_temperature_std_minus_background_full_minus_median']['gdfgsdg']	# I want them all
				saved_file_dict_short = np.load(laser_to_analyse[:-4]+'_short.npz')
				temperature_minus_background = saved_file_dict_short['temperature_minus_background']
			except:
				temp = np.abs(parameters_available_int_time-laser_int_time/1000)<0.1
				framerate = np.array(parameters_available_framerate)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
				int_time = np.array(parameters_available_int_time)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
				temp = np.array(parameters_available)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
				print('parameters selected '+temp)

				# Load parameters
				temp = pathparams+'/'+temp+'/numcoeff'+str(n)+'/average'
				fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
				params_dict=np.load(fullpathparams)
				params = params_dict['coeff']
				errparams = params_dict['errcoeff']

				temp_counts = []
				temp_bad_pixels_flag = []
				temp_time = []
				temp_ref_counts = []
				temp_ref_counts_std = []
				full_saved_file_dict['only_pulse_data'] = dict([])
				# filter_plot_index = 1
				fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False)
				for i in range(len(laser_digitizer_ID)):
					# laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_corrected[i]],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=10/(laser_framerate/len(laser_digitizer_ID)),oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True)[0]
					# laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_corrected[i]],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=10/(laser_framerate/len(laser_digitizer_ID)),oscillation_search_window_end=2,plot_conparison=True)[0]
					# laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_corrected[i]],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=6,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True)[0]
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])] = dict([])
					time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
					if external_clock_marker:
						select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])-start_time_of_pulse>0,time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])-start_time_of_pulse<1.5)	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
						full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['time'] = (time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])-start_time_of_pulse)[select_time]
						if False:	# big failure, it fails to clean up the oscillation
							# with external clocks I still have the counts ramp up after the pulse, so I need to take only the flattest and part of the record
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_corrected[i]],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=8,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2],ROI=[30,240,30,290])[0]
							for j in range(2):
								fig = matplotlib.pyplot.gcf()
								fig.set_size_inches(15, 10, forward=True)
								plt.savefig(laser_to_analyse[:-4]+'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
								filter_plot_index+=1
								plt.close()
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=8,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2],ROI=[30,240,30,290])[0]
							for j in range(2):
								fig = matplotlib.pyplot.gcf()
								fig.set_size_inches(15, 10, forward=True)
								plt.savefig(laser_to_analyse[:-4]+'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
								filter_plot_index+=1
								plt.close()
							if (len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID))>14:	# the longer the record is the more single frequencies have to be removed to erase complitely the oshillation, so I think this is reasonable.
								laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=8,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2],ROI=[30,240,30,290])[0]
								for j in range(2):
									fig = matplotlib.pyplot.gcf()
									fig.set_size_inches(15, 10, forward=True)
									plt.savefig(laser_to_analyse[:-4]+'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
									filter_plot_index+=1
									plt.close()
						else:	# bare bone approach, maybe this work
							filter_agent = np.mean(laser_counts_corrected[i],axis=(-1,-2)) - generic_filter(np.mean(laser_counts_corrected[i],axis=(-1,-2)),np.nanmean,size=[7])
							laser_counts_filtered = (laser_counts_corrected[i].T - filter_agent).T
							filter_agent = np.mean(laser_counts_filtered,axis=(-1,-2)) - generic_filter(np.mean(laser_counts_filtered,axis=(-1,-2)),np.nanmean,size=[15])
							laser_counts_filtered = (laser_counts_filtered.T - filter_agent).T
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
					else:
						select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse>0,time_of_experiment_digitizer_ID_seconds-start_time_of_pulse<1.5)
						full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['time'] = (time_of_experiment_digitizer_ID_seconds-start_time_of_pulse)[select_time]
						filter_plot_index = 1
						if False:	# big failure, it fails to clean up the oscillation
							# with internal clocks there is no ramp up and I can use a larger portion of the record after the poulse, I only have to avoid the cooling of the foil
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_corrected[i]],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
							for trash in range(2):
								fig = matplotlib.pyplot.gcf()
								fig.set_size_inches(15, 10, forward=True)
								plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
								filter_plot_index+=1
								plt.close()
							for trash in range(5):
								laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
								for trash in range(2):
									fig = matplotlib.pyplot.gcf()
									fig.set_size_inches(15, 10, forward=True)
									plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
									filter_plot_index+=1
									plt.close()
						else:	# bare bone approach, maybe this work
							filter_agent = np.mean(laser_counts_corrected[i],axis=(-1,-2)) - generic_filter(np.mean(laser_counts_corrected[i],axis=(-1,-2)),np.nanmean,size=[7])
							laser_counts_filtered = (laser_counts_corrected[i].T - filter_agent).T
							filter_agent = np.mean(laser_counts_filtered,axis=(-1,-2)) - generic_filter(np.mean(laser_counts_filtered,axis=(-1,-2)),np.nanmean,size=[15])
							laser_counts_filtered = (laser_counts_filtered.T - filter_agent).T
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
							laser_counts_filtered = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False,which_plot=[1,2,3],force_poscentre=[np.shape(laser_counts_corrected[i])[1]-30,30],window=1000)[0]#,ROI=[30,240,30,290])[0]
					full_average = np.mean(laser_counts_corrected[i],axis=(-1,-2))
					full_spectra = np.fft.fft(full_average)
					full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
					full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
					ax[0,0].plot(full_freq,full_magnitude/(100**i),color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
					temp = (full_average.max()-full_average.min())
					ax[1,0].plot(time_of_experiment_digitizer_ID_seconds,full_average-temp*i,color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
					full_average = np.mean(laser_counts_filtered,axis=(-1,-2))
					full_spectra = np.fft.fft(full_average)
					full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
					full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
					ax[0,0].plot(full_freq,full_magnitude/(100**i),'k--')
					ax[1,0].plot(time_of_experiment_digitizer_ID_seconds,full_average-temp*i,'k--')

					temp_time.append(full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['time'])
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['counts'] = np.float16(laser_counts_filtered[select_time])
					temp_counts.append(laser_counts_filtered[select_time])
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['ref_counts'] = np.float16(np.mean(laser_counts_filtered[-int(1*laser_framerate/len(laser_digitizer_ID)):],axis=0))
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['ref_counts_std'] = np.float16(np.std(laser_counts_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
					bad_pixels_flag = coleval.find_dead_pixels([laser_counts_corrected[i][-int(seconds_for_bad_pixels*laser_framerate/len(laser_digitizer_ID)):]],treshold_for_bad_difference=100)
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['bad_pixels_flag'] = bad_pixels_flag
					temp_bad_pixels_flag.append(bad_pixels_flag)

					temp_ref_counts.append(np.mean(laser_counts_filtered[-int(1*laser_framerate/len(laser_digitizer_ID)):],axis=0))
					temp_ref_counts_std.append(np.std(laser_counts_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))	# need to modify count_to_temp_poly_multi_digitizer to accept 2 backgrounds rather than 1 for both digitizers
				fig.suptitle('Removal of unwanted oscillations')

				ax[0,0].semilogy()
				ax[0,0].set_xlabel('freq [Hz]')
				ax[0,0].set_ylabel('amplitude [au]')
				ax[0,0].grid()
				ax[0,0].set_xlim(left=0)
				ax[1,0].set_xlabel('time [s]')
				ax[1,0].set_ylabel('average counts [au]')
				ax[1,0].grid()
				ax[0,0].legend(loc='best', fontsize='x-small')
				plt.savefig(laser_to_analyse[:-4]+'_filter.eps', bbox_inches='tight')
				plt.close('all')

				laser_temperature,laser_temperature_std = coleval.count_to_temp_poly_multi_digitizer(temp_counts,params,errparams,laser_digitizer_ID,number_cpu_available,report=1)
				reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_poly_multi_digitizer(temp_ref_counts,params,errparams,laser_digitizer_ID,number_cpu_available,counts_std=temp_ref_counts_std,report=0,parallelised=False)
				laser_temperature_minus_background = [laser_temperature[i]-reference_background_temperature[i] for i in range(len(laser_digitizer_ID))]
				laser_temperature_std_minus_background = [(laser_temperature_std[i]**2+reference_background_temperature_std[i]**2)**0.5 for i in range(len(laser_digitizer_ID))]

				laser_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature)]
				laser_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature_std)]
				laser_temperature_minus_background_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature_minus_background)]
				laser_temperature_std_minus_background_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature_std_minus_background)]
				reference_background_temperature_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,reference_background_temperature)]
				reference_background_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,reference_background_temperature_std)]
				temp_counts = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,temp_counts)]

				for i in range(len(laser_digitizer_ID)):
					# full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels'] = np.float16(laser_temperature_no_dead_pixels[i])
					# full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels'] = np.float16(laser_temperature_std_no_dead_pixels[i])
					# full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_minus_background_no_dead_pixels'] = np.float16(laser_temperature_minus_background_no_dead_pixels[i])
					# full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_std_minus_background_no_dead_pixels'] = np.float16(laser_temperature_std_minus_background_no_dead_pixels[i])
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['reference_background_temperature_no_dead_pixels'] = np.array(reference_background_temperature_no_dead_pixels[i])
					full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['reference_background_temperature_std_no_dead_pixels'] = np.array(reference_background_temperature_std_no_dead_pixels[i])
					try:
						del full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels']
						del full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels']
						del full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_minus_background_no_dead_pixels']
						del full_saved_file_dict['only_pulse_data'][str(laser_digitizer_ID[i])]['laser_temperature_std_minus_background_no_dead_pixels']
					except:
						print('no legacy elements to erase')


				temp_time_full = np.sort(np.concatenate(temp_time))
				laser_full = [(temp_counts[0][np.abs(temp_time[0]-time).argmin()]) if time in temp_time[0] else (temp_counts[1][np.abs(temp_time[1]-time).argmin()]) for time in temp_time_full]
				laser_temperature_full = [(laser_temperature_no_dead_pixels[0][np.abs(temp_time[0]-time).argmin()]) if time in temp_time[0] else (laser_temperature_no_dead_pixels[1][np.abs(temp_time[1]-time).argmin()]) for time in temp_time_full]
				laser_temperature_std_full = [(laser_temperature_std_no_dead_pixels[0][np.abs(temp_time[0]-time).argmin()]) if time in temp_time[0] else (laser_temperature_std_no_dead_pixels[1][np.abs(temp_time[1]-time).argmin()]) for time in temp_time_full]
				laser_temperature_minus_background_full = [(laser_temperature_minus_background_no_dead_pixels[0][np.abs(temp_time[0]-time).argmin()]) if time in temp_time[0] else (laser_temperature_minus_background_no_dead_pixels[1][np.abs(temp_time[1]-time).argmin()]) for time in temp_time_full]
				laser_temperature_std_minus_background_full = [(laser_temperature_std_minus_background_no_dead_pixels[0][np.abs(temp_time[0]-time).argmin()]) if time in temp_time[0] else (laser_temperature_std_minus_background_no_dead_pixels[1][np.abs(temp_time[1]-time).argmin()]) for time in temp_time_full]
				full_saved_file_dict['only_pulse_data']['laser_temperature_full_median'] = np.median(laser_temperature_full,axis=0)
				full_saved_file_dict['only_pulse_data']['laser_temperature_std_full_median'] = np.median(laser_temperature_std_full,axis=0)
				full_saved_file_dict['only_pulse_data']['laser_temperature_full_minus_median'] = np.float16(laser_temperature_full-np.median(laser_temperature_full,axis=0))
				full_saved_file_dict['only_pulse_data']['laser_temperature_std_full_minus_median'] = np.float16(laser_temperature_std_full-np.median(laser_temperature_std_full,axis=0))
				full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full_median'] = np.median(laser_temperature_minus_background_full,axis=0)
				full_saved_file_dict['only_pulse_data']['laser_temperature_std_minus_background_full_median'] = np.median(laser_temperature_std_minus_background_full,axis=0)
				full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full_minus_median'] = np.float16(laser_temperature_minus_background_full-np.median(laser_temperature_minus_background_full,axis=0))
				full_saved_file_dict['only_pulse_data']['laser_temperature_std_minus_background_full_minus_median'] = np.float16(laser_temperature_std_minus_background_full-np.median(laser_temperature_std_minus_background_full,axis=0))
				if len(np.unique(np.diff(temp_time_full)))<5:
					temp = np.polyval(np.polyfit(np.arange(len(temp_time_full)),temp_time_full,1),np.arange(len(temp_time_full)))
					temp_time_full = temp - (temp[0]-temp_time_full[0])
					full_saved_file_dict['only_pulse_data']['time_full_mode'] = 'the time series was eiterpolated to increase resolution. time intervals were '+str(np.diff(temp_time_full))
				else:
					full_saved_file_dict['only_pulse_data']['time_full_mode'] = 'no time refinement required'
				full_saved_file_dict['only_pulse_data']['time_full'] = np.array(temp_time_full)

				try:
					del full_saved_file_dict['only_pulse_data']['laser_temperature_full']
					del full_saved_file_dict['only_pulse_data']['laser_temperature_std_full']
					del full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full']
					del full_saved_file_dict['only_pulse_data']['laser_temperature_std_minus_background_full']
				except:
					print('no legacy elements to erase')

				np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)

				try:
					saved_file_dict_short = np.load(laser_to_analyse[:-4]+'_short.npz')
					a=saved_file_dict_short['counts']
					saved_file_dict_short = dict(saved_file_dict_short)
				except:
					saved_file_dict_short = dict([])
				saved_file_dict_short['counts'] = np.float32(laser_full)
				saved_file_dict_short['temperature_minus_background'] = np.float32(laser_temperature_minus_background_full)
				saved_file_dict_short['temperature_minus_background_std'] = np.float32(laser_temperature_std_minus_background_full)
				saved_file_dict_short['temperature'] = np.float32(laser_temperature_full)
				saved_file_dict_short['temperature_std'] = np.float32(laser_temperature_std_full)
				saved_file_dict_short['time_full'] = np.array(temp_time_full)
				np.savez_compressed(laser_to_analyse[:-4]+'_short',**saved_file_dict_short)

			for i in laser_digitizer_ID:
				laser_counts_corrected[i] = laser_counts_corrected[i].tolist()
			laser_counts_corrected = list(laser_counts_corrected)

			if not every_pixel_independent:
				for i in range(len(laser_digitizer_ID)):
					time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
					laser_counts_corrected[i] = np.flip(np.transpose(laser_counts_corrected[i],(0,2,1)),axis=2).astype(np.float)
					plt.figure(figsize=(8, 5))
					plt.imshow(np.mean(laser_counts_corrected[i][-int(1*(laser_framerate/len(laser_digitizer_ID))):],axis=0),'rainbow',origin='lower')
					temp = np.shape(laser_counts_corrected[i])[2]
					plt.plot([temp-ROI_horizontal[0],temp-ROI_horizontal[0],temp-ROI_horizontal[1],temp-ROI_horizontal[1],temp-ROI_horizontal[0]],[ROI_horizontal[2],ROI_horizontal[3],ROI_horizontal[3],ROI_horizontal[2],ROI_horizontal[2]],'--k')
					plt.plot([temp-ROI_vertical[0],temp-ROI_vertical[0],temp-ROI_vertical[1],temp-ROI_vertical[1],temp-ROI_vertical[0]],[ROI_vertical[2],ROI_vertical[3],ROI_vertical[3],ROI_vertical[2],ROI_vertical[2]],'--k')
					plt.plot([0,temp],[220]*2,'--k')
					plt.plot([temp-30,temp-240,temp-240,temp-30,temp-30],[30,30,290,290,30],'--k')
					plt.colorbar().set_label('counts [au]')
					plt.xlabel('Horizontal axis [pixles]')
					plt.ylabel('Vertical axis [pixles]')
					plt.title('Counts Background digitizer '+str(laser_digitizer_ID[i]))
					plt.savefig(laser_to_analyse[:-4]+'_back_dig_'+str(laser_digitizer_ID[i])+'.eps', bbox_inches='tight')
					plt.close('all')
					laser_counts_corrected[i]-=np.mean(laser_counts_corrected[i][-int(1*(laser_framerate/len(laser_digitizer_ID))):],axis=0)
					# laser_counts_corrected[i] = median_filter(laser_counts_corrected[i],size=[1,3,3])
					# if not(os.path.exists(laser_to_analyse[:-4]+'_digitizer'+str(i) + '.mp4')):
					if external_clock_marker:
						select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds-aggregated_correction_coefficients[i,4]-start_time_of_pulse>0,time_of_experiment_digitizer_ID_seconds-aggregated_correction_coefficients[i,4]-start_time_of_pulse<1.5)
						ani = coleval.movie_from_data(np.array([laser_counts_corrected[i][select_time]]), laser_framerate/len(laser_digitizer_ID), integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='adjusted counts [au]',prelude='shot '+laser_to_analyse[-9:-4]+'digitizer '+str(i)+'\n')
					else:
						select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse>0,time_of_experiment_digitizer_ID_seconds-start_time_of_pulse<1.5)
						ani = coleval.movie_from_data(np.array([laser_counts_corrected[i][select_time]]), laser_framerate/len(laser_digitizer_ID), integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='adjusted counts digitizer '+str(i)+' [au]',prelude='shot '+laser_to_analyse[-9:-4]+'digitizer '+str(i)+'\n')
					# plt.pause(0.01)
					ani.save(laser_to_analyse[:-4]+'_digitizer'+str(i) + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
					plt.close('all')
				laser_counts_corrected_merged = np.array([(laser_counts_corrected[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if time in time_of_experiment_digitizer_ID[0] else (laser_counts_corrected[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time in time_of_experiment])
				# if not(os.path.exists(laser_to_analyse[:-4] + '.mp4')):
				if external_clock_marker:
					select_time = np.logical_and((time_of_experiment - time_of_experiment[0])*1e-6-np.mean(aggregated_correction_coefficients[:,4])-start_time_of_pulse>0,(time_of_experiment - time_of_experiment[0])*1e-6-np.mean(aggregated_correction_coefficients[:,4])-start_time_of_pulse<1.5)
				else:
					select_time = np.logical_and((time_of_experiment - time_of_experiment[0])*1e-6-start_time_of_pulse>0,(time_of_experiment - time_of_experiment[0])*1e-6-start_time_of_pulse<1.5)
				ani = coleval.movie_from_data(np.array([laser_counts_corrected_merged[select_time]]), laser_framerate,integration=laser_int_time / 1000, extvmin=0,extvmax=laser_counts_corrected_merged[select_time][:,:220].max(),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='adjusted counts [au]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
				ani.save(laser_to_analyse[:-4]+ '_counts.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
				plt.close('all')
				# plt.pause(0.01)
				laser_temperature_minus_background_full = full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full_median'] + full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full_minus_median'].astype(np.float64)
				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_temperature_minus_background_full,(0,2,1)),axis=2)]), laser_framerate,integration=laser_int_time / 1000, extvmin=0,extvmax=laser_temperature_minus_background_full[10:,:,:220].max(),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='temp increase [K]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
				ani.save(laser_to_analyse[:-4]+ '_temp.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
				plt.close('all')
				laser_counts_merged = np.array([(laser_counts[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_counts[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)])
				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_counts_merged,(0,2,1)),axis=2)]), laser_framerate, integration=laser_int_time/1000,time_offset=-start_time_of_pulse,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]')
				# plt.pause(0.01)
				ani.save(laser_to_analyse[:-4]+ '_raw.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
				plt.close('all')

				print('completed ' + laser_to_analyse)

			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power2.py").read())

		except Exception as e:
			print('FAILED ' + laser_to_analyse)
			logging.exception('with error: ' + str(e))
			# print('with error: ' + str(e))
