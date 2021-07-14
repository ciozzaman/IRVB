# Created 2021-07-08
# in order to facilitate a proper binning and remotion of the oscillation only after that, here I will only:
# do the initial adjustment of the ramp up, convert to temperature
print('starting ' + laser_to_analyse)

try:
	laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
	laser_dict.allow_pickle=True
	laser_digitizer_ID = laser_dict['uniques_digitizer_ID']
	test = laser_dict['only_foil'].all()[str(laser_digitizer_ID[0])]['laser_temperature_no_dead_pixels_crop_minus_median']
except:
	print('missing '+laser_to_analyse[:-4]+'.npz'+' file. rigenerated')
	if laser_to_analyse[-4:]=='.ats':
		full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse)
	else:
		full_saved_file_dict = coleval.ptw_to_dict(laser_to_analyse)
	np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
	laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
	laser_dict.allow_pickle=True
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
	select_space = np.logical_or(np.logical_or(vertical_coord<30,vertical_coord>240),np.logical_or(horizontal_coord<30,horizontal_coord>290))
	startup_correction = []
	laser_counts_corrected = []
	temp=0
	exponential = lambda t,c1,c2,c3,c4,t0: -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))
	exponential_biased = lambda t,c1,c2,c3,c4,c5,t0: -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))+c5*(t-t0)
	external_clock_marker = False
	for i in range(len(laser_digitizer_ID)):
		reference = np.mean(laser_counts[i][:,select_space],axis=-1)
		if np.abs(reference[-1]-reference[10])>30:
			external_clock_marker = True
		laser_counts_corrected.append(laser_counts[i].astype(np.float64))
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
			if aggregated_correction_coefficients_present:
				fit = [aggregated_correction_coefficients[i]]
			else:
				fit = curve_fit(exponential_biased, time_of_experiment_digitizer_ID_seconds[select_time], reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]), p0=guess,bounds=bds,maxfev=int(1e6))
				aggregated_correction_coefficients[i] = fit[0]
			ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,exponential_biased(time_of_experiment_digitizer_ID_seconds,*fit[0])+np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),':',color=color[i],label='fit of out of foil area')
			# startup_correction.append(np.mean(reference[(time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6>12]) - reference)
			# startup_correction.append(-exponential((time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6,*fit[0]))
			laser_counts_corrected[i] = (laser_counts_corrected[i].astype(np.float64).T -exponential_biased(time_of_experiment_digitizer_ID_seconds,*fit[0])).T
			ax[plot_index,0].axhline(y=np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),linestyle=':',color=color[i])
			ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,np.mean(laser_counts_corrected[i][:,select_space],axis=-1),'-.',color=color[i],label='CORRECTED out of foil area DIG'+str(laser_digitizer_ID[i]))
	ax[plot_index,0].set_ylabel('mean counts [au]')
	fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz\n' %(laser_int_time/1000,laser_framerate) + str(aggregated_correction_coefficients))
	ax[plot_index,0].set_ylim(top=temp)
	ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
	ax[plot_index,0].grid()
	ax[plot_index,0].legend(loc='best', fontsize='x-small')

	plot_index += 1
	ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse,full_saved_file_dict['SensorTemp_0'])
	ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
	ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
	ax[plot_index,0].grid()
	plot_index += 1
	ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse,full_saved_file_dict['SensorTemp_3'])
	ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
	ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
	ax[plot_index,0].grid()
	plot_index += 1
	ax[plot_index,0].plot((time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse,full_saved_file_dict['DetectorTemp'])
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
		plt.savefig(laser_to_analyse[:-4]+'_1.eps', bbox_inches='tight')
	np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
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
		test = full_saved_file_dict['only_foil'].all()[str(laser_digitizer_ID[0])]['laser_temperature_no_dead_pixels_crop_minus_median']
		time_full = full_saved_file_dict['full_frame'].all()['time_full']
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
		params_dict.allow_pickle=True
		params = params_dict['coeff']
		errparams = params_dict['errcoeff']

		temp_counts = []
		temp_bad_pixels_flag = []
		time_partial = []
		temp_ref_counts = []
		temp_ref_counts_std = []
		full_saved_file_dict['full_frame'] = dict([])
		horizontal_coord = np.arange(np.shape(laser_counts[0])[2])
		vertical_coord = np.arange(np.shape(laser_counts[0])[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select_space = np.logical_or(np.logical_or(vertical_coord<30,vertical_coord>240),np.logical_or(horizontal_coord<30,horizontal_coord>290))
		frames_per_oscillation = max(1,int(round(laser_framerate/len(laser_digitizer_ID)/30)))

		seconds_for_bad_pixels = 2	# s
		fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False)
		for i in range(len(laser_digitizer_ID)):
			full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])] = dict([])
			time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
			if external_clock_marker:
				time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
				select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse>0-0.5,time_of_experiment_digitizer_ID_seconds-start_time_of_pulse<1.5)	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
			else:
				select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse>0-0.5,time_of_experiment_digitizer_ID_seconds-start_time_of_pulse<1.5)
			if True:	# I'm not creating my problems by doing this before binning, but I have to do many cleanings
				filter_agent = np.mean(laser_counts_corrected[i][:,select_space],axis=-1) - generic_filter(np.mean(laser_counts_corrected[i][:,select_space],axis=-1),np.nanmean,size=[frames_per_oscillation])
				laser_counts_filtered = (laser_counts_corrected[i].T - filter_agent).T
				filter_agent = np.mean(laser_counts_filtered[:,select_space],axis=-1) - generic_filter(np.mean(laser_counts_filtered[:,select_space],axis=-1),np.nanmean,size=[frames_per_oscillation*2])
				laser_counts_filtered = (laser_counts_filtered.T - filter_agent).T
				filter_agent = np.mean(laser_counts_filtered[:,select_space],axis=-1) - generic_filter(np.mean(laser_counts_filtered[:,select_space],axis=-1),np.nanmean,size=[7])
				laser_counts_filtered = (laser_counts_filtered.T - filter_agent).T
				window = np.min(np.shape(laser_counts_filtered)[1:])//6
				filter_plot_index = 1
				if laser_framerate/len(laser_digitizer_ID)/2>32:
					laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,output_noise=True)
					laser_counts_filtered = laser_counts_filtered[0]
					for trash in range(2):
						fig = matplotlib.pyplot.gcf()
						fig.set_size_inches(15, 10, forward=True)
						plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
						filter_plot_index+=1
						plt.close()
					while peak_value_post_filter>3.2*median_noise:
						laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,output_noise=True)
						laser_counts_filtered = laser_counts_filtered[0]
						for trash in range(2):
							fig = matplotlib.pyplot.gcf()
							fig.set_size_inches(15, 10, forward=True)
							plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
							filter_plot_index+=1
							plt.close()
				if laser_framerate/len(laser_digitizer_ID)/2>65:
					laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=50,max_frequency_to_erase=70,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,output_noise=True)
					laser_counts_filtered = laser_counts_filtered[0]
					for trash in range(2):
						fig = matplotlib.pyplot.gcf()
						fig.set_size_inches(15, 10, forward=True)
						plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
						filter_plot_index+=1
						plt.close()
					while peak_value_post_filter>4*median_noise:
						laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=50,max_frequency_to_erase=70,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,output_noise=True)
						laser_counts_filtered = laser_counts_filtered[0]
						for trash in range(2):
							fig = matplotlib.pyplot.gcf()
							fig.set_size_inches(15, 10, forward=True)
							plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
							filter_plot_index+=1
							plt.close()
				if laser_framerate/len(laser_digitizer_ID)/2>90:
					laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=75,max_frequency_to_erase=102,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,output_noise=True)
					laser_counts_filtered = laser_counts_filtered[0]
					for trash in range(2):
						fig = matplotlib.pyplot.gcf()
						fig.set_size_inches(15, 10, forward=True)
						plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
						filter_plot_index+=1
						plt.close()
					while peak_value_post_filter>3.2*median_noise:
						laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=75,max_frequency_to_erase=102,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,output_noise=True)
						laser_counts_filtered = laser_counts_filtered[0]
						for trash in range(2):
							fig = matplotlib.pyplot.gcf()
							fig.set_size_inches(15, 10, forward=True)
							plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
							filter_plot_index+=1
							plt.close()
			else:
				laser_counts_filtered = laser_counts_corrected[i]
			full_average = np.mean(laser_counts_corrected[i][:,np.logical_not(select_space)][select_time],axis=(-1))
			full_spectra = np.fft.fft(full_average)
			full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
			full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
			ax[0,0].plot(full_freq,full_magnitude/(100**i),color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
			temp = (full_average.max()-full_average.min())
			ax[1,0].plot(time_of_experiment_digitizer_ID_seconds[select_time],full_average-temp*i,color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
			# ax[1,0].plot(time_of_experiment_digitizer_ID_seconds,full_average,color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
			full_average = np.mean(laser_counts_filtered[:,np.logical_not(select_space)][select_time],axis=(-1))
			full_spectra = np.fft.fft(full_average)
			full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
			full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
			ax[0,0].plot(full_freq,full_magnitude/(100**i),'k--')
			ax[1,0].plot(time_of_experiment_digitizer_ID_seconds[select_time],full_average-temp*i,'k--')
			# ax[1,0].plot(time_of_experiment_digitizer_ID_seconds,full_average,'k--')

			full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['time'] = (time_of_experiment_digitizer_ID_seconds-start_time_of_pulse)
			time_partial.append(full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['time'])
			temp_counts.append(laser_counts_filtered)
			bad_pixels_flag = coleval.find_dead_pixels([laser_counts_corrected[i][-int(seconds_for_bad_pixels*laser_framerate/len(laser_digitizer_ID)):]],treshold_for_bad_difference=200)
			full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['bad_pixels_flag'] = bad_pixels_flag
			temp_bad_pixels_flag.append(bad_pixels_flag)
		fig.suptitle('Removal of unwanted oscillations\naverage of all frame except the frame')

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
		laser_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature)]
		laser_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature_std)]

		# for i in range(len(laser_digitizer_ID)):
		# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_minus_median'] = np.float16(laser_temperature_no_dead_pixels[i].T-np.median(laser_temperature_no_dead_pixels[i],axis=(-1,-2))).T
		# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_median'] = np.median(laser_temperature_no_dead_pixels[i],axis=(-1,-2))
		# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_minus_median'] = np.float16(laser_temperature_std_no_dead_pixels[i].T-np.median(laser_temperature_std_no_dead_pixels[i],axis=(-1,-2))).T
		# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_median'] = np.median(laser_temperature_std_no_dead_pixels[i],axis=(-1,-2))

		time_full = np.sort(np.concatenate(time_partial))
		if len(np.unique(np.diff(time_full)))<5:
			temp = np.polyval(np.polyfit(np.arange(len(time_full)),time_full,1),np.arange(len(time_full)))
			time_full = temp - (temp[0]-time_full[0])
			full_saved_file_dict['full_frame']['time_full_mode'] = 'the time series was eiterpolated to increase resolution. time intervals were '+str(np.diff(temp_time_full))
		else:
			full_saved_file_dict['full_frame']['time_full_mode'] = 'no time refinement required'
		full_saved_file_dict['full_frame']['time_full'] = time_full

		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process3.py").read())

	for i in laser_digitizer_ID:
		laser_counts_corrected[i] = laser_counts_corrected[i].tolist()
	laser_counts_corrected = list(laser_counts_corrected)

	ROI_horizontal = [220,238,50,100]
	# ROI_horizontal = [35,45,68,73]
	ROI_vertical = [100,170,28,45]
	if not every_pixel_independent:
		for i in range(len(laser_digitizer_ID)):
			time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
			laser_counts_corrected[i] = np.flip(np.transpose(laser_counts_corrected[i],(0,2,1)),axis=2).astype(np.float64)
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
		# laser_temperature_minus_background_full = full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full_median'] + full_saved_file_dict['only_pulse_data']['laser_temperature_minus_background_full_minus_median'].astype(np.float64)
		# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_temperature_minus_background_full,(0,2,1)),axis=2)]), laser_framerate,integration=laser_int_time / 1000, extvmin=0,extvmax=laser_temperature_minus_background_full[10:,:,:220].max(),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='temp increase [K]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
		# ani.save(laser_to_analyse[:-4]+ '_temp.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		# plt.close('all')
		laser_counts_merged = np.array([(laser_counts[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_counts[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)])
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_counts_merged,(0,2,1)),axis=2)]), laser_framerate, integration=laser_int_time/1000,time_offset=-start_time_of_pulse,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
		# plt.pause(0.01)
		ani.save(laser_to_analyse[:-4]+ '_raw.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		plt.close('all')
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_counts_merged[select_time],(0,2,1)),axis=2)]), laser_framerate, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
		# plt.pause(0.01)
		ani.save(laser_to_analyse[:-4]+ '_raw_short.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

		print('completed ' + laser_to_analyse)

	# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power2.py").read())

except Exception as e:
	print('FAILED ' + laser_to_analyse)
	logging.exception('with error: ' + str(e))
	# print('with error: ' + str(e))
