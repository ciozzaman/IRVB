def MASTU_pulse_process_FAST(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,seconds_for_reference_frame)

	# Here I do a fast analysis of the unfiltered data so I can look at it during experiments
		foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
	temp_ref_counts = []
	temp_counts_minus_background = []
	timesteps = np.inf
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if external_clock_marker:
			time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
		temp_ref_counts.append(np.mean(laser_counts_corrected[i][-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
		select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds[-1]>=0,time_of_experiment_digitizer_ID_seconds[-1]<=1.5)
		temp_counts_minus_background.append(laser_counts_filtered[select_time]-temp_ref_counts)
		time_partial.append(time_of_experiment_digitizer_ID_seconds[select_time])
		timesteps = min(timesteps,len(temp_counts_minus_background[-1]))

	for i in range(len(laser_digitizer_ID)):
		temp_counts_minus_background[i] = temp_counts_minus_background[i][:timesteps]
		time_partial[i] = time_partial[i][:timesteps]
	temp_counts_minus_background = np.nanmean(temp_counts_minus_background,axis=0)
	temp_ref_counts = np.nanmean(temp_ref_counts,axis=0)
	time_partial = np.nanmean(time_partial,axis=0)

	# I'm going to use the reference frames for foil position
	foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx = get_rotation_crop_parameters(temp_ref_counts,foil_position_dict,laser_to_analyse,temp_counts_minus_background,time_partial,foilhorizw=0.09,foilvertw=0.07)

	# rotation and crop
	temp_counts_minus_background_rot=rotate(temp_counts_minus_background[i],foilrotdeg,axes=(-1,-2))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		temp_counts_minus_background_rot*=out_of_ROI_mask
		temp_counts_minus_background_rot[np.logical_and(temp_counts_minus_background_rot<np.nanmin(temp_counts_minus_background[i]),temp_counts_minus_background_rot>np.nanmax(temp_counts_minus_background[i]))]=0
	FAST_counts_minus_background_crop = temp_counts_minus_background_rot[:,foildw:foilup,foillx:foilrx]

	print('completed FAST rotating/cropping ' + laser_to_analyse)

return foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop
