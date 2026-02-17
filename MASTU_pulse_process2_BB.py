# Created 2021-07-08
# in order to facilitate a proper binning and remotion of the oscillation only after that, here I will only:
# do the initial adjustment of the ramp up, convert to temperature
# print('starting ' + laser_to_analyse + '\n of \n' + str(shot_available[i_day]))


try:
	trash = cp.deepcopy(only_plot_brightness)
except:
	only_plot_brightness = False

if os.path.exists(laser_to_analyse):	# I pick the right filename from the start
	pass
elif os.path.exists(laser_to_analyse[:-4]+'.ptw'):	# additional stage: change the file extension if found
	laser_to_analyse = laser_to_analyse[:-4]+'.ptw'
elif os.path.exists(laser_to_analyse[:-4]+'.ats'):	# additional stage: change the file extension if found
	laser_to_analyse = laser_to_analyse[:-4]+'.ats'
else:
	print(laser_to_analyse+' source file missing.')

print('starting ' + laser_to_analyse + '\n of \n' + str(to_do))


try:
	laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
	laser_dict.allow_pickle=True
	laser_digitizer_ID = laser_dict['uniques_digitizer_ID']
	if laser_to_analyse[-4:]=='.ptw':
		test = laser_dict['discarded_frames']
except:
	print('missing '+laser_to_analyse[:-4]+'.npz'+' file.')
	if os.path.exists(laser_to_analyse):
		print(laser_to_analyse[:-4]+'.npz'+' file will be generated')
	elif os.path.exists(laser_to_analyse[:-4]+'.ptw'):	# additional stage: change the file extension if found
		laser_to_analyse = laser_to_analyse[:-4]+'.ptw'
		print(laser_to_analyse[:-4]+'.npz'+' file will be generated')
	elif os.path.exists(laser_to_analyse[:-4]+'.ats'):	# additional stage: change the file extension if found
		laser_to_analyse = laser_to_analyse[:-4]+'.ats'
		print(laser_to_analyse[:-4]+'.npz'+' file will be generated')
	else:
		print('file missing, I try to look for it in the temp storage')

		# 2025-12-18 I try to recover the file from the storage
		try:
			archive_path,name = coleval.find_shot_file_and_path_in_temp_storage(shotnumber)	# I over write name, so if the file type was wrong it can get fixed
			# to copy the files
			import shutil

			if not os.path.exists(path+day):
				os.mkdir(path+day)
			if not os.path.isfile(path+day + '/' + name):
				shutil.copyfile(archive_path + '/' + name, path+day + '/' + name)
				print('copied from \n'+archive_path + '/' + name + '\nto\n' +path+day + '/' + name)
				laser_to_analyse=path+day+'/'+name

			if os.path.exists(laser_to_analyse):
				print(laser_to_analyse[:-4]+'.npz'+' file will be generated')
			elif os.path.exists(laser_to_analyse[:-4]+'.ptw'):	# additional stage: change the file extension if found
				laser_to_analyse = laser_to_analyse[:-4]+'.ptw'
				print(laser_to_analyse[:-4]+'.npz'+' file will be generated')
			elif os.path.exists(laser_to_analyse[:-4]+'.ats'):	# additional stage: change the file extension if found
				laser_to_analyse = laser_to_analyse[:-4]+'.ats'
				print(laser_to_analyse[:-4]+'.npz'+' file will be generated')
			# here we go back to the normal code

		except:
			print(laser_to_analyse+' file missing, analysis halted.')
			bla = sgna	# I want this to generate an error. te camera file is not present.

	if laser_to_analyse[-4:]=='.ats':
		full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse)
	else:
		full_saved_file_dict = coleval.ptw_to_dict(laser_to_analyse,max_time_s = 30)
	np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
	print(laser_to_analyse[:-4]+ '.npz' + ' created and saved')
	laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
	laser_dict.allow_pickle=True
full_saved_file_dict = dict(laser_dict)

if laser_to_analyse[-3:] != 'npz':
	try:
		os.remove(laser_to_analyse)
		print('source file ' + laser_to_analyse + ' deleted to save Freia space')
	except:
		pass


if laser_to_analyse[-9:-4] in list(MASTU_shots_timing.keys()):	# manual exceptions at the beginning of IRVB life
	start_time_of_pulse = MASTU_shots_timing[laser_to_analyse[-9:-4]]['pulse_start']	# s
else:
	start_time_of_pulse = 2.5	# s
seconds_for_reference_frame = 5	# s

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
	time_of_experiment = full_saved_file_dict['time_of_measurement']	# microseconds
	laser_digitizer = full_saved_file_dict['digitizer_ID']
	SensorTemp_0 = full_saved_file_dict['SensorTemp_0']
	SensorTemp_3 = full_saved_file_dict['SensorTemp_3']
	DetectorTemp = full_saved_file_dict['DetectorTemp']
	width = full_saved_file_dict['width']
	height = full_saved_file_dict['height']
	time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)	# microseconds
	for i in range(len(laser_digitizer_ID)):
		if np.diff(time_of_experiment_digitizer_ID[i])[0]<0:
			laser_counts[i] = laser_counts[i][1:]
			SensorTemp_0 = SensorTemp_0[time_of_experiment_digitizer_ID[i][0] != time_of_experiment]
			SensorTemp_3 = SensorTemp_3[time_of_experiment_digitizer_ID[i][0] != time_of_experiment]
			DetectorTemp = DetectorTemp[time_of_experiment_digitizer_ID[i][0] != time_of_experiment]
			time_of_experiment_digitizer_ID[i] = time_of_experiment_digitizer_ID[i][1:]
	# now I have to add the caveat that if there is a big gap between two section of data, that is the demarcation from internal and external clocks
	if np.nanmax(np.diff(time_of_experiment))>np.nanmedian(np.diff(time_of_experiment))*3 and \
	((int(laser_to_analyse[-9:-4])>49476 and int(laser_to_analyse[-9:-4])<50646) or \
	(int(laser_to_analyse[-9:-4])>51639 and int(laser_to_analyse[-9:-4])<np.inf)): 	# 3 arbitrary limit
		# first I reduce the file size
		last_good_frame = (np.diff(time_of_experiment_digitizer_ID) == np.nanmax(np.diff(time_of_experiment_digitizer_ID))).argmax()
		time_of_experiment = time_of_experiment[:last_good_frame]
		laser_digitizer = laser_digitizer[:last_good_frame]
		SensorTemp_0 = SensorTemp_0[:last_good_frame]
		SensorTemp_3 = SensorTemp_3[:last_good_frame]
		DetectorTemp = DetectorTemp[:last_good_frame]
		DetectorTemp = DetectorTemp[:last_good_frame]
		laser_counts = [laser_counts[0][:last_good_frame]]
		time_of_experiment_digitizer_ID = [time_of_experiment_digitizer_ID[0][:last_good_frame]]
		exernal_time = coleval.import_external_time(laser_to_analyse[-9:-4])
		if not len(exernal_time)==1 and np.nanmax(np.diff(time_of_experiment_digitizer_ID))<np.median(np.diff(time_of_experiment_digitizer_ID))*2:	# meaning reading successfull
			time_of_experiment = exernal_time[-len(time_of_experiment):]*1e6
			time_of_experiment_digitizer_ID = [exernal_time[-len(time_of_experiment):]*1e6]
			frame_time_succesfully_rigidly_determined = True
			print('proper new way of tining the frames achieved')
		else:	# if the reading fails I can still use the last pulse as marker, and the clock of the camera to determine the real framerate. this effectively uses the internal clock of the camera, but with an external final reference
			laser_framerate = 1e6/np.mean(np.sort(np.diff(time_of_experiment))[2:-2])
			# 2025/05/22 I still have no idea why this happens, but the hole is always 4 seconds long, showing up each ~24.5 minutes possibly
			# if the hole is 4 seconds long i automatically everything is fine
			# if int(laser_to_analyse[-9:-4]) in [51804,51807]:	# MANUAL CORRECTION the data seems fine, but there is a problem with the timestamp that I can't figure out
			if (np.nanmax(np.diff(time_of_experiment_digitizer_ID)) > (4-2/laser_framerate)*1e6) and (np.nanmax(np.diff(time_of_experiment_digitizer_ID)) < (4+2/laser_framerate)*1e6):
				time_of_experiment = np.arange(len(time_of_experiment))*(1/laser_framerate)*1e6
			if not len(exernal_time)==1 and np.nanmax(np.diff(time_of_experiment))<np.median(np.diff(time_of_experiment))*2:	# meaning reading successfull
				time_of_experiment = exernal_time[-len(time_of_experiment):]*1e6
				time_of_experiment_digitizer_ID = [exernal_time[-len(time_of_experiment):]*1e6]
				frame_time_succesfully_rigidly_determined = True
			else:
				time_of_experiment = time_of_experiment-time_of_experiment[-1] +(- 2.5 + 6038*(1/laser_framerate))*1e6	# since the beginning I send 7000 pulses at 960Hz. this might be changed if I switch to 0.5ms 2kHz. 2025/5/20 in reality the real start is about 1 second after the beginning of external clocks, at -1.5
				frame_time_succesfully_rigidly_determined = False
			# time_of_experiment = time_of_experiment*1e6
			time_of_experiment_digitizer_ID = [time_of_experiment]
			print('second best option for getting the frame times achieved\nI can still use the last pulse as marker, and the clock of the camera to determine the real framerate.\nthis effectively uses the internal clock of the camera, but with an external final reference')
		frame_time_rigidly_determined = True
		start_time_of_pulse = -np.min(time_of_experiment)*1e-6	# s
	else:
		frame_time_succesfully_rigidly_determined = False
		frame_time_rigidly_determined = False
	time_of_experiment = np.sort(np.concatenate(time_of_experiment_digitizer_ID))
	laser_framerate = 1e6/np.mean(np.sort(np.diff(time_of_experiment))[2:-2])
	if full_saved_file_dict['IntegrationTime']<100:	# for some reason some files contain already int time in microseconds
		laser_int_time = full_saved_file_dict['IntegrationTime']*1000	# microseconds
	else:
		laser_int_time = full_saved_file_dict['IntegrationTime']	# microseconds

	if True and (int(laser_to_analyse[-9:-4])>52500 or int(laser_to_analyse[-9:-4])==51653):	# this bit it to detect badness in the camera image as found in the FLIR X6980 in summer 2025	# introduced 51653 as test case to test effect of missing data
		window_for_bad_stripes = 400
		laser_counts_median_filter_initial = [median_filter(value,size=[3,1,1]) for value in np.array(laser_counts)[:,:window_for_bad_stripes]]
		neutron_caused_blips_initial = [(np.abs(laser_counts_median_filter_initial[i] - np.array(laser_counts)[i,:window_for_bad_stripes])/laser_counts_median_filter_initial[i])>0.025 for i in range(len(laser_counts))]	# 2.5% arbitrary threshold that seems to remove most badness
		laser_counts_median_filter_end = [median_filter(value,size=[3,1,1]) for value in np.array(laser_counts)[:,-window_for_bad_stripes*2:]]
		neutron_caused_blips_end = [(np.abs(laser_counts_median_filter_end[i] - np.array(laser_counts)[i,-window_for_bad_stripes*2:])/laser_counts_median_filter_end[i])>0.025 for i in range(len(laser_counts))]	# 2.5% arbitrary threshold that seems to remove most badness
		neutron_caused_blips_initial = np.sum(neutron_caused_blips_initial,axis=(0,1))
		neutron_caused_blips_end = np.sum(neutron_caused_blips_end,axis=(0,1))
		plt.figure(figsize=(12, 8))
		# plt.imshow(np.sum(neutron_caused_blips_initial,axis=(0,1))>0)
		plt.imshow(neutron_caused_blips_end+neutron_caused_blips_initial,'rainbow')
		plt.colorbar().set_label('over deviation from median detected [au]')
		rows_to_replace = np.arange(len(np.sum(neutron_caused_blips_end+neutron_caused_blips_initial>10,axis=0)>2))[np.sum(neutron_caused_blips_end+neutron_caused_blips_initial>10,axis=0)>2]
		if int(laser_to_analyse[-9:-4])==51653:	# modyfying dood data to see what is the effect of removing columns in groups of 2 as with the damaged FLIR X6980 camera
			print('WARNING 51653 was manually modified to see if removing the column data would worsen the inversion')
			rows_to_replace = np.array([  6 ,  7,  22,  23,  38,  39,  54,  55,  70,  71, 86, 87,102,103,118,119,134,135,150,151,166,167,182,183,198,199,214,215,230,231,246,247,262,263,278,279,294,295,310,311,326,327,342,343])
		for i__ in rows_to_replace:
			plt.axvline(x=i__,color='k',linestyle=':',alpha=0.1)
		plt.title(day+'/'+name+'\nint time %.3gms, framerate %.3gHz' %(laser_int_time/1000,laser_framerate) +'\nreplaced columns '+str( rows_to_replace ))
		plt.savefig(laser_to_analyse[:-4]+'_column_badness.eps')
		plt.close('all')
		# plt.figure()
		# plt.plot(np.sum(neutron_caused_blips_end+neutron_caused_blips_initial>10,axis=0)>2)
		temp = np.array(laser_counts).astype(np.float16)
		temp[:,:,:,np.sum(neutron_caused_blips_end+neutron_caused_blips_initial>10,axis=0)>2] = np.nan
		if int(laser_to_analyse[-9:-4])==51653:	# modyfying dood data to see what is the effect of removing columns in groups of 2 as with the damaged FLIR X6980 camera
			for i_ in range(len(laser_counts)):
				for i__ in rows_to_replace:
					temp[i_][:,:,i__] = np.nan
		if True:
			for i_ in range(len(laser_counts)):
				for i__ in rows_to_replace:
					laser_counts[i_][:,:,i__] = np.nanmedian(temp[i_][:,:,i__-1:i__+2],axis=-1)
		else:
			# temp_filtered = [median_filter(value[:100],size=[3,3,3]) for value in temp]
			temp_filtered = [generic_filter(value,np.nanmean,size=[1,3,3]) for value in temp]
			for i in range(len(laser_counts)):
				laser_counts[i][:,:,np.sum(neutron_caused_blips_end+neutron_caused_blips_initial>10,axis=0)>2] = (temp_filtered[i]).astype(int)[:,:,np.sum(neutron_caused_blips_end+neutron_caused_blips_initial>10,axis=0)>2]
		print('replaced columns '+str( rows_to_replace ))
	# in the FLIR X6980 data i think i see a bunch of dynamically changing bad pixels i think due to neutrons, i eliminate them with a temporal median filter
	# I don't see the issue to leave it in all cases
	laser_counts_median_filter = [median_filter(value,size=[3,1,1]) for value in laser_counts]
	neutron_caused_blips = [(np.abs(laser_counts_median_filter[i] - laser_counts[i])/laser_counts_median_filter[i])>0.025 for i in range(len(laser_counts))]	# 2.5% arbitrary threshold that seems to remove most badness
	for i in range(len(laser_counts)):
		laser_counts[i][neutron_caused_blips[i]] = laser_counts_median_filter[i][neutron_caused_blips[i]]
	try:	# if the length of the data in the 2 digitizers is fdifferent this can collapse, and i don't want this to stop everything
		fig, ax = plt.subplots( 2,1,figsize=(10, 18), squeeze=False)
		fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz\nneutron induced bad pixels ' %(laser_int_time/1000,laser_framerate))
		im = ax[0,0].imshow(np.sum(neutron_caused_blips,axis=(0,1)))
		fig.colorbar(im, ax=ax[0, 0]).set_label('number of times pixel affected [au]')
		# im.colorbar().set_label('number of times pixel effected [au]')
		ax[1,0].plot((np.mean(time_of_experiment_digitizer_ID,axis=0)-np.mean(time_of_experiment_digitizer_ID,axis=0).min())*1e-6,np.sum(neutron_caused_blips,axis=(0,2,3)))
		ax[1,0].plot((np.mean(time_of_experiment_digitizer_ID,axis=0)-np.mean(time_of_experiment_digitizer_ID,axis=0).min())*1e-6,generic_filter(np.sum(neutron_caused_blips,axis=(0,2,3)).astype(float),np.nanmean,size=[max(1,int(0.050*(laser_framerate/len(laser_counts))))]),label='50ms smoothing')
		ax[1,0].legend(loc='best', fontsize='x-small')
		ax[1,0].set_xlim(left =1, right = 4)
		ax[1,0].set_xlabel('time from start of recording [s]')
		ax[1,0].set_ylabel('number of pixel affected [au]')
		ax[1,0].grid()
		# ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse)
		plt.savefig(laser_to_analyse[:-4]+'_neutron_induced_bad_pixels.eps')
		plt.close('all')
	except:
		pass


	if False:
		# using the new startup of the camera I reach steady state way before the pulse, so I can use the first frames as reference without any correction
		flag_use_of_first_frames_as_reference = np.std((full_saved_file_dict['data']+full_saved_file_dict['data_median'])[np.logical_and((time_of_experiment-time_of_experiment[0])*1e-6>0.1,(time_of_experiment-time_of_experiment[0])*1e-6-start_time_of_pulse-0.5<0)],axis=0).min()<10
	# Nope: doing this I don't subtract enough and there is a significant residual component of BB after the pulse that is just wrong.
	flag_use_of_first_frames_as_reference = True
	# I set this like this because at the end of the recording period there is always an offset that is significantly larger than any error I could have from subtracting the initial reading
	# this is true even in the early shots where I had a very strong ramp at the beginning. with the new subtraction technique (subtracting the average of the counts outside the foil to the whole image rather than fitting)
	# this seems reasonable

	try:
		full_correction_coefficients = full_saved_file_dict['full_correction_coeff']
		full_correction_coefficients_present = True
	except:
		full_correction_coefficients_present = False
		full_correction_coefficients = np.zeros((2,*np.shape(laser_counts[0])[1:],6))*np.nan

	# try:
	# 	temp = full_saved_file_dict['aggregated_correction_coeff'][0][4]
	# 	if len(full_saved_file_dict['aggregated_correction_coeff'][0])>5:
	# 		bla=sgna
	# 	aggregated_correction_coefficients_present = True
	# 	aggregated_correction_coefficients = full_saved_file_dict['aggregated_correction_coeff']
	# except:
	# it takes so little to do it that it doesn't matter
	aggregated_correction_coefficients_present = False
	aggregated_correction_coefficients = np.zeros((2,5))*np.nan

	fig, ax = plt.subplots( 4,1,figsize=(10, 24), squeeze=False, sharex=True)
	plot_index = 0
	horizontal_coord = np.arange(width)
	vertical_coord = np.arange(height)
	horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
	select_space_foil = np.logical_and(np.logical_and(vertical_coord>50,vertical_coord<220),np.logical_and(horizontal_coord>50,horizontal_coord<275))
	select_space1 = np.logical_and(vertical_coord<20,np.logical_and(horizontal_coord>30,horizontal_coord<width-10))
	select_space2 = np.logical_and(vertical_coord>height-10,np.logical_and(horizontal_coord>30,horizontal_coord<width-10))
	select_space3 = np.logical_and(horizontal_coord<20,np.logical_and(vertical_coord>75,vertical_coord<height-10))
	select_space4 = np.logical_and(horizontal_coord>width-10,np.logical_and(vertical_coord>75,vertical_coord<height-10))
	# select_space = np.logical_or(np.logical_or(vertical_coord<30,vertical_coord>240),np.logical_or(horizontal_coord<30,horizontal_coord>290))
	select_space = np.logical_or(np.logical_or(select_space1,select_space2),np.logical_or(select_space3,select_space4))
	startup_correction = []
	laser_counts_corrected = []
	max_counts_for_plot=0
	exponential = lambda t,c1,c2,c3,c4,t0: -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))
	exponential_biased = lambda t,c1,c2,c3,c5,t0: -c1*np.exp(-c2*(np.maximum(0,(t-t0))**c3))+c5*(t-t0)
	def exponential_biased_with_grad(coeffs,*args):
		print(coeffs)
		c1,c2,c3,c5,t0 = coeffs
		counts = args[0]
		t = args[1]
		exponential_biased = -c1*np.exp(-c2*(np.maximum(0,(t-t0))**c3))+c5*(t-t0)
		residuals = np.nansum((counts-exponential_biased)**2)
		temp = c2*c3*(np.maximum(0,(t-t0))**(c3-1))
		temp[t-t0<0] = 0
		grad = -2*np.nansum((counts-exponential_biased)*np.array([-np.exp(-c2*(np.maximum(0,(t-t0))**c3)) , -c1*np.exp(-c2*(np.maximum(0,(t-t0))**c3))*(-(np.maximum(0,(t-t0))**c3)) , -c1*np.exp(-c2*(np.maximum(0,(t-t0))**c3))*(-temp) , (t-t0) , -c1*np.exp(-c2*(np.maximum(0,(t-t0))**c3))*(temp*(t-t0>0))-c5]),axis=1)
		print(residuals,grad)
		return residuals,grad
	# def exponential_biased(t,c1,c2,c3,c4,c5,t0):
	# 	print(c1,c2,c3,c4,c5,t0)
	# 	return -c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))+c5*(t-t0)
	external_clock_marker = False
	for i in range(len(laser_digitizer_ID)):
		reference = np.mean(laser_counts[i][:,select_space],axis=-1)
		reference1 = np.mean(laser_counts[i][:,select_space1],axis=-1)
		reference2 = np.mean(laser_counts[i][:,select_space2],axis=-1)
		reference3 = np.mean(laser_counts[i][:,select_space3],axis=-1)
		reference4 = np.mean(laser_counts[i][:,select_space4],axis=-1)
		if np.abs(reference[-1]-reference[10])>80:	# it was 30	# 2024-12-02 it was 50
			external_clock_marker = True
		laser_counts_corrected.append(laser_counts[i].astype(np.float64))
		max_counts_for_plot = max(max_counts_for_plot,np.max(np.mean(laser_counts[i][10:],axis=(-1,-2))))
		if laser_framerate<60:
			time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i][:np.min( [len(value) for value in laser_counts] )]-time_of_experiment[0])*1e-6
		else:
			time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
		select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds<start_time_of_pulse+30,np.logical_or(time_of_experiment_digitizer_ID_seconds<start_time_of_pulse-0.2,time_of_experiment_digitizer_ID_seconds>start_time_of_pulse+5))	# I avoid the period with the plasma
		select_time[:8]=False	# to avoid the very firsts frames that can have unreasonable counts
		bds = [[0,0,0,-np.inf,start_time_of_pulse-0.1],[np.inf,np.inf,np.inf,np.inf,start_time_of_pulse+0.1]]
		guess=[np.abs((reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]))[0]),1,1.5,0,start_time_of_pulse]
		# temp = np.mean(laser_counts[i],axis=(-1,-2))
		# ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,temp-temp[0],color=color[i],label='all foil mean DIG'+str(laser_digitizer_ID[i]))
		ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,reference-reference[4],'--',color=color[i],label='out of foil area DIG'+str(laser_digitizer_ID[i]))
		ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,reference1-reference1[4],'.-',color=color[i])
		ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,reference2-reference2[4],'.-')
		ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,reference3-reference3[4],'.-')
		ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,reference4-reference4[4],'.-')
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
				if i==1:
					guess = fit[0]
				fit = curve_fit(exponential_biased, time_of_experiment_digitizer_ID_seconds[select_time], reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]), p0=guess,bounds=bds,maxfev=int(1e6))

				if False:	# only for testinf the prob_and_gradient function
					target = 2
					scale = 1e-1
					args = [reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),time_of_experiment_digitizer_ID_seconds[select_time]]
					# guess[target] = 1e5
					temp1 = exponential_biased_with_grad(guess,*args)
					guess[target] +=scale
					temp2 = exponential_biased_with_grad(guess,*args)
					guess[target] += -2*scale
					temp3 = exponential_biased_with_grad(guess,*args)
					guess[target] += scale
					print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))
					guess=[np.abs((reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]))[0]),1,1.5,0,start_time_of_pulse]
					bds = [[0,np.inf],[0,np.inf],[0,np.inf],[0,np.inf],[0,start_time_of_pulse+0.1]]
					fit, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(exponential_biased_with_grad, x0=guess, args = [reference[select_time]-np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):]),time_of_experiment_digitizer_ID_seconds[select_time]] ,bounds = bds) #,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
					fit = [fit.tolist()]

				aggregated_correction_coefficients[i] = fit[0]
			temp = exponential_biased(time_of_experiment_digitizer_ID_seconds,*fit[0])+np.mean(reference[-int(1*laser_framerate/len(laser_digitizer_ID)):])
			ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,temp-temp[4],':',color=color[i],label='fit of out of foil area')
			# startup_correction.append(np.mean(reference[(time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6>12]) - reference)
			# startup_correction.append(-exponential((time_of_experiment_digitizer_ID[i]-time_of_experiment_digitizer_ID[i][0])*1e-6,*fit[0]))
			ax[plot_index,0].axhline(y=np.mean(reference[select_time][-int(1*laser_framerate/len(laser_digitizer_ID)):])-reference[4],linestyle=':',color=color[i])

			temp = np.mean(laser_counts_corrected[i][:,select_space_foil],axis=-1)
			ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,temp-temp[4],'-.',color=color[i],label='correction NOT applied\nfoil area DIG'+str(laser_digitizer_ID[i]))
			# this increase could actually be due to the frame heating up! this is only by ~0.3C, that is low, and could make sense.
			# the weird thing is that the increase starts almost at the sale time as the foil heating. (expected delay across 2 foil holder places ~0.3s)
			laser_counts_corrected[i] = laser_counts_corrected[i].astype(np.float64)#.T -reference+reference[4]).T
			temp = np.mean(laser_counts_corrected[i][:,select_space_foil],axis=-1)
			ax[plot_index,0].plot(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse,temp-temp[4],'-',color=color[i],label='correction applied\nfoil area DIG'+str(laser_digitizer_ID[i]))
	ax[plot_index,0].set_ylabel('mean counts [au]')
	fig.suptitle(day+'/'+name+'\nint time %.3gms, framerate %.3gHz\n' %(laser_int_time/1000,laser_framerate) + str(aggregated_correction_coefficients))
	# ax[plot_index,0].set_ylim(top=max_counts_for_plot)
	ax[plot_index,0].set_xlim(left=time_of_experiment_digitizer_ID_seconds[10]-10/(laser_framerate/len(laser_digitizer_ID))-start_time_of_pulse,right=min(time_of_experiment_digitizer_ID_seconds.max(),25))
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
		full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(np.maximum(0,(t-t0))**c3))+c5*(t-t0)'
		# if aggregated_correction_coefficients_present:
		# 	if np.sum((full_saved_file_dict['aggregated_correction_coeff']!=aggregated_correction_coefficients))>1:
		# 		np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
		# else:
		# 	full_saved_file_dict['aggregated_correction_coeff'] = aggregated_correction_coefficients
		# 	full_saved_file_dict['correction_equation'] = '-c1*np.exp(-c2*(t-t0)-c3*((t-t0)**2)-c4*((t-t0)**3))'
		plt.savefig(laser_to_analyse[:-4]+'_0.eps', bbox_inches='tight')
	# np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)
	plt.close('all')

	# 27/09/2024 I reduce the area of the foil for this, as I found that there is a signal from -0.4s to +0.1s due to some geometrical deformation of the camera mount
	# I reduce the size of select_space_foil so I avoid the problematic section at the edge of the foil
	select_space_foil = np.logical_and(np.logical_and(vertical_coord>100,vertical_coord<height-100),np.logical_and(horizontal_coord>100,horizontal_coord<width-150))


	# added 8/3/22 I need to find the real beginning of the pulse. if framerate is 50H the error is just too large
	if laser_framerate<60:
		plt.figure(figsize=(12, 8))
		real_start_time_of_pulse = []
		for i in range(len(laser_digitizer_ID)):
			time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
			# temp=np.sort(median_filter(laser_counts_corrected[i],[1,5,5]).reshape((len(laser_counts_corrected[i]),np.shape(laser_counts_corrected[i])[1]*np.shape(laser_counts_corrected[i])[2])),axis=(1))
			# temp = np.diff(np.mean(temp[:,-40:],axis=1))
			temp = np.mean(laser_counts_corrected[i][:,select_space_foil],axis=(1))
			temp = np.diff(temp)
			time = time_of_experiment_digitizer_ID_seconds[:-1]+np.mean(np.diff(time_of_experiment_digitizer_ID_seconds))
			std = np.std(temp[time<2][4:])
			plt.plot(time,temp)
			# plt.axhline(y=-std*5,color='y')
			plt.axhline(y=std*5,color='y')
			plt.axvline(x=time[(temp>5*std).argmax()-1],color='k')
			real_start_time_of_pulse.append(time[(temp>5*std).argmax()-1])
		start_time_of_pulse = np.mean(real_start_time_of_pulse)
		plt.xlim(left=start_time_of_pulse-1,right=start_time_of_pulse+1)
		plt.grid()
		plt.title(day+'/'+name+'\nint time %.3gms, framerate %.3gHz\nstart time = mean ' %(laser_int_time/1000,laser_framerate) +str(real_start_time_of_pulse))
		plt.savefig(laser_to_analyse[:-4]+'_0.5.eps', bbox_inches='tight')
		plt.close()

	plt.figure(figsize=(12, 8))
	real_start_time_of_pulse = []
	real_start_time_of_pulse_uncertainty = []
	# plt.figure()
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6
		temp = np.mean(laser_counts_corrected[i][:,select_space_foil],axis=(1))
		full_spectra = np.fft.fft(temp[np.logical_and(time_of_experiment_digitizer_ID_seconds<start_time_of_pulse-0.2,time_of_experiment_digitizer_ID_seconds>start_time_of_pulse-1)])
		full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
		full_freq = np.fft.fftfreq(len(full_magnitude), d=np.mean(np.diff(time_of_experiment_digitizer_ID_seconds)))
		to_filter = full_freq[full_freq>10][(full_magnitude[full_freq>10]).argmax()]
		a = plt.plot(time_of_experiment_digitizer_ID_seconds,temp,':')
		temp = generic_filter(temp,np.mean,size=[max(1,int(1/to_filter/np.mean(np.diff(time_of_experiment_digitizer_ID_seconds))))])

		if False:
			temp2 = temp[np.logical_and(time_of_experiment_digitizer_ID_seconds<2.5-0.05,time_of_experiment_digitizer_ID_seconds>2.5-0.05-0.2)]
			temp2 -= np.mean(temp2)
			def sinusoid(x, amplitude, frequency, phase,offset):
				return amplitude * np.sin(frequency * x + phase+offset)

			# Use curve_fit to fit the sinusoidal function to the data
			initial_guess = [2, 7*30, 0,0]  # Initial guess for amplitude, frequency, phase, and offset
			bounds = ([1, 0, 0,-np.inf], [np.inf, np.inf, 2 * np.pi, np.inf])
			params, params_covariance = curve_fit(sinusoid, time_of_experiment_digitizer_ID_seconds[np.logical_and(time_of_experiment_digitizer_ID_seconds<2.5-0.05,time_of_experiment_digitizer_ID_seconds>2.5-0.05-0.2)], temp2, p0=initial_guess, bounds=bounds)

			full_spectra = np.fft.fft(temp2)
			full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
			full_freq = np.fft.fftfreq(len(full_magnitude), d= np.mean(np.diff(time_of_experiment_digitizer_ID_seconds)) )

		periodicity = 1/to_filter
		if laser_framerate<60:
			add_time = 0.15	# s
		else:
			add_time = 0.9*periodicity	# 1 waves with frequency ~ 29.1Hz
		def check(args):
			c0,c1,c2,t0=args
			select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds<=t0+1*add_time,time_of_experiment_digitizer_ID_seconds>=t0-2*add_time)
			# select_time = np.logical_and(select_time,np.logical_or(time_of_experiment_digitizer_ID_seconds>=t0,time_of_experiment_digitizer_ID_seconds<=t0-periodicity/2))	# I want to try excluding the end of the last SS wave as that is effected by the start of the rise	# EDIT: not really necessary
			# select_time = np.logical_or(time_of_experiment_digitizer_ID_seconds<=t0-0.4,np.logical_and(time_of_experiment_digitizer_ID_seconds<=t0+0.4,time_of_experiment_digitizer_ID_seconds>=t0+0.1))
			positive_time = np.maximum(0,time_of_experiment_digitizer_ID_seconds[select_time]-t0)
			out = np.sum((temp[select_time]-c0-c1*positive_time-c2*positive_time**2)**2)
			# print(out)
			return out
		guess = [np.median(temp[time_of_experiment_digitizer_ID_seconds<start_time_of_pulse-0.4][10:]),max(15,(temp[np.abs(time_of_experiment_digitizer_ID_seconds-0.2-start_time_of_pulse).argmin()]-temp[np.abs(time_of_experiment_digitizer_ID_seconds-start_time_of_pulse).argmin()])/0.2),0,start_time_of_pulse-0.02]
		# guess[2] = 0#guess[1]**0.5
		# for some reasons I can only have a linear fit, otherwise the fit fails
		# Also I have to pin pretty hard the SS value, otherwise it moved it too much
		fit, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(check, x0=guess,approx_grad=True,bounds = [[guess[0]-0.1,guess[0]+0.1],[15,np.inf],[-1E-8,1E-8],[start_time_of_pulse-0.04,start_time_of_pulse+0.020]],maxls=100,pgtol=1e-8,factr=10000) #,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
		# fit, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(check, x0=guess,approx_grad=True,bounds = [[guess[0]-0.1,guess[0]+0.1],[0,np.inf],[0,np.inf],[start_time_of_pulse-0.1,start_time_of_pulse+0.1]]) #,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
		# print(opt_info)	with the linear fit, even it the fit fails the solution that it finds in the meantime is good already
		# print(fit)
		plt.plot(time_of_experiment_digitizer_ID_seconds,temp,color=a[0].get_color())
		plt.plot(time_of_experiment_digitizer_ID_seconds,fit[0]+fit[1]*np.maximum(0,time_of_experiment_digitizer_ID_seconds-fit[-1])+fit[2]*np.maximum(0,time_of_experiment_digitizer_ID_seconds-fit[-1])**2,'--',color=a[0].get_color())
		plt.axvline(x=fit[-1]+add_time,color=a[0].get_color())
		plt.axvline(x=fit[-1],color=a[0].get_color())
		plt.axhline(y=guess[0],color=a[0].get_color())
		real_start_time_of_pulse.append(fit[-1])
		real_start_time_of_pulse_uncertainty.append(opt_info['grad'][-1])
	# EXCLUDED 2024-11-04 it turns out the it is not necessary, the difference is of the order of a few ms for framerate = 383Hz	# 2024/11/15 not really, rectified now with proper quantification
	if not (frame_time_rigidly_determined and frame_time_succesfully_rigidly_determined):
		# start_time_of_pulse = np.mean(real_start_time_of_pulse)
		start_time_of_pulse = np.sum(np.array(real_start_time_of_pulse)/np.abs(real_start_time_of_pulse_uncertainty))/np.sum(1/np.abs(real_start_time_of_pulse_uncertainty))
		additional_correction_from_disruptions = -10*1e-3
		time_uncertainty_down = 10*1e-3
		time_uncertainty_up = 15*1e-3
		if int(laser_to_analyse[-9:-4])<45514:	# MU01
			additional_correction_from_disruptions = -10.3*1e-3
			time_uncertainty_down = 15*1e-3
			time_uncertainty_up = 30*1e-3
		elif int(laser_to_analyse[-9:-4])<47175:	# MU02
			additional_correction_from_disruptions = -9.9*1e-3
			time_uncertainty_down = 10*1e-3
			time_uncertainty_up = 15*1e-3
		elif int(laser_to_analyse[-9:-4])<49476:	# MU03
			additional_correction_from_disruptions = -11*1e-3
			time_uncertainty_down = 10*1e-3
			time_uncertainty_up = 15*1e-3
		print('time axis not successfully determined, so it is established with the start of the rise of the signal plus the additional correction factors coming from looking at disruptions')
	else:
		if frame_time_succesfully_rigidly_determined:
			additional_correction_from_disruptions = 0*1e-3
			time_uncertainty_down = 1*1e-3
			time_uncertainty_up = 1*1e-3
		else:
			additional_correction_from_disruptions = 0*1e-3
			time_uncertainty_down = 3/laser_framerate
			time_uncertainty_up = 3/laser_framerate
		external_clock_marker = True
	plt.axvline(x=start_time_of_pulse,color='k')
	plt.axvline(x=start_time_of_pulse+additional_correction_from_disruptions,color='k',linestyle='--')
	plt.xlim(left=start_time_of_pulse-2*add_time,right=start_time_of_pulse+2*add_time)
	plt.ylim(bottom=temp[time_of_experiment_digitizer_ID_seconds<start_time_of_pulse+add_time].min()-3,top=temp[time_of_experiment_digitizer_ID_seconds<start_time_of_pulse+2*add_time].max()+3)
	plt.grid()
	try:
		plt.title(day+'/'+name+'\nint time %.3gms, framerate %.3gHz, oscill. found %.3gHz\nstart time = weigh mean([%.5gs+/-%.3gms,%.5gs+/-%.3gms])=%.5gs\nadditional correction from VDEs = %.3g/+%.3g-%.3gms, final start time=%.5g' %(laser_int_time/1000,laser_framerate,to_filter,real_start_time_of_pulse[0],real_start_time_of_pulse_uncertainty[0]*1000,real_start_time_of_pulse[1],real_start_time_of_pulse_uncertainty[1]*1000,start_time_of_pulse,additional_correction_from_disruptions*1e3,time_uncertainty_down*1e3,time_uncertainty_up*1e3,start_time_of_pulse+additional_correction_from_disruptions))
	except:
		plt.title(day+'/'+name+'\nint time %.3gms, framerate %.3gHz, oscill. found %.3gHz\nstart time = weigh mean([%.5gs+/-%.3gms])=%.5gs\nadditional correction from VDEs = %.3g/+%.3g-%.3gms, final start time=%.5g' %(laser_int_time/1000,laser_framerate,to_filter,real_start_time_of_pulse[0],real_start_time_of_pulse_uncertainty[0]*1000,start_time_of_pulse,additional_correction_from_disruptions*1e3,time_uncertainty_down*1e3,time_uncertainty_up*1e3,start_time_of_pulse+additional_correction_from_disruptions))
	plt.savefig(laser_to_analyse[:-4]+'_1.eps', bbox_inches='tight')
	plt.close()
	start_time_of_pulse = start_time_of_pulse+additional_correction_from_disruptions

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

	# if none of these is activated the used parameters are the one from the old FLIR SC7500 camera
	if (int(laser_to_analyse[-9:-4])>49476 and int(laser_to_analyse[-9:-4])<50646):
		parameters_available_int_time_BB_int = parameters_available_int_time_BB_X6980
		parameters_available_framerate_BB_int = parameters_available_framerate_BB_X6980
		parameters_available_BB_int = parameters_available_BB_X6980
		pathparams_BB_int = pathparams_BB_X6980

		parameters_available_int_time_int = parameters_available_int_time_BB_X6980
		parameters_available_framerate_int = parameters_available_framerate_BB_X6980
		parameters_available_int = parameters_available_BB_X6980
		pathparams_int = pathparams_BB_X6980
	elif (int(laser_to_analyse[-9:-4])>51639 and int(laser_to_analyse[-9:-4])<52778):
		parameters_available_int_time_BB_int = parameters_available_int_time_BB_X6980_2
		parameters_available_framerate_BB_int = parameters_available_framerate_BB_X6980_2
		parameters_available_BB_int = parameters_available_BB_X6980_2
		pathparams_BB_int = pathparams_BB_X6980_2

		parameters_available_int_time_int = parameters_available_int_time_BB_X6980_2
		parameters_available_framerate_int = parameters_available_framerate_BB_X6980_2
		parameters_available_int = parameters_available_BB_X6980_2
		pathparams_int = pathparams_BB_X6980_2
	elif (int(laser_to_analyse[-9:-4])>53316 and int(laser_to_analyse[-9:-4])<99999):
		parameters_available_int_time_BB_int = parameters_available_int_time_BB_X6980_3
		parameters_available_framerate_BB_int = parameters_available_framerate_BB_X6980_3
		parameters_available_BB_int = parameters_available_BB_X6980_3
		pathparams_BB_int = pathparams_BB_X6980_3

		parameters_available_int_time_int = parameters_available_int_time_BB_X6980_3
		parameters_available_framerate_int = parameters_available_framerate_BB_X6980_3
		parameters_available_int = parameters_available_BB_X6980_3
		pathparams_int = pathparams_BB_X6980_3
	else:
		parameters_available_int_time_BB_int = parameters_available_int_time_BB
		parameters_available_framerate_BB_int = parameters_available_framerate_BB
		parameters_available_BB_int = parameters_available_BB
		pathparams_BB_int = pathparams_BB

		parameters_available_int_time_int = parameters_available_int_time
		parameters_available_framerate_int = parameters_available_framerate
		parameters_available_int = parameters_available
		pathparams_int = pathparams


	temp = np.abs(parameters_available_int_time_int-laser_int_time/1000)<0.1
	framerate = np.array(parameters_available_framerate_int)[temp][np.abs(parameters_available_framerate_int[temp]-laser_framerate).argmin()]
	int_time = np.array(parameters_available_int_time_int)[temp][np.abs(parameters_available_framerate_int[temp]-laser_framerate).argmin()]
	temp = np.array(parameters_available_int)[temp][np.abs(parameters_available_framerate_int[temp]-laser_framerate).argmin()]
	print('parameters selected '+temp)

	# Load parameters
	# temp = pathparams+'/'+temp+'/numcoeff'+str(n)+'/average'
	temp = pathparams_int+'/'+temp+'/numcoeff'+str(n)
	fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
	params_dict=np.load(fullpathparams)
	params_dict.allow_pickle=True
	params = params_dict['coeff']
	errparams = params_dict['errcoeff']

	time_partial = []
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		laser_counts_corrected[i] = laser_counts_corrected[i][time_of_experiment_digitizer_ID_seconds<14.5]
		laser_counts[i] = laser_counts[i][time_of_experiment_digitizer_ID_seconds<14.5]
		time_of_experiment_digitizer_ID[i] = time_of_experiment_digitizer_ID[i][time_of_experiment_digitizer_ID_seconds<14.5]
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		# if external_clock_marker:	# commented 2025/5/20 because who know what was really this
		# 	time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.nanmean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
		time_partial.append(time_of_experiment_digitizer_ID_seconds)
	time_full_int = np.sort(np.concatenate(time_partial))

	# this is not filtered, but I guess it's well enough for the FAST process
	temp_ref_counts = []
	temp_ref_counts_std = []
	for i in range(len(laser_digitizer_ID)):
		time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
		if flag_use_of_first_frames_as_reference:
			temp_ref_counts.append(np.mean(laser_counts_corrected[i][np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>time_of_experiment_digitizer_ID_seconds.min()+0.5)],axis=0))
			temp_ref_counts_std.append(np.std(laser_counts_corrected[i][np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>time_of_experiment_digitizer_ID_seconds.min()+0.5)],axis=0))
		else:
			temp_ref_counts.append(np.mean(laser_counts_corrected[i][-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
			temp_ref_counts_std.append(np.std(laser_counts_corrected[i][-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
	reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_poly_multi_digitizer(temp_ref_counts,params,errparams,laser_digitizer_ID,number_cpu_available,counts_std=temp_ref_counts_std,report=0,parallelised=False)

	ref_temperature = coleval.retrive_vessel_average_temp_archve(int(laser_to_analyse[-9:-4]))
	ref_temperature_std = 0.25	# coming from the fact that there is no noise during transition, so the std must be quite below 1K
	if np.isnan(ref_temperature):
		ref_temperature = 25	# np.mean(reference_background_temperature)	# 2023/06/27 modified because I verified that it is not trustworthy to use the camera as thermometer, it's good only for differences
		ref_temperature_std = 1	# (np.sum(np.array(reference_background_temperature_std)**2)**0.5 / len(np.array(reference_background_temperature).flatten()))

	# Load BB parameters
	temp = np.abs(parameters_available_int_time_BB_int-laser_int_time/1000)<0.1
	framerate = np.array(parameters_available_framerate_BB_int)[temp][np.abs(parameters_available_framerate_BB_int[temp]-laser_framerate).argmin()]
	int_time = np.array(parameters_available_int_time_BB_int)[temp][np.abs(parameters_available_framerate_BB_int[temp]-laser_framerate).argmin()]
	temp = np.array(parameters_available_BB_int)[temp][np.abs(parameters_available_framerate_BB_int[temp]-laser_framerate).argmin()]
	print('parameters selected '+temp)

	# temp = pathparams+'/'+temp+'/numcoeff'+str(n)+'/average'
	temp = pathparams_BB_int+'/'+temp+'/numcoeff'+str(n)
	fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
	params_dict=np.load(fullpathparams)
	params_dict.allow_pickle=True
	params_BB = params_dict['coeff2']
	errparams_BB = params_dict['errcoeff2']
	BB_proportional,BB_proportional_std,constant_offset,constant_offset_std,photon_dict = coleval.calc_BB_coefficients_multi_digitizer(params_BB,errparams_BB,laser_digitizer_ID,temp_ref_counts,temp_ref_counts_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=laser_int_time/1000)
	photon_flux_over_temperature_interpolator = photon_dict['photon_flux_over_temperature_interpolator']

	if int(laser_to_analyse[-9:-4]) >= 53317:	# MU05 third time with FLIR X6980
		foil_position_dict = dict([('angle',0.4),('foilcenter',[197,129]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',297)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) >= 52778:	# MU05 start with old FLIR SC7500 camera
		foil_position_dict = dict([('angle',1),('foilcenter',[141,139]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',246)])
	elif int(laser_to_analyse[-9:-4]) >= 52277:	# MU04 second time with FLIR X6980, FOV somehow moved
		foil_position_dict = dict([('angle',0.4),('foilcenter',[167,142]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',297)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) >= 51792:	# MU04 second time with FLIR X6980, FOV somehow moved
		foil_position_dict = dict([('angle',0.4),('foilcenter',[167,143]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',297)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) >= 51773:	# MU04 second time with FLIR X6980
		foil_position_dict = dict([('angle',0.4),('foilcenter',[167,145]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',297)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) >= 51639:	# MU04 second time with FLIR X6980
		foil_position_dict = dict([('angle',0.4),('foilcenter',[170,147]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',297)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) > 50714:	# MU04
		foil_position_dict = dict([('angle',1),('foilcenter',[152,143]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',246)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) > 47174:	# MU03
		foil_position_dict = dict([('angle',1),('foilcenter',[155,137]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',246)])	# identified ~2023-08 after checking what is the actual result of the rotation
	elif int(laser_to_analyse[-9:-4]) > 45517:	# MU02
		foil_position_dict = dict([('angle',1),('foilcenter',[159,137]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',246)])	# identified ~2023-08 after checking what is the actual result of the rotation
		# foil_position_dict = dict([('angle',1),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',246)])	# identified 2022-11-08 for MU02
	else:
		# foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix
		foil_position_dict = dict([('angle',0.6),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',242)])	# identified ~2023-08 after checking what is the actual result of the rotation

	try:
		full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST.npz')
		full_saved_file_dict_FAST.allow_pickle=True
		full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
		full_saved_file_dict_FAST_covariance = np.load(laser_to_analyse[:-4]+'_FAST_covariance.npz')
		full_saved_file_dict_FAST_covariance.allow_pickle=True
		full_saved_file_dict_FAST_covariance = dict(full_saved_file_dict_FAST_covariance)
		if override_FAST_analysis:
			bla=asgas #	I want an error to happen to override the FAST analysis
		inverted_dict = full_saved_file_dict_FAST['first_pass'].all()['inverted_dict']
		test = inverted_dict[str(2)]['regolarisation_coeff_all']
		foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx = coleval.get_rotation_crop_parameters(temp_ref_counts[-1],foil_position_dict,laser_to_analyse,laser_counts_corrected[-1]-temp_ref_counts[-1],time_of_experiment_digitizer_ID_seconds)
		print('FAST creation skipped')
	except:
		print('generating FAST'+' '+str(datetime.today()))
		try:
			start = tm.time()
			try:	# I differenciate between first and second pass, so no need now to redo the first pass if the second is missing
				test = full_saved_file_dict_FAST['first_pass']
				test = full_saved_file_dict_FAST_covariance['first_pass']
			except:
				full_saved_file_dict_FAST = dict([])
				full_saved_file_dict_FAST_covariance = dict([])
			pass_number = 0

			try:
				test = full_saved_file_dict_FAST['first_pass']
				test = full_saved_file_dict_FAST_covariance['first_pass']
				if override_first_pass:
					sgans = dsjfd	# I want this path to fail to regenerate the first pass
				try:
					time_of_last_first_pass = full_saved_file_dict_FAST['first_pass']['processing_start_time']
				except:
					time_of_last_first_pass = full_saved_file_dict_FAST['first_pass'].all()['processing_start_time']
				if datetime.strptime(time_of_last_first_pass,'%Y-%m-%d %H:%M:%S.%f') < datetime(2025,10,18):	# date threshold to deonstrate the validity of the inversion to be "new" enough
					print('first pass redone as '+str(datetime.strptime(time_of_last_first_pass,'%Y-%m-%d %H:%M:%S.%f'))+' is too old compared to the set threshold of ' + str(datetime(2025,10,18)))
					sgans = dsjfd	# I want this path to fail to regenerate the first pass
				print('last pass '+str(pass_number)+' done '+time_of_last_first_pass+' therefore inversion skipped\npost analysis only done instead')
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
			except:
				# foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop,FAST_counts_minus_background_crop_time = coleval.MASTU_pulse_process_FAST(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,laser_dict['height'],laser_dict['width'],flag_use_of_first_frames_as_reference)
				foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop,time_binned,powernoback,brightness,binning_type,inverted_dict,covariance_dict,temperature_minus_background_crop_dt,temperature_minus_background_crop_dt_time \
				= coleval.MASTU_pulse_process_FAST3_BB(
				laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,
				laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,laser_dict['height'],laser_dict['width'],
				flag_use_of_first_frames_as_reference,params,errparams,params_BB,errparams_BB,photon_flux_over_temperature_interpolator,BB_proportional,BB_proportional_std,foil_position_dict,
				pass_number = pass_number,disruption_check=True,x_point_region_radious=0.2,wavewlength_top=5.1,wavelength_bottom=1.5,only_plot_brightness = only_plot_brightness)
				full_saved_file_dict_FAST['first_pass'] = dict([])
				full_saved_file_dict_FAST['first_pass']['FAST_counts_minus_background_crop'] = np.float16(FAST_counts_minus_background_crop)
				full_saved_file_dict_FAST['first_pass']['FAST_powernoback'] = np.float16(powernoback)
				full_saved_file_dict_FAST['first_pass']['FAST_brightness'] = np.float32(brightness)
				full_saved_file_dict_FAST['first_pass']['FAST_time_binned'] = time_binned
				full_saved_file_dict_FAST['first_pass']['FAST_binning_type'] = binning_type
				full_saved_file_dict_FAST['first_pass']['inverted_dict'] = inverted_dict
				full_saved_file_dict_FAST['first_pass']['time_full_full'] = time_full_int
				full_saved_file_dict_FAST['time_counts'] = time_full_int
				full_saved_file_dict_FAST['first_pass']['temperature_minus_background_crop_dt'] = np.float16(temperature_minus_background_crop_dt)
				full_saved_file_dict_FAST['first_pass']['temperature_minus_background_crop_dt_time'] = np.float16(temperature_minus_background_crop_dt_time)
				full_saved_file_dict_FAST['first_pass']['processing_start_time'] = str(datetime.fromtimestamp(start))
				full_saved_file_dict_FAST['first_pass']['processing_end_time'] = str(datetime.fromtimestamp(tm.time()))
				full_saved_file_dict_FAST['first_pass']['time_uncertainty_down'] = time_uncertainty_down
				full_saved_file_dict_FAST['first_pass']['time_uncertainty_up'] = time_uncertainty_up
				try:
					full_saved_file_dict_FAST['neutron_caused_blips'] = neutron_caused_blips
				except:
					pass
				# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)

				full_saved_file_dict_FAST_covariance['first_pass'] = covariance_dict
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST_covariance',**full_saved_file_dict_FAST_covariance)

				print('generated FAST first pass in %.3g min' %((tm.time()-start)/60)+' '+str(datetime.today()))
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_multi_instrument.py").read())
				# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				print('generated FAST first pass further analysis in %.3g min' %((tm.time()-start)/60)+' '+str(datetime.today()))

			start = tm.time()
			pass_number = 1

			if skip_second_pass:
				print('temporary: second FAST pass skipped to do them quicker'+' '+str(datetime.today()))
				# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
				stra = second_fast_skipped	# just to cause an error

			try:
				test = full_saved_file_dict_FAST['second_pass']
				test = full_saved_file_dict_FAST_covariance['second_pass']
				if override_second_pass:
					sgans = dsjfd	# I want this path to fail to regenerate the second pass
				try:
					time_of_last_second_pass = full_saved_file_dict_FAST['second_pass']['processing_start_time']
				except:
					time_of_last_second_pass = full_saved_file_dict_FAST['second_pass'].all()['processing_start_time']
				if datetime.strptime(time_of_last_second_pass,'%Y-%m-%d %H:%M:%S.%f') < datetime(2025,10,18):	# date threshold to deonstrate the validity of the inversion to be "new" enough
					print('second pass redone as '+str(datetime.strptime(time_of_last_second_pass,'%Y-%m-%d %H:%M:%S.%f'))+' is too old compared to the set threshold of ' + str(datetime(2025,10,18)))
					sgans = dsjfd	# I want this path to fail to regenerate the second pass
				print('last pass '+str(pass_number)+' done '+time_of_last_second_pass+' therefore inversion skipped\npost analysis only done instead')
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
			except:
				foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop,time_binned,powernoback,brightness,binning_type,inverted_dict,covariance_dict = coleval.MASTU_pulse_process_FAST3_BB(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,laser_dict['height'],laser_dict['width'],flag_use_of_first_frames_as_reference,params,errparams,params_BB,errparams_BB,photon_flux_over_temperature_interpolator,BB_proportional,BB_proportional_std,foil_position_dict,pass_number = pass_number,x_point_region_radious=0.2,wavewlength_top=5.1,wavelength_bottom=1.5,override_second_pass=True)
				full_saved_file_dict_FAST['second_pass'] = dict([])
				full_saved_file_dict_FAST['second_pass']['FAST_counts_minus_background_crop'] = np.float16(FAST_counts_minus_background_crop)
				full_saved_file_dict_FAST['second_pass']['FAST_powernoback'] = np.float16(powernoback)
				full_saved_file_dict_FAST['second_pass']['FAST_brightness'] = np.float32(brightness)
				full_saved_file_dict_FAST['second_pass']['FAST_time_binned'] = time_binned
				full_saved_file_dict_FAST['second_pass']['FAST_binning_type'] = binning_type
				full_saved_file_dict_FAST['second_pass']['inverted_dict'] = inverted_dict
				full_saved_file_dict_FAST['second_pass']['time_full_full'] = time_full_int
				full_saved_file_dict_FAST['second_pass']['processing_start_time'] = str(datetime.fromtimestamp(start))
				full_saved_file_dict_FAST['second_pass']['processing_end_time'] = str(datetime.fromtimestamp(tm.time()))
				full_saved_file_dict_FAST['second_pass']['time_uncertainty_down'] = time_uncertainty_down
				full_saved_file_dict_FAST['second_pass']['time_uncertainty_up'] = time_uncertainty_up
				# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)

				full_saved_file_dict_FAST_covariance['second_pass'] = covariance_dict
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST_covariance',**full_saved_file_dict_FAST_covariance)

				print('generated FAST second pass in %.3g min' %((tm.time()-start)/60)+' '+str(datetime.today()))
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_multi_instrument.py").read())
				# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				print('generated FAST second pass further analysis in %.3g min' %((tm.time()-start)/60)+' '+str(datetime.today()))

			start = tm.time()
			pass_number = 2

			if skip_third_pass:
				print('temporary: third FAST pass skipped to do them quicker')
				# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
				stra = third_fast_skipped	# just to cause an error

			try:
				test = full_saved_file_dict_FAST['third_pass']
				test = full_saved_file_dict_FAST_covariance['third_pass']
				if override_third_pass:
					sgans = dsjfd	# I want this path to fail to regenerate the third pass
				try:
					time_of_last_third_pass = full_saved_file_dict_FAST['third_pass']['processing_start_time']
				except:
					time_of_last_third_pass = full_saved_file_dict_FAST['third_pass'].all()['processing_start_time']
				if datetime.strptime(time_of_last_third_pass,'%Y-%m-%d %H:%M:%S.%f') < datetime(2025,7,4):	# date threshold to deonstrate the validity of the inversion to be "new" enough
					print('third pass redone as '+str(datetime.strptime(time_of_last_third_pass,'%Y-%m-%d %H:%M:%S.%f'))+' is too old compared to the set threshold of ' + str(datetime(2025,7,4)))
					sgans = dsjfd	# I want this path to fail to regenerate the third pass

				# given I don't do an inversion, it doesn't make sense to run any post processing
				# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
			except:
				foilrotdeg,out_of_ROI_mask,foildw,foilup,foillx,foilrx,FAST_counts_minus_background_crop,time_binned,powernoback,brightness,binning_type,inverted_dict,covariance_dict = coleval.MASTU_pulse_process_FAST3_BB(laser_counts_corrected,time_of_experiment_digitizer_ID,time_of_experiment,external_clock_marker,aggregated_correction_coefficients,laser_framerate,laser_digitizer_ID,laser_int_time,seconds_for_reference_frame,start_time_of_pulse,laser_to_analyse,laser_dict['height'],laser_dict['width'],flag_use_of_first_frames_as_reference,params,errparams,params_BB,errparams_BB,photon_flux_over_temperature_interpolator,BB_proportional,BB_proportional_std,foil_position_dict,pass_number = pass_number,x_point_region_radious=0.2,wavewlength_top=5.1,wavelength_bottom=1.5,override_third_pass=True)
				full_saved_file_dict_FAST['third_pass'] = dict([])
				full_saved_file_dict_FAST['third_pass']['FAST_counts_minus_background_crop'] = np.float16(FAST_counts_minus_background_crop)
				full_saved_file_dict_FAST['third_pass']['FAST_powernoback'] = np.float16(powernoback)
				full_saved_file_dict_FAST['third_pass']['FAST_brightness'] = np.float32(brightness)
				full_saved_file_dict_FAST['third_pass']['FAST_time_binned'] = time_binned
				full_saved_file_dict_FAST['third_pass']['FAST_binning_type'] = binning_type
				full_saved_file_dict_FAST['third_pass']['inverted_dict'] = inverted_dict
				full_saved_file_dict_FAST['third_pass']['time_full_full'] = time_full_int
				full_saved_file_dict_FAST['third_pass']['processing_start_time'] = str(datetime.fromtimestamp(start))
				full_saved_file_dict_FAST['third_pass']['processing_end_time'] = str(datetime.fromtimestamp(tm.time()))
				full_saved_file_dict_FAST['third_pass']['time_uncertainty_down'] = time_uncertainty_down
				full_saved_file_dict_FAST['third_pass']['time_uncertainty_up'] = time_uncertainty_up
				# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)

				full_saved_file_dict_FAST_covariance['third_pass'] = covariance_dict
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST_covariance',**full_saved_file_dict_FAST_covariance)

				print('generated FAST third pass in %.3g min' %((tm.time()-start)/60)+' '+str(datetime.today()))
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_multi_instrument.py").read())
				# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
				print('generated FAST third pass further analysis in %.3g min' %((tm.time()-start)/60)+' '+str(datetime.today()))

		except Exception as e:
			print('FAST FAILED ' + laser_to_analyse)
			logging.exception('with error: ' + str(e))
	# print('with error: ' + str(e))

	try:
		del full_saved_file_dict_FAST, full_saved_file_dict_FAST_covariance
		print('full_saved_file_dict_FAST, full_saved_file_dict_FAST_covariance\ndeleted ahead of next shot analysis')
	except:
		pass

	filter_overwritten = False
	try:
		if overwrite_oscillation_filter==True:
			blagna=sbluppa	# I want them all
		else:
			test = np.load(laser_to_analyse[:-4]+'_FAST.npz')
			test.allow_pickle=True
			inverted_dict = dict(test)
			inverted_dict = inverted_dict['inverted_dict'].all()
			test = inverted_dict[str(2)]['regolarisation_coeff_all']
		test = full_saved_file_dict['only_foil'].all()[str(laser_digitizer_ID[0])]['BB_proportional_no_dead_pixels_crop']
		test = full_saved_file_dict['only_foil'].all()[str(laser_digitizer_ID[0])]['laser_temperature_no_dead_pixels_crop_minus_median']
		if test.shape[1:]!=(foilup-foildw,foilrx-foillx):
			bla = sdu+2	# random stuff to generate an error
		time_full = full_saved_file_dict['full_frame'].all()['time_full_full']
		print('data filtering skipped')
	except:
		print('Filtering data')
		if continue_after_FAST == True:
			filter_overwritten = True
			full_saved_file_dict['only_foil'] = dict([])
			temp_counts = []
			temp_counts_full = []
			temp_bad_pixels_flag = []
			time_partial = []
			temp_ref_counts = []
			temp_ref_counts_std = []
			full_saved_file_dict['full_frame'] = dict([])
			horizontal_coord = np.arange(np.shape(laser_counts[0])[2])
			vertical_coord = np.arange(np.shape(laser_counts[0])[1])
			horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
			select_space = np.logical_or(np.logical_or(vertical_coord<30,vertical_coord>240),np.logical_or(horizontal_coord<30,horizontal_coord>290))
			frames_per_oscillation_1 = max(1,int(np.floor(laser_framerate/len(laser_digitizer_ID)/30)))
			frames_per_oscillation_2 = max(1,int(np.ceil(laser_framerate/len(laser_digitizer_ID)/30)))

			seconds_for_bad_pixels = 2	# s
			fig, ax = plt.subplots( 2,1,figsize=(8, 15), squeeze=False)
			for i in range(len(laser_digitizer_ID)):
				full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])] = dict([])
				time_of_experiment_digitizer_ID_seconds = (time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6-start_time_of_pulse
				if external_clock_marker:
					time_of_experiment_digitizer_ID_seconds = time_of_experiment_digitizer_ID_seconds-np.mean(aggregated_correction_coefficients[:,4])	# I use the mean of the coefficients because I want to avoid small unpredictable differences between the digitisers
				select_time = np.logical_and(time_of_experiment_digitizer_ID_seconds>0-0.5,time_of_experiment_digitizer_ID_seconds<1.5)
				# if True:	# I'm not creating my problems by doing this before binning, but I have to do many cleanings
				if laser_framerate>60:	# I skip this for low framerate as there is no t really oshillation in this case
					if np.shape(laser_counts_corrected[i][0]) == (256, 320):
						force_poscentre = [204,160]	# among the points that I monitor this is the one with the highest oscillations
					else:
						force_poscentre=[int(np.shape(laser_counts_corrected[i])[1]//2),int(np.shape(laser_counts_corrected[i])[2]//2)]
					if False:	# just for test
						# freq_to_erase_frames_up = int(np.ceil(laser_framerate/len(laser_digitizer_ID)/27))
						# freq_to_erase_frames_down = int(np.floor(laser_framerate/len(laser_digitizer_ID)/38))
						freq_to_erase_frames_up = int(np.ceil(laser_framerate/len(laser_digitizer_ID)/31))
						freq_to_erase_frames_down = int(np.floor(laser_framerate/len(laser_digitizer_ID)/31))
						freq_to_erase_all = np.linspace(freq_to_erase_frames_down,freq_to_erase_frames_up,freq_to_erase_frames_up-freq_to_erase_frames_down+1).astype(int)
						filter_agent = generic_filter(laser_counts_corrected[i],np.mean,size=[freq_to_erase_all[0],1,1])
						# filter_agent = generic_filter(laser_counts_corrected[i],np.mean,size=[round(laser_framerate/len(laser_digitizer_ID)/30),1,1])
						laser_counts_filtered = laser_counts_corrected[i] - filter_agent
						for i_ in range(len(freq_to_erase_all)-1):
							temp = generic_filter(laser_counts_filtered,np.mean,size=[freq_to_erase_all[i_+1],1,1])
							filter_agent += temp
							laser_counts_filtered -= temp
					elif False:	# mean filter exactly of the size I need and perfectly centred on each pixel. reflection at the borders
						freq_to_erase_frames = laser_framerate/len(laser_digitizer_ID)/31
						filter_agent = cp.deepcopy(laser_counts_corrected[i])
						padded_by = int(np.ceil(freq_to_erase_frames/2))
						temp = np.pad(filter_agent, ((padded_by,padded_by),(0,0),(0,0)), mode='reflect')
						for ii in range(-int(np.floor((freq_to_erase_frames-1)/2)),int(np.floor((freq_to_erase_frames-1)/2))+1):
							if ii!=0:
								filter_agent += temp[padded_by+ii:-padded_by+ii]
						filter_agent += temp[padded_by-int(np.floor((freq_to_erase_frames-1)/2))-1:-padded_by-int(np.floor((freq_to_erase_frames-1)/2))-1]*((freq_to_erase_frames-1)/2-int(np.floor((freq_to_erase_frames-1)/2)))
						filter_agent += temp[padded_by+int(np.floor((freq_to_erase_frames-1)/2))+1:-padded_by+int(np.floor((freq_to_erase_frames-1)/2))+1]*((freq_to_erase_frames-1)/2-int(np.floor((freq_to_erase_frames-1)/2)))
						filter_agent/=freq_to_erase_frames
						laser_counts_filtered = laser_counts_corrected[i] - filter_agent
						del temp
						# this function is now incorporated inside clear_oscillation_central5
					# full_average = np.mean(laser_counts_filtered[:,np.logical_not(select_space)],axis=(-1))
					# full_spectra = np.fft.fft(full_average)
					# full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
					# full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
					# ax[0,0].plot(full_freq,full_magnitude/(100**i),'g--')
					# ax[1,0].plot(time_of_experiment_digitizer_ID_seconds[select_time],full_average[select_time]-temp*i,'g--')
					window = np.min(np.shape(laser_counts_corrected[i])[1:])//6
					filter_plot_index = 1
					if laser_framerate/len(laser_digitizer_ID)/2>32:
						laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central5([laser_counts_corrected[i]],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=25,max_frequency_to_erase=38,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_corrected[i])-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,force_poscentre=force_poscentre,output_noise=True,multiple_frequencies_cleaned=3)
						laser_counts_filtered = laser_counts_filtered[0]
						for trash in range(2):
							fig = matplotlib.pyplot.gcf()
							fig.set_size_inches(15, 10, forward=True)
							plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
							filter_plot_index+=1
							plt.close()
						# laser_counts_filtered += filter_agent
						# while peak_value_post_filter>2.8*median_noise:
						# 	laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=27,max_frequency_to_erase=38,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,force_poscentre=force_poscentre,output_noise=True,multiple_frequencies_cleaned=3)
						# 	laser_counts_filtered = laser_counts_filtered[0]
						# 	for trash in range(2):
						# 		fig = matplotlib.pyplot.gcf()
						# 		fig.set_size_inches(15, 10, forward=True)
						# 		plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
						# 		filter_plot_index+=1
						# 		plt.close()
					if laser_framerate/len(laser_digitizer_ID)/2>65:
						if np.shape(laser_counts_corrected[i][0]) == (256, 320):
							force_poscentre = [204,160]	# among the points that I monitor this is the one with the highest oscillations
						else:
							force_poscentre=[int(np.shape(laser_counts_corrected[i])[1]//2),int(np.shape(laser_counts_corrected[i])[2]//2)]
						laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central5([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=55,max_frequency_to_erase=70,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,force_poscentre=force_poscentre,output_noise=True,multiple_frequencies_cleaned=1)
						laser_counts_filtered = laser_counts_filtered[0]
						for trash in range(2):
							fig = matplotlib.pyplot.gcf()
							fig.set_size_inches(15, 10, forward=True)
							plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
							filter_plot_index+=1
							plt.close()
						# while peak_value_post_filter>2*median_noise:
						# 	laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=55,max_frequency_to_erase=70,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,force_poscentre=force_poscentre,output_noise=True,multiple_frequencies_cleaned=3)
						# 	laser_counts_filtered = laser_counts_filtered[0]
						# 	for trash in range(2):
						# 		fig = matplotlib.pyplot.gcf()
						# 		fig.set_size_inches(15, 10, forward=True)
						# 		plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
						# 		filter_plot_index+=1
						# 		plt.close()
					if laser_framerate/len(laser_digitizer_ID)/2>90:
						if np.shape(laser_counts_corrected[i][0]) == (256, 320):
							force_poscentre = [204,160]	# among the points that I monitor this is the one with the highest oscillations
						else:
							force_poscentre=[int(np.shape(laser_counts_corrected[i])[1]//2),int(np.shape(laser_counts_corrected[i])[2]//2)]
						laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central5([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=80,max_frequency_to_erase=102,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,force_poscentre=force_poscentre,output_noise=True,multiple_frequencies_cleaned=1)
						laser_counts_filtered = laser_counts_filtered[0]
						for trash in range(2):
							fig = matplotlib.pyplot.gcf()
							fig.set_size_inches(15, 10, forward=True)
							plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
							filter_plot_index+=1
							plt.close()
						# while peak_value_post_filter>2*median_noise:
						# 	laser_counts_filtered,peak_value_pre_filter,peak_value_post_filter,max_noise,median_noise = coleval.clear_oscillation_central2([laser_counts_filtered],laser_framerate/len(laser_digitizer_ID),min_frequency_to_erase=80,max_frequency_to_erase=102,oscillation_search_window_begin=0,oscillation_search_window_end=(len(laser_counts_filtered)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=True,which_plot=[1,2,3],window=window,force_poscentre=force_poscentre,output_noise=True,multiple_frequencies_cleaned=3)
						# 	laser_counts_filtered = laser_counts_filtered[0]
						# 	for trash in range(2):
						# 		fig = matplotlib.pyplot.gcf()
						# 		fig.set_size_inches(15, 10, forward=True)
						# 		plt.savefig(laser_to_analyse[:-4]+'_digitizer'+str(i) +'_filter'+str(filter_plot_index)+'.eps', bbox_inches='tight')
						# 		filter_plot_index+=1
						# 		plt.close()
				else:
					laser_counts_filtered = laser_counts_corrected[i]
				full_average = np.mean(laser_counts_corrected[i][:,np.logical_not(select_space)],axis=(-1))
				full_spectra = np.fft.fft(full_average)
				full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
				full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
				ax[0,0].plot(full_freq,full_magnitude/(100**i),color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
				temp = (full_average.max()-full_average.min())
				ax[1,0].plot(time_of_experiment_digitizer_ID_seconds[select_time],full_average[select_time]-temp*i,color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
				# ax[1,0].plot(time_of_experiment_digitizer_ID_seconds,full_average,color=color[i],label='DIG'+str(laser_digitizer_ID[i]))
				full_average = np.mean(laser_counts_filtered[:,np.logical_not(select_space)],axis=(-1))
				full_spectra = np.fft.fft(full_average)
				full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
				full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
				ax[0,0].plot(full_freq,full_magnitude/(100**i),'k--')
				ax[1,0].plot(time_of_experiment_digitizer_ID_seconds[select_time],full_average[select_time]-temp*i,'k--')
				if True:	# ramp up correction applied afterwards to facilitate oscillation removal
					# laser_counts_filtered = (laser_counts_filtered.T -exponential_biased((time_of_experiment_digitizer_ID[i]-time_of_experiment[0])*1e-6,*aggregated_correction_coefficients[i])).T
					full_average = np.mean(laser_counts_filtered[:,np.logical_not(select_space)],axis=(-1))
					full_spectra = np.fft.fft(full_average)
					full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
					full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (framerate/len(laser_digitizer_ID)))
					ax[0,0].plot(full_freq,full_magnitude/(100**i),'y--')
					ax[1,0].plot(time_of_experiment_digitizer_ID_seconds[select_time],full_average[select_time]-temp*i,'y--')

				# ax[1,0].plot(time_of_experiment_digitizer_ID_seconds,full_average,'k--')

				if flag_use_of_first_frames_as_reference:
					if False:	# from following plots it's clear that it's absolutely arbitrary the length of the interval to consider for the average. I pick 0.5 seconds just to be close to t=0
						plt.figure()
						plt.imshow(np.flip(np.transpose(np.mean(laser_counts_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0),(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
						plt.colorbar().set_label('counts [au]')
						plt.xlabel('Horizontal axis [pixles]')
						plt.ylabel('Vertical axis [pixles]')
						plt.pause(0.01)

						temp0 = np.mean(laser_counts_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0)
						temp1 = np.mean(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<0,time_of_experiment_digitizer_ID_seconds>-0.5)],axis=0)
						temp2 = np.mean(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.1,time_of_experiment_digitizer_ID_seconds>-0.6)],axis=0)
						temp3 = np.mean(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-1,time_of_experiment_digitizer_ID_seconds>-1.5)],axis=0)
						plt.figure()
						plt.imshow(np.flip(np.transpose( coleval.proper_homo_binning_2D(np.mean(laser_counts_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0)-np.mean(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>-1)],axis=0),10),(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
						plt.colorbar().set_label('counts [au]')
						plt.xlabel('Horizontal axis [pixles]')
						plt.ylabel('Vertical axis [pixles]')
						plt.pause(0.01)


						gna = coleval.proper_homo_binning_2D( temp1-temp2 ,10)
						foil_shape = np.shape(gna)
						ROI = np.array([[0.2,0.85],[0.1,0.9]])
						ROI = np.round((ROI.T*foil_shape).T).astype(int)
						plt.figure()
						plt.imshow(np.flip(np.transpose( gna,(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
						plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
						plt.colorbar().set_label('counts [au]')
						plt.xlabel('Horizontal axis [pixles]')
						plt.ylabel('Vertical axis [pixles]')
						plt.pause(0.01)

						plt.figure()
						temp = generic_filter(np.mean(laser_counts_filtered[:,100:110,170:180],axis=(1,2)),np.mean,footprint=np.concatenate([np.ones((100))/100,np.zeros((100))]))
						plt.plot(time_of_experiment_digitizer_ID_seconds,temp-temp.min())
						temp = generic_filter(np.mean(laser_counts_filtered[:,100:110,170-100:180-100],axis=(1,2)),np.mean,footprint=np.concatenate([np.ones((100))/100,np.zeros((100))]))
						plt.plot(time_of_experiment_digitizer_ID_seconds,temp-temp.min())
						temp = generic_filter(np.mean(laser_counts_filtered[:,100:110,170+100:180+100],axis=(1,2)),np.mean,footprint=np.concatenate([np.ones((100))/100,np.zeros((100))]))
						plt.plot(time_of_experiment_digitizer_ID_seconds,temp-temp.min())
						plt.grid()
						plt.pause(0.01)

						plt.figure()
						plt.imshow(np.flip(np.transpose(coleval.proper_homo_binning_2D(temp1-np.mean(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<0.1,time_of_experiment_digitizer_ID_seconds>-0.4)],axis=0),5),(1,0)),axis=1),'rainbow',interpolation='none',origin='lower',vmin=-2)#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
						plt.colorbar().set_label('counts [au]')
						plt.xlabel('Horizontal axis [pixles]')
						plt.ylabel('Vertical axis [pixles]')
						plt.pause(0.01)

						plt.figure()
						plt.imshow(np.flip(np.transpose(params[0,:,:,2],(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
						plt.colorbar().set_label('counts [au]')
						plt.xlabel('Horizontal axis [pixles]')
						plt.ylabel('Vertical axis [pixles]')
						plt.pause(0.01)
					elif False:
						2
					# temp_ref_counts.append(np.mean(laser_counts_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0))
					temp_ref_counts.append(np.mean(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>time_of_experiment_digitizer_ID_seconds.min()+0.5)],axis=0))
					# temp_ref_counts_std.append(np.std(laser_counts_filtered[time_of_experiment_digitizer_ID_seconds<0],axis=0))
					temp_ref_counts_std.append(np.std(laser_counts_filtered[np.logical_and(time_of_experiment_digitizer_ID_seconds<-0.5,time_of_experiment_digitizer_ID_seconds>time_of_experiment_digitizer_ID_seconds.min()+0.5)],axis=0))
				else:
					temp_ref_counts.append(np.mean(laser_counts_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))
					temp_ref_counts_std.append(np.std(laser_counts_filtered[-int(seconds_for_reference_frame*laser_framerate/len(laser_digitizer_ID)):],axis=0))

				time_partial.append(time_of_experiment_digitizer_ID_seconds)
				select_time = np.logical_and(time_partial[-1]>=-0.1,time_partial[-1]<=1.5)
				full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['time'] = time_partial[-1][select_time]
				temp_counts.append(laser_counts_filtered[select_time])
				temp_counts_full.append(laser_counts_filtered)
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

			reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_poly_multi_digitizer(temp_ref_counts,params,errparams,laser_digitizer_ID,number_cpu_available,counts_std=temp_ref_counts_std,report=0,parallelised=False)

			ref_temperature = coleval.retrive_vessel_average_temp_archve(int(laser_to_analyse[-9:-4]))
			ref_temperature_std = 0.25	# coming from the fact that there is no noise during transition, so the std must be quite below 1K
			if np.isnan(ref_temperature):
				print('for this phase I want the true vessel temperature but the reading failed\nwrong shot number of pyuda fails')
				exit()
				ref_temperature = np.mean(reference_background_temperature)
				ref_temperature_std = (np.sum(np.array(reference_background_temperature_std)**2)**0.5 / len(np.array(reference_background_temperature).flatten()))

			# I recalculate coleval.calc_BB_coefficients_multi_digitizer here so it uses filtered data
			BB_proportional,BB_proportional_std,constant_offset,constant_offset_std,photon_dict = coleval.calc_BB_coefficients_multi_digitizer(params_BB,errparams_BB,laser_digitizer_ID,temp_ref_counts,temp_ref_counts_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=laser_int_time/1000)

			laser_temperature,laser_temperature_std = coleval.count_to_temp_BB_multi_digitizer(temp_counts,params_BB,errparams_BB,laser_digitizer_ID,reference_background=temp_ref_counts,reference_background_std=temp_ref_counts_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=laser_int_time/1000)
			temp_counts_std = []
			for i in range(len(laser_digitizer_ID)):
				temp_counts_std.append(coleval.estimate_counts_std(temp_counts[i],int_time=laser_int_time/1000))

			laser_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature)]
			laser_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,laser_temperature_std)]
			reference_background_temperature_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,reference_background_temperature)]
			reference_background_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,reference_background_temperature_std)]
			temp_counts_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,temp_counts)]
			temp_counts_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(temp_bad_pixels_flag,temp_counts_std)]
			temp_ref_counts_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,temp_ref_counts)]
			temp_ref_counts_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,temp_ref_counts_std)]
			BB_proportional_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,BB_proportional)]
			BB_proportional_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(temp_bad_pixels_flag,BB_proportional_std)]

			# for i in range(len(laser_digitizer_ID)):
			# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_minus_median'] = np.float16(laser_temperature_no_dead_pixels[i].T-np.median(laser_temperature_no_dead_pixels[i],axis=(-1,-2))).T
			# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_median'] = np.median(laser_temperature_no_dead_pixels[i],axis=(-1,-2))
			# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_minus_median'] = np.float16(laser_temperature_std_no_dead_pixels[i].T-np.median(laser_temperature_std_no_dead_pixels[i],axis=(-1,-2))).T
			# 	full_saved_file_dict['full_frame'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_median'] = np.median(laser_temperature_std_no_dead_pixels[i],axis=(-1,-2))
			time_full_int = np.sort(np.concatenate(time_partial))
			full_saved_file_dict['full_frame']['time_full_full'] = time_full_int
			counts_full = [(temp_counts_full[0][np.abs(time_partial[0]-time).argmin()]) if time in time_partial[0] else (temp_counts_full[1][np.abs(time_partial[1]-time).argmin()]) for time in time_full_int]	# I keep this only to look at the movement of the foil frame
			time_full = time_full_int[np.logical_and(time_full_int>=0,time_full_int<=1.5)]
			for i in range(len(laser_digitizer_ID)):	#	I only save the usefull data, given the oscillation filtering is already done
				time_partial[i] = time_partial[i][np.logical_and(time_partial[i]>=0,time_partial[i]<=1.5)]
			time_full = np.sort(np.concatenate(time_partial))
			if len(np.unique(np.diff(time_full)))<5:
				full_saved_file_dict['full_frame']['time_full_mode'] = 'the time series was eiterpolated to increase resolution. time intervals were '+str(np.unique(np.diff(time_full)))
				temp = np.polyval(np.polyfit(np.arange(len(time_full)),time_full,1),np.arange(len(time_full)))
				time_full = temp - (temp[0]-time_full[0])
			else:
				full_saved_file_dict['full_frame']['time_full_mode'] = 'no time refinement required'
			full_saved_file_dict['full_frame']['time_full'] = time_full
			full_saved_file_dict['ref_temperature'] = ref_temperature
			full_saved_file_dict['ref_temperature_std'] = ref_temperature_std
			full_saved_file_dict['photon_flux_over_temperature_interpolator'] = photon_flux_over_temperature_interpolator
			print('data filtering done')
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process3_BB.py").read())
		else:
			print('analysis halted at FAST to save memory')

	if continue_after_FAST == True or filter_overwritten:
		if False:	# there is no reason to keep doing these
			if not(os.path.exists(laser_to_analyse[:-4]+ '_raw_short.mp4')):
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
					ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_counts_merged[::2,::2,::2],(0,2,1)),axis=2)]), laser_framerate/2, integration=laser_int_time/1000,time_offset=-start_time_of_pulse,xlabel='horizontal coord [pixels]\nprinted 1 pixel every 2',ylabel='vertical coord [pixels]\nprinted 1 pixel every 2',barlabel='raw counts [au]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
					# plt.pause(0.01)
					ani.save(laser_to_analyse[:-4]+ '_raw.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
					plt.close('all')
					ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_counts_merged[select_time],(0,2,1)),axis=2)]), laser_framerate, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
					# plt.pause(0.01)
					ani.save(laser_to_analyse[:-4]+ '_raw_short.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
					plt.close('all')

					del laser_counts_corrected_merged

		else:
			pass
		del laser_counts_corrected

		try:
			if overwrite_binning or filter_overwritten:
				tregna = sba	# I want this to fail
			saved_file_dict_short = np.load(laser_to_analyse[:-4]+'_short.npz')
			saved_file_dict_short.allow_pickle=True
			saved_file_dict_short = dict(saved_file_dict_short)
			test = saved_file_dict_short['bin7x2x2'].all()['laser_framerate_binned']
			print('binning skipped')
		except:
			print('binning data')
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3_BB.py").read())
			print('binning done')

		if do_inversions:
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_power_to_emissivity.py").read())

		print('completed ' + laser_to_analyse)

except Exception as e:
	print('FAILED ' + laser_to_analyse)
	logging.exception('with error: ' + str(e))
	# print('with error: ' + str(e))
