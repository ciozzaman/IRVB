# Created 17/09/2021
# Fabio Federici
# created from Laser_data_analysis3_2.py . I make only one cycle in which I fitt all parameters

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 14
figure_index=0
path_where_to_save_everything = '/home/ffederic/work/irvb/laser/results/foil_properties/'

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']

f = []
f1 = []
for (dirpath, dirnames, filenames) in os.walk('/home/ffederic/work/irvb/laser/results'):
	f1.append(dirnames)
for path in f1[0]:
	if path=='foil_properties' or path=='foil_properties_in_steps':
		continue
	f2 = []
	for (dirpath, dirnames, filenames) in os.walk('/home/ffederic/work/irvb/laser/results/'+path):
		f2.append(dirnames)
	for f3 in f2[0]:
		if f3[-3:]!='old':
			f.append([path,f3])

cases_to_include = np.array(f)[:,1]
for i_name,name in enumerate(cases_to_include):
	try:
		freqlaser = []
		voltlaser = []
		laser_location = []
		FrameRate = []
		for i_laser_to_analyse,laser_to_analyse in enumerate(collection_of_records[name]['path_files_laser']):
			try:
				laser_dict = np.load(laser_to_analyse+'.npz')
				laser_dict.allow_pickle = True
				try:
					laser_location.append(laser_dict['laser_location'])
				except:
					laser_location.append(laser_dict['NUC_plate'].all()['laser_location'])
				freqlaser.append(collection_of_records[name]['freqlaser'][i_laser_to_analyse])
				voltlaser.append(collection_of_records[name]['voltlaser'][i_laser_to_analyse])
				FrameRate.append(laser_dict['FrameRate'])
			except Exception as e:
				print('missing ' + laser_to_analyse+'.npz')
				logging.exception('with error: ' + str(e))
		f[i_name].extend(np.array(laser_location)[np.array(voltlaser)==np.max(voltlaser)][(np.array(freqlaser)[np.array(voltlaser)==np.max(voltlaser)]).argmax()])
		f[i_name].extend([np.array(FrameRate)[np.array(voltlaser)==np.max(voltlaser)][(np.array(freqlaser)[np.array(voltlaser)==np.max(voltlaser)]).argmax()]])
	except Exception as e:
		print('missing ' + name)
		logging.exception('with error: ' + str(e))
		f[i_name].extend(np.array([0,0,0,0,0]))
		f[i_name].extend([np.array(0)])

f = np.array(f)
f = np.array([[y1,y2,y3,y4,y5,y6,y7,y8] for _,y1,y2,y3,y4,y5,y6,y7,y8 in sorted(zip(np.round(f[:,-1].astype(np.float)), *f.T))])


foilhorizwpixel=240
foilhorizw=0.09	# m
foilvertw=0.07	# m
dx=foilhorizw/(foilhorizwpixel-1)
plt.figure(figsize=(60, 40))
plt.title('Location of the sampled points')
plt.xlabel('horizontal direction [m]')
plt.ylabel('vertical direction [m]')
index1=0
index2=0
index3=0
for array in f:
	x=float(array[2])*dx+np.random.random()/4*dx
	y=float(array[4])*dx+np.random.random()/4*dx
	if array[0]=='fully_defocused':
		plt.plot(x,y,'o',fillstyle='none',color=color[index1],label='full def'+' FR%.5gHz ' %(float(array[-1])) + array[1],markersize=5+index1*3)
		index1+=1
	elif array[0]=='partially_defocused':
		plt.plot(x,y,'v',fillstyle='none',color=color[index2],label='part def'+' FR%.5gHz ' %(float(array[-1])) + array[1],markersize=5+index2*3)
		index2+=1
	else:
		plt.plot(x,y,'P',fillstyle='none',color=color[index3],label='focus'+' FR%.5gHz ' %(float(array[-1])) + array[1],markersize=5+index3*3)
		index3+=1
	if np.random.random()>0.5:
		x_=1
	else:
		x_=-5
	if np.random.random()>0.5:
		y_=1
	else:
		y_=-5
	plt.annotate(array[1][-2:],(x+x_*1e-4,y+y_*1e-4),fontsize=10)
plt.plot([0,foilhorizw,foilhorizw,0,0],[0,0,foilvertw,foilvertw,0],'k')
plt.grid()
plt.legend(loc='best', fontsize='x-small',ncol=2)
figure_index+=1
plt.savefig(path_where_to_save_everything + str(figure_index)+ '_' + 'foil_map'  +'.eps', bbox_inches='tight')
plt.close()


power_reduction_window = (6.4+6.3)/2/6.8	# known from a dedicated investigation done 2021/09/14
# threshold_freq_list = [1,10,60,100,160,240,300]
minimum_ON_period_list = 0.0003	# seconds
minimum_frames_for_ON_phase = 2# 4 frames
# cases_to_include = ['laser17']
# cases_to_include = ['laser39','laser37']	# FR ~2kHz
# cases_to_include = ['laser38','laser36']	# FR ~1kHz
# cases_to_include = ['laser34','laser35']	# FR ~383Hz
# all_cases_to_include = [['laser34','laser35'] , ['laser38','laser36'] , ['laser39','laser37']]
# all_cases_to_include = [['laser34','laser35'], ['laser34'], ['laser35'] , ['laser38','laser36'], ['laser38'], ['laser36'] , ['laser39','laser37'], ['laser39'], ['laser37']]

sample_properties = dict([])
# sample_properties['thickness'] = 3.2*1e-6
# sample_properties['emissivity'] = 1
# sample_properties['diffusivity'] = 1.32*1e-5
sample_properties['thickness'] = 2.093616658223934e-06
sample_properties['emissivity'] = 1
sample_properties['diffusivity'] = 1.03*1e-5

# WARNING 'laser20' has bad data in it
# also laser25 101 has something wrong
# apparently laser 35 is too different from the other defocused ones


# cases_to_include = ['laser22','laser25','laser33','laser30']	# FR ~383Hz
# cases_to_include = ['laser15','laser16','laser23','laser21']	# FR ~1kHz
# cases_to_include = ['laser24','laser26','laser27','laser31','laser28','laser29']	# FR ~2kHz
all_cases_to_include = [['laser22','laser25','laser33','laser30'] , ['laser15','laser16','laser23','laser21'] , ['laser24','laser26','laser27','laser31','laser28','laser29'],['laser22','laser25','laser33','laser30','laser15','laser16','laser23','laser21','laser24','laser26','laser27','laser31','laser28','laser29']]
all_cases_to_include = [['laser34','laser35'], ['laser38','laser36'] , ['laser39','laser37'], ['laser34','laser35','laser38','laser36','laser39','laser37']]
all_cases_to_include = [['laser22','laser25','laser33','laser30','laser34','laser35']]
all_cases_to_include = [['laser22','laser25','laser34']]	# defocused scans avoided for now
all_cases_to_include = [['laser17','laser18','laser19','laser22','laser32','laser34','laser30','laser33']]	# all cases with correct fraq and no windowing
all_cases_to_include = [['laser19','laser22','laser30','laser33']]	# all cases in the same location
all_cases_to_include = [['laser19','laser22','laser30','laser33']]	# all cases in the same location
all_cases_to_include = [['laser22','laser33']]	# all cases in the same location
include_large_area_data = False
# type_of_calibration = 'NUC_plate'
type_of_calibration = 'BB_source_w/o_window'
# type_of_calibration = 'BB_source_w_window'
# all_cases_to_include = [['laser34','laser35']]	# all cases in the same location
for cases_to_include in all_cases_to_include:
	figure_index = 0
	coefficients = []
	coefficients_first_stage = []
	all_R2 = []
	all_sharpness_first = []
	all_sharpness_second = []
	# all_all_focus_status = []
	all_scan_type = []

	all_case_ID = []
	all_laser_to_analyse = []
	all_laser_to_analyse_ROI = []
	all_laser_to_analyse_frequency = []
	all_laser_to_analyse_voltage = []
	all_laser_to_analyse_duty = []
	all_focus_status = []
	all_power_interpolator = []
	for name in cases_to_include:
		all_case_ID.extend([name] * len(collection_of_records[name]['path_files_laser']))
		all_laser_to_analyse.extend(collection_of_records[name]['path_files_laser'])
		all_laser_to_analyse_ROI.extend(collection_of_records[name]['laserROI'])
		all_laser_to_analyse_frequency.extend(collection_of_records[name]['freqlaser'])
		all_laser_to_analyse_voltage.extend(collection_of_records[name]['voltlaser'])
		all_laser_to_analyse_duty.extend(collection_of_records[name]['dutylaser'])
		all_focus_status.extend(collection_of_records[name]['focus_status'])
		all_power_interpolator.extend(collection_of_records[name]['power_interpolator'])
		# all_all_focus_status.extend([collection_of_records[name]['focus_status'][0]]*len(collection_of_records[name]['path_files_laser']))
		all_scan_type.append(collection_of_records[name]['scan_type'])

	minimum_ON_period = minimum_ON_period_list

	all_partial_BBrad = []
	all_partial_BBrad_std = []
	all_partial_diffusion = []
	all_partial_diffusion_std = []
	all_partial_timevariation = []
	all_partial_timevariation_std = []
	all_partial_BBrad_large = []
	all_partial_BBrad_std_large = []
	all_partial_diffusion_large = []
	all_partial_diffusion_std_large = []
	all_partial_timevariation_large = []
	all_partial_timevariation_std_large = []
	all_partial_BBrad_small = []
	all_partial_BBrad_std_small = []
	all_partial_diffusion_small = []
	all_partial_diffusion_std_small = []
	all_partial_timevariation_small = []
	all_partial_timevariation_std_small = []
	all_time_of_experiment = []
	all_laser_framerate = []
	all_laser_to_analyse_frequency_end = []
	all_laser_to_analyse_voltage_end = []
	all_laser_to_analyse_duty_end = []
	all_laser_to_analyse_power_end = []
	all_focus_status_end = []
	all_case_ID_end = []
	all_laser_to_analyse_end = []
	all_laser_location_end = []
	# sharpness_degradation_high_frequency = []

	for index in range(len(all_laser_to_analyse)):
		laser_to_analyse = all_laser_to_analyse[index]
		experimental_laser_frequency = all_laser_to_analyse_frequency[index]
		experimental_laser_voltage = all_laser_to_analyse_voltage[index]
		experimental_laser_duty = all_laser_to_analyse_duty[index]
		power_interpolator = all_power_interpolator[index]
		focus_status = all_focus_status[index]
		case_ID = all_case_ID[index]
		laser_to_analyse = all_laser_to_analyse[index]
		# focus_status = all_all_focus_status[index]

		print('STARTING '+laser_to_analyse)
		if not(os.path.exists(laser_to_analyse+'.npz')):
			print('missing .npz file, aborted')
			continue

		if 1/experimental_laser_frequency*experimental_laser_duty<minimum_ON_period:
			print('skipped for ON period too short')
			continue

		# if focus_status!='focused':
		# 	print('temporary skip because the data is not ready yet')
		# 	continue

		success_read = False
		try:
			laser_dict = np.load(laser_to_analyse+'.npz')
			laser_dict.allow_pickle=True
			temp = laser_dict[type_of_calibration].all()
			# temp = laser_dict['BB_source_w/o_window'].all()
			# temp = laser_dict['BB_source_w_window'].all()
			partial_BBrad_small = temp['partial_BBrad_small']
			partial_BBrad_std_small = temp['partial_BBrad_std_small']
			partial_diffusion_small = temp['partial_diffusion_small']
			partial_diffusion_std_small = temp['partial_diffusion_std_small']
			partial_timevariation_small = temp['partial_timevariation_small']
			partial_timevariation_std_small = temp['partial_timevariation_std_small']
			partial_BBrad_large = temp['partial_BBrad']
			partial_BBrad_std_large = temp['partial_BBrad_std']
			partial_diffusion_large = temp['partial_diffusion']
			partial_diffusion_std_large = temp['partial_diffusion_std']
			partial_timevariation_large = temp['partial_timevariation']
			partial_timevariation_std_large = temp['partial_timevariation_std']
			if type_of_calibration=='NUC_plate':
				partial_timevariation_small /=2
				partial_timevariation_std_small /=2
				partial_timevariation_large /=2
				partial_timevariation_std_large /=2
			partial_timevariation_large *=1.15
			partial_timevariation_std_large *=1.15
			time_of_experiment = temp['time_of_experiment']	# microseconds
			if np.diff(time_of_experiment).max()>np.median(np.diff(time_of_experiment))*1.1:
				hole_pos = np.diff(time_of_experiment).argmax()
				if hole_pos<len(time_of_experiment)/2:
					time_of_experiment = time_of_experiment[hole_pos+1:]
				else:
					time_of_experiment = time_of_experiment[:-(hole_pos+1)]
			laser_framerate = laser_dict['FrameRate']	# Hz
			try:
				laser_location = temp['laser_location']
			except:
				laser_location = laser_dict['laser_location']
			success_read = True
		except Exception as e:
			print('failed reading '+laser_to_analyse+'.npz')
			logging.exception('with error: ' + str(e))

		if success_read==False:
			continue


		frames_for_one_pulse = laser_framerate/experimental_laser_frequency
		if frames_for_one_pulse*experimental_laser_duty<minimum_frames_for_ON_phase:
			print('skipped for laser frequency too high compared to camera framerate')
			continue

		# if experimental_laser_frequency>threshold_freq_list[0]:
		# 	sharpness_degradation_high_frequency.append(2)
		# elif experimental_laser_frequency>threshold_freq_list[1]:
		# 	sharpness_degradation_high_frequency.append(10)
		# else:
		# 	sharpness_degradation_high_frequency.append(1)

		all_partial_BBrad.append(partial_BBrad_small)
		all_partial_BBrad_std.append(partial_BBrad_std_small)
		all_partial_diffusion.append(partial_diffusion_small)
		all_partial_diffusion_std.append(partial_diffusion_std_small)
		all_partial_timevariation.append(partial_timevariation_small)
		all_partial_timevariation_std.append(partial_timevariation_std_small)
		if include_large_area_data:
			all_partial_BBrad.append(partial_BBrad_large)
			all_partial_BBrad_std.append(partial_BBrad_std_large)
			all_partial_diffusion.append(partial_diffusion_large)
			all_partial_diffusion_std.append(partial_diffusion_std_large)
			all_partial_timevariation.append(partial_timevariation_large)
			all_partial_timevariation_std.append(partial_timevariation_std_large)
		all_partial_BBrad_small.append(partial_BBrad_small)
		all_partial_BBrad_std_small.append(partial_BBrad_std_small)
		all_partial_diffusion_small.append(partial_diffusion_small)
		all_partial_diffusion_std_small.append(partial_diffusion_std_small)
		all_partial_timevariation_small.append(partial_timevariation_small)
		all_partial_timevariation_std_small.append(partial_timevariation_std_small)
		all_partial_BBrad_large.append(partial_BBrad_large)
		all_partial_BBrad_std_large.append(partial_BBrad_std_large)
		all_partial_diffusion_large.append(partial_diffusion_large)
		all_partial_diffusion_std_large.append(partial_diffusion_std_large)
		all_partial_timevariation_large.append(partial_timevariation_large)
		all_partial_timevariation_std_large.append(partial_timevariation_std_large)
		all_time_of_experiment.append(time_of_experiment)
		all_laser_framerate.append(laser_framerate)
		all_laser_to_analyse_frequency_end.append(experimental_laser_frequency)
		all_laser_to_analyse_voltage_end.append(experimental_laser_voltage)
		all_laser_to_analyse_duty_end.append(experimental_laser_duty)
		if include_large_area_data:
			all_time_of_experiment.append(time_of_experiment)
			all_laser_framerate.append(laser_framerate)
			all_laser_to_analyse_frequency_end.append(experimental_laser_frequency)
			all_laser_to_analyse_voltage_end.append(experimental_laser_voltage)
			all_laser_to_analyse_duty_end.append(experimental_laser_duty)
		if focus_status=='fully_defocused':
			all_laser_to_analyse_power_end.append(power_interpolator(experimental_laser_voltage)*1)
			if include_large_area_data:
				all_laser_to_analyse_power_end.append(power_interpolator(experimental_laser_voltage)*1)
		else:
			all_laser_to_analyse_power_end.append(power_interpolator(experimental_laser_voltage))
			if include_large_area_data:
				all_laser_to_analyse_power_end.append(power_interpolator(experimental_laser_voltage))
		all_focus_status_end.append(focus_status)
		all_case_ID_end.append(case_ID)
		all_laser_to_analyse_end.append(laser_to_analyse)
		all_laser_location_end.append(laser_location)
		if include_large_area_data:
			all_focus_status_end.append(focus_status)
			all_case_ID_end.append(case_ID)
			all_laser_to_analyse_end.append(laser_to_analyse)
			all_laser_location_end.append(laser_location)

		print('FINISHED '+laser_to_analyse)

	path_where_to_save_everything_int = path_where_to_save_everything+ 'FR' + str(int(all_laser_framerate[0])) + '_' +str(cases_to_include)+'/'
	if not(os.path.exists(path_where_to_save_everything_int)):
		os.makedirs(path_where_to_save_everything_int)


	all_laser_to_analyse_frequency_end = np.array(all_laser_to_analyse_frequency_end)
	all_laser_to_analyse_voltage_end = np.array(all_laser_to_analyse_voltage_end)
	all_laser_to_analyse_duty_end = np.array(all_laser_to_analyse_duty_end)
	all_laser_to_analyse_power_end = np.array(all_laser_to_analyse_power_end)
	all_focus_status_end = np.array(all_focus_status_end)
	all_case_ID_end = np.array(all_case_ID_end)
	all_laser_to_analyse_end = np.array(all_laser_to_analyse_end)
	laser_location = np.nanmean(all_laser_location_end,axis=0)
	# sharpness_degradation_high_frequency = np.array(sharpness_degradation_high_frequency)

	# power_reduction_window = 0.9
	def calculate_laser_power_given_parameters_1(trash,search_emissivity,search_thickness,search_thickness_over_diffusivity,search_defocused_to_focused_power,search_laser_through_mirror_correction):
		search_diffusivity = search_thickness/search_thickness_over_diffusivity
		print([search_emissivity,search_thickness,search_diffusivity,search_defocused_to_focused_power,search_laser_through_mirror_correction])
		# thickness_over_diffusivity = search_thickness/search_diffusivity
		# search_diffusivity = Ptthermaldiffusivity
		all_fitted_power = []
		all_fitted_zero_power = []
		for index in range(len(all_laser_to_analyse_power_end)):
			time_of_experiment = all_time_of_experiment[index]
			partial_BBrad = all_partial_BBrad[index]
			# partial_BBrad_std = all_partial_BBrad_std[index]
			partial_diffusion = all_partial_diffusion[index]
			# partial_diffusion_std = all_partial_diffusion_std[index]
			partial_timevariation = all_partial_timevariation[index]
			# partial_timevariation_std = all_partial_timevariation_std[index]
			laser_framerate = all_laser_framerate[index]/2
			experimental_laser_frequency = all_laser_to_analyse_frequency_end[index]
			experimental_laser_voltage = all_laser_to_analyse_voltage_end[index]
			experimental_laser_duty = all_laser_to_analyse_duty_end[index]
			focus_status = all_focus_status_end[index]
			case_ID = all_case_ID_end[index]
			if focus_status == 'fully_defocused':
				defocused_to_focused_power = search_defocused_to_focused_power
			else:
				defocused_to_focused_power = 1
			# if int(case_ID[-2:]) >=41 and int(case_ID[-2:]) <=47:
			defocused_to_focused_power *= search_laser_through_mirror_correction
			BBrad = partial_BBrad * search_emissivity
			# BBrad_std = partial_BBrad_std * 1
			diffusion = partial_diffusion * search_thickness
			# diffusion_std = partial_diffusion_std * search_thickness
			timevariation = search_thickness_over_diffusivity * partial_timevariation
			# timevariation_std = (1/search_diffusivity)*search_thickness * partial_timevariation_std
			powernoback = diffusion + timevariation + BBrad
			# powernoback = timevariation + BBrad
			# powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5
			frames_for_one_pulse = int(laser_framerate/experimental_laser_frequency)
			time_ON_after_SS = max(1,int(frames_for_one_pulse*experimental_laser_duty - minimum_ON_period*laser_framerate))
			if False:

				temp = max(1,int((len(powernoback)-len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)/2))
				footprint = np.concatenate([np.ones((time_ON_after_SS)),np.zeros((time_ON_after_SS))])
				totalpower_filtered_1 = generic_filter(powernoback,np.mean,footprint=footprint)
				totalpower_filtered_1 = generic_filter(totalpower_filtered_1,np.mean,size=[max(time_ON_after_SS//10,3)])
				footprint = np.concatenate([np.ones((int(frames_for_one_pulse*experimental_laser_duty))),np.zeros((int(frames_for_one_pulse*experimental_laser_duty)))])
				totalpower_filtered_2 = generic_filter(powernoback,np.mean,footprint=footprint)
				peaks_high = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				peaks_high = peaks_high[np.logical_and(peaks_high>frames_for_one_pulse,peaks_high<len(powernoback)-10)].tolist()
				peaks_low = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				peaks_low = peaks_low[np.logical_and(peaks_low>frames_for_one_pulse,peaks_low<len(powernoback)-10)].tolist()
				all_fitted_power.append(np.array([totalpower_filtered_1.max()-totalpower_filtered_1.min(),np.mean(totalpower_filtered_2[peaks_low])+all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power])/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power))
			elif True:
				time_axis = time_of_experiment
				totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty)-2)])
				totalpower_filtered_2 = totalpower_filtered_2_full#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
				time_axis_crop = time_axis#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
				temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
				totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
				time_axis_crop = time_axis_crop[temp:-temp]
				peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
				if len(peaks_loc)==0:
					peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
					if len(peaks_loc)==0:
						peaks_loc = totalpower_filtered_2.argmax()
					else:
						peaks_loc = peaks_loc[0]
				if len(peaks_loc)>=3:
					peaks_loc = peaks_loc[np.logical_and(peaks_loc>peaks_loc.min(),peaks_loc<peaks_loc.max())]
				peaks = totalpower_filtered_2[peaks_loc]
				peak = np.mean(peaks)

				if experimental_laser_duty!=0.5:
					totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
					totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
					time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
					temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
					totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
				throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				throughs_loc = throughs_loc[np.logical_and(throughs_loc>1,throughs_loc<len(totalpower_filtered_2)-1)]
				if len(throughs_loc)==0:
					throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
					if len(throughs_loc)==0:
						throughs_loc = totalpower_filtered_2.argmin()
					else:
						throughs_loc[0]
				if len(throughs_loc)>=3:
					throughs_loc = throughs_loc[np.logical_and(throughs_loc>throughs_loc.min(),throughs_loc<throughs_loc.max())]
				throughs = totalpower_filtered_2[throughs_loc]
				through = np.mean(throughs)
				footprint = np.concatenate([np.ones((time_ON_after_SS)),np.zeros((time_ON_after_SS))])
				# totalpower_filtered_1 = generic_filter(powernoback,np.mean,footprint=footprint)
				# totalpower_filtered_1 = generic_filter(totalpower_filtered_1,np.mean,size=[max(time_ON_after_SS//10,3)])
				# sharpness_indicator = ((np.std(totalpower_filtered_1[totalpower_filtered_1>(peak+through)/2]) + np.std(totalpower_filtered_1[totalpower_filtered_1<(peak+through)/2]) + (all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)))/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)
				# sharpness_indicator = ((np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]) -np.mean(np.diff(powernoback)**2)**0.5 + (peak-through)))/(peak-through)
				sharpness_indicator = (np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]))/(np.mean(np.diff(powernoback)**2)**0.5)

				# all_fitted_power.append((np.array([peak,through+all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power])/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)).tolist()+[sharpness_indicator])
				all_fitted_power.append((np.array([peak,peak-through])/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)).tolist()+[sharpness_indicator])

			# totalpower_filtered_1 = totalpower_filtered_1[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
			# plt.figure()
			# plt.plot(time_of_experiment[1:-1],generic_filter(timevariation,np.mean,size=[10]),'--')
			# plt.plot(time_of_experiment[1:-1],BBrad)
			# search_thickness = 2.5e-6
			# diffusion = partial_diffusion * search_thickness
			# plt.plot(time_of_experiment[1:-1],diffusion)
			# # plt.plot(time_of_experiment[1:-1][int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))],totalpower_filtered_1)
			# plt.plot(time_of_experiment[1:-1],totalpower_filtered_1)
			# plt.plot(time_of_experiment[1:-1],totalpower_filtered_2)
			# plt.plot(time_of_experiment[1:-1][peaks_low+peaks_high],totalpower_filtered_2[peaks_low+peaks_high],'o')
			# plt.pause(0.01)

			# all_fitted_power.append(np.array([totalpower_filtered_1.max()-totalpower_filtered_1.min(),np.mean(totalpower_filtered_2[peaks_low])+all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power])/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power))
			# all_fitted_zero_power.append(totalpower_filtered_1.min())
		all_fitted_power = np.concatenate(all_fitted_power)
		return all_fitted_power

	x = np.arange(len(all_laser_to_analyse_power_end)*3)
	y = np.concatenate([np.ones_like(all_laser_to_analyse_power_end),np.ones_like(all_laser_to_analyse_power_end),np.ones_like(all_laser_to_analyse_power_end)])
	# weigth = np.array([np.ones_like(all_laser_to_analyse_power_end)*1,np.ones_like(all_laser_to_analyse_power_end)*0.1,np.ones_like(all_laser_to_analyse_power_end)*0.001*sharpness_degradation_high_frequency]).T.flatten()
	# sigma = np.ones_like(y)
	# sigma = ((np.abs(all_laser_to_analyse_power_end-all_laser_to_analyse_power_end.max()) + all_laser_to_analyse_power_end.max())/np.array(all_laser_to_analyse_frequency_end))**2
	sigma = []
	for index,value in enumerate(all_laser_to_analyse_power_end):
		case_ID = all_case_ID_end[index]
		max_power = all_laser_to_analyse_power_end[case_ID == np.array(all_case_ID_end)].max()
		temp = (np.abs(value-max_power) + max_power)/max_power
		sigma.append(temp)
		# sigma.append(1)
	sigma = np.array(sigma)# * ((all_laser_to_analyse_frequency_end/(all_laser_to_analyse_frequency_end.min()))**0.1)
	sigma = np.array([sigma,sigma*3,sigma*np.inf]).T.flatten()
	# bds = [[0.7,0.2*2.5e-6,0.2*Ptthermaldiffusivity,0.4,0.8],[1,5*2.5e-6,5*Ptthermaldiffusivity,1.5,1.5]]
	bds = [[0.0,0.2*2.5e-6,0.2*2.5e-6/Ptthermaldiffusivity,0.4,0.99999991],[1,5*2.5e-6,5*2.5e-6/Ptthermaldiffusivity,1.5,1.0000002]]
	# guess=[0.98,2.5e-6,Ptthermaldiffusivity,1]
	# guess=[0.98,2.0e-6,1e-5,0.753,1]
	guess=[0.9,2.0e-6,2.0e-6/1e-5,0.78,1]
	fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/100,xtol=1e-10)
	guess = fit[0]
	fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/10000,xtol=1e-10)
	guess = fit[0]
	fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/1000000,xtol=1e-10)
	# fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=fit[0],bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(fit[0])/10000,xtol=1e-10)
	best_power = calculate_laser_power_given_parameters_1(1,*fit[0])
	guess_best_power = calculate_laser_power_given_parameters_1(1,*guess)
	emissivity_first_stage, thickness_first_stage,thickness_over_diffusivity_first_stage,defocused_to_focused_power_first_stage,laser_through_mirror_correction_first_stage = fit[0]
	diffusivity_first_stage = thickness_first_stage/thickness_over_diffusivity_first_stage
	emissivity_first_stage_sigma, thickness_first_stage_sigma, thickness_over_diffusivity_first_stage_sigma, defocused_to_focused_power_first_stage_sigma, laser_through_mirror_correction_first_stage_sigma = np.diag(fit[1])**0.5
	diffusivity_first_stage_sigma = thickness_first_stage/thickness_over_diffusivity_first_stage *( (thickness_first_stage_sigma/thickness_first_stage)**2 + (thickness_over_diffusivity_first_stage_sigma/thickness_over_diffusivity_first_stage)**2 )**0.5
	# coefficients_first_stage.append(fit[0])
	# all_sharpness_first.append(np.nanmean(best_sharpness))
	total_diffusivity_first_stage_sigma = ((diffusivity_first_stage_sigma/diffusivity_first_stage)**2 + (np.std(foilthickness)/np.mean(foilthickness))**2)**0.5
	total_emissivity_first_stage_sigma = ((emissivity_first_stage_sigma/emissivity_first_stage)**2 + (np.std(foilemissivity)/np.mean(foilemissivity))**2)**0.5
	total_thickness_first_stage_sigma = ((thickness_first_stage_sigma/thickness_first_stage)**2 + (np.std(foilthickness)/np.mean(foilthickness))**2)**0.5


	if cases_to_include==['laser22', 'laser25', 'laser34']:
		diffusivity_first_stage = 9.887773210977352e-06
		thickness_first_stage = 1.9939777848995856e-06
		emissivity_first_stage = 0.9998841926593125
		defocused_to_focused_power_first_stage = 1
		# modified by hand

	if cases_to_include==['laser22', 'laser25', 'laser33', 'laser30', 'laser34', 'laser35']:
		diffusivity_first_stage = 1.0283685197530968e-05
		thickness_first_stage = 2.0531473351462095e-06
		emissivity_first_stage = 0.9999999999999
		defocused_to_focused_power_first_stage = 0.826382122298828
		# this to me is the best I ever got. I'll use this

	if cases_to_include==['laser19', 'laser22', 'laser30', 'laser33']:
		diffusivity_first_stage = 1.0748873883996856e-05
		thickness_first_stage = 2.041275758399619e-06
		emissivity_first_stage = 0.9957284275401461
		defocused_to_focused_power_first_stage = 0.7591477777823928
		# 23/08/2022 used the BB temperature. this seems the better one for now

	if cases_to_include==['laser19', 'laser22', 'laser33']:	# with max frew 10Hz, BB source
		diffusivity_first_stage = 1.0541064256459692e-05
		thickness_first_stage = 1.8212362455836903e-06
		emissivity_first_stage = 0.9999720335475922
		defocused_to_focused_power_first_stage = 0.9023132100212834
		laser_through_mirror_correction_first_stage = 1
		# 02/09/2022 I will need to use this, but for now it is not set yet

	if cases_to_include==['laser19', 'laser22', 'laser33']:	# with max frew 10Hz, NUC poly
		diffusivity_first_stage = 1.0834175364809539e-05
		thickness_first_stage = 2.171960404217933e-06
		emissivity_first_stage = 0.9968454239843706
		defocused_to_focused_power_first_stage = 0.8438514098053702
		laser_through_mirror_correction_first_stage = 1
		# 02/09/2022 I will need to use this, but for now it is not set yet

	if cases_to_include==['laser22', 'laser33']:	# with max frew 10Hz, NUC BB formula, small area
		diffusivity_first_stage = 1.3449081299985357e-05
		thickness_first_stage = 2.740001604038667e-06
		emissivity_first_stage = 0.9999999320220466
		defocused_to_focused_power_first_stage =1.022145803922534
		laser_through_mirror_correction_first_stage = 1
		# 29/09/2022 I use this now
		# 29/09/2022 now i maintained consistency between digitizers from temperature calibration and laser calibration
		diffusivity_first_stage =1.347180363576353e-05
		thickness_first_stage =2.6854735451543735e-06
		emissivity_first_stage =0.9999999999
		defocused_to_focused_power_first_stage =1.0250421275462596
		laser_through_mirror_correction_first_stage =1
		total_diffusivity_first_stage_sigma = 0.13893985595434794
		total_emissivity_first_stage_sigma =0.08610352405403708
		total_thickness_first_stage_sigma =0.1357454922343387




	if include_large_area_data:
		temp=int(len(all_laser_to_analyse_power_end)/2)
	else:
		temp=len(all_laser_to_analyse_power_end)
	for index in np.arange(temp):
		# index=2
		# index=len(all_case_ID_end)-1
		case_ID = all_case_ID_end[index]
		laser_to_analyse = all_laser_to_analyse_end[index]
		partial_BBrad = all_partial_BBrad_small[index]
		partial_BBrad_std = all_partial_BBrad_std_small[index]
		partial_diffusion = all_partial_diffusion_small[index]
		partial_diffusion_std = all_partial_diffusion_std_small[index]
		partial_timevariation = all_partial_timevariation_small[index]
		partial_timevariation_std = all_partial_timevariation_std_small[index]
		if include_large_area_data:
			partial_BBrad_large = all_partial_BBrad_large[index]
			partial_BBrad_std_large = all_partial_BBrad_std_large[index]
			partial_diffusion_large = all_partial_diffusion_large[index]
			partial_diffusion_std_large = all_partial_diffusion_std_large[index]
			partial_timevariation_large = all_partial_timevariation_large[index]
			partial_timevariation_std_large = all_partial_timevariation_std_large[index]
			time_of_experiment = all_time_of_experiment[index*2]
			laser_framerate = all_laser_framerate[index*2]/2
			experimental_laser_frequency = all_laser_to_analyse_frequency_end[index*2]
			experimental_laser_voltage = all_laser_to_analyse_voltage_end[index*2]
			experimental_laser_duty = all_laser_to_analyse_duty_end[index*2]
			laser_to_analyse_power_end = all_laser_to_analyse_power_end[index*2]
			focus_status = all_focus_status_end[index*2]
		else:
			time_of_experiment = all_time_of_experiment[index]
			laser_framerate = all_laser_framerate[index]/2
			experimental_laser_frequency = all_laser_to_analyse_frequency_end[index]
			experimental_laser_voltage = all_laser_to_analyse_voltage_end[index]
			experimental_laser_duty = all_laser_to_analyse_duty_end[index]
			laser_to_analyse_power_end = all_laser_to_analyse_power_end[index]
			focus_status = all_focus_status_end[index]
		# defocused_to_focused_power = 1
		if focus_status=='fully_defocused':
			defocused_to_focused_power = defocused_to_focused_power_first_stage
		else:
			defocused_to_focused_power = 1
		# if int(case_ID[-2:]) >=41 and int(case_ID[-2:]) <=47:
		defocused_to_focused_power *= laser_through_mirror_correction_first_stage
		# power_reduction_window = 0.9
		BBrad = partial_BBrad * emissivity_first_stage
		BBrad_first = partial_BBrad * emissivity_first_stage
		BBrad_std = partial_BBrad_std * emissivity_first_stage
		diffusion_first = partial_diffusion * thickness_first_stage
		diffusion = partial_diffusion * thickness_first_stage
		diffusion_std = partial_diffusion_std * thickness_first_stage
		timevariation = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation
		timevariation_first = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation
		timevariation_std = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation_std
		powernoback = diffusion + timevariation + BBrad
		powernoback_first = diffusion_first + timevariation_first + BBrad_first
		powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5
		if include_large_area_data:
			BBrad_large = partial_BBrad_large * emissivity_first_stage
			diffusion_large = partial_diffusion_large * thickness_first_stage
			timevariation_large = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation_large
			powernoback_large = BBrad_large + diffusion_large + timevariation_large

		frames_for_one_pulse = int(laser_framerate/experimental_laser_frequency)
		time_ON_after_SS = max(1,int(frames_for_one_pulse*experimental_laser_duty - minimum_ON_period*laser_framerate))
		# totalpower_filtered_1 = generic_filter(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])

		# part_powernoback = np.sort(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)])
		# # fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
		# fitted_power_2 = [np.median(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]

		time_axis = time_of_experiment
		totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty)-2)])
		totalpower_filtered_2 = totalpower_filtered_2_full#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		time_axis_crop = time_axis#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
		totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		# time_axis_crop = time_axis_crop[temp:-temp]
		peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
		if len(peaks_loc)==0:
			peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			if len(peaks_loc)==0:
				peaks_loc = totalpower_filtered_2.argmax()
			else:
				peaks_loc = peaks_loc[0]
		if len(peaks_loc)>=3:
			peaks_loc = peaks_loc[np.logical_and(peaks_loc>peaks_loc.min(),peaks_loc<peaks_loc.max())]
		peaks = totalpower_filtered_2[peaks_loc]
		time_peaks = time_axis_crop[peaks_loc+temp]
		peak = np.mean(peaks)
		peaks_std = np.std(peaks)

		if experimental_laser_duty!=0.5:
			totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
			totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
			totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		throughs_loc = throughs_loc[np.logical_and(throughs_loc>1,throughs_loc<len(totalpower_filtered_2)-1)]
		if len(throughs_loc)==0:
			throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			if len(throughs_loc)==0:
				throughs_loc = totalpower_filtered_2.argmin()
			else:
				throughs_loc[0]
		if len(throughs_loc)>=3:
			throughs_loc = throughs_loc[np.logical_and(throughs_loc>throughs_loc.min(),throughs_loc<throughs_loc.max())]
		throughs = totalpower_filtered_2[throughs_loc]
		time_throughs = time_axis_crop[throughs_loc+temp]
		through = np.mean(throughs)
		throughs_std = np.std(throughs)

		footprint = np.concatenate([np.ones((time_ON_after_SS)),np.zeros((time_ON_after_SS))])
		totalpower_filtered_1 = generic_filter(powernoback,np.mean,footprint=footprint)
		totalpower_filtered_1 = generic_filter(totalpower_filtered_1,np.mean,size=[max(time_ON_after_SS//10,3)])
		# sharpness_indicator = ((np.std(totalpower_filtered_1[totalpower_filtered_1>(peak+through)/2]) + np.std(totalpower_filtered_1[totalpower_filtered_1<(peak+through)/2]) + (all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)))/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)
		# sharpness_indicator = ((np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]) -np.mean(np.diff(powernoback)**2)**0.5 + (peak-through)))/(peak-through)
		sharpness_indicator = (np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]))/(np.mean(np.diff(powernoback)**2)**0.5)

		noise_amplitude = 4e-5
		fitted_power_2 = [through,throughs_std,peak,peaks_std]

		if include_large_area_data:
			totalpower_filtered_2_full_large = generic_filter(powernoback_large,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty)-2)])

		if True:	# plots for me to see how things are
			plt.figure(figsize=(20, 10))
			plt.plot(time_axis,powernoback_first,'--',label='totalpower first pass')
			plt.plot(time_axis,totalpower_filtered_1,'--',label='totalpower filtered first pass')
			ax,=plt.plot(time_axis,powernoback,label='totalpower')
			if include_large_area_data:
				plt.plot(time_axis,generic_filter(powernoback_large,np.mean,size=max(int(0.01*frames_for_one_pulse*experimental_laser_duty),2)),'--',label='totalpower_large',color=ax.get_color())
			ax,=plt.plot(time_axis,BBrad,label='totalBBrad')
			if include_large_area_data:
				plt.plot(time_axis,generic_filter(BBrad_large,np.mean,size=max(int(0.01*frames_for_one_pulse*experimental_laser_duty),2)),'--',label='totalBBrad_large',color=ax.get_color())
			ax,=plt.plot(time_axis,diffusion,label='totaldiffusion')
			if include_large_area_data:
				plt.plot(time_axis,generic_filter(diffusion_large,np.mean,size=max(int(0.01*frames_for_one_pulse*experimental_laser_duty),2)),'--',label='totaldiffusion_large',color=ax.get_color())
			ax,=plt.plot(time_axis,timevariation,label='totaltimevariation')
			if include_large_area_data:
				plt.plot(time_axis,generic_filter(timevariation_large,np.mean,size=max(int(0.01*frames_for_one_pulse*experimental_laser_duty),2)),'--',label='totaltimevariation',color=ax.get_color())
			if experimental_laser_duty!=0.5:
				plt.plot(time_axis,totalpower_filtered_1,linewidth=3,label='totalpower filtered ON')
				plt.plot(time_axis,totalpower_filtered_1,linewidth=3,label='totalpower filtered OFF')
				ax,=plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='totalpower super filtered')
				plt.plot(time_axis,totalpower_filtered_2_full_negative,linewidth=3,color=ax.get_color())
			else:
				plt.plot(time_axis,totalpower_filtered_1,linewidth=3,label='totalpower filtered')
				ax,=plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='totalpower super filtered')
			if include_large_area_data:
				plt.plot(time_axis,totalpower_filtered_2_full_large,'--',linewidth=1,label='totalpower super filtered_large',color=ax.get_color())
			plt.plot(time_peaks,peaks,'o',markersize=10,label='peaks')
			plt.plot(time_throughs,throughs,'o',markersize=10,label='through')
			plt.errorbar([time_axis[0],time_axis[-1]],[fitted_power_2[0]]*2,yerr=[fitted_power_2[1]]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
			plt.errorbar([time_axis[0],time_axis[-1]],[fitted_power_2[2]]*2,yerr=[fitted_power_2[3]]*2,color='k',linestyle=':',linewidth=2)
			plt.plot([time_axis[0],time_axis[-1]],[through+noise_amplitude]*2,'--k',label='sharpness limits')
			plt.plot([time_axis[0],time_axis[-1]],[through-noise_amplitude]*2,'--k')
			plt.plot([time_axis[0],time_axis[-1]],[peak-noise_amplitude]*2,'--k')
			plt.plot([time_axis[0],time_axis[-1]],[peak+noise_amplitude]*2,'--k')
			plt.plot([time_axis[0],time_axis[-1]],[laser_to_analyse_power_end*defocused_to_focused_power*power_reduction_window]*2,'--r',label='power input')
			# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[0]]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
			# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[1]]*2,color='k',linestyle=':',linewidth=2)
			plt.legend(loc='best', fontsize='small')
			plt.xlabel('time [s]')
			plt.ylabel('power [W]')
			plt.grid()
			plt.title(laser_to_analyse+' in '+case_ID+'\nInput '+focus_status+', spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' power %.3gW, freq %.3gHz, duty%.3g\nHigh Power=%.3g+/-%.3gW, Low Power=%.3g+/-%.3gW, sharpness=%.3g\ndiffusivity %.3g,thickness %.3g,emissivity %.3g,defocused_to_focused %.3g,laser_through_mirror_correction %.3g' %(laser_to_analyse_power_end*defocused_to_focused_power*power_reduction_window,experimental_laser_frequency,experimental_laser_duty,fitted_power_2[2],fitted_power_2[3],fitted_power_2[0],fitted_power_2[1],sharpness_indicator,diffusivity_first_stage,thickness_first_stage,emissivity_first_stage,defocused_to_focused_power_first_stage,laser_through_mirror_correction_first_stage))
			# plt.pause(0.01)
			figure_index+=1
			plt.savefig(path_where_to_save_everything_int + 'example_' + str(figure_index) +'.eps', bbox_inches='tight')
			plt.close()
		else:	# these are for the paper
			plt.figure(figsize=(13, 8))
			plt.rcParams.update({'font.size': 20})
			# plt.plot(time_axis,powernoback_first,'--',label='totalpower first pass')
			# plt.plot(time_axis,totalpower_filtered_1,'--',label='totalpower filtered first pass')
			plt.plot(time_axis,powernoback*1e3,label='P',alpha=1)
			plt.plot(time_axis,BBrad*1e3,label=r'$P_{BB}$',alpha=0.7)
			plt.plot(time_axis,diffusion*1e3,label=r'$P_{\Delta T}$',alpha=0.7)
			plt.plot(time_axis,timevariation*1e3,label=r'$P_{\frac {\partial T} {\partial t}}$',alpha=0.7)
			# if experimental_laser_duty!=0.5:
			# 	plt.plot(time_axis,totalpower_filtered_1,linewidth=3,label='totalpower filtered ON')
			# 	plt.plot(time_axis,totalpower_filtered_1,linewidth=3,label='totalpower filtered OFF')
			# 	ax,=plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='totalpower filtered')
			# 	plt.plot(time_axis,totalpower_filtered_2_full_negative,linewidth=3,color=ax.get_color())
			# else:
			# 	# plt.plot(time_axis,totalpower_filtered_1,linewidth=3,label='totalpower filtered')
			# 	plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='P filtered')
			plt.plot(time_peaks,peaks*1e3,'o',markersize=10,label=r'$P_{h}$')
			plt.plot(time_throughs,throughs*1e3,'o',markersize=10,label=r'$P_{l}$')
			# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[0]]*2,color='k',linestyle=':',linewidth=2,label='power high/low')
			# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[2]]*2,color='k',linestyle=':',linewidth=2)
			# plt.plot([time_axis[0],time_axis[-1]],[through+noise_amplitude]*2,'--k',label='sharpness limits')
			# plt.plot([time_axis[0],time_axis[-1]],[through-noise_amplitude]*2,'--k')
			# plt.plot([time_axis[0],time_axis[-1]],[peak-noise_amplitude]*2,'--k')
			# plt.plot([time_axis[0],time_axis[-1]],[peak+noise_amplitude]*2,'--k')
			plt.axhline(y=laser_to_analyse_power_end*defocused_to_focused_power*power_reduction_window*1e3,linestyle='--',color='k',label=r'$P_{in}$')
			plt.axhline(y=0,linestyle='--',color='k')
			# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[0]]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
			# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[1]]*2,color='k',linestyle=':',linewidth=2)
			plt.legend(loc='best', fontsize='medium')
			plt.xlabel('time [s]')
			plt.ylabel('power [mW]')
			plt.grid()
			plt.title(laser_to_analyse+' in '+case_ID+'\nInput '+focus_status+', spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' power %.3gW, freq %.3gHz, duty%.3g\nHigh Power=%.3g+/-%.3gW, Low Power=%.3g+/-%.3gW, sharpness=%.3g\ndiffusivity %.3g,thickness %.3g,emissivity %.3g,defocused_to_focused %.3g,laser_through_mirror_correction %.3g' %(laser_to_analyse_power_end*defocused_to_focused_power*power_reduction_window,experimental_laser_frequency,experimental_laser_duty,fitted_power_2[2],fitted_power_2[3],fitted_power_2[0],fitted_power_2[1],sharpness_indicator,diffusivity_first_stage,thickness_first_stage,emissivity_first_stage,defocused_to_focused_power_first_stage,laser_through_mirror_correction_first_stage))
			# plt.pause(0.01)
			figure_index+=1
			plt.savefig(path_where_to_save_everything_int + 'example_for_paper_' + str(figure_index) +'.png', bbox_inches='tight')
			plt.close()



	all_peak = []
	all_peak_std = []
	all_through = []
	all_through_std = []
	all_sharpness_indicator = []
	for index in np.arange(len(all_laser_to_analyse_power_end)):
		# index=2
		# index=len(all_case_ID_end)-1
		case_ID = all_case_ID_end[index]
		laser_to_analyse = all_laser_to_analyse_end[index]
		partial_BBrad = all_partial_BBrad[index]
		partial_BBrad_std = all_partial_BBrad_std[index]
		partial_diffusion = all_partial_diffusion[index]
		partial_diffusion_std = all_partial_diffusion_std[index]
		partial_timevariation = all_partial_timevariation[index]
		partial_timevariation_std = all_partial_timevariation_std[index]
		time_of_experiment = all_time_of_experiment[index]
		laser_framerate = all_laser_framerate[index]/2
		experimental_laser_frequency = all_laser_to_analyse_frequency_end[index]
		experimental_laser_voltage = all_laser_to_analyse_voltage_end[index]
		experimental_laser_duty = all_laser_to_analyse_duty_end[index]
		laser_to_analyse_power_end = all_laser_to_analyse_power_end[index]
		focus_status = all_focus_status_end[index]
		# defocused_to_focused_power = 1
		if focus_status=='fully_defocused':
			defocused_to_focused_power = defocused_to_focused_power_first_stage
		else:
			defocused_to_focused_power = 1
		# if int(case_ID[-2:]) >=41 and int(case_ID[-2:]) <=47:
		defocused_to_focused_power *= laser_through_mirror_correction_first_stage
		# power_reduction_window = 0.9
		BBrad = partial_BBrad * emissivity_first_stage
		BBrad_first = partial_BBrad * emissivity_first_stage
		BBrad_std = partial_BBrad_std * emissivity_first_stage
		diffusion_first = partial_diffusion * thickness_first_stage
		diffusion = partial_diffusion * thickness_first_stage
		diffusion_std = partial_diffusion_std * thickness_first_stage
		timevariation = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation
		timevariation_first = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation
		timevariation_std = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation_std
		powernoback = diffusion + timevariation + BBrad
		powernoback_first = diffusion_first + timevariation_first + BBrad_first
		powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5

		frames_for_one_pulse = int(laser_framerate/experimental_laser_frequency)
		time_ON_after_SS = max(1,int(frames_for_one_pulse*experimental_laser_duty - minimum_ON_period*laser_framerate))
		# totalpower_filtered_1 = generic_filter(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])

		# part_powernoback = np.sort(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)])
		# # fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
		# fitted_power_2 = [np.median(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]

		time_axis = time_of_experiment
		totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty)-2)])
		totalpower_filtered_2 = totalpower_filtered_2_full#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
		totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		time_axis_crop = time_axis_crop[temp:-temp]
		peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
		if len(peaks_loc)==0:
			peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			if len(peaks_loc)==0:
				peaks_loc = totalpower_filtered_2.argmax()
			else:
				peaks_loc = peaks_loc[0]
		if len(peaks_loc)>=3:
			peaks_loc = peaks_loc[np.logical_and(peaks_loc>peaks_loc.min(),peaks_loc<peaks_loc.max())]
		peaks = totalpower_filtered_2[peaks_loc]
		peak = np.mean(peaks)
		peaks_std = np.std(peaks)

		if experimental_laser_duty!=0.5:
			totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
			totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
			totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		throughs_loc = throughs_loc[np.logical_and(throughs_loc>1,throughs_loc<len(totalpower_filtered_2)-1)]
		if len(throughs_loc)==0:
			throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			if len(throughs_loc)==0:
				throughs_loc = totalpower_filtered_2.argmin()
			else:
				throughs_loc[0]
		if len(throughs_loc)>=3:
			throughs_loc = throughs_loc[np.logical_and(throughs_loc>throughs_loc.min(),throughs_loc<throughs_loc.max())]
		throughs = totalpower_filtered_2[throughs_loc]
		through = np.mean(throughs)
		throughs_std = np.std(throughs)

		sharpness_indicator = (np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]))/(np.mean(np.diff(powernoback)**2)**0.5)

		all_peak.append(peak)
		all_peak_std.append(peaks_std)
		all_through.append(through)
		all_through_std.append(throughs_std)
		all_sharpness_indicator.append(sharpness_indicator)
	all_peak = np.array(all_peak)/laser_through_mirror_correction_first_stage
	all_peak_std = np.array(all_peak_std)/laser_through_mirror_correction_first_stage
	all_through = np.array(all_through)/laser_through_mirror_correction_first_stage
	all_through_std = np.array(all_through_std)/laser_through_mirror_correction_first_stage
	all_sharpness_indicator = np.array(all_sharpness_indicator)

	capsize = 5
	plt.rcParams.update({'font.size': 20})
	fig, ax = plt.subplots( 2,1,figsize=(12, 14), squeeze=False,sharex=False)
	select = np.logical_and(all_focus_status_end!='fully_defocused',all_laser_to_analyse_frequency_end==all_laser_to_analyse_frequency_end.min())
	ax[0,0].errorbar(np.sort(all_laser_to_analyse_power_end[select]*power_reduction_window)/(np.pi*(0.001**2)),[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], (all_peak[select]-all_through[select])/(np.pi*(0.001**2))))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], (all_peak_std[select]**2+all_through_std[select]**2)**0.5/(np.pi*(0.001**2))))],fmt='^',color='r',mfc='none',linestyle='--',capsize=capsize)
	# ax[1,0].errorbar(np.sort(all_laser_to_analyse_power_end[select]*power_reduction_window)/(np.pi*(0.001**2)),[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], all_through[select]/(np.pi*(0.001**2))))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], all_through_std[select]/(np.pi*(0.001**2))))],fmt='v',color='r',mfc='none',linestyle='--',capsize=capsize)
	ax[0,0].plot([0,all_laser_to_analyse_power_end.max()*power_reduction_window/(np.pi*(0.001**2))],[0,all_laser_to_analyse_power_end.max()*power_reduction_window/(np.pi*(0.001**2))],'k--')
	# ax[1,0].plot([0,all_laser_to_analyse_power_end.max()*power_reduction_window/(np.pi*(0.001**2))],[0,0],'k--')

	# ax2 = ax[0,0].twinx()  # instantiate a second axes that shares the same x-axis
	# ax2.plot(all_laser_to_analyse_power_end[select]*power_reduction_window,all_sharpness_indicator[select],marker='s',color='r',linestyle="None",mfc='none')
	# ax2.plot([0,all_laser_to_analyse_power_end.max()*power_reduction_window],[1,1],'k--')

	select = np.logical_and(all_focus_status_end=='fully_defocused',all_laser_to_analyse_frequency_end==all_laser_to_analyse_frequency_end.min())
	ax[0,0].errorbar(np.sort(all_laser_to_analyse_power_end[select]*power_reduction_window*defocused_to_focused_power_first_stage)/(np.pi*(0.0026**2)),[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], (all_peak[select]-all_through[select])/(np.pi*(0.0026**2))))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], (all_peak_std[select]**2+all_through_std[select]**2)**0.5/(np.pi*(0.0026**2))))],fmt='^',color='b',mfc='none',linestyle='--',capsize=capsize)
	# ax[1,0].errorbar(np.sort(all_laser_to_analyse_power_end[select]*power_reduction_window*defocused_to_focused_power_first_stage)/(np.pi*(0.0026**2)),[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], all_through[select]/(np.pi*(0.0026**2))))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_power_end[select], all_through_std[select]/(np.pi*(0.0026**2))))],fmt='v',color='b',mfc='none',linestyle='--',capsize=capsize)
	# ax2.plot(all_laser_to_analyse_power_end[select]*power_reduction_window*defocused_to_focused_power_first_stage,all_sharpness_indicator[select],marker='s',color='b',linestyle="None",mfc='none')
	# ax2.set_ylabel('Interested area found [mm2]', color='k')  # we already handled the x-label with ax1
	# ax2.tick_params(axis='y', labelcolor='k')
	# # ax2.set_ylim(bottom=min(ax[0,0].get_ylim()))
	ax[0,0].semilogx()
	ax[0,0].semilogy()
	# ax[1,0].semilogx()

	select = np.logical_and(all_focus_status_end!='fully_defocused',all_laser_to_analyse_voltage_end>=0.5)
	ax[1,0].errorbar(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], ((all_peak-all_through)/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], ((all_peak_std**2+all_through_std**2)**0.5/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],fmt='^',color='r',mfc='none',linestyle='-',capsize=capsize)
	# ax[2,0].errorbar(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through_std/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],fmt='v',color='r',mfc='none',linestyle='-',capsize=capsize)
	# ax[1,0].plot(all_laser_to_analyse_frequency_end[select]*power_reduction_window,all_sharpness_indicator[select],marker='s',color='r',linestyle="None",mfc='none')
	ax[1,0].plot([0,all_laser_to_analyse_frequency_end.max()],[1,1],'k--')
	ax[1,0].plot([0,all_laser_to_analyse_frequency_end.max()],[0,0],'k--')

	select = np.logical_and(all_focus_status_end=='fully_defocused',all_laser_to_analyse_voltage_end>=0.5)
	ax[1,0].errorbar(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], ((all_peak-all_through)/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], ((all_peak_std**2+all_through_std**2)**0.5/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],fmt='^',color='b',mfc='none',linestyle='-',capsize=capsize)
	# ax[2,0].errorbar(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],yerr=[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through_std/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],fmt='v',color='b',mfc='none',linestyle='-',capsize=capsize)
	# ax[1,0].plot(all_laser_to_analyse_frequency_end[select]*power_reduction_window,all_sharpness_indicator[select],marker='s',color='b',linestyle="None",mfc='none')
	ax[0,0].grid()
	ax[0,0].set_xlabel(r'$P_{in}/area \; [W/m^2]$')
	ax[0,0].set_ylabel(r'$(P_{h}-P_{l})/area \; [W/m^2]$')
	# ax[1,0].grid()
	ax[1,0].semilogx()
	ax[1,0].grid()
	ax[1,0].set_xlabel('frequency [Hz]')
	ax[1,0].set_ylabel(r'$(P_{h}-P_{l})/P_{in} [au]$')
	# plt.pause(0.01)
	fig.suptitle('nominal properties\n'+'spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' \ndiffusivity %.3g,thickness %.3g,emissivity %.3g,defocused_to_focused %.3g,laser_through_mirror_correction %.3g' %(diffusivity_first_stage,thickness_first_stage,emissivity_first_stage,defocused_to_focused_power_first_stage,laser_through_mirror_correction_first_stage))

	figure_index+=1
	plt.savefig(path_where_to_save_everything_int + 'example_for_paper_' + str(figure_index) +'.png', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots( 2,1,figsize=(8, 15), squeeze=False,sharex=False)
	capsize = 0
	style = [':','-','--']
	colors1 = ['r','k','r']
	colors1 = ['b','k','b']
	for i_,thickness_first_stage_scan in enumerate([thickness_first_stage*1.2,thickness_first_stage,thickness_first_stage*0.8]):
		all_peak = []
		all_peak_std = []
		all_through = []
		all_through_std = []
		all_sharpness_indicator = []
		for index in np.arange(len(all_laser_to_analyse_power_end)):
			# index=2
			# index=len(all_case_ID_end)-1
			case_ID = all_case_ID_end[index]
			laser_to_analyse = all_laser_to_analyse_end[index]
			partial_BBrad = all_partial_BBrad[index]
			partial_BBrad_std = all_partial_BBrad_std[index]
			partial_diffusion = all_partial_diffusion[index]
			partial_diffusion_std = all_partial_diffusion_std[index]
			partial_timevariation = all_partial_timevariation[index]
			partial_timevariation_std = all_partial_timevariation_std[index]
			time_of_experiment = all_time_of_experiment[index]
			laser_framerate = all_laser_framerate[index]/2
			experimental_laser_frequency = all_laser_to_analyse_frequency_end[index]
			experimental_laser_voltage = all_laser_to_analyse_voltage_end[index]
			experimental_laser_duty = all_laser_to_analyse_duty_end[index]
			laser_to_analyse_power_end = all_laser_to_analyse_power_end[index]
			focus_status = all_focus_status_end[index]
			# defocused_to_focused_power = 1
			if focus_status=='fully_defocused':
				defocused_to_focused_power = defocused_to_focused_power_first_stage
			else:
				defocused_to_focused_power = 1
			if int(case_ID[-2:]) >=41 and int(case_ID[-2:]) <=47:
				defocused_to_focused_power *= laser_through_mirror_correction_first_stage
			# power_reduction_window = 0.9
			BBrad = partial_BBrad * emissivity_first_stage
			BBrad_first = partial_BBrad * emissivity_first_stage
			BBrad_std = partial_BBrad_std * emissivity_first_stage
			diffusion_first = partial_diffusion * thickness_first_stage_scan
			diffusion = partial_diffusion * thickness_first_stage_scan
			diffusion_std = partial_diffusion_std * thickness_first_stage_scan
			timevariation = (1/diffusivity_first_stage)*thickness_first_stage_scan * partial_timevariation
			timevariation_first = (1/Ptthermaldiffusivity)*thickness_first_stage_scan * partial_timevariation
			timevariation_std = (1/diffusivity_first_stage)*thickness_first_stage_scan * partial_timevariation_std
			powernoback = diffusion + timevariation + BBrad
			powernoback_first = diffusion_first + timevariation_first + BBrad_first
			powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5

			frames_for_one_pulse = int(laser_framerate/experimental_laser_frequency)
			time_ON_after_SS = max(1,int(frames_for_one_pulse*experimental_laser_duty - minimum_ON_period*laser_framerate))
			# totalpower_filtered_1 = generic_filter(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])

			# part_powernoback = np.sort(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)])
			# # fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
			# fitted_power_2 = [np.median(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]

			time_axis = time_of_experiment
			totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
			totalpower_filtered_2 = totalpower_filtered_2_full#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
			time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
			temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
			totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
			time_axis_crop = time_axis_crop[temp:-temp]
			peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
			if len(peaks_loc)==0:
				peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				if len(peaks_loc)==0:
					peaks_loc = totalpower_filtered_2.argmax()
				else:
					peaks_loc = peaks_loc[0]
			peaks = totalpower_filtered_2[peaks_loc]
			peak = np.mean(peaks)
			peaks_std = np.std(peaks)

			if experimental_laser_duty!=0.5:
				totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
				totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
				time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
				temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
				totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
			throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			throughs_loc = throughs_loc[np.logical_and(throughs_loc>1,throughs_loc<len(totalpower_filtered_2)-1)]
			if len(throughs_loc)==0:
				throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				if len(throughs_loc)==0:
					throughs_loc = totalpower_filtered_2.argmin()
				else:
					throughs_loc[0]
			throughs = totalpower_filtered_2[throughs_loc]
			through = np.mean(throughs)
			throughs_std = np.std(throughs)

			sharpness_indicator = (np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]))/(np.mean(np.diff(powernoback)**2)**0.5)

			all_peak.append(peak)
			all_peak_std.append(peaks_std)
			all_through.append(through)
			all_through_std.append(throughs_std)
			all_sharpness_indicator.append(sharpness_indicator)
		all_peak = np.array(all_peak)
		all_peak_std = np.array(all_peak_std)
		all_through = np.array(all_through)
		all_through_std = np.array(all_through_std)
		all_sharpness_indicator = np.array(all_sharpness_indicator)

		select = np.logical_and(all_focus_status_end!='fully_defocused',all_laser_to_analyse_power_end==all_laser_to_analyse_power_end.max())
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_peak/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],marker='^',color='r',mfc='none',linestyle=style[i_])
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],marker='v',color='r',mfc='none',linestyle=style[i_])
		# ax[1,0].plot(all_laser_to_analyse_frequency_end[select]*power_reduction_window,all_sharpness_indicator[select],marker='s',color='r',linestyle="None",mfc='none')
		ax[1,0].plot([0,all_laser_to_analyse_frequency_end.max()],[1,1],'k--')
		ax[1,0].plot([0,all_laser_to_analyse_frequency_end.max()],[0,0],'k--')

		select = np.logical_and(all_focus_status_end=='fully_defocused',all_laser_to_analyse_power_end==all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused'].max())
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_peak/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],marker='^',color='b',mfc='none',linestyle=style[i_])
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],marker='v',color='b',mfc='none',linestyle=style[i_])

	ax[0,0].grid()
	ax[1,0].semilogx()
	ax[1,0].grid()
	# plt.pause(0.01)
	fig.suptitle('variation in thickness\n'+'spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' \ndiffusivity %.3g,thickness %.3g,emissivity %.3g,defocused_to_focused %.3g,laser_through_mirror_correction %.3g' %(diffusivity_first_stage,thickness_first_stage,emissivity_first_stage,defocused_to_focused_power_first_stage,laser_through_mirror_correction_first_stage))
	figure_index+=1
	plt.savefig(path_where_to_save_everything_int + 'example_for_paper_' + str(figure_index) +'.eps', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False,sharex=False)
	capsize = 0
	style = [':','-','--']
	for i_,diffusivity_first_stage_scan in enumerate([diffusivity_first_stage*1.2,diffusivity_first_stage,diffusivity_first_stage*0.8]):
		all_peak = []
		all_peak_std = []
		all_through = []
		all_through_std = []
		all_sharpness_indicator = []
		for index in np.arange(len(all_laser_to_analyse_power_end)):
			# index=2
			# index=len(all_case_ID_end)-1
			case_ID = all_case_ID_end[index]
			laser_to_analyse = all_laser_to_analyse_end[index]
			partial_BBrad = all_partial_BBrad[index]
			partial_BBrad_std = all_partial_BBrad_std[index]
			partial_diffusion = all_partial_diffusion[index]
			partial_diffusion_std = all_partial_diffusion_std[index]
			partial_timevariation = all_partial_timevariation[index]
			partial_timevariation_std = all_partial_timevariation_std[index]
			time_of_experiment = all_time_of_experiment[index]
			laser_framerate = all_laser_framerate[index]/2
			experimental_laser_frequency = all_laser_to_analyse_frequency_end[index]
			experimental_laser_voltage = all_laser_to_analyse_voltage_end[index]
			experimental_laser_duty = all_laser_to_analyse_duty_end[index]
			laser_to_analyse_power_end = all_laser_to_analyse_power_end[index]
			focus_status = all_focus_status_end[index]
			# defocused_to_focused_power = 1
			if focus_status=='fully_defocused':
				defocused_to_focused_power = defocused_to_focused_power_first_stage
			else:
				defocused_to_focused_power = 1
			if int(case_ID[-2:]) >=41 and int(case_ID[-2:]) <=47:
				defocused_to_focused_power *= laser_through_mirror_correction_first_stage
			# power_reduction_window = 0.9
			BBrad = partial_BBrad * emissivity_first_stage
			BBrad_first = partial_BBrad * emissivity_first_stage
			BBrad_std = partial_BBrad_std * emissivity_first_stage
			diffusion_first = partial_diffusion * thickness_first_stage
			diffusion = partial_diffusion * thickness_first_stage
			diffusion_std = partial_diffusion_std * thickness_first_stage
			timevariation = (1/diffusivity_first_stage_scan)*thickness_first_stage * partial_timevariation
			timevariation_first = (1/Ptthermaldiffusivity)*thickness_first_stage * partial_timevariation
			timevariation_std = (1/diffusivity_first_stage_scan)*thickness_first_stage * partial_timevariation_std
			powernoback = diffusion + timevariation + BBrad
			powernoback_first = diffusion_first + timevariation_first + BBrad_first
			powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5

			frames_for_one_pulse = int(laser_framerate/experimental_laser_frequency)
			time_ON_after_SS = max(1,int(frames_for_one_pulse*experimental_laser_duty - minimum_ON_period*laser_framerate))
			# totalpower_filtered_1 = generic_filter(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])

			# part_powernoback = np.sort(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)])
			# # fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
			# fitted_power_2 = [np.median(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]

			time_axis = time_of_experiment
			totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
			totalpower_filtered_2 = totalpower_filtered_2_full#[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
			time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
			temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
			totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
			time_axis_crop = time_axis_crop[temp:-temp]
			peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
			if len(peaks_loc)==0:
				peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				if len(peaks_loc)==0:
					peaks_loc = totalpower_filtered_2.argmax()
				else:
					peaks_loc = peaks_loc[0]
			peaks = totalpower_filtered_2[peaks_loc]
			peak = np.mean(peaks)
			peaks_std = np.std(peaks)

			if experimental_laser_duty!=0.5:
				totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
				totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
				time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
				temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
				totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
			throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
			throughs_loc = throughs_loc[np.logical_and(throughs_loc>1,throughs_loc<len(totalpower_filtered_2)-1)]
			if len(throughs_loc)==0:
				throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				if len(throughs_loc)==0:
					throughs_loc = totalpower_filtered_2.argmin()
				else:
					throughs_loc[0]
			throughs = totalpower_filtered_2[throughs_loc]
			through = np.mean(throughs)
			throughs_std = np.std(throughs)

			sharpness_indicator = (np.std(powernoback[powernoback>through+(peak-through)/2]) + np.std(powernoback[powernoback<through+(peak-through)/2]))/(np.mean(np.diff(powernoback)**2)**0.5)

			all_peak.append(peak)
			all_peak_std.append(peaks_std)
			all_through.append(through)
			all_through_std.append(throughs_std)
			all_sharpness_indicator.append(sharpness_indicator)
		all_peak = np.array(all_peak)
		all_peak_std = np.array(all_peak_std)
		all_through = np.array(all_through)
		all_through_std = np.array(all_through_std)
		all_sharpness_indicator = np.array(all_sharpness_indicator)

		select = np.logical_and(all_focus_status_end!='fully_defocused',all_laser_to_analyse_power_end==all_laser_to_analyse_power_end.max())
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_peak/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],marker='^',color='r',mfc='none',linestyle=style[i_])
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through/(all_laser_to_analyse_power_end*power_reduction_window))[select]))],marker='v',color='r',mfc='none',linestyle=style[i_])
		# ax[1,0].plot(all_laser_to_analyse_frequency_end[select]*power_reduction_window,all_sharpness_indicator[select],marker='s',color='r',linestyle="None",mfc='none')
		ax[1,0].plot([0,all_laser_to_analyse_frequency_end.max()],[1,1],'k--')
		ax[1,0].plot([0,all_laser_to_analyse_frequency_end.max()],[0,0],'k--')

		select = np.logical_and(all_focus_status_end=='fully_defocused',all_laser_to_analyse_power_end==all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused'].max())
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_peak/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],marker='^',color='b',mfc='none',linestyle=style[i_])
		ax[1,0].plot(np.sort(all_laser_to_analyse_frequency_end[select]),[y for _, y in sorted(zip(all_laser_to_analyse_frequency_end[select], (all_through/(all_laser_to_analyse_power_end*power_reduction_window*defocused_to_focused_power_first_stage))[select]))],marker='v',color='b',mfc='none',linestyle=style[i_])

	ax[0,0].grid()
	ax[1,0].semilogx()
	ax[1,0].grid()
	# plt.pause(0.01)
	fig.suptitle('variation in diffusivity\n'+'spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' \ndiffusivity %.3g,thickness %.3g,emissivity %.3g,defocused_to_focused %.3g,laser_through_mirror_correction %.3g' %(diffusivity_first_stage,thickness_first_stage,emissivity_first_stage,defocused_to_focused_power_first_stage,laser_through_mirror_correction_first_stage))
	figure_index+=1
	plt.savefig(path_where_to_save_everything_int + 'example_for_paper_' + str(figure_index) +'.eps', bbox_inches='tight')
	plt.close()
