# Created 17/09/2021
# Fabio Federici
# created from Laser_data_analysis3_2.py . I make only one cycle in which I fitt all parameters

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

from scipy.interpolate import interpn

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
					laser_location.append(laser_dict['BB_source_w_window'].all()['laser_location'])
					break
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
f = np.array([[y1,y2,y3,y4,y5,y6,y7,y8] for _,y1,y2,y3,y4,y5,y6,y7,y8 in sorted(zip(np.round(f[:,-1].astype(float)), *f.T))])


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
include_large_area_data = True
# type_of_calibration = 'NUC_plate'
# type_of_calibration = 'BB_source_w/o_window'
type_of_calibration = 'BB_source_w_window'
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

	all_time_axis = []
	all_emissivity_range = []
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
			time_axis = np.arange(len(temp['totalpower_nominal_properties']))
			emissivity_range = np.flip(temp['emissivity_range'],axis=0)
			temp1=[gna[11:] for gna in list(temp.keys()) if (gna[:5]=='emiss' and gna[-5:]!='range')]
			partial_BBrad_small = []
			partial_BBrad_std_small = []
			partial_diffusion_small = []
			partial_diffusion_std_small = []
			partial_timevariation_small = []
			partial_timevariation_std_small = []
			partial_BBrad_large = []
			partial_BBrad_std_large = []
			partial_diffusion_large = []
			partial_diffusion_std_large = []
			partial_timevariation_large = []
			partial_timevariation_std_large = []
			for emissivity in emissivity_range:
				temp2 = temp['emissivity='+temp1[np.abs(np.array(temp1).astype(float)-emissivity).argmin()]]
				partial_BBrad_small.append(temp2['partial_BBrad_small'])
				partial_BBrad_std_small.append(temp2['partial_BBrad_std_small'])
				partial_diffusion_small.append(temp2['partial_diffusion_small'])
				partial_diffusion_std_small.append(temp2['partial_diffusion_std_small'])
				partial_timevariation_small.append(temp2['partial_timevariation_small'])
				partial_timevariation_std_small.append(temp2['partial_timevariation_std_small'])
				partial_BBrad_large.append(temp2['partial_BBrad'])
				partial_BBrad_std_large.append(temp2['partial_BBrad_std'])
				partial_diffusion_large.append(temp2['partial_diffusion'])
				partial_diffusion_std_large.append(temp2['partial_diffusion_std'])
				partial_timevariation_large.append(temp2['partial_timevariation'])
				partial_timevariation_std_large.append(temp2['partial_timevariation_std'])
			if type_of_calibration=='NUC_plate':
				partial_timevariation_small /=2
				partial_timevariation_std_small /=2
				partial_timevariation_large /=2
				partial_timevariation_std_large /=2

			# partial_BBrad_small = interp2d(time_axis,emissivity_range,partial_BBrad_small)
			# partial_BBrad_std_small = interp2d(time_axis,emissivity_range,partial_BBrad_std_small)
			# partial_diffusion_small = interp2d(time_axis,emissivity_range,partial_diffusion_small)
			# partial_diffusion_std_small = interp2d(time_axis,emissivity_range,partial_diffusion_std_small)
			# partial_timevariation_small = interp2d(time_axis,emissivity_range,partial_timevariation_small)
			# partial_timevariation_std_small = interp2d(time_axis,emissivity_range,partial_timevariation_std_small)
			# partial_BBrad_large = interp2d(time_axis,emissivity_range,partial_BBrad_large)
			# partial_BBrad_std_large = interp2d(time_axis,emissivity_range,partial_BBrad_std_large)
			# partial_diffusion_large = interp2d(time_axis,emissivity_range,partial_diffusion_large)
			# partial_diffusion_std_large = interp2d(time_axis,emissivity_range,partial_diffusion_std_large)
			# partial_timevariation_large = interp2d(time_axis,emissivity_range,partial_timevariation_large)
			# partial_timevariation_std_large = interp2d(time_axis,emissivity_range,partial_timevariation_std_large)

			# partial_timevariation_large *=1.15	# 2023-06-26 no idea now why I did this correction, removed
			# partial_timevariation_std_large *=1.15
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

		all_emissivity_range.append(emissivity_range)
		all_time_axis.append(time_axis)
		all_partial_BBrad.append(partial_BBrad_small)
		all_partial_BBrad_std.append(partial_BBrad_std_small)
		all_partial_diffusion.append(partial_diffusion_small)
		all_partial_diffusion_std.append(partial_diffusion_std_small)
		all_partial_timevariation.append(partial_timevariation_small)
		all_partial_timevariation_std.append(partial_timevariation_std_small)
		if include_large_area_data:
			all_emissivity_range.append(emissivity_range)
			all_time_axis.append(time_axis)
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
			# print(index)
			time_of_experiment = all_time_of_experiment[index]
			emissivity_range = all_emissivity_range[index]
			time_axis = all_time_axis[index]
			partial_BBrad = interpn((emissivity_range,time_axis),all_partial_BBrad[index],np.array([[search_emissivity]*len(time_axis),time_axis]).T)
			# partial_BBrad = all_partial_BBrad[index]
			# partial_BBrad_std = all_partial_BBrad_std[index]
			partial_diffusion = interpn((emissivity_range,time_axis),all_partial_diffusion[index],np.array([[search_emissivity]*len(time_axis),time_axis]).T)
			# partial_diffusion = all_partial_diffusion[index]
			# partial_diffusion_std = all_partial_diffusion_std[index]
			partial_timevariation = interpn((emissivity_range,time_axis),all_partial_timevariation[index],np.array([[search_emissivity]*len(time_axis),time_axis]).T)
			# partial_timevariation = all_partial_timevariation[index]
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
	bds = [[0.6,0.2*2.5e-6,0.2*2.5e-6/Ptthermaldiffusivity,0.4,0.99999991],[1.2,5*2.5e-6,5*2.5e-6/Ptthermaldiffusivity,1.5,1.0000002]]
	# guess=[0.98,2.5e-6,Ptthermaldiffusivity,1]
	# guess=[0.98,2.0e-6,1e-5,0.753,1]
	guess=[0.8,2.0e-6,2.0e-6/1e-5,0.78,1]
	# fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/100,xtol=1e-10)
	# guess = fit[0]
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
		# partial_BBrad = all_partial_BBrad_small[index]
		# partial_BBrad_std = all_partial_BBrad_std_small[index]
		# partial_diffusion = all_partial_diffusion_small[index]
		# partial_diffusion_std = all_partial_diffusion_std_small[index]
		# partial_timevariation = all_partial_timevariation_small[index]
		# partial_timevariation_std = all_partial_timevariation_std_small[index]
		if include_large_area_data:
			emissivity_range = all_emissivity_range[index*2]
			time_axis = all_time_axis[index*2]
		else:
			emissivity_range = all_emissivity_range[index]
			time_axis = all_time_axis[index]
		partial_BBrad = interpn((emissivity_range,time_axis),all_partial_BBrad_small[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
		partial_BBrad_std = interpn((emissivity_range,time_axis),all_partial_BBrad_std_small[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
		partial_diffusion = interpn((emissivity_range,time_axis),all_partial_diffusion_small[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
		partial_diffusion_std = interpn((emissivity_range,time_axis),all_partial_diffusion_std_small[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
		partial_timevariation = interpn((emissivity_range,time_axis),all_partial_timevariation_small[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
		partial_timevariation_std = interpn((emissivity_range,time_axis),all_partial_timevariation_std_small[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
		if include_large_area_data:
			# partial_BBrad_large = all_partial_BBrad_large[index]
			# partial_BBrad_std_large = all_partial_BBrad_std_large[index]
			# partial_diffusion_large = all_partial_diffusion_large[index]
			# partial_diffusion_std_large = all_partial_diffusion_std_large[index]
			# partial_timevariation_large = all_partial_timevariation_large[index]
			# partial_timevariation_std_large = all_partial_timevariation_std_large[index]
			partial_BBrad_large = interpn((emissivity_range,time_axis),all_partial_BBrad_large[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
			partial_BBrad_std_large = interpn((emissivity_range,time_axis),all_partial_BBrad_std_large[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
			partial_diffusion_large = interpn((emissivity_range,time_axis),all_partial_diffusion_large[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
			partial_diffusion_std_large = interpn((emissivity_range,time_axis),all_partial_diffusion_std_large[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
			partial_timevariation_large = interpn((emissivity_range,time_axis),all_partial_timevariation_large[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
			partial_timevariation_std_large = interpn((emissivity_range,time_axis),all_partial_timevariation_std_large[index],np.array([[emissivity_first_stage]*len(time_axis),time_axis]).T)
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








# 2023-07-16 new thing.
# I want to use as much info I can take from a single laser experiment for findint the foil properties

index = 4
laser_to_analyse = all_laser_to_analyse[index]
experimental_laser_frequency = all_laser_to_analyse_frequency[index]
experimental_laser_voltage = all_laser_to_analyse_voltage[index]
experimental_laser_duty = all_laser_to_analyse_duty[index]
power_interpolator = all_power_interpolator[index]
focus_status = all_focus_status[index]
case_ID = all_case_ID[index]
laser_to_analyse = all_laser_to_analyse[index]
laser_to_analyse_power = power_interpolator(experimental_laser_voltage)
laser_to_analyse_power = laser_to_analyse_power * power_reduction_window


# laser_dict = coleval.read_IR_file(laser_to_analyse)
laser_dict = np.load(laser_to_analyse+'.npz')
laser_dict.allow_pickle=True
# laser_counts_filtered = laser_dict['laser_counts_filtered']
full_saved_file_dict = dict(laser_dict)
type_of_calibration = 'BB_source_w_window'

try:
	full_saved_file_dict[type_of_calibration] = full_saved_file_dict[type_of_calibration].all()
except:
	pass

aggregated_emissivity_range = full_saved_file_dict[type_of_calibration]['emissivity_range']
reference_temperature_range = full_saved_file_dict[type_of_calibration]['reference_temperature_range']
time_partial = full_saved_file_dict[type_of_calibration]['time_of_experiment']

if False:	# these are only the plots for the paper

	i_emissivity,aggregated_emissivity = list(enumerate(aggregated_emissivity_range))[8]
	try:
		full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
	except:
		pass
	nuc_plate_emissivity = 1.0
	emissivity = aggregated_emissivity * nuc_plate_emissivity
	i_reference_temperature,reference_temperature = list(enumerate(reference_temperature_range))[5]

	try:
		full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
	except:
		pass

	laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']
	framerate = 1/np.median(np.diff(time_partial))
	frames_for_one_pulse = framerate/experimental_laser_frequency
	partial_timevariation = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
	partial_timevariation_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
	partial_timevariation_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
	partial_timevariation_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
	partial_BBrad = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
	partial_BBrad_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
	partial_BBrad_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
	partial_BBrad_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
	partial_diffusion = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
	partial_diffusion_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']


	diffusivity = 9.466307489300497e-06
	thickness_over_diffusivity = 0.07502143143067985	# this is the real measurement I obtain
	thickness = thickness_over_diffusivity*diffusivity	# m
	emissivity = 0.6311559794933064


	sigmaSB=5.6704e-08 #[W/(m2 K4)]
	dx=0.09/240
	timevariation = partial_timevariation * thicknesse-6/Ptthermaldiffusivity
	timevariation_small = partial_timevariation_small * thicknesse-6/Ptthermaldiffusivity
	BB = partial_BBrad *emissivity
	BB_small = partial_BBrad_small *emissivity
	diff = partial_diffusion * thickness_over_diffusivity * Ptthermaldiffusivity
	diff_small = partial_diffusion_small * thickness_over_diffusivity * Ptthermaldiffusivity
	timevariation_and_BB = timevariation + BB
	timevariation_and_BB_small = timevariation_small + BB_small
	timevariation_and_BB_and_diff = timevariation + BB + diff
	BB_and_diff_small = BB_small + diff_small
	BB_and_diff = BB + diff

	# I would prefer not to, but I really need to do this, otherwise I can't see anything with the known noise
	timevariation_filtered = coleval.butter_lowpass_filter(timevariation,15,383/2,3)
	timevariation_small_filtered = coleval.butter_lowpass_filter(timevariation_small,15,383/2,3)


	plt.figure(figsize=(10, 5))
	start = time_partial[569-7]
	end = time_partial[300+748-7]
	plt.plot([0,start,start,end,end,10],[0,0,laser_to_analyse_power,laser_to_analyse_power,0,0],'--r')
	a1, = plt.plot(time_partial,timevariation_small_filtered,label=r'$\delta T/\delta t$')
	plt.plot(time_partial,timevariation_filtered,a1.get_color(),linestyle='--')
	a3, = plt.plot(time_partial,diff_small,label=r'$\Delta T$')
	plt.plot(time_partial,diff,a3.get_color(),linestyle='--')
	a2, = plt.plot(time_partial,BB_small,label=r'$BB$')
	plt.plot(time_partial,BB,a2.get_color(),linestyle='--')
	plt.xlabel('time [s]')
	plt.ylabel('Integrated power [W]')
	plt.grid()
	plt.xlim(left=2.5,right=7.5)
	plt.legend(loc='best')
	# plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG_for_paper_2'+'.eps', bbox_inches='tight')
	plt.savefig(path_where_to_save_everything + 'FIG_for_paper_3'  +'.eps', bbox_inches='tight')



if False:	# here I try to estimate thickness_over_diffusivity from an analytic formula on the peak temperature, but I don't seems to have enough freedom to do the rest
	# reference_temperature_range = reference_temperature_range[2:]
	aggregated_emissivity_range = aggregated_emissivity_range[:-2]


	nuc_plate_emissivity = 1.0
	reference_temperature_correction = -0.0

	plt.figure()
	linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']
	collect_peak_heating = []
	collect_peak_dT_dt = []
	collect_aggregated_emissivity = []
	collect_reference_temperature = []
	collect_thickness_over_diffusivity = []
	for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
	# for i_emissivity,aggregated_emissivity in enumerate([aggregated_emissivity_range[2]]):
	# for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range[::2]):
		# aggregated_emissivity = aggregated_emissivity_range[5]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
		except:
			pass

		emissivity = aggregated_emissivity * nuc_plate_emissivity

		for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range):
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[2]]):
		# for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range[::3]):
		# reference_temperature = reference_temperature_range[0]
			try:
				full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
			except:
				pass

			try:
				laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']
			except:
				continue
			framerate = 1/np.median(np.diff(time_partial))
			frames_for_one_pulse = framerate/experimental_laser_frequency
			temp = generic_filter(laser_temperature_minus_background_crop_max,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty*0.01))])
			temp = np.diff(temp)
			from scipy.signal import find_peaks, peak_prominences as get_proms
			start_loc = find_peaks(temp,distance=frames_for_one_pulse*0.95)[0]
			end_loc = find_peaks(-temp,distance=frames_for_one_pulse*0.95)[0]

			peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
			start_loc-=2
			frames_for_one_pulse_ON = [frames_for_one_pulse*experimental_laser_duty//2*2]
			for i in range(len(start_loc)):
				try:
					frames_for_one_pulse_ON.append(end_loc[i]-start_loc[i])
				except:
					pass
			frames_for_one_pulse_ON = np.min(frames_for_one_pulse_ON).astype(int)

			heated_phase = np.zeros((frames_for_one_pulse_ON))
			for loc in start_loc:
				loc -= max(0,int(frames_for_one_pulse*experimental_laser_duty*0.00/2))
				heated_phase += laser_temperature_minus_background_crop_max[loc:loc+frames_for_one_pulse_ON]
			heated_phase /= len(start_loc)

			heated_phase = heated_phase[max(0,int(frames_for_one_pulse*experimental_laser_duty*0.00)):-int(frames_for_one_pulse*experimental_laser_duty*0.05)]

			# eta is supposed to be constant, but BB depends on temperature, so I fit only below a certain treshold, let's say until eta is 5% wrong
			real_eta_component = 4*((273.15+reference_temperature+reference_temperature_correction)**3) + 6*((273.15+reference_temperature+reference_temperature_correction)**2)*heated_phase + 4*((273.15+reference_temperature+reference_temperature_correction)**1)*(heated_phase**2) + (heated_phase**3)
			partial_eta_component = 4*((273.15+reference_temperature+reference_temperature_correction)**3)
			# upper_treshold = max(min(len(heated_phase-int(frames_for_one_pulse*experimental_laser_duty*0.05)) , np.abs(real_eta_component/partial_eta_component-1.1).argmin()),40)
			sigma = np.abs(real_eta_component/partial_eta_component)

			time = np.arange(len(heated_phase))/framerate
			from scipy.special import expi
			def func_(t,t0,t_mult,csi,max):
				# print(t0,t_mult,csi,max)
				return max*(expi(-csi*(1+4*t_mult*(t-t0))) - expi(-csi))
			bds = [[-np.inf,0.,0.,0.],[0.,np.inf,np.inf,np.inf]]
			guess = [0.,1,0.1,1.]
			fit = curve_fit(func_, time,heated_phase, sigma=sigma, p0=guess, bounds = bds, maxfev=100000000)
			# plt.figure()
			# plt.plot(time,heated_phase)
			# plt.plot(np.arange(len(time)*10)*np.median(np.diff(time)),func_(np.arange(len(time)*10)*np.median(np.diff(time)),*fit[0]),'--')
			# plt.xlabel('time [s]')
			# plt.ylabel(r'$\Delta T$'+' [K]')
			# plt.xlim(left=-0.1,right=2.6)
			# # plt.plot(np.arange(len(temp))/(framerate/2),temp)
			# # # plt.plot(np.arange(len(temp))/(framerate/2),func_(np.arange(len(temp))/(framerate/2),*guess),':')
			# # plt.plot(np.arange(len(temp))/(framerate/2),func_(np.arange(len(temp))/(framerate/2),*fit[0]),'--')
			# plt.pause(0.1)
			# print(fit[0])
			# Ptthermaldiffusivity=Ptthermalconductivity/(Ptspecificheat*Ptdensity)    #m2/s
			w0 = (Ptthermaldiffusivity/fit[0][1])**0.5
			hs_star = 2.5E-6/w0
			eta = 1*emissivity*5.67E-8*4*((273.15+reference_temperature+reference_temperature_correction)**3)	# this is calculated wia a taylor expansion of the BB radiation term, using only the stronger dependency. for dT<10 the error is <6%
			eta_error = 1*emissivity*5.67E-8*(6*((273.15+reference_temperature+reference_temperature_correction)**2)*(heated_phase.max()) + 4*(273+reference_temperature+reference_temperature_correction)*((heated_phase.max())**2) + (heated_phase.max())**3)
			eta_star = eta*w0/Ptthermalconductivity
			csi = eta_star/(2*hs_star)
			# print([emissivity,reference_temperature])
			# print([csi,fit[0][2],csi/fit[0][2]])
			hs = eta/(2*Ptspecificheat*Ptdensity) * 1/fit[0][1] * 1/fit[0][2]
			print(hs)

			#fit = t0,t_mult,csi,max

			from uncertainties import correlated_values,ufloat
			from uncertainties.unumpy import nominal_values,std_devs,uarray
			fit_ = correlated_values(fit[0],fit[1])
			thickness_over_diffusivity = ufloat(eta,eta_error)/(2*Ptthermalconductivity) * 1/(fit_[1] *fit_[2])
			print(thickness_over_diffusivity)

			partial_timevariation = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
			partial_timevariation_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
			partial_timevariation_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
			partial_timevariation_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
			partial_BBrad = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
			partial_BBrad_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
			partial_BBrad_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
			partial_BBrad_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
			partial_diffusion = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
			partial_diffusion_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

			sigmaSB=5.6704e-08 #[W/(m2 K4)]
			dx=0.09/240
			timevariation = partial_timevariation * nominal_values(thickness_over_diffusivity)
			timevariation_small = partial_timevariation_small * nominal_values(thickness_over_diffusivity)
			BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
			BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
			diff = partial_diffusion * nominal_values(thickness_over_diffusivity) * Ptthermaldiffusivity
			diff_small = partial_diffusion_small * nominal_values(thickness_over_diffusivity) * Ptthermaldiffusivity
			timevariation_and_BB = timevariation + BB
			timevariation_and_BB_small = timevariation_small + BB_small
			timevariation_and_BB_and_diff = timevariation + BB + diff
			BB_and_diff_small = BB_small + diff_small
			BB_and_diff = BB + diff

			timevariation_filtered = generic_filter(timevariation,np.mean,size=[7])
			# timevariation_small_filtered = generic_filter(timevariation_small,np.mean,size=[7])
			timevariation_small_filtered = coleval.butter_lowpass_filter(timevariation_small,15,383/2,3)
			# timevariation_and_BB_small_filtered = generic_filter(timevariation_and_BB_small,np.mean,size=[7])
			timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)
			timevariation_and_BB_filtered = generic_filter(timevariation_and_BB,np.mean,size=[7])
			# timevariation_and_BB_filtered = coleval.butter_lowpass_filter(timevariation_and_BB,15,383/2,3)
			BB_and_diff_filtered = generic_filter(BB_and_diff,np.mean,size=[7])
			BB_filtered = generic_filter(BB,np.mean,size=[7])
			BB_small_filtered = generic_filter(BB_small,np.mean,size=[7])
			temp = find_peaks(timevariation_and_BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
			peak_heating = np.median(timevariation_and_BB_filtered[temp])
			peak_dT_dt = np.median(partial_timevariation_small[temp])
			# peak_heating = np.median(timevariation_and_BB_filtered[temp])

			# coleval.find_reliable_peaks_and_throughs(generic_filter(timevariation,np.mean,size=[7]),time_partial,experimental_laser_frequency,experimental_laser_duty)
			collect_peak_heating.append(peak_heating)
			collect_peak_dT_dt.append(peak_dT_dt)
			collect_aggregated_emissivity.append(aggregated_emissivity)
			collect_reference_temperature.append(reference_temperature)
			collect_thickness_over_diffusivity.append(nominal_values(thickness_over_diffusivity))

			# plt.plot(generic_filter(timevariation,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			plt.plot(timevariation_and_BB_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.plot(generic_filter(timevariation_and_BB_and_diff,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
			# plt.plot(timevariation_and_BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.plot(timevariation_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
			# plt.plot(BB_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
			# plt.plot(BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.plot(laser_temperature_minus_background_crop_max,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.plot(diff,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
			# plt.plot(diff_small,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
			# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
			# plt.plot(generic_filter(partial_diffusion*nominal_values(thickness_over_diffusivity)*Ptthermaldiffusivity,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.plot(generic_filter(laser_temperature_minus_background_crop_max,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			# plt.figure()
			# a1, = plt.plot(time_partial,timevariation_small_filtered,label=r'$\delta T/\delta t$')
			# plt.plot(time_partial,timevariation_filtered,a1.get_color(),linestyle='--')
			# a3, = plt.plot(time_partial,diff_small,label=r'$\nabla T$')
			# plt.plot(time_partial,diff,a3.get_color(),linestyle='--')
			# a2, = plt.plot(time_partial,BB_small_filtered,label=r'$BB$')
			# plt.plot(time_partial,BB_filtered,a2.get_color(),linestyle='--')
			# plt.xlabel('time [s]')
			# plt.ylabel('Power [W]')
			# plt.grid()
			# plt.xlim(left=2.5,right=8.5)
			# plt.legend(loc='best')
			# plt.figure()
			# a1, = plt.plot(time_partial,timevariation_small_filtered+diff_small+BB_small_filtered)
			# plt.plot(time_partial,timevariation_filtered+diff+BB_filtered,linestyle='--')
			# plt.xlabel('time [s]')
			# plt.ylabel('Power [W]')
			# plt.grid()
			# plt.xlim(left=2.5,right=8.5)




	collect_peak_heating = np.array(collect_peak_heating)
	collect_peak_dT_dt = np.array(collect_peak_dT_dt)
	collect_aggregated_emissivity = np.array(collect_aggregated_emissivity)
	collect_reference_temperature = np.array(collect_reference_temperature)
	collect_thickness_over_diffusivity = np.array(collect_thickness_over_diffusivity)
	plt.legend(loc='best', fontsize='xx-small')
	plt.axhline(y=laser_to_analyse_power,color='k')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity))
	plt.close()

	# plt.figure()
	# plt.scatter(collect_emissivity,collect_reference_temperature,c=collect_peak_heating)
	# plt.colorbar()


	# plt.figure()
	# inferred_aggregated_emissivity = []
	# for i_reference_temperature,reference_temperature in enumerate(np.unique(collect_reference_temperature)):
	# 	aggregated_emissivity = collect_aggregated_emissivity[collect_reference_temperature==reference_temperature]
	# 	peak_heating = collect_peak_heating[collect_reference_temperature==reference_temperature]
	# 	# if laser_to_analyse_power>peak_heating.max() or laser_to_analyse_power<peak_heating.min():
	# 	# 	fit = np.polyfit(peak_heating,aggregated_emissivity,1)
	# 	# else:
	# 	fit = np.polyfit(aggregated_emissivity,peak_heating,4)
	# 	fit[-1] -=laser_to_analyse_power
	# 	sols = np.roots(fit)
	# 	fit[-1] +=laser_to_analyse_power
	# 	if np.sum(np.isreal(sols)) == 0:
	# 		inferred_aggregated_emissivity.append(np.nan)
	# 	else:
	# 		inferred_aggregated_emissivity.append(sols[np.abs(sols-1).argmin()])
	# 	plt.plot(aggregated_emissivity,peak_heating,color='C'+str(i_reference_temperature),label='ref_temp=%.3g' %(reference_temperature))
	# 	plt.plot(np.linspace(0.3,3,20),np.polyval(fit,np.linspace(0.3,3,20)),'--',color='C'+str(i_reference_temperature))
	# 	plt.axhline(y=laser_to_analyse_power,color='C'+str(i_reference_temperature))
	# 	plt.axvline(x=inferred_aggregated_emissivity[-1],color='C'+str(i_reference_temperature))
	# inferred_aggregated_emissivity = np.array(inferred_aggregated_emissivity)
	# inferred_aggregated_emissivity[inferred_aggregated_emissivity<0] = np.nan
	# plt.legend(loc='best', fontsize='xx-small')
	# plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity))

	plt.figure()
	inferred_reference_temperature = []
	for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
		emissivity = aggregated_emissivity*nuc_plate_emissivity	# this is equivalent to setting the NUC plate emissivity to 0.75
		reference_temperature = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
		peak_heating = collect_peak_heating[collect_aggregated_emissivity==aggregated_emissivity]
		# if laser_to_analyse_power>peak_heating.max() or laser_to_analyse_power<peak_heating.min():
		# 	fit = np.polyfit(peak_heating,aggregated_emissivity,1)
		# else:
		fit = np.polyfit(reference_temperature,peak_heating,2)
		fit[-1] -=laser_to_analyse_power
		sols = np.roots(fit)
		fit[-1] +=laser_to_analyse_power
		if np.sum(np.isreal(sols)) == 0:
			inferred_reference_temperature.append(np.nan)
		else:
			# inferred_reference_temperature.append(sols[np.abs(sols-np.mean(reference_temperature)).argmin()])
			inferred_reference_temperature.append(min(sols))
		plt.plot(reference_temperature,peak_heating,color='C'+str(i_emissivity),label='emissivity=%.3g' %(emissivity))
		plt.plot(np.linspace(15,35,20),np.polyval(fit,np.linspace(15,35,20)),'--',color='C'+str(i_emissivity))
		plt.axhline(y=laser_to_analyse_power,color='C'+str(i_emissivity))
		plt.axvline(x=inferred_reference_temperature[-1],color='C'+str(i_emissivity))
	inferred_reference_temperature = np.array(inferred_reference_temperature)
	inferred_reference_temperature[inferred_reference_temperature<0] = np.nan
	plt.legend(loc='best', fontsize='xx-small')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity))
	plt.close()

	plt.figure()
	for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
		emissivity = aggregated_emissivity*nuc_plate_emissivity	# this is equivalent to setting the NUC plate emissivity to 0.75
		reference_temperature = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
		thickness_over_diffusivity = collect_thickness_over_diffusivity[collect_aggregated_emissivity==aggregated_emissivity]
		plt.plot(reference_temperature,thickness_over_diffusivity,color='C'+str(i_emissivity),label='emissivity=%.3g' %(emissivity))
	plt.legend(loc='best', fontsize='xx-small')
	plt.ylabel('thickness_over_diffusivity')
	plt.xlabel('reference_temperature')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity))
	plt.close()

	plt.figure()
	for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
		emissivity = aggregated_emissivity*nuc_plate_emissivity	# this is equivalent to setting the NUC plate emissivity to 0.75
		reference_temperature = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
		peak_dT_dt = collect_peak_dT_dt[collect_aggregated_emissivity==aggregated_emissivity]
		plt.plot(reference_temperature,peak_dT_dt,color='C'+str(i_emissivity),label='emissivity=%.3g' %(emissivity))
	plt.legend(loc='best', fontsize='xx-small')
	plt.ylabel('peak_dT_dt')
	plt.xlabel('reference_temperature')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity))
	plt.close()

	plt.figure()
	# plt.scatter(collect_aggregated_emissivity,collect_reference_temperature,c=collect_peak_heating)
	plt.scatter(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature,c=collect_thickness_over_diffusivity,cmap='rainbow')
	plt.plot(aggregated_emissivity_range*nuc_plate_emissivity,inferred_reference_temperature)
	# collect_reference_temperature_fit = np.polyfit(inferred_aggregated_emissivity[np.isfinite(inferred_aggregated_emissivity)],np.unique(collect_reference_temperature)[np.isfinite(inferred_aggregated_emissivity)],2)
	collect_reference_temperature_fit = np.polyfit(aggregated_emissivity_range[np.isfinite(inferred_reference_temperature)],inferred_reference_temperature[np.isfinite(inferred_reference_temperature)],3)
	plt.plot(np.unique(collect_aggregated_emissivity)*nuc_plate_emissivity,np.polyval(collect_reference_temperature_fit,np.unique(collect_aggregated_emissivity)),'--')
	# thickness_over_diffusivity_interpolator = interp2d(collect_aggregated_emissivity,collect_reference_temperature,collect_thickness_over_diffusivity)
	thickness_over_diffusivity_interpolator = RectBivariateSpline(np.unique(collect_aggregated_emissivity),np.unique(collect_reference_temperature),np.flip(collect_thickness_over_diffusivity.reshape((len(np.unique(collect_aggregated_emissivity)),len(np.unique(collect_reference_temperature)))),axis=0))
	plt.scatter(np.unique(collect_aggregated_emissivity)*nuc_plate_emissivity,np.polyval(collect_reference_temperature_fit,np.unique(collect_aggregated_emissivity)),c=thickness_over_diffusivity_interpolator(np.unique(collect_aggregated_emissivity),np.polyval(collect_reference_temperature_fit,np.unique(collect_aggregated_emissivity)),grid=False),marker='s',vmin=collect_thickness_over_diffusivity.min(),vmax=collect_thickness_over_diffusivity.max(),cmap='rainbow')
	plt.colorbar()
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity))
	plt.close()



	plt.figure()
	collect_real_thermaldiffusivity = []
	# for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
	for i_emissivity,emissivity in enumerate(np.arange(0.6,1+0.05,0.05)):
	# for i_emissivity,emissivity in enumerate([emissivity_range[0]]):
		# emissivity = emissivity_range[5]

		# emissivity = aggregated_emissivity*nuc_plate_emissivity
		aggregated_emissivity = emissivity/nuc_plate_emissivity

		real_reference_temperature_fitted = np.polyval(collect_reference_temperature_fit,aggregated_emissivity)
		real_thickness_over_diffusivity = thickness_over_diffusivity_interpolator(aggregated_emissivity,real_reference_temperature_fitted,grid=False)

		# I want to plot only the ones that are remotely reasonable
		if emissivity>1.5 or emissivity<0.5:
			continue


		temp = np.abs(aggregated_emissivity_range-aggregated_emissivity)
		temp1 = aggregated_emissivity_range[temp<=np.sort(temp)[1]]
		z = (aggregated_emissivity-temp1[1])/(temp1[1]-temp1[0])
		coeff_0 = -z
		coeff_1 = 1+z


		aggregated_emissivity = cp.deepcopy(temp1[0])

		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
		except:
			pass

		reference_temperature_fitted = np.polyval(collect_reference_temperature_fit,aggregated_emissivity)
		thickness_over_diffusivity = thickness_over_diffusivity_interpolator(aggregated_emissivity,reference_temperature_fitted,grid=False)
		temp = np.abs(reference_temperature_range-reference_temperature_fitted)
		temp2 = reference_temperature_range[temp<=np.sort(temp)[1]]
		z = (reference_temperature_fitted-temp2[1])/(temp2[1]-temp2[0])
		coeff_0_0 = -z
		coeff_0_1 = 1+z


		reference_temperature = cp.deepcopy(temp2[0])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

		reference_temperature = cp.deepcopy(temp2[1])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']


		aggregated_emissivity = cp.deepcopy(temp1[1])

		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
		except:
			pass

		reference_temperature_fitted = np.polyval(collect_reference_temperature_fit,aggregated_emissivity)
		thickness_over_diffusivity = thickness_over_diffusivity_interpolator(aggregated_emissivity,reference_temperature_fitted,grid=False)
		temp = np.abs(reference_temperature_range-reference_temperature_fitted)
		temp2 = reference_temperature_range[temp<=np.sort(temp)[1]]
		z = (reference_temperature_fitted-temp2[1])/(temp2[1]-temp2[0])
		coeff_1_0 = -z
		coeff_1_1 = 1+z



		reference_temperature = cp.deepcopy(temp2[0])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

		reference_temperature = cp.deepcopy(temp2[1])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

		partial_timevariation = partial_timevariation_0_0*coeff_0*coeff_0_0 + partial_timevariation_0_1*coeff_0*coeff_0_1 + partial_timevariation_1_0*coeff_1*coeff_1_0 + partial_timevariation_1_1*coeff_1*coeff_1_1
		partial_timevariation_small = partial_timevariation_small_0_0*coeff_0*coeff_0_0 + partial_timevariation_small_0_1*coeff_0*coeff_0_1 + partial_timevariation_small_1_0*coeff_1*coeff_1_0 + partial_timevariation_small_1_1*coeff_1*coeff_1_1
		partial_BBrad = 1*(partial_BBrad_0_0*coeff_0*coeff_0_0 + partial_BBrad_0_1*coeff_0*coeff_0_1 + partial_BBrad_1_0*coeff_1*coeff_1_0 + partial_BBrad_1_1*coeff_1*coeff_1_1)
		partial_BBrad_small = 1*(partial_BBrad_small_0_0*coeff_0*coeff_0_0 + partial_BBrad_small_0_1*coeff_0*coeff_0_1 + partial_BBrad_small_1_0*coeff_1*coeff_1_0 + partial_BBrad_small_1_1*coeff_1*coeff_1_1)
		partial_diffusion = partial_diffusion_0_0*coeff_0*coeff_0_0 + partial_diffusion_0_1*coeff_0*coeff_0_1 + partial_diffusion_1_0*coeff_1*coeff_1_0 + partial_diffusion_1_1*coeff_1*coeff_1_1
		partial_diffusion_small = partial_diffusion_small_0_0*coeff_0*coeff_0_0 + partial_diffusion_small_0_1*coeff_0*coeff_0_1 + partial_diffusion_small_1_0*coeff_1*coeff_1_0 + partial_diffusion_small_1_1*coeff_1*coeff_1_1


		timevariation = partial_timevariation * nominal_values(thickness_over_diffusivity)
		timevariation_small = partial_timevariation_small * nominal_values(thickness_over_diffusivity)
		BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
		BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
		diff_small = partial_diffusion_small * nominal_values(thickness_over_diffusivity) * Ptthermaldiffusivity
		diff = partial_diffusion * nominal_values(thickness_over_diffusivity) * Ptthermaldiffusivity
		timevariation_and_BB = timevariation + BB
		timevariation_and_BB_small = timevariation_small + BB_small
		timevariation_and_BB_and_diff = timevariation + BB + diff
		timevariation_and_BB_and_diff_small = timevariation_small + BB_small + diff_small
		BB_and_diff_small = BB_small + diff_small

		diff_small_filtered = generic_filter(diff_small,np.mean,size=[21])
		# diff_small_filtered = coleval.butter_lowpass_filter(diff_small,15,383/2,3)
		timevariation_and_BB_small_filtered = generic_filter(timevariation_and_BB_small,np.mean,size=[21])
		# timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)
		BB_small_filtered = generic_filter(BB_small,np.mean,size=[21])
		# BB_small_filtered = coleval.butter_lowpass_filter(BB_small,15,383/2,3)

		temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
		# real_thermaldiffusivity = np.median(Ptthermaldiffusivity*(laser_to_analyse_power-timevariation_and_BB_small_filtered[temp])/(diff_small_filtered[temp]) )
		real_thermaldiffusivity = np.median(Ptthermaldiffusivity*(laser_to_analyse_power-BB_small_filtered[temp])/(diff_small_filtered[temp]) )
		collect_real_thermaldiffusivity.append(real_thermaldiffusivity)
		diff_small = partial_diffusion_small * nominal_values(thickness_over_diffusivity) * real_thermaldiffusivity
		diff = partial_diffusion * nominal_values(thickness_over_diffusivity) * real_thermaldiffusivity
		timevariation_and_BB_and_diff_small = timevariation_small + BB_small + diff_small
		timevariation_and_BB_and_diff_small_filtered = generic_filter(timevariation_and_BB_and_diff_small,np.mean,size=[21])
		timevariation_and_BB_and_diff = timevariation + BB + diff
		# timevariation_and_BB_and_diff_filtered = generic_filter(timevariation_and_BB_and_diff,np.mean,size=[32])
		timevariation_and_BB_and_diff_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_and_diff,10,383/2,3)
		BB_and_diff_filtered = generic_filter(BB + diff,np.mean,size=[7])
		BB_and_diff_small_filtered = generic_filter(BB_small + diff_small,np.mean,size=[7])



		plt.plot(timevariation_and_BB_and_diff_small_filtered,color='C'+str(i_emissivity),linestyle='-',label='emis=%.3g, T0=%.3g, thick=%.3g, diffus=%.3g' %(emissivity,real_reference_temperature_fitted,real_thickness_over_diffusivity*real_thermaldiffusivity,real_thermaldiffusivity))
		plt.plot(timevariation_and_BB_and_diff_filtered,color='C'+str(i_emissivity),linestyle='--')
		# plt.plot(BB_and_diff_small_filtered,color='C'+str(i_emissivity),linestyle='-',label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature_fitted))
		# plt.plot(BB_and_diff_filtered,color='C'+str(i_emissivity),linestyle='--')
	plt.legend(loc='best', fontsize='xx-small')
	plt.axhline(y=laser_to_analyse_power,color='k')
	plt.axhline(y=0,color='k')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nreference_temperature_correction '+str(reference_temperature_correction))

else:	# here I want to fit thickness_over_diffusivity from the peak time derivative alone
	nuc_plate_emissivity = 1.0
	reference_temperature_correction = -0.0
	nuc_plate_emissivity_range = np.arange(0.5,1.65,0.1)
	# aggregated_emissivity_range = np.array(np.linspace(3.5,1.5,6).tolist() + np.linspace(1.2,0.3,10).tolist())
	from scipy.signal import savgol_filter

	collect_collect_thickness_over_diffusivity_from_peak_match = []
	collect_collect_real_thermaldiffusivity = []
	collect_full_error=[]
	for nuc_plate_emissivity in nuc_plate_emissivity_range:


		plt.figure()
		linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']
		collect_peak_heating = []
		collect_peak_dT_dt = []
		collect_aggregated_emissivity = []
		collect_reference_temperature = []
		collect_thickness_over_diffusivity = []
		collect_thickness_over_diffusivity_from_peak_match = []
		for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
		# for i_emissivity,aggregated_emissivity in enumerate([aggregated_emissivity_range[2]]):
		# for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range[::2]):
			# aggregated_emissivity = aggregated_emissivity_range[5]
			try:
				full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
			except:
				pass

			emissivity = aggregated_emissivity * nuc_plate_emissivity

			for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range):
			# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[2]]):
			# for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range[::2]):
			# reference_temperature = reference_temperature_range[0]
				try:
					full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
				except:
					pass

				if False:	# I add this as an a posteriori check of the inferrence from the totals making sense
					try:
						laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']
					except:
						continue
					framerate = 1/np.median(np.diff(time_partial))
					frames_for_one_pulse = framerate/experimental_laser_frequency
					temp = generic_filter(laser_temperature_minus_background_crop_max,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty*0.01))])
					temp = np.diff(temp)
					from scipy.signal import find_peaks, peak_prominences as get_proms
					start_loc = find_peaks(temp,distance=frames_for_one_pulse*0.95)[0]
					end_loc = find_peaks(-temp,distance=frames_for_one_pulse*0.95)[0]

					peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
					start_loc-=2
					frames_for_one_pulse_ON = [frames_for_one_pulse*experimental_laser_duty//2*2]
					for i in range(len(start_loc)):
						try:
							frames_for_one_pulse_ON.append(end_loc[i]-start_loc[i])
						except:
							pass
					frames_for_one_pulse_ON = np.min(frames_for_one_pulse_ON).astype(int)

					heated_phase = np.zeros((frames_for_one_pulse_ON))
					for loc in start_loc:
						loc -= max(0,int(frames_for_one_pulse*experimental_laser_duty*0.00/2))
						heated_phase += laser_temperature_minus_background_crop_max[loc:loc+frames_for_one_pulse_ON]
					heated_phase /= len(start_loc)

					heated_phase = heated_phase[max(0,int(frames_for_one_pulse*experimental_laser_duty*0.00)):-int(frames_for_one_pulse*experimental_laser_duty*0.01)]

					# eta is supposed to be constant, but BB depends on temperature, so I fit only below a certain treshold, let's say until eta is 5% wrong
					real_eta_component = 4*((273.15+reference_temperature+reference_temperature_correction)**3) + 6*((273.15+reference_temperature+reference_temperature_correction)**2)*heated_phase + 4*((273.15+reference_temperature+reference_temperature_correction)**1)*(heated_phase**2) + (heated_phase**3)
					partial_eta_component = 4*((273.15+reference_temperature+reference_temperature_correction)**3)
					# upper_treshold = max(min(len(heated_phase-int(frames_for_one_pulse*experimental_laser_duty*0.05)) , np.abs(real_eta_component/partial_eta_component-1.1).argmin()),40)
					sigma = np.abs(real_eta_component/partial_eta_component)**4

					time = np.arange(len(heated_phase))/framerate
					from scipy.special import expi
					def func_(t,t0,t_mult,csi,max):
						# print(t0,t_mult,csi,max)
						return max*(expi(-csi*(1+4*t_mult*(t-t0))) - expi(-csi))
					bds = [[-np.inf,0.,0.,0.],[0.,np.inf,np.inf,np.inf]]
					guess = [0.,1,0.1,1.]
					fit = curve_fit(func_, time,heated_phase, sigma=sigma, p0=guess, bounds = bds, maxfev=100000000)
					# plt.figure()
					# plt.plot(time,heated_phase)
					# plt.plot(np.arange(len(time)*10)*np.median(np.diff(time)),func_(np.arange(len(time)*10)*np.median(np.diff(time)),*fit[0]),'--')
					# # plt.plot(np.arange(len(temp))/(framerate/2),temp)
					# # # plt.plot(np.arange(len(temp))/(framerate/2),func_(np.arange(len(temp))/(framerate/2),*guess),':')
					# # plt.plot(np.arange(len(temp))/(framerate/2),func_(np.arange(len(temp))/(framerate/2),*fit[0]),'--')
					# plt.pause(0.1)
					# print(fit[0])
					w0 = (Ptthermaldiffusivity/fit[0][1])**0.5
					hs_star = 2.5E-6/w0
					eta = 1*emissivity*5.67E-8*4*((273.15+reference_temperature+reference_temperature_correction)**3)	# this is calculated wia a taylor expansion of the BB radiation term, using only the stronger dependency. for dT<10 the error is <6%
					eta_error = 1*emissivity*5.67E-8*(6*((273.15+reference_temperature+reference_temperature_correction)**2)*(heated_phase.max()) + 4*(273+reference_temperature+reference_temperature_correction)*((heated_phase.max())**2) + (heated_phase.max())**3)
					eta_star = eta*w0/Ptthermalconductivity
					csi = eta_star/(2*hs_star)
					# print([emissivity,reference_temperature])
					# print([csi,fit[0][2],csi/fit[0][2]])
					hs = eta/(2*Ptspecificheat*Ptdensity) * 1/fit[0][1] * 1/fit[0][2]
					print(hs)



					from uncertainties import correlated_values,ufloat
					from uncertainties.unumpy import nominal_values,std_devs,uarray
					fit_ = correlated_values(fit[0],fit[1])
					thickness_over_diffusivity = ufloat(eta,eta_error)/(2*Ptthermalconductivity) * 1/(fit_[1] *fit_[2])
					print(thickness_over_diffusivity)
					collect_thickness_over_diffusivity.append(thickness_over_diffusivity)
				else:
					pass


				partial_timevariation = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
				partial_timevariation_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
				partial_timevariation_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
				partial_timevariation_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
				partial_BBrad = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
				partial_BBrad_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
				partial_BBrad_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
				partial_BBrad_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
				partial_diffusion = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
				partial_diffusion_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

				sigmaSB=5.6704e-08 #[W/(m2 K4)]
				dx=0.09/240

				BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
				BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity

				partial_timevariation_small_filtered = generic_filter(partial_timevariation_small,np.mean,size=[7])
				partial_timevariation_filtered = generic_filter(partial_timevariation,np.mean,size=[7])
				# partial_timevariation_small_filtered = savgol_filter(partial_timevariation_small,7*1,1)
				# partial_timevariation_filtered = savgol_filter(partial_timevariation,7*3,2)
				BB_small_filtered = generic_filter(BB_small,np.mean,size=[7])
				BB_filtered = generic_filter(BB,np.mean,size=[7])
				framerate = 1/np.median(np.diff(time_partial))
				frames_for_one_pulse = framerate/experimental_laser_frequency
				temp = find_peaks(partial_timevariation_small_filtered,distance=frames_for_one_pulse*0.9)[0]
				thickness_over_diffusivity_from_peak_match = np.median(((laser_to_analyse_power*emissivity-BB_small_filtered)/partial_timevariation_small_filtered)[temp])
				# in thepry I should use the large area for this so the diffusion component is as little as possible
				# but it takes time for the heat signal to move, and the difference at the beginning is negligible
				# and the noise is so high that a lot of smoothing is required, defeating the purpose
				# thickness_over_diffusivity_from_peak_match = np.median(((laser_to_analyse_power*emissivity-BB_filtered)/partial_timevariation_filtered)[temp])

				timevariation = partial_timevariation * nominal_values(thickness_over_diffusivity_from_peak_match)
				timevariation_small = partial_timevariation_small * nominal_values(thickness_over_diffusivity_from_peak_match)
				BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
				diff = partial_diffusion * nominal_values(thickness_over_diffusivity_from_peak_match) * Ptthermaldiffusivity
				diff_small = partial_diffusion_small * nominal_values(thickness_over_diffusivity_from_peak_match) * Ptthermaldiffusivity
				timevariation_and_BB = timevariation + BB
				timevariation_and_BB_small = timevariation_small + BB_small
				timevariation_and_BB_and_diff = timevariation + BB + diff
				timevariation_and_BB_and_diff_small = timevariation_small + BB_small + diff_small
				BB_and_diff_small = BB_small + diff_small
				BB_and_diff = BB + diff

				timevariation_filtered = generic_filter(timevariation,np.mean,size=[7])
				# timevariation_small_filtered = generic_filter(timevariation_small,np.mean,size=[7])
				timevariation_small_filtered = coleval.butter_lowpass_filter(timevariation_small,15,383/2,3)
				# timevariation_and_BB_small_filtered = generic_filter(timevariation_and_BB_small,np.mean,size=[7])
				timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)
				# timevariation_and_BB_filtered = generic_filter(timevariation_and_BB,np.mean,size=[7])
				timevariation_and_BB_filtered = coleval.butter_lowpass_filter(timevariation_and_BB,15,383/2,3)
				BB_and_diff_filtered = generic_filter(BB_and_diff,np.mean,size=[7])
				BB_filtered = generic_filter(BB,np.mean,size=[7])
				BB_small_filtered = generic_filter(BB_small,np.mean,size=[7])
				temp = find_peaks(timevariation_and_BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
				peak_heating = np.median(timevariation_and_BB_filtered[temp])
				peak_dT_dt = np.median(partial_timevariation_small[temp])
				# peak_heating = np.median(timevariation_and_BB_filtered[temp])

				# coleval.find_reliable_peaks_and_throughs(generic_filter(timevariation,np.mean,size=[7]),time_partial,experimental_laser_frequency,experimental_laser_duty)
				collect_peak_heating.append(peak_heating)
				collect_peak_dT_dt.append(peak_dT_dt)
				collect_aggregated_emissivity.append(aggregated_emissivity)
				collect_reference_temperature.append(reference_temperature)
				collect_thickness_over_diffusivity_from_peak_match.append(nominal_values(thickness_over_diffusivity_from_peak_match))

				# plt.plot(generic_filter(timevariation,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(timevariation_and_BB_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				plt.plot(generic_filter(timevariation_and_BB_and_diff,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				plt.plot(generic_filter(timevariation_and_BB_and_diff_small,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(timevariation_and_BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(timevariation_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(BB_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(laser_temperature_minus_background_crop_max,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(diff,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(diff_small,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(generic_filter(partial_diffusion*nominal_values(thickness_over_diffusivity)*Ptthermaldiffusivity,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(generic_filter(laser_temperature_minus_background_crop_max,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))

		collect_peak_heating = np.array(collect_peak_heating)
		collect_peak_dT_dt = np.array(collect_peak_dT_dt)
		collect_aggregated_emissivity = np.array(collect_aggregated_emissivity)
		collect_reference_temperature = np.array(collect_reference_temperature)
		# collect_thickness_over_diffusivity = np.array(collect_thickness_over_diffusivity)
		collect_thickness_over_diffusivity_from_peak_match = np.array(collect_thickness_over_diffusivity_from_peak_match)
		plt.legend(loc='best', fontsize='xx-small')
		# plt.axhline(y=laser_to_analyse_power,color='k')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nreference_temperature_correction '+str(reference_temperature_correction) + '\nlaser power %.3gW' %(laser_to_analyse_power) +'\n from peak fit')
		plt.close()

		plt.figure()
		inferred_reference_temperature = []
		for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
			emissivity = aggregated_emissivity*nuc_plate_emissivity	# this is equivalent to setting the NUC plate emissivity to 0.75
			reference_temperature = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
			peak_heating = collect_peak_heating[collect_aggregated_emissivity==aggregated_emissivity]
			# if laser_to_analyse_power>peak_heating.max() or laser_to_analyse_power<peak_heating.min():
			# 	fit = np.polyfit(peak_heating,aggregated_emissivity,1)
			# else:
			fit = np.polyfit(reference_temperature,peak_heating,2)
			fit[-1] -=laser_to_analyse_power*emissivity
			sols = np.roots(fit)
			fit[-1] +=laser_to_analyse_power*emissivity
			if np.sum(np.isreal(sols)) == 0:
				inferred_reference_temperature.append(np.nan)
			else:
				inferred_reference_temperature.append(sols[np.abs(sols-np.mean(reference_temperature)).argmin()])
			plt.plot(reference_temperature,peak_heating,color='C'+str(i_emissivity),label='emissivity=%.3g' %(emissivity))
			plt.plot(np.linspace(15,35,20),np.polyval(fit,np.linspace(15,35,20)),'--',color='C'+str(i_emissivity))
			# plt.axhline(y=laser_to_analyse_power,color='C'+str(i_emissivity))
			plt.axvline(x=inferred_reference_temperature[-1],color='C'+str(i_emissivity))
		inferred_reference_temperature = np.array(inferred_reference_temperature)
		inferred_reference_temperature[inferred_reference_temperature<0] = np.nan
		plt.legend(loc='best', fontsize='xx-small')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power) +'\n from peak fit')
		plt.close()

		plt.figure()
		for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
			emissivity = aggregated_emissivity*nuc_plate_emissivity	# this is equivalent to setting the NUC plate emissivity to 0.75
			reference_temperature = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
			thickness_over_diffusivity = collect_thickness_over_diffusivity_from_peak_match[collect_aggregated_emissivity==aggregated_emissivity]
			plt.plot(reference_temperature,thickness_over_diffusivity,color='C'+str(i_emissivity),label='emissivity=%.3g' %(emissivity))
		plt.legend(loc='best', fontsize='xx-small')
		plt.ylabel('thickness_over_diffusivity')
		plt.xlabel('reference_temperature')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power) +'\n from peak fit')
		plt.close()


		# now I find the diffusivity

		plt.figure()
		collect_high_level_std = []
		collect_high_level_std_small = []
		collect_low_level_std = []
		collect_low_level_std_small = []
		collect_real_thermaldiffusivity = []
		collect_late_high_level_large = []
		for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
		# for i_emissivity,aggregated_emissivity in enumerate([aggregated_emissivity_range[2]]):
		# for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range[::2]):
			# aggregated_emissivity = aggregated_emissivity_range[5]
			try:
				full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
			except:
				pass

			emissivity = aggregated_emissivity * nuc_plate_emissivity
			# if emissivity>1.5:
			# 	continue

			for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range):
			# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[2]]):
			# for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range[::2]):
			# reference_temperature = reference_temperature_range[0]

				thickness_over_diffusivity = collect_thickness_over_diffusivity_from_peak_match[collect_aggregated_emissivity==aggregated_emissivity]
				temp = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
				thickness_over_diffusivity = thickness_over_diffusivity[temp==reference_temperature][0]

				try:
					full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
				except:
					pass

				try:
					laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']
				except:
					continue

				partial_timevariation = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
				partial_timevariation_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
				partial_timevariation_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
				partial_timevariation_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
				partial_BBrad = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
				partial_BBrad_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
				partial_BBrad_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
				partial_BBrad_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
				partial_diffusion = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
				partial_diffusion_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

				sigmaSB=5.6704e-08 #[W/(m2 K4)]
				dx=0.09/240

				BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
				diff_small = partial_diffusion_small * thickness_over_diffusivity * Ptthermaldiffusivity

				diff_small_filtered = generic_filter(diff_small,np.mean,size=[21])
				# diff_small_filtered = coleval.butter_lowpass_filter(diff_small,15,383/2,3)
				BB_small_filtered = generic_filter(BB_small,np.mean,size=[21])
				# timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)

				temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
				real_thermaldiffusivity = np.median(Ptthermaldiffusivity*((laser_to_analyse_power*emissivity-BB_small_filtered)/diff_small_filtered)[temp] )

				timevariation = partial_timevariation * thickness_over_diffusivity
				timevariation_small = partial_timevariation_small * thickness_over_diffusivity
				BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
				BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
				diff = partial_diffusion * thickness_over_diffusivity * real_thermaldiffusivity
				diff_small = partial_diffusion_small * thickness_over_diffusivity * real_thermaldiffusivity
				timevariation_and_BB_diff = timevariation + BB + diff
				timevariation_and_BB_diff_small = timevariation_small + BB_small + diff_small

				timevariation_and_BB_diff_filtered = generic_filter(timevariation_and_BB_diff,np.mean,size=[7])
				timevariation_and_BB_diff_small_filtered = generic_filter(timevariation_and_BB_diff_small,np.mean,size=[7])
				# timevariation_and_BB_small_filtered = generic_filter(timevariation_and_BB_small,np.mean,size=[7])


				peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
				start_loc+=5
				end_loc-=7

				high_level_std = []
				high_level_std_small = []
				for i in range(len(start_loc)):
					high_level_std.append((np.sum((timevariation_and_BB_diff_filtered[start_loc[i]:end_loc[i]]-laser_to_analyse_power*emissivity)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
					high_level_std_small.append((np.sum((timevariation_and_BB_diff_small_filtered[start_loc[i]:end_loc[i]]-laser_to_analyse_power*emissivity)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
				high_level_std = np.median(high_level_std)
				high_level_std_small = np.median(high_level_std_small)

				temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
				late_high_level_large = np.median(timevariation_and_BB_diff_filtered[temp] )

				peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(-laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
				start_loc+=5
				end_loc-=7

				low_level_std = []
				low_level_std_small = []
				for i in range(len(start_loc)):
					low_level_std.append((np.sum((timevariation_and_BB_diff_filtered[start_loc[i]:end_loc[i]]-0)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
					low_level_std_small.append((np.sum((timevariation_and_BB_diff_small_filtered[start_loc[i]:end_loc[i]]-0)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
				low_level_std = np.median(low_level_std)
				low_level_std_small = np.median(low_level_std_small)

				collect_high_level_std.append(high_level_std/(laser_to_analyse_power*emissivity))
				collect_high_level_std_small.append(high_level_std_small/(laser_to_analyse_power*emissivity))
				collect_low_level_std.append(low_level_std/(laser_to_analyse_power*emissivity))
				collect_low_level_std_small.append(low_level_std_small/(laser_to_analyse_power*emissivity))
				collect_real_thermaldiffusivity.append(real_thermaldiffusivity)
				collect_late_high_level_large.append(late_high_level_large)

				# plt.plot(generic_filter(timevariation,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				plt.plot(timevariation_and_BB_diff_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				plt.plot(timevariation_and_BB_diff_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(timevariation_and_BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(timevariation_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(BB_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(laser_temperature_minus_background_crop_max,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(diff,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(diff_small,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
				# plt.plot(generic_filter(partial_diffusion*nominal_values(thickness_over_diffusivity)*Ptthermaldiffusivity,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
				# plt.plot(generic_filter(laser_temperature_minus_background_crop_max,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
		collect_high_level_std = np.array(collect_high_level_std)
		collect_high_level_std_small = np.array(collect_high_level_std_small)
		collect_low_level_std = np.array(collect_low_level_std)
		collect_low_level_std_small = np.array(collect_low_level_std_small)
		collect_real_thermaldiffusivity = np.array(collect_real_thermaldiffusivity)
		collect_late_high_level_large = np.array(collect_late_high_level_large)
		plt.legend(loc='best', fontsize='xx-small')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power)+'\n from peak fit')
		plt.close()

		select = np.logical_and(collect_aggregated_emissivity*nuc_plate_emissivity<1.5,collect_aggregated_emissivity*nuc_plate_emissivity>0.2)

		plt.figure()
		# plt.scatter(collect_aggregated_emissivity,collect_reference_temperature,c=collect_peak_heating)
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=25,cmap='rainbow')
		plt.colorbar().set_label('large_level_std')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=25,colors='k')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c='k',marker='+')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel('emissivity')
		# plt.xlim(right=1.5)
		plt.ylabel('reference_temperature')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		plt.close()

		plt.figure(figsize=(7, 6))
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=15,cmap='rainbow')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
		plt.colorbar().set_label('fit error [au]')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=15,colors='k')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel(r'$\varepsilon$ [au]')
		# plt.xlim(right=1.5)
		plt.ylabel(r'$T_0 [K]$')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW\n' %(laser_to_analyse_power))
		# plt.savefig(path_where_to_save_everything + 'FIG_for_paper_4'  +'.eps')
		plt.close()

		plt.figure()
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=20,cmap='rainbow')
		# plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
		plt.colorbar().set_label('small_level_std')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,colors='k')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c='k',marker='+')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel('emissivity')
		# plt.xlim(right=1.5)
		plt.ylabel('reference_temperature')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		plt.close()

		plt.figure()
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=20,cmap='rainbow')
		# plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
		plt.colorbar().set_label('large_level_std')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,colors='k')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c='k',marker='+')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel('emissivity')
		# plt.xlim(right=1.5)
		plt.ylabel('reference_temperature')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		plt.close()

		plt.figure(figsize=(7, 6))
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_thickness_over_diffusivity_from_peak_match[select]*Ptthermalconductivity,levels=15,cmap='rainbow')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=collect_thickness_over_diffusivity_from_peak_match[select]*Ptthermalconductivity,cmap='rainbow')
		plt.colorbar().set_label(r'$k t_f / \kappa$ $[w/m^2 s/K]$')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_thickness_over_diffusivity_from_peak_match[select]*Ptthermalconductivity,levels=15,colors='k')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel(r'$\varepsilon$ [au]')
		# plt.xlim(right=1.5)
		plt.ylabel(r'$T_0 [K]$')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW\n' %(laser_to_analyse_power))
		# plt.savefig(path_where_to_save_everything + 'FIG_for_paper_5'  +'.eps')
		plt.close()

		plt.figure(figsize=(7, 6))
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_real_thermaldiffusivity[select]*collect_thickness_over_diffusivity_from_peak_match[select]*Ptthermalconductivity,levels=15,cmap='rainbow')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=collect_real_thermaldiffusivity[select]*collect_thickness_over_diffusivity_from_peak_match[select]*Ptthermalconductivity,cmap='rainbow')
		plt.colorbar().set_label(r'$k t_f$ $[W/K]$')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_real_thermaldiffusivity[select]*collect_thickness_over_diffusivity_from_peak_match[select]*Ptthermalconductivity,levels=15,colors='k')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel(r'$\varepsilon$ [au]')
		# plt.xlim(right=1.5)
		plt.ylabel(r'$T_0 [K]$')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW\n' %(laser_to_analyse_power))
		# plt.savefig(path_where_to_save_everything + 'FIG_for_paper_6'  +'.eps')
		plt.close()

		# plt.figure()
		# # plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_thickness_over_diffusivity_from_peak_match/nominal_values(collect_thickness_over_diffusivity))[select],levels=14,cmap='rainbow')
		# plt.colorbar().set_label('from peak vs profile fitted thickness_over_diffusivity')
		# plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_thickness_over_diffusivity_from_peak_match/nominal_values(collect_thickness_over_diffusivity))[select],levels=14,colors='k')
		# plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_thickness_over_diffusivity_from_peak_match/nominal_values(collect_thickness_over_diffusivity))[select],cmap='rainbow')
		# plt.axhline(y=27,color='k')
		# plt.axvline(x=1,color='k')
		# plt.xlabel('emissivity')
		# # plt.xlim(right=1.5)
		# plt.ylabel('reference_temperature')
		# plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		# plt.close()

		collect_collect_thickness_over_diffusivity_from_peak_match.append(collect_thickness_over_diffusivity_from_peak_match)
		collect_collect_real_thermaldiffusivity.append(collect_real_thermaldiffusivity)
		collect_full_error.append(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)


	thickness_over_diffusivity_per_NUC = []
	thickness_over_diffusivity_per_NUC_std = []
	cond_thickness_over_diffusivity_per_NUC = []
	cond_thickness_over_diffusivity_per_NUC_std = []
	thickness_per_NUC = []
	thickness_per_NUC_std = []
	cond_thickness_per_NUC = []
	cond_thickness_per_NUC_std = []
	thermaldiffusivity_per_NUC = []
	thermaldiffusivity_per_NUC_std = []
	emissivity_per_NUC = []
	emissivity_per_NUC_std = []
	reference_temperature = []
	reference_temperature_std = []
	from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator
	for i_,nuc_plate_emissivity in enumerate(nuc_plate_emissivity_range):

		full_error_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_full_error[i_])
		thickness_over_diffusivity_from_peak_match_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_collect_thickness_over_diffusivity_from_peak_match[i_])
		thickness_from_peak_match_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_collect_thickness_over_diffusivity_from_peak_match[i_]*collect_collect_real_thermaldiffusivity[i_])
		thermaldiffusivity_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_collect_real_thermaldiffusivity[i_])
		emissivity_ = np.linspace(min(collect_aggregated_emissivity*nuc_plate_emissivity),max(collect_aggregated_emissivity*nuc_plate_emissivity),num=100)
		reference_temperature_ = np.linspace(min(collect_reference_temperature),max(collect_reference_temperature),num=100)
		emissivity,reference_temperature__ = np.meshgrid(emissivity_,reference_temperature_)
		full_error = full_error_interpolator(emissivity,reference_temperature__)
		# select = full_error<=(np.min(full_error)*1.5)
		select = full_error<=np.sort(full_error.flatten())[200]	# changed to have a more consistent method with later
		# plt.figure()
		# plt.tricontourf(emissivity.flatten(),reference_temperature__.flatten(),full_error_interpolator(emissivity,reference_temperature__).flatten(),levels=20,cmap='rainbow')
		# # plt.scatter(emissivity.flatten(),reference_temperature.flatten(),c=full_error_interpolator(emissivity,reference_temperature)<=(np.min(full_error_interpolator(emissivity,reference_temperature))*1.5))
		# # plt.tricontourf(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature,collect_full_error[i_],levels=20,cmap='rainbow')
		# # plt.tricontourf(emissivity[select].flatten(),reference_temperature__[select].flatten(),thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)[select].flatten(),levels=20,cmap='rainbow')
		# # plt.tricontourf(emissivity[select],reference_temperature[select],thermaldiffusivity_interpolator(emissivity,reference_temperature)[select],levels=20,cmap='rainbow')
		# # plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c='k',marker='+')
		# plt.colorbar()
		# plt.scatter(emissivity[select],reference_temperature__[select],c='k',marker='+')

		# thickness_over_diffusivity_per_NUC.append(np.mean(collect_collect_thickness_over_diffusivity_from_peak_match[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# thickness_over_diffusivity_per_NUC_std.append(np.std(collect_collect_thickness_over_diffusivity_from_peak_match[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# thermaldiffusivity_per_NUC.append(np.mean(collect_collect_real_thermaldiffusivity[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# thermaldiffusivity_per_NUC_std.append(np.std(collect_collect_real_thermaldiffusivity[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# emissivity_per_NUC.append(np.mean((collect_aggregated_emissivity*nuc_plate_emissivity)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# emissivity_per_NUC_std.append(np.std((collect_aggregated_emissivity*nuc_plate_emissivity)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# reference_temperature.append(np.mean((collect_reference_temperature)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
		# reference_temperature_std.append(np.std((collect_reference_temperature)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))

		cond_thickness_over_diffusivity_per_NUC.append(np.sum((Ptthermalconductivity*thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
		cond_thickness_over_diffusivity_per_NUC_std.append(np.std((Ptthermalconductivity*thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
		thickness_over_diffusivity_per_NUC.append(np.sum((thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
		thickness_over_diffusivity_per_NUC_std.append(np.std((thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
		cond_thickness_per_NUC.append(np.sum((Ptthermalconductivity*thickness_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
		cond_thickness_per_NUC_std.append(np.std((Ptthermalconductivity*thickness_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
		thickness_per_NUC.append(np.sum((thickness_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
		thickness_per_NUC_std.append(np.std((thickness_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
		thermaldiffusivity_per_NUC.append(np.sum((thermaldiffusivity_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
		thermaldiffusivity_per_NUC_std.append(np.std((thermaldiffusivity_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
		emissivity_per_NUC.append(np.sum((emissivity/full_error)[select])/np.sum(1/full_error[select]))
		emissivity_per_NUC_std.append(np.std((emissivity/full_error)[select])/np.mean(1/full_error[select]))
		reference_temperature.append(np.sum((reference_temperature__/full_error)[select])/np.sum(1/full_error[select]))
		reference_temperature_std.append(np.std((reference_temperature__/full_error)[select])/np.mean(1/full_error[select]))


	plt.figure()
	plt.errorbar(nuc_plate_emissivity_range,thickness_over_diffusivity_per_NUC,yerr=thickness_over_diffusivity_per_NUC_std,label='thickness_over_diffusivity_per_NUC')
	fit = np.polyfit(nuc_plate_emissivity_range,thickness_over_diffusivity_per_NUC,2,w=1/np.array(thickness_over_diffusivity_per_NUC_std))
	plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'--')
	plt.grid()
	plt.xlabel('NUC emissivity')
	plt.ylabel('thickness_over_diffusivity')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
	plt.figure()
	plt.errorbar(nuc_plate_emissivity_range,thickness_per_NUC,yerr=thickness_per_NUC_std,label='thickness_per_NUC')
	fit = np.polyfit(nuc_plate_emissivity_range,thickness_per_NUC,2,w=1/np.array(thickness_per_NUC_std))
	plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'--')
	plt.grid()
	plt.xlabel('NUC emissivity')
	plt.ylabel('thickness')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
	plt.figure()
	plt.errorbar(nuc_plate_emissivity_range,thermaldiffusivity_per_NUC,yerr=thermaldiffusivity_per_NUC_std,label='thermaldiffusivity_per_NUC')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
	plt.grid()
	plt.xlabel('NUC emissivity')
	plt.ylabel('thermaldiffusivity')
	plt.figure()
	plt.errorbar(nuc_plate_emissivity_range,emissivity_per_NUC,yerr=emissivity_per_NUC_std,label='emissivity_per_NUC')
	fit = np.polyfit(nuc_plate_emissivity_range,emissivity_per_NUC,2,w=1/np.array(thermaldiffusivity_per_NUC_std))
	plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'--')
	plt.grid()
	plt.xlabel('NUC emissivity')
	plt.ylabel('emissivity')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
	plt.figure()
	plt.errorbar(nuc_plate_emissivity_range,reference_temperature,yerr=reference_temperature_std,label='reference_temperature')
	plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
	plt.grid()
	plt.xlabel('NUC emissivity')
	plt.ylabel('reference_temperature')

	# this seems to work quite well. the problem is that I cannot find a way to figure out the backgroud temperature and NUC plate emissivity as a result of the fits.
	# lookig at the resul, though, the room temperature is not influent at all, so it's an irrelevant parameter
	# for this reason I consider (for now, because I don't have any neasurement of it) the NUC plate emissivity to be =1
	# the results are:
	if False:	# 'laser22', '/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000005'
		diffusivity = 9.466307489300497e-06
		thickness_over_diffusivity = 0.07502143143067985	# this is the real measurement I obtain
		thickness = thickness_over_diffusivity*diffusivity	# m
		emissivity = 0.6311559794933064

		# I don't have a measurement of anysotropy of the foil, so I use the one given us by the japanese group
		thickness_variability_japanese_measurement = np.std(foilthickness)/np.mean(foilthickness)
		emissivity_variability_japanese_measurement = np.std(foilemissivity)/np.mean(foilemissivity)

		# these quantities are ok to be relative
		sigma_diffusivity = 8.366535349083484e-07/diffusivity
		sigma_thickness_over_diffusivity = 0.009226369537711715/thickness_over_diffusivity
		sigma_thickness_over_diffusivity = (sigma_thickness_over_diffusivity**2 + thickness_variability_japanese_measurement**2)**0.5
		sigma_thickness = (sigma_diffusivity**2+sigma_thickness_over_diffusivity**2)**0.5
		sigma_emissivity = 0.08937985599750203/emissivity
		sigma_emissivity = (sigma_emissivity**2 + emissivity_variability_japanese_measurement**2)**0.5



	# I waht to check with another experiment, to see if it works
	index = 4
	laser_to_analyse = all_laser_to_analyse[index]
	experimental_laser_frequency = all_laser_to_analyse_frequency[index]
	experimental_laser_voltage = all_laser_to_analyse_voltage[index]
	experimental_laser_duty = all_laser_to_analyse_duty[index]
	power_interpolator = all_power_interpolator[index]
	focus_status = all_focus_status[index]
	case_ID = all_case_ID[index]
	laser_to_analyse = all_laser_to_analyse[index]
	laser_to_analyse_power = power_interpolator(experimental_laser_voltage)
	laser_to_analyse_power = laser_to_analyse_power * power_reduction_window


	# laser_dict = coleval.read_IR_file(laser_to_analyse)
	laser_dict = np.load(laser_to_analyse+'.npz')
	laser_dict.allow_pickle=True
	# laser_counts_filtered = laser_dict['laser_counts_filtered']
	full_saved_file_dict = dict(laser_dict)
	type_of_calibration = 'BB_source_w_window'

	try:
		full_saved_file_dict[type_of_calibration] = full_saved_file_dict[type_of_calibration].all()
	except:
		pass

	aggregated_emissivity_range = full_saved_file_dict[type_of_calibration]['emissivity_range']
	reference_temperature_range = full_saved_file_dict[type_of_calibration]['reference_temperature_range']
	time_partial = full_saved_file_dict[type_of_calibration]['time_of_experiment']

	if False:	# following lines are to redo all the analysis

		collect_large_error_fully_defocused = []
		collect_full_error_fully_defocused = []
		for i_nuc_plate_emissivity,nuc_plate_emissivity in enumerate(nuc_plate_emissivity_range):
			plt.figure()
			collect_high_level_std = []
			collect_high_level_std_small = []
			collect_low_level_std = []
			collect_low_level_std_small = []
			# collect_real_thermaldiffusivity = []
			collect_late_high_level_large = []
			for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range):
			# for i_emissivity,aggregated_emissivity in enumerate([aggregated_emissivity_range[2]]):
			# for i_emissivity,aggregated_emissivity in enumerate(aggregated_emissivity_range[::2]):
				# aggregated_emissivity = aggregated_emissivity_range[5]
				try:
					full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
				except:
					pass

				emissivity = aggregated_emissivity * nuc_plate_emissivity
				# if emissivity>1.5:
				# 	continue

				for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range):
				# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[2]]):
				# for i_reference_temperature,reference_temperature in enumerate(reference_temperature_range[::2]):
				# reference_temperature = reference_temperature_range[0]

					collect_thickness_over_diffusivity_from_peak_match = collect_collect_thickness_over_diffusivity_from_peak_match[i_nuc_plate_emissivity]
					collect_real_thermaldiffusivity = collect_collect_real_thermaldiffusivity[i_nuc_plate_emissivity]
					thickness_over_diffusivity = collect_thickness_over_diffusivity_from_peak_match[collect_aggregated_emissivity==aggregated_emissivity]
					real_thermaldiffusivity = collect_real_thermaldiffusivity[collect_aggregated_emissivity==aggregated_emissivity]
					temp = collect_reference_temperature[collect_aggregated_emissivity==aggregated_emissivity]
					thickness_over_diffusivity = thickness_over_diffusivity[temp==reference_temperature][0]
					real_thermaldiffusivity = real_thermaldiffusivity[temp==reference_temperature][0]

					try:
						full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
					except:
						pass

					try:
						laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']
					except:
						continue

					partial_timevariation = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
					partial_timevariation_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
					partial_timevariation_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
					partial_timevariation_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
					partial_BBrad = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
					partial_BBrad_std = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
					partial_BBrad_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
					partial_BBrad_std_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
					partial_diffusion = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
					partial_diffusion_small = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

					sigmaSB=5.6704e-08 #[W/(m2 K4)]
					dx=0.09/240

					BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
					diff_small = partial_diffusion_small * thickness_over_diffusivity * Ptthermaldiffusivity

					diff_small_filtered = generic_filter(diff_small,np.mean,size=[21])
					# diff_small_filtered = coleval.butter_lowpass_filter(diff_small,15,383/2,3)
					BB_small_filtered = generic_filter(BB_small,np.mean,size=[21])
					# timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)

					# temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
					# real_thermaldiffusivity = np.median(Ptthermaldiffusivity*((laser_to_analyse_power*emissivity-BB_small_filtered)/diff_small_filtered)[temp] )

					timevariation = partial_timevariation * thickness_over_diffusivity
					timevariation_small = partial_timevariation_small * thickness_over_diffusivity
					BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
					BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
					diff = partial_diffusion * thickness_over_diffusivity * real_thermaldiffusivity
					diff_small = partial_diffusion_small * thickness_over_diffusivity * real_thermaldiffusivity
					timevariation_and_BB_diff = timevariation + BB + diff
					timevariation_and_BB_diff_small = timevariation_small + BB_small + diff_small

					timevariation_and_BB_diff_filtered = generic_filter(timevariation_and_BB_diff,np.mean,size=[7])
					timevariation_and_BB_diff_small_filtered = generic_filter(timevariation_and_BB_diff_small,np.mean,size=[7])
					# timevariation_and_BB_small_filtered = generic_filter(timevariation_and_BB_small,np.mean,size=[7])


					peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
					start_loc+=5
					end_loc-=7

					high_level_std = []
					high_level_std_small = []
					for i in range(len(start_loc)):
						high_level_std.append((np.sum((timevariation_and_BB_diff_filtered[start_loc[i]:end_loc[i]]-laser_to_analyse_power*emissivity)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
						high_level_std_small.append((np.sum((timevariation_and_BB_diff_small_filtered[start_loc[i]:end_loc[i]]-laser_to_analyse_power*emissivity)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
					high_level_std = np.median(high_level_std)
					high_level_std_small = np.median(high_level_std_small)

					temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
					late_high_level_large = np.median(timevariation_and_BB_diff_filtered[temp] )

					peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(-laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
					start_loc+=5
					end_loc-=7

					low_level_std = []
					low_level_std_small = []
					for i in range(len(start_loc)):
						low_level_std.append((np.sum((timevariation_and_BB_diff_filtered[start_loc[i]:end_loc[i]]-0)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
						low_level_std_small.append((np.sum((timevariation_and_BB_diff_small_filtered[start_loc[i]:end_loc[i]]-0)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
					low_level_std = np.median(low_level_std)
					low_level_std_small = np.median(low_level_std_small)

					collect_high_level_std.append(high_level_std/(laser_to_analyse_power*emissivity))
					collect_high_level_std_small.append(high_level_std_small/(laser_to_analyse_power*emissivity))
					collect_low_level_std.append(low_level_std/(laser_to_analyse_power*emissivity))
					collect_low_level_std_small.append(low_level_std_small/(laser_to_analyse_power*emissivity))
					# collect_real_thermaldiffusivity.append(real_thermaldiffusivity)
					collect_late_high_level_large.append(late_high_level_large)

					# plt.plot(generic_filter(timevariation,np.mean,size=[21]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					plt.plot(timevariation_and_BB_diff_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					plt.plot(timevariation_and_BB_diff_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
					# plt.plot(timevariation_and_BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					# plt.plot(timevariation_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
					# plt.plot(BB_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
					# plt.plot(BB_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					# plt.plot(laser_temperature_minus_background_crop_max,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					# plt.plot(diff,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
					# plt.plot(diff_small,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
					# plt.plot(timevariation_small_filtered,color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity])
					# plt.plot(generic_filter(partial_diffusion*nominal_values(thickness_over_diffusivity)*Ptthermaldiffusivity,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
					# plt.plot(generic_filter(laser_temperature_minus_background_crop_max,np.mean,size=[7]),color='C'+str(i_reference_temperature),linestyle=linestyles[i_emissivity],label='emis=%.3g, T0=%.3g' %(emissivity,reference_temperature))
			collect_high_level_std = np.array(collect_high_level_std)
			collect_high_level_std_small = np.array(collect_high_level_std_small)
			collect_low_level_std = np.array(collect_low_level_std)
			collect_low_level_std_small = np.array(collect_low_level_std_small)
			collect_real_thermaldiffusivity = np.array(collect_real_thermaldiffusivity)
			collect_late_high_level_large = np.array(collect_late_high_level_large)
			plt.legend(loc='best', fontsize='xx-small')
			plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power)+'\n from peak fit')
			plt.close()
			collect_full_error_fully_defocused.append(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)
			collect_large_error_fully_defocused.append(collect_high_level_std+collect_low_level_std)

			select = collect_aggregated_emissivity*nuc_plate_emissivity<1.5

			plt.figure()
			# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
			plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_large_error_fully_defocused[-1])[select],levels=20,cmap='rainbow')
			# plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
			plt.colorbar().set_label('small+large_level_std_small')
			plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_large_error_fully_defocused[-1])[select],levels=20,colors='k')
			plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c='k',marker='+')
			plt.axhline(y=27,color='k')
			plt.axvline(x=1,color='k')
			plt.xlabel('emissivity')
			# plt.xlim(right=1.5)
			plt.ylabel('reference_temperature')
			plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
			plt.close()


		thickness_over_diffusivity_per_NUC_fully_defocused = []
		thickness_over_diffusivity_per_NUC_std_fully_defocused = []
		thermaldiffusivity_per_NUC_fully_defocused = []
		thermaldiffusivity_per_NUC_std_fully_defocused = []
		emissivity_per_NUC_fully_defocused = []
		emissivity_per_NUC_std_fully_defocused = []
		reference_temperature_fully_defocused = []
		reference_temperature_std_fully_defocused = []
		from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator
		for i_,nuc_plate_emissivity in enumerate(nuc_plate_emissivity_range):

			full_error_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_full_error_fully_defocused[i_])
			thickness_over_diffusivity_from_peak_match_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_collect_thickness_over_diffusivity_from_peak_match[i_])
			thermaldiffusivity_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_collect_real_thermaldiffusivity[i_])
			emissivity_ = np.linspace(min(collect_aggregated_emissivity*nuc_plate_emissivity),max(collect_aggregated_emissivity*nuc_plate_emissivity),num=100)
			reference_temperature_ = np.linspace(min(collect_reference_temperature),max(collect_reference_temperature),num=100)
			emissivity,reference_temperature__ = np.meshgrid(emissivity_,reference_temperature_)
			full_error = full_error_interpolator(emissivity,reference_temperature__)
			select = full_error<=(np.min(full_error)*1.5)
			# plt.figure()
			# # plt.tricontourf(emissivity.flatten(),reference_temperature__.flatten(),full_error_interpolator(emissivity,reference_temperature__).flatten(),levels=20,cmap='rainbow')
			# # plt.scatter(emissivity.flatten(),reference_temperature.flatten(),c=full_error_interpolator(emissivity,reference_temperature)<=(np.min(full_error_interpolator(emissivity,reference_temperature))*1.5))
			# plt.tricontourf(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature,collect_full_error[i_],levels=20,cmap='rainbow')
			# # plt.tricontourf(emissivity[select],reference_temperature[select],thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature)[select],levels=20,cmap='rainbow')
			# # plt.tricontourf(emissivity[select],reference_temperature__[select],thermaldiffusivity_interpolator(emissivity,reference_temperature__)[select],levels=20,cmap='rainbow')
			# plt.scatter(emissivity[select],reference_temperature__[select],c='k',marker='+')
			# plt.colorbar()
			# plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power)+'\n from peak fit')

			# thickness_over_diffusivity_per_NUC.append(np.mean(collect_collect_thickness_over_diffusivity_from_peak_match[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# thickness_over_diffusivity_per_NUC_std.append(np.std(collect_collect_thickness_over_diffusivity_from_peak_match[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# thermaldiffusivity_per_NUC.append(np.mean(collect_collect_real_thermaldiffusivity[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# thermaldiffusivity_per_NUC_std.append(np.std(collect_collect_real_thermaldiffusivity[i_][collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# emissivity_per_NUC.append(np.mean((collect_aggregated_emissivity*nuc_plate_emissivity)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# emissivity_per_NUC_std.append(np.std((collect_aggregated_emissivity*nuc_plate_emissivity)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# reference_temperature.append(np.mean((collect_reference_temperature)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))
			# reference_temperature_std.append(np.std((collect_reference_temperature)[collect_full_error[i_]<=(np.min(collect_full_error[i_])*2)]))

			thickness_over_diffusivity_per_NUC_fully_defocused.append(np.sum((thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
			thickness_over_diffusivity_per_NUC_std_fully_defocused.append(np.std((thickness_over_diffusivity_from_peak_match_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
			thermaldiffusivity_per_NUC_fully_defocused.append(np.sum((thermaldiffusivity_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.sum(1/full_error[select]))
			thermaldiffusivity_per_NUC_std_fully_defocused.append(np.std((thermaldiffusivity_interpolator(emissivity,reference_temperature__)/full_error)[select])/np.mean(1/full_error[select]))
			emissivity_per_NUC_fully_defocused.append(np.sum((emissivity/full_error)[select])/np.sum(1/full_error[select]))
			emissivity_per_NUC_std_fully_defocused.append(np.std((emissivity/full_error)[select])/np.mean(1/full_error[select]))
			reference_temperature_fully_defocused.append(np.sum((reference_temperature__/full_error)[select])/np.sum(1/full_error[select]))
			reference_temperature_std_fully_defocused.append(np.std((reference_temperature__/full_error)[select])/np.mean(1/full_error[select]))


		plt.figure()
		plt.errorbar(nuc_plate_emissivity_range,thickness_over_diffusivity_per_NUC_fully_defocused,yerr=thickness_over_diffusivity_per_NUC_std_fully_defocused,label='thickness_over_diffusivity_per_NUC_fully_defocused')
		fit = np.polyfit(nuc_plate_emissivity_range,thickness_over_diffusivity_per_NUC_fully_defocused,2,w=1/np.array(thickness_over_diffusivity_per_NUC_std_fully_defocused))
		plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'k')
		plt.errorbar(nuc_plate_emissivity_range,thickness_over_diffusivity_per_NUC,yerr=thickness_over_diffusivity_per_NUC_std,label='thickness_over_diffusivity_per_NUC_fully_defocused',linestyle='--')
		fit = np.polyfit(nuc_plate_emissivity_range,thickness_over_diffusivity_per_NUC,2,w=1/np.array(thickness_over_diffusivity_per_NUC_std))
		plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'--k')
		plt.grid()
		plt.xlabel('NUC emissivity')
		plt.ylabel('thickness_over_diffusivity')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		plt.figure()
		plt.errorbar(nuc_plate_emissivity_range,thermaldiffusivity_per_NUC_fully_defocused,yerr=thermaldiffusivity_per_NUC_std_fully_defocused,label='thermaldiffusivity_per_NUC_fully_defocused')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		plt.figure()
		plt.errorbar(nuc_plate_emissivity_range,emissivity_per_NUC_fully_defocused,yerr=emissivity_per_NUC_std_fully_defocused,label='emissivity_per_NUC_fully_defocused')
		fit = np.polyfit(nuc_plate_emissivity_range,emissivity_per_NUC_fully_defocused,2,w=1/np.array(thermaldiffusivity_per_NUC_std_fully_defocused))
		plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'--')
		plt.errorbar(nuc_plate_emissivity_range,emissivity_per_NUC,yerr=emissivity_per_NUC_std,linestyle='--')
		fit = np.polyfit(nuc_plate_emissivity_range,emissivity_per_NUC,2,w=1/np.array(emissivity_per_NUC_std))
		plt.plot(nuc_plate_emissivity_range,np.polyval(fit,nuc_plate_emissivity_range),'--k')
		plt.xlabel('NUC emissivity')
		plt.ylabel('emissivity')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		plt.figure()
		plt.errorbar(nuc_plate_emissivity_range,reference_temperature_fully_defocused,yerr=reference_temperature_std_fully_defocused,label='reference_temperature')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
	else:	# here i just want to use the found values beforehand to see it it is stuly ok

		nuc_plate_emissivity=1	# this is because I don't know it, and the search does not help
		reference_temperature_all = 24.1	# this is because I don't know it, and the search does not help, and also does not matter
		reference_temperature_correction=0

		diffusivity = 9.475792174791206e-06
		thickness_over_diffusivity = 0.07522113087718789	# this is the real measurement I obtain
		thickness = thickness_over_diffusivity*diffusivity	# m
		emissivity = 0.6274716761572996


		aggregated_emissivity = emissivity/nuc_plate_emissivity

		temp = np.abs(aggregated_emissivity_range-aggregated_emissivity)
		temp1 = aggregated_emissivity_range[temp<=np.sort(temp)[1]]
		z = (aggregated_emissivity-temp1[1])/(temp1[1]-temp1[0])
		coeff_0 = -z
		coeff_1 = 1+z


		aggregated_emissivity = cp.deepcopy(temp1[0])

		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
		except:
			pass

		temp = np.abs(reference_temperature_range-reference_temperature_all)
		temp2 = reference_temperature_range[temp<=np.sort(temp)[1]]
		z = (reference_temperature_all-temp2[1])/(temp2[1]-temp2[0])
		coeff_0_0 = -z
		coeff_0_1 = 1+z


		reference_temperature = cp.deepcopy(temp2[0])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_0_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

		reference_temperature = cp.deepcopy(temp2[1])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_0_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']


		aggregated_emissivity = cp.deepcopy(temp1[1])

		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)].all()
		except:
			pass

		temp = np.abs(reference_temperature_range-reference_temperature_all)
		temp2 = reference_temperature_range[temp<=np.sort(temp)[1]]
		z = (reference_temperature_all-temp2[1])/(temp2[1]-temp2[0])
		coeff_1_0 = -z
		coeff_1_1 = 1+z



		reference_temperature = cp.deepcopy(temp2[0])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_1_0 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

		reference_temperature = cp.deepcopy(temp2[1])
		# for i_reference_temperature,reference_temperature in enumerate([reference_temperature_range[0]]):
		# reference_temperature = reference_temperature_range[0]
		try:
			full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)].all()
		except:
			pass

		laser_temperature_minus_background_crop_max = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']

		partial_timevariation_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']
		partial_timevariation_std_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']
		partial_timevariation_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']
		partial_timevariation_std_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']
		partial_BBrad_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']	# this is already multiplied by 2
		partial_BBrad_std_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']
		partial_BBrad_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']
		partial_BBrad_std_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']
		partial_diffusion_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']
		partial_diffusion_small_1_1 = full_saved_file_dict[type_of_calibration]['emissivity='+str(aggregated_emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']

		partial_timevariation = partial_timevariation_0_0*coeff_0*coeff_0_0 + partial_timevariation_0_1*coeff_0*coeff_0_1 + partial_timevariation_1_0*coeff_1*coeff_1_0 + partial_timevariation_1_1*coeff_1*coeff_1_1
		partial_timevariation_small = partial_timevariation_small_0_0*coeff_0*coeff_0_0 + partial_timevariation_small_0_1*coeff_0*coeff_0_1 + partial_timevariation_small_1_0*coeff_1*coeff_1_0 + partial_timevariation_small_1_1*coeff_1*coeff_1_1
		partial_BBrad = 1*(partial_BBrad_0_0*coeff_0*coeff_0_0 + partial_BBrad_0_1*coeff_0*coeff_0_1 + partial_BBrad_1_0*coeff_1*coeff_1_0 + partial_BBrad_1_1*coeff_1*coeff_1_1)
		partial_BBrad_small = 1*(partial_BBrad_small_0_0*coeff_0*coeff_0_0 + partial_BBrad_small_0_1*coeff_0*coeff_0_1 + partial_BBrad_small_1_0*coeff_1*coeff_1_0 + partial_BBrad_small_1_1*coeff_1*coeff_1_1)
		partial_diffusion = partial_diffusion_0_0*coeff_0*coeff_0_0 + partial_diffusion_0_1*coeff_0*coeff_0_1 + partial_diffusion_1_0*coeff_1*coeff_1_0 + partial_diffusion_1_1*coeff_1*coeff_1_1
		partial_diffusion_small = partial_diffusion_small_0_0*coeff_0*coeff_0_0 + partial_diffusion_small_0_1*coeff_0*coeff_0_1 + partial_diffusion_small_1_0*coeff_1*coeff_1_0 + partial_diffusion_small_1_1*coeff_1*coeff_1_1

		timevariation = partial_timevariation * thickness_over_diffusivity
		timevariation_small = partial_timevariation_small * thickness_over_diffusivity
		BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature_all)**4-(273.15+reference_temperature_all+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
		BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature_all)**4-(273.15+reference_temperature_all+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
		diff = partial_diffusion * thickness_over_diffusivity * diffusivity
		diff_small = partial_diffusion_small * thickness_over_diffusivity * diffusivity


		plt.figure()
		a1, = plt.plot(time_partial,timevariation_small+BB_small+diff_small)
		plt.plot(time_partial,timevariation+BB+diff,linestyle='--')
		plt.axhline(y=laser_to_analyse_power*emissivity,linestyle='--',color='k')
		plt.axhline(y=0,linestyle='--',color='k')
		plt.xlabel('time [s]')
		plt.ylabel('Power [W]')
		plt.grid()
		# plt.xlim(left=2.5,right=8.5)

			# I would prefer not to, but I really need to do this, otherwise I can't see anything with the known noise
			timevariation_filtered = coleval.butter_lowpass_filter(timevariation,15,383/2,3)
			timevariation_small_filtered = coleval.butter_lowpass_filter(timevariation_small,15,383/2,3)
			# timevariation_filtered = cp.deepcopy(timevariation)
			# timevariation_small_filtered = cp.deepcopy(timevariation_small)


			plt.figure(figsize=(10, 5))
			start = time_partial[569-7]
			end = time_partial[300+748-7]
			a1, = plt.plot(time_partial,timevariation_small_filtered,label=r'$P_{\delta T/\delta t}$')
			plt.plot(time_partial,timevariation_filtered,a1.get_color(),linestyle='--')
			a3, = plt.plot(time_partial,diff_small,label=r'$P_{\Delta T}$')
			plt.plot(time_partial,diff,a3.get_color(),linestyle='--')
			a2, = plt.plot(time_partial,BB_small,label=r'$P_{BB}$')
			plt.plot(time_partial,BB,a2.get_color(),linestyle='--')
			a4, = plt.plot(time_partial,BB_small+timevariation_small_filtered+diff_small,label=r'$P_{foil}$')
			plt.plot(time_partial,BB+timevariation_filtered+diff,a4.get_color(),linestyle='--')
			reduced_framerate = 1/np.median(np.diff(time_partial))	# the /7 is the smoothing applied to the temperature derivaive
			full_framerate = 2*reduced_framerate
			pixels = 240*187
			area = 0.09*0.07
			noise = 10**0.5 * Ptthermalconductivity*thickness*0.03/((full_framerate*pixels)**0.5) * ( pixels**3 *reduced_framerate/ area**2 + pixels * reduced_framerate**3 / (5 *diffusivity**2) )**0.5 *(np.pi * 0.0033**2)/((np.pi * 0.0033**2/area*pixels)**0.5)
			plt.plot([0,start,start,end,end,10],np.array([0,0,laser_to_analyse_power*emissivity,laser_to_analyse_power*emissivity,0,0])+noise,'-',color='gray')
			plt.plot([0,start,start,end,end,10],np.array([0,0,laser_to_analyse_power*emissivity,laser_to_analyse_power*emissivity,0,0])-noise,'-',color='gray')
			noise = 10**0.5 * Ptthermalconductivity*thickness*0.03/((full_framerate*pixels)**0.5) * ( pixels**3 *reduced_framerate/ area**2 + pixels * reduced_framerate**3 / (5 *diffusivity**2) )**0.5 *(np.pi * 0.0075**2)/((np.pi * 0.0075**2/area*pixels)**0.5)
			plt.plot([0,start,start,end,end,10],np.array([0,0,laser_to_analyse_power*emissivity,laser_to_analyse_power*emissivity,0,0])+noise,'--',color='gray')
			plt.plot([0,start,start,end,end,10],np.array([0,0,laser_to_analyse_power*emissivity,laser_to_analyse_power*emissivity,0,0])-noise,'--',color='gray')
			plt.plot([0,start,start,end,end,10],[0,0,laser_to_analyse_power*emissivity,laser_to_analyse_power*emissivity,0,0],'--k',label=r'$\varepsilon P_{laser}$')

			plt.xlabel('time [s]')
			plt.ylabel('Integrated power [W]')
			plt.grid()
			plt.xlim(left=2.5,right=7.5)
			plt.legend(loc='best')
			# plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG_for_paper_2'+'.eps', bbox_inches='tight')
			plt.savefig(path_where_to_save_everything + 'FIG_for_paper_3'  +'.eps', bbox_inches='tight')


#
