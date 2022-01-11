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
		f.append([path,f3])

cases_to_include = np.array(f)[:,1]
for i_name,name in enumerate(cases_to_include):
	freqlaser = []
	voltlaser = []
	laser_location = []
	FrameRate = []
	for i_laser_to_analyse,laser_to_analyse in enumerate(collection_of_records[name]['path_files_laser']):
		try:
			laser_dict = np.load(laser_to_analyse+'.npz')
			laser_location.append(laser_dict['laser_location'])
			freqlaser.append(collection_of_records[name]['freqlaser'][i_laser_to_analyse])
			voltlaser.append(collection_of_records[name]['voltlaser'][i_laser_to_analyse])
			FrameRate.append(laser_dict['FrameRate'])
		except Exception as e:
			print('missing ' + laser_to_analyse+'.npz')
			logging.exception('with error: ' + str(e))
	f[i_name].extend(np.array(laser_location)[np.array(voltlaser)==np.max(voltlaser)][(np.array(freqlaser)[np.array(voltlaser)==np.max(voltlaser)]).argmax()])
	f[i_name].extend([np.array(FrameRate)[np.array(voltlaser)==np.max(voltlaser)][(np.array(freqlaser)[np.array(voltlaser)==np.max(voltlaser)]).argmax()]])

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
		plt.plot(x,y,'o',fillstyle='none',color=color[index1],label='full def'+' FR%.5gHz ' %(float(array[-1])) + array[1],markersize=5+index1*2)
		index1+=1
	elif array[0]=='partially_defocused':
		plt.plot(x,y,'v',fillstyle='none',color=color[index2],label='part def'+' FR%.5gHz ' %(float(array[-1])) + array[1],markersize=5+index2*2)
		index2+=1
	else:
		plt.plot(x,y,'+',color=color[index3],label='focus'+' FR%.5gHz ' %(float(array[-1])) + array[1],markersize=5+index3*2)
		index3+=1
	plt.annotate(array[1][-2:],(x+1e-4,y+1e-4),fontsize=10)
plt.plot([0,foilhorizw,foilhorizw,0,0],[0,0,foilvertw,foilvertw,0],'k')
plt.grid()
plt.legend(loc='best', fontsize='x-small',ncol=2)
figure_index+=1
plt.savefig(path_where_to_save_everything + str(figure_index)+ '_' + 'foil_map'  +'.eps', bbox_inches='tight')
plt.close()


power_reduction_window = (6.4+6.3)/2/6.8	# known from a dedicated investigation done 2021/09/14
# threshold_freq_list = [1,10,60,100,160,240,300]
minimum_ON_period_list = 0.1	# seconds
# cases_to_include = ['laser17']
# cases_to_include = ['laser39','laser37']	# FR ~2kHz
# cases_to_include = ['laser38','laser36']	# FR ~1kHz
# cases_to_include = ['laser34','laser35']	# FR ~383Hz
# all_cases_to_include = [['laser34','laser35'] , ['laser38','laser36'] , ['laser39','laser37']]
# all_cases_to_include = [['laser34','laser35'], ['laser34'], ['laser35'] , ['laser38','laser36'], ['laser38'], ['laser36'] , ['laser39','laser37'], ['laser39'], ['laser37']]

# cases_to_include = ['laser22','laser25','laser33','laser30']	# FR ~383Hz
# cases_to_include = ['laser15','laser16','laser23','laser21']	# FR ~1kHz
# cases_to_include = ['laser24','laser26','laser27','laser31','laser28','laser29']	# FR ~2kHz
all_cases_to_include = [['laser22','laser25','laser33','laser30'] , ['laser15','laser16','laser23','laser21'] , ['laser24','laser26','laser27','laser31','laser28','laser29'],['laser22','laser25','laser33','laser30','laser15','laser16','laser23','laser21','laser24','laser26','laser27','laser31','laser28','laser29']]
all_cases_to_include = [['laser34','laser35'], ['laser38','laser36'] , ['laser39','laser37'], ['laser34','laser35','laser38','laser36','laser39','laser37']]
all_cases_to_include = [['laser22','laser25','laser33','laser30','laser34','laser35']]
all_cases_to_include = [['laser22','laser25','laser34']]	# defocused scans avoided for now
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
			partial_BBrad = laser_dict['partial_BBrad_small']
			partial_BBrad_std = laser_dict['partial_BBrad_std_small']
			partial_diffusion = laser_dict['partial_diffusion_small']
			partial_diffusion_std = laser_dict['partial_diffusion_std_small']
			partial_timevariation = laser_dict['partial_timevariation_small']
			partial_timevariation_std = laser_dict['partial_timevariation_std_small']
			time_of_experiment = laser_dict['time_of_experiment']	# microseconds
			if np.diff(time_of_experiment).max()>np.median(np.diff(time_of_experiment))*1.1:
				hole_pos = np.diff(time_of_experiment).argmax()
				if hole_pos<len(time_of_experiment)/2:
					time_of_experiment = time_of_experiment[hole_pos+1:]
				else:
					time_of_experiment = time_of_experiment[:-(hole_pos+1)]
			laser_framerate = laser_dict['FrameRate']	# Hz
			laser_location = laser_dict['laser_location']
			success_read = True
		except Exception as e:
			print('failed reading '+laser_to_analyse+'.npz')
			logging.exception('with error: ' + str(e))

		if success_read==False:
			continue


		frames_for_one_pulse = laser_framerate/experimental_laser_frequency
		if frames_for_one_pulse*experimental_laser_duty<4:
			print('skipped for laser frequency too high compared to camera framerate')
			continue

		# if experimental_laser_frequency>threshold_freq_list[0]:
		# 	sharpness_degradation_high_frequency.append(2)
		# elif experimental_laser_frequency>threshold_freq_list[1]:
		# 	sharpness_degradation_high_frequency.append(10)
		# else:
		# 	sharpness_degradation_high_frequency.append(1)

		all_partial_BBrad.append(partial_BBrad)
		all_partial_BBrad_std.append(partial_BBrad_std)
		all_partial_diffusion.append(partial_diffusion)
		all_partial_diffusion_std.append(partial_diffusion_std)
		all_partial_timevariation.append(partial_timevariation)
		all_partial_timevariation_std.append(partial_timevariation_std)
		all_time_of_experiment.append(time_of_experiment)
		all_laser_framerate.append(laser_framerate)
		all_laser_to_analyse_frequency_end.append(experimental_laser_frequency)
		all_laser_to_analyse_voltage_end.append(experimental_laser_voltage)
		all_laser_to_analyse_duty_end.append(experimental_laser_duty)
		if focus_status=='fully_defocused':
			all_laser_to_analyse_power_end.append(power_interpolator(experimental_laser_voltage)*1)
		else:
			all_laser_to_analyse_power_end.append(power_interpolator(experimental_laser_voltage))
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
	def calculate_laser_power_given_parameters_1(trash,search_emissivity,search_thickness,search_diffusivity,search_defocused_to_focused_power):
		print([search_emissivity,search_thickness,search_diffusivity,search_defocused_to_focused_power])
		thickness_over_diffusivity = search_thickness/search_diffusivity
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
			laser_framerate = all_laser_framerate[index]
			experimental_laser_frequency = all_laser_to_analyse_frequency_end[index]
			experimental_laser_voltage = all_laser_to_analyse_voltage_end[index]
			experimental_laser_duty = all_laser_to_analyse_duty_end[index]
			focus_status = all_focus_status_end[index]
			if focus_status == 'fully_defocused':
				defocused_to_focused_power = search_defocused_to_focused_power
			else:
				defocused_to_focused_power = 1
			BBrad = partial_BBrad * search_emissivity
			# BBrad_std = partial_BBrad_std * 1
			diffusion = partial_diffusion * search_thickness
			# diffusion_std = partial_diffusion_std * search_thickness
			timevariation = thickness_over_diffusivity * partial_timevariation
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
			else:
				time_axis = (time_of_experiment[1:-1]-time_of_experiment[1])/1e6
				totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
				totalpower_filtered_2 = totalpower_filtered_2_full[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
				time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
				temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
				totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
				time_axis_crop = time_axis_crop[temp:-temp]
				peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
				peaks = totalpower_filtered_2[peaks_loc]
				peak = np.mean(peaks)

				if experimental_laser_duty!=0.5:
					totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
					totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
					time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
					temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
					totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
				throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				throughs_loc = throughs_loc[np.logical_and(throughs_loc>5,throughs_loc<len(totalpower_filtered_2)-6)]
				throughs = totalpower_filtered_2[throughs_loc]
				through = np.mean(throughs)
				footprint = np.concatenate([np.ones((time_ON_after_SS)),np.zeros((time_ON_after_SS))])
				totalpower_filtered_1 = generic_filter(powernoback,np.mean,footprint=footprint)
				totalpower_filtered_1 = generic_filter(totalpower_filtered_1,np.mean,size=[max(time_ON_after_SS//10,3)])
				sharpness_indicator = ((np.std(totalpower_filtered_1[totalpower_filtered_1>(peak+through)/2]) + np.std(totalpower_filtered_1[totalpower_filtered_1<(peak+through)/2]) + (peak-through)))/(peak-through)

				all_fitted_power.append((np.array([peak-through,through+all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power])/(all_laser_to_analyse_power_end[index]*power_reduction_window*defocused_to_focused_power)).tolist()+[sharpness_indicator])

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
	sigma = (np.abs(all_laser_to_analyse_power_end-all_laser_to_analyse_power_end.max()) + all_laser_to_analyse_power_end.max())/np.array(all_laser_to_analyse_frequency_end)
	sigma = np.array([sigma,sigma*2,sigma]).T.flatten()
	bds = [[0.7,0.1*2.5e-6,0.1*Ptthermaldiffusivity,0.8],[1,10*2.5e-6,10*Ptthermaldiffusivity,1.2]]
	guess=[0.98,2.5e-6,Ptthermaldiffusivity,1]
	fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/100,xtol=1e-10)
	guess = fit[0]
	fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/10000,xtol=1e-10)
	guess = fit[0]
	fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=guess,bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(guess)/1000000,xtol=1e-10)
	# fit = curve_fit(calculate_laser_power_given_parameters_1, x, y, sigma=sigma, p0=fit[0],bounds=bds,maxfev=int(1e6),verbose=2,diff_step=np.array(fit[0])/10000,xtol=1e-10)
	best_power = calculate_laser_power_given_parameters_1(1,*fit[0])
	guess_best_power = calculate_laser_power_given_parameters_1(1,*guess)
	emissivity_first_stage, thickness_first_stage, diffusivity_first_stage,defocused_to_focused_power_first_stage = fit[0]
	coefficients_first_stage.append(fit[0])
	# all_sharpness_first.append(np.nanmean(best_sharpness))


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
		laser_framerate = all_laser_framerate[index]
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
		# power_reduction_window = 0.9
		BBrad = partial_BBrad * emissivity_first_stage
		BBrad_first = partial_BBrad * emissivity_first_stage
		BBrad_std = partial_BBrad_std * emissivity_first_stage
		diffusion_first = partial_diffusion * thickness_first_stage
		diffusion = partial_diffusion * thickness_first_stage
		diffusion_std = partial_diffusion_std * thickness_first_stage
		timevariation = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation
		timevariation_first = (1/Ptthermaldiffusivity)*thickness_first_stage * partial_timevariation
		timevariation_std = (1/diffusivity_first_stage)*thickness_first_stage * partial_timevariation_std
		powernoback = diffusion + timevariation + BBrad
		powernoback_first = diffusion_first + timevariation_first + BBrad_first
		powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5

		frames_for_one_pulse = laser_framerate/experimental_laser_frequency
		# totalpower_filtered_1 = generic_filter(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])

		# part_powernoback = np.sort(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)])
		# # fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
		# fitted_power_2 = [np.median(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]

		time_axis = (time_of_experiment[1:-1]-time_of_experiment[1])/1e6
		totalpower_filtered_2_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
		totalpower_filtered_2_full_first = generic_filter(powernoback_first,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
		totalpower_filtered_2 = totalpower_filtered_2_full[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
		totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		time_axis_crop = time_axis_crop[temp:-temp]
		peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
		peaks = totalpower_filtered_2[peaks_loc]
		peak = np.mean(peaks)
		time_peaks =  time_axis_crop[peaks_loc]
		peaks_std = np.std(peaks)

		if experimental_laser_duty!=0.5:
			totalpower_filtered_2_full_negative = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))])
			totalpower_filtered_2 = totalpower_filtered_2_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
			totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		throughs_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		throughs_loc = throughs_loc[np.logical_and(throughs_loc>5,throughs_loc<len(totalpower_filtered_2)-6)]
		throughs = totalpower_filtered_2[throughs_loc]
		through = np.mean(throughs)
		time_throughs =  time_axis_crop[throughs_loc]
		throughs_std = np.std(throughs)
		fitted_power_2 = [through,throughs_std,peak,peaks_std]

		totalpower_filtered_1_full = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])
		totalpower_filtered_1_full_first = generic_filter(powernoback_first,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])
		totalpower_filtered_1 = totalpower_filtered_1_full[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.5))]
		totalpower_filtered_1 = totalpower_filtered_1[temp:-temp]
		len_totalpower_filtered_1 = len(totalpower_filtered_1)
		# noise_amplitude = min(max(np.std(nominal_values(powernoback) - totalpower_filtered_1_full)*2,(peaks-through)*0.1),(peaks-through)*0.4)
		# noise_amplitude = (peaks-through)*0.03
		noise_amplitude = 4e-5
		# sharpness_indicator = 1-np.sum(np.logical_and(totalpower_filtered_1 > (peaks-through)*0.1+through , totalpower_filtered_1 < (peaks-through)*0.9+through))/len(totalpower_filtered_1)
		# sharpness_indicator = np.logical_and(totalpower_filtered_1 > through+noise_amplitude , totalpower_filtered_1 < peaks-noise_amplitude)
		# sharpness_indicator = np.logical_or(sharpness_indicator,totalpower_filtered_1<through-noise_amplitude)
		# sharpness_indicator = np.logical_or(sharpness_indicator,totalpower_filtered_1>peaks+noise_amplitude)
		sharpness_indicator = np.sum(np.logical_and(totalpower_filtered_1 > peak-noise_amplitude , totalpower_filtered_1 < peak+noise_amplitude))
		if experimental_laser_duty!=0.5:
			totalpower_filtered_1_full_negative = generic_filter(powernoback,np.mean,size=[max(2,int(frames_for_one_pulse*(1-experimental_laser_duty)/15//2*2+1))])
			totalpower_filtered_1 = totalpower_filtered_1_full_negative[int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5):-max(1,int(max(1,int(frames_for_one_pulse*(1-experimental_laser_duty)))*0.5))]
			totalpower_filtered_1 = totalpower_filtered_1[temp:-temp]
		sharpness_indicator = sharpness_indicator + np.sum(np.logical_and(totalpower_filtered_1 > through-noise_amplitude , totalpower_filtered_1 < through+noise_amplitude))
		len_totalpower_filtered_1 = max(len_totalpower_filtered_1,len(totalpower_filtered_1))
		sharpness_indicator = sharpness_indicator/len_totalpower_filtered_1


		plt.figure(figsize=(20, 10))
		plt.plot()
		plt.plot(time_axis,powernoback_first,'--',label='totalpower first pass')
		plt.plot(time_axis,totalpower_filtered_1_full_first,'--',label='totalpower filtered first pass')
		plt.plot(time_axis,powernoback,label='totalpower')
		plt.plot(time_axis,BBrad,label='totalBBrad')
		plt.plot(time_axis,diffusion,label='totaldiffusion')
		plt.plot(time_axis,timevariation,label='totaltimevariation')
		if experimental_laser_duty!=0.5:
			plt.plot(time_axis,totalpower_filtered_1_full,linewidth=3,label='totalpower filtered ON')
			plt.plot(time_axis,totalpower_filtered_1_full_negative,linewidth=3,label='totalpower filtered OFF')
			ax,=plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='totalpower super filtered')
			plt.plot(time_axis,totalpower_filtered_2_full_negative,linewidth=3,color=ax.get_color())
		else:
			plt.plot(time_axis,totalpower_filtered_1_full,linewidth=3,label='totalpower filtered')
			plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='totalpower super filtered')
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
		plt.title(laser_to_analyse+' in '+case_ID+'\nInput '+focus_status+', spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' power %.3gW, freq %.3gHz, duty%.3g\nHigh Power=%.3g+/-%.3gW, Low Power=%.3g+/-%.3gW, sharpness=%.3g\ndiffusivity %.3g,thickness %.3g,emissivity %.3g,defocused_to_focused %.3g' %(laser_to_analyse_power_end*defocused_to_focused_power*power_reduction_window,experimental_laser_frequency,experimental_laser_duty,fitted_power_2[2],fitted_power_2[3],fitted_power_2[0],fitted_power_2[1],sharpness_indicator,diffusivity_first_stage,thickness_first_stage,emissivity_first_stage,defocused_to_focused_power_first_stage))
		# plt.pause(0.01)
		figure_index+=1
		plt.savefig(path_where_to_save_everything_int + 'example_' + str(figure_index) +'.eps', bbox_inches='tight')
		plt.close()
