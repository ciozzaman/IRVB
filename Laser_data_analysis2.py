# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 14
figure_index=0
path_where_to_save_everything = '/home/ffederic/work/irvb/laser/results/foil_properties/fixed_diffusivity/'

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']

f = []
f1 = []
for (dirpath, dirnames, filenames) in os.walk('/home/ffederic/work/irvb/laser/results'):
	f1.append(dirnames)
for path in f1[0]:
	if path=='foil_properties':
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
		except:
			print('missing ' + laser_to_analyse+'.npz')
	f[i_name].extend(np.array(laser_location)[np.array(voltlaser)==np.max(voltlaser)][(np.array(freqlaser)[np.array(voltlaser)==np.max(voltlaser)]).argmax()])
	f[i_name].extend([np.array(FrameRate)[np.array(voltlaser)==np.max(voltlaser)][(np.array(freqlaser)[np.array(voltlaser)==np.max(voltlaser)]).argmax()]])

f = np.array(f)
f = np.array([[y1,y2,y3,y4,y5,y6,y7,y8] for _,y1,y2,y3,y4,y5,y6,y7,y8 in sorted(zip(np.round(f[:,-1].astype(np.float)), *f.T))])


foilhorizwpixel=240
foilhorizw=0.09	# m
foilvertw=0.07	# m
dx=foilhorizw/(foilhorizwpixel-1)
plt.figure(figsize=(30, 20))
plt.title('Location of the sampled points')
plt.xlabel('horizontal direction [m]')
plt.ylabel('vertical direction [m]')
index1=0
index2=0
index3=0
for array in f:
	x=np.float(array[2])*dx+np.random.random()/4*dx
	y=np.float(array[4])*dx+np.random.random()/4*dx
	if array[0]=='fully_defocused':
		plt.plot(x,y,'o',fillstyle='none',color=color[index1],label='full def'+' FR%.5gHz ' %(np.float(array[-1])) + array[1],markersize=5+index1*2)
		index1+=1
	elif array[0]=='partially_defocused':
		plt.plot(x,y,'v',fillstyle='none',color=color[index2],label='part def'+' FR%.5gHz ' %(np.float(array[-1])) + array[1],markersize=5+index2*2)
		index2+=1
	else:
		plt.plot(x,y,'+',color=color[index3],label='focus'+' FR%.5gHz ' %(np.float(array[-1])) + array[1],markersize=5+index3*2)
		index3+=1
	plt.annotate(array[1][-2:],(x+1e-4,y+1e-4),fontsize=10)
plt.plot([0,foilhorizw,foilhorizw,0,0],[0,0,foilvertw,foilvertw,0],'k')
plt.grid()
plt.legend(loc='best', fontsize='x-small',ncol=2)
figure_index+=1
plt.savefig(path_where_to_save_everything + str(figure_index)+ '_' + 'foil_map'  +'.eps', bbox_inches='tight')
plt.close()



threshold_freq_list = [1,10,60,100,160,240,300]
# cases_to_include = ['laser17']
# cases_to_include = ['laser39','laser37']	# FR ~2kHz
# cases_to_include = ['laser38','laser36']	# FR ~1kHz
# cases_to_include = ['laser34','laser35']	# FR ~383Hz
# all_cases_to_include = [['laser34','laser35'] , ['laser38','laser36'] , ['laser39','laser37']]
# all_cases_to_include = [['laser34','laser35'], ['laser34'], ['laser35'] , ['laser38','laser36'], ['laser38'], ['laser36'] , ['laser39','laser37'], ['laser39'], ['laser37']]

# cases_to_include = ['laser22','laser25','laser33','laser30']	# FR ~383Hz
# cases_to_include = ['laser15','laser16','laser23','laser21']	# FR ~1kHz
# cases_to_include = ['laser24','laser26','laser27','laser31','laser28','laser29']	# FR ~2kHz
all_cases_to_include = [['laser22','laser25','laser33','laser30'] , ['laser15','laser16','laser23','laser21'] , ['laser24','laser26','laser27','laser31','laser28','laser29']]
# all_cases_to_include = [['laser22','laser25','laser33','laser30'] , ['laser22','laser25'], ['laser33','laser30'], ['laser15','laser23','laser16','laser21'] , ['laser15','laser23'], ['laser16','laser21'], ['laser24','laser26','laser27','laser31','laser28','laser29'], ['laser24','laser26','laser27'], ['laser31','laser28','laser29']]
for cases_to_include in all_cases_to_include:
	figure_index = 0
	coefficients = []
	all_R2 = []
	for max_laser_frequency_focused in threshold_freq_list:
		# max_laser_frequency_focused = 100
		max_laser_frequency_defocused = max_laser_frequency_focused
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

		for index in range(len(all_laser_to_analyse)):
			laser_to_analyse = all_laser_to_analyse[index]
			experimental_laser_frequency = all_laser_to_analyse_frequency[index]
			experimental_laser_voltage = all_laser_to_analyse_voltage[index]
			experimental_laser_duty = all_laser_to_analyse_duty[index]
			power_interpolator = all_power_interpolator[index]
			focus_status = all_focus_status[index]
			case_ID = all_case_ID[index]
			laser_to_analyse = all_laser_to_analyse[index]

			print('STARTING '+laser_to_analyse)
			if not(os.path.exists(laser_to_analyse+'.npz')):
				print('missing .npz file, aborted')
				continue

			if not(focus_status=='fully_defocused'):
				if experimental_laser_frequency>max_laser_frequency_focused:
					print('skipped for laser frequency too high')
					continue
			else:
				if experimental_laser_frequency>max_laser_frequency_defocused:
					print('skipped for laser frequency too high')
					continue

			if experimental_laser_frequency==30:
				print('not considered because close to oscillation frequency')
				continue


			success_read = False
			try:
				laser_dict = np.load(laser_to_analyse+'.npz')
				partial_BBrad = laser_dict['partial_BBrad']
				partial_BBrad_std = laser_dict['partial_BBrad_std']
				partial_diffusion = laser_dict['partial_diffusion']
				partial_diffusion_std = laser_dict['partial_diffusion_std']
				partial_timevariation = laser_dict['partial_timevariation']
				partial_timevariation_std = laser_dict['partial_timevariation_std']
				time_of_experiment = laser_dict['time_of_measurement']	# microseconds
				laser_framerate = laser_dict['FrameRate']	# Hz
				laser_location = laser_dict['laser_location']
				success_read = True
			except:
				print('failed reading '+laser_to_analyse+'.npz')

			if success_read==False:
				continue


			frames_for_one_pulse = laser_framerate/experimental_laser_frequency
			if frames_for_one_pulse*experimental_laser_duty<4:
				print('skipped for laser frequency too high compared to camera framerate')
				continue


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

		def calculate_laser_power_given_parameters(trash,search_thickness,search_emissivity,search_diffusivity,power_reduction_window,defocused_to_focesed_power):
			all_fitted_power = []
			for index in range(len(all_laser_to_analyse_power_end)):
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
				BBrad = partial_BBrad * search_emissivity
				# BBrad_std = partial_BBrad_std * search_emissivity
				diffusion = partial_diffusion * search_thickness
				# diffusion_std = partial_diffusion_std * search_thickness
				timevariation = (1/search_diffusivity)*search_thickness * partial_timevariation
				# timevariation_std = (1/search_diffusivity)*search_thickness * partial_timevariation_std
				powernoback = diffusion + timevariation + BBrad
				# powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5
				frames_for_one_pulse = laser_framerate/experimental_laser_frequency

				# totalpower_filtered_1_std = generic_filter((totalpower_std**2)/max(1,int(framerate/experimental_laser_frequency/30//2*2+1)),np.sum,size=[max(1,int(framerate/experimental_laser_frequency/30//2*2+1))])**0.5

				# temp = max(int((len(powernoback)-len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)/2),1)
				# part_powernoback = np.sort(powernoback[temp:-temp])
				# # fitted_power_2 = [np.median(np.sort(powernoback)[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(np.sort(powernoback)[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(np.sort(powernoback)[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(np.sort(powernoback)[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]
				# totalpower_filtered_1 = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])[temp:-temp]
				# # fitted_power_2 = [np.mean(part_powernoback[:-int(len(part_powernoback)*experimental_laser_duty)]),np.std(part_powernoback[:-int(len(part_powernoback)*experimental_laser_duty)]),np.mean(part_powernoback[-int(len(part_powernoback)*experimental_laser_duty):]),np.std(part_powernoback[-int(len(part_powernoback)*experimental_laser_duty):])]
				# sharpness_indicator = np.sum(np.logical_and(totalpower_filtered_1-np.min(totalpower_filtered_1)>(np.mean(totalpower_filtered_1)-np.min(totalpower_filtered_1))/experimental_laser_duty/2*0.2,totalpower_filtered_1-np.min(totalpower_filtered_1)<(np.mean(totalpower_filtered_1)-np.min(totalpower_filtered_1))/experimental_laser_duty/2*1.8))/len(totalpower_filtered_1)
				# # all_fitted_power.extend(fitted_power_2)

				totalpower_filtered_2 = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
				totalpower_filtered_2 = totalpower_filtered_2[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9))]
				temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
				totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]

				peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
				peaks = np.mean(totalpower_filtered_2[peaks_loc])
				through_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
				through_loc = through_loc[np.logical_and(through_loc>5,through_loc<len(totalpower_filtered_2)-6)]
				through = np.mean(totalpower_filtered_2[through_loc])

				noise_amplitude = (peaks-through)*0.05
				totalpower_filtered_1 = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])
				totalpower_filtered_1 = totalpower_filtered_1[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9))]
				totalpower_filtered_1 = totalpower_filtered_1[temp:-temp]
				sharpness_indicator = np.logical_and(totalpower_filtered_1 > through+noise_amplitude , totalpower_filtered_1 < peaks-noise_amplitude)
				sharpness_indicator = np.logical_or(sharpness_indicator,totalpower_filtered_1<through-noise_amplitude)
				sharpness_indicator = np.logical_or(sharpness_indicator,totalpower_filtered_1>peaks+noise_amplitude)
				sharpness_indicator = 1-np.sum(sharpness_indicator)/len(totalpower_filtered_1)

				# totalpower_filtered_2 = generic_filter(powernoback,np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])[temp:-temp]
				# peaks = np.mean(totalpower_filtered_2[find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.9)[0]])
				# through = np.mean(totalpower_filtered_2[find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.9)[0]])
				if focus_status=='fully_defocused':
					peaks *= defocused_to_focesed_power
					through *= defocused_to_focesed_power
				# all_fitted_power.extend([through,fitted_power_2[1],peaks,fitted_power_2[3]])
				# all_fitted_power.extend([through,peaks])
				all_fitted_power.extend([through,peaks,sharpness_indicator])

			all_fitted_power = np.array(all_fitted_power)
			# all_fitted_power[1::4] = all_fitted_power[1::4]/all_laser_to_analyse_power_end
			# all_fitted_power[3::4] = all_fitted_power[3::4]/all_laser_to_analyse_power_end
			# all_fitted_power[0::4] = all_fitted_power[0::4]/all_laser_to_analyse_power_end
			# all_fitted_power[2::4] = all_fitted_power[2::4]/all_laser_to_analyse_power_end
			# all_fitted_power[0::2] = all_fitted_power[0::2]/all_laser_to_analyse_power_end
			# all_fitted_power[1::2] = all_fitted_power[1::2]/all_laser_to_analyse_power_end
			all_fitted_power[0::3] = all_fitted_power[0::3]/(all_laser_to_analyse_power_end*power_reduction_window)
			all_fitted_power[1::3] = all_fitted_power[1::3]/(all_laser_to_analyse_power_end*power_reduction_window)
			return all_fitted_power



		# x = np.arange(len(all_laser_to_analyse_power_end)*4)
		# y = np.array([np.zeros_like(all_laser_to_analyse_power_end),np.zeros_like(all_laser_to_analyse_power_end),np.ones_like(all_laser_to_analyse_power_end),np.zeros_like(all_laser_to_analyse_power_end)]).T.flatten()
		# weigth = np.array([np.ones_like(all_laser_to_analyse_power_end)*10,np.ones_like(all_laser_to_analyse_power_end)*1,np.ones_like(all_laser_to_analyse_power_end)*10,np.ones_like(all_laser_to_analyse_power_end)*1]).T.flatten()
		# x = np.arange(len(all_laser_to_analyse_power_end)*2)
		# y = np.array([np.zeros_like(all_laser_to_analyse_power_end),np.ones_like(all_laser_to_analyse_power_end)]).T.flatten()
		# weigth = np.ones_like(y)
		x = np.arange(len(all_laser_to_analyse_power_end)*3)
		y = np.array([np.zeros_like(all_laser_to_analyse_power_end),np.ones_like(all_laser_to_analyse_power_end),np.ones_like(all_laser_to_analyse_power_end)]).T.flatten()
		# weigth = np.ones_like(y)
		# weigth = np.array([(1/(all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)).tolist()]*3).T.flatten()
		weigth = np.array([(np.ones_like(all_laser_to_analyse_power_end)*np.min(1/(all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)))]*2+[(1/(all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end))]).T.flatten()
		bds = [[0.5*2.5/1000000,0.5,0.99999*Ptthermaldiffusivity,0.2,0.5],[5*2.5/1000000,1,1.0001*Ptthermaldiffusivity,1,2]]
		# bds = [[1e-9,1e-9,1e-9,0.2],[10*2.5/1000000,1,10*Ptthermaldiffusivity,1]]
		guess=[2.5/1000000,1,Ptthermaldiffusivity,0.9,1]
		# guess = fitted_coefficients_2
		x_scale=[2.5/1000000,1,Ptthermaldiffusivity,1,1]
		fit = curve_fit(calculate_laser_power_given_parameters, x, y, sigma=weigth, p0=guess,x_scale=np.abs(x_scale),bounds=bds,maxfev=int(1e6),verbose=2,ftol=1e-15,xtol=1e-15,gtol=1e-15)

		fitted_coefficients = correlated_values(fit[0],fit[1])
		print('found coefficients: thickness '+str(fitted_coefficients[0])+'m, emissivity '+str(fitted_coefficients[1])+', thermal diffusivity '+str(fitted_coefficients[2])+'m2/s')
		fitted_coefficients_2 = fit[0]
		total_output = calculate_laser_power_given_parameters(1,*fitted_coefficients_2)
		coefficients.append(fitted_coefficients_2)
		# R2 = rsquared(y,total_output)
		R2 = 1-np.sum(((y-total_output)/weigth)**2)/np.sum(((y-np.mean(y))/weigth)**2)
		all_R2.append(R2)
		# total_output[0::4] = total_output[0::4]*all_laser_to_analyse_power_end
		# total_output[1::4] = total_output[1::4]*all_laser_to_analyse_power_end
		# total_output[2::4] = total_output[2::4]*all_laser_to_analyse_power_end
		# total_output[3::4] = total_output[3::4]*all_laser_to_analyse_power_end
		sharpness_indicator = total_output[2::3]
		total_output = total_output[[True,True,False]*len(all_laser_to_analyse_power_end)]
		total_output[0::2] = total_output[0::2]*all_laser_to_analyse_power_end*fit[0][-1]
		total_output[1::2] = total_output[1::2]*all_laser_to_analyse_power_end*fit[0][-1]

		fig, ax1 = plt.subplots(figsize=(20, 10))
		ax1.set_title('found coefficients: thickness '+str(fitted_coefficients[0])+'m, emissivity '+str(fitted_coefficients[1])+'\nthermal diffusivity '+str(fitted_coefficients[2])+'m2/s, window trasmissivity ' +str(fitted_coefficients[3])+ ', defocused/focused factor ' +str(fitted_coefficients[4])+' \nin '+ str(np.unique(all_case_ID)) + ', spot location '+str([laser_location[0],laser_location[2]])+', max freq focused %.3gHz, defocused %.3gHz, camera FR%.3gHz, R2%.3g' %(max_laser_frequency_focused,max_laser_frequency_defocused,laser_framerate,R2))
		ax1.set_xlabel('input power [W]')
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		# plt.errorbar(all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused'],total_output[2::4][all_focus_status_end=='fully_defocused'],yerr=total_output[3::4][all_focus_status_end=='fully_defocused'],fmt='o',color='r')
		# plt.errorbar(all_laser_to_analyse_power_end[np.logical_not(all_focus_status_end=='fully_defocused')],total_output[2::4][np.logical_not(all_focus_status_end=='fully_defocused')],yerr=total_output[3::4][np.logical_not(all_focus_status_end=='fully_defocused')],fmt='+',color='r')
		# plt.errorbar(all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused'],total_output[0::4][all_focus_status_end=='fully_defocused'],yerr=total_output[1::4][all_focus_status_end=='fully_defocused'],fmt='o',color='b')
		# plt.errorbar(all_laser_to_analyse_power_end[np.logical_not(all_focus_status_end=='fully_defocused')],total_output[0::4][np.logical_not(all_focus_status_end=='fully_defocused')],yerr=total_output[1::4][np.logical_not(all_focus_status_end=='fully_defocused')],fmt='+',color='b')
		ax1.plot(all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused']*fit[0][-1],total_output[1::2][all_focus_status_end=='fully_defocused'],'o',color='r')
		ax1.plot(all_laser_to_analyse_power_end[all_focus_status_end=='partially_defocused']*fit[0][-1],total_output[1::2][all_focus_status_end=='partially_defocused'],'v',color='r')
		ax1.plot(all_laser_to_analyse_power_end[all_focus_status_end=='focused']*fit[0][-1],total_output[1::2][all_focus_status_end=='focused'],'+',color='r')
		ax1.plot(all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused']*fit[0][-1],total_output[0::2][all_focus_status_end=='fully_defocused'],'o',color='b')
		ax1.plot(all_laser_to_analyse_power_end[all_focus_status_end=='partially_defocused']*fit[0][-1],total_output[0::2][all_focus_status_end=='partially_defocused'],'v',color='b')
		ax1.plot(all_laser_to_analyse_power_end[all_focus_status_end=='focused']*fit[0][-1],total_output[0::2][all_focus_status_end=='focused'],'+',color='b')
		ax1.plot(np.sort(all_laser_to_analyse_power_end)*fit[0][-1],np.sort(all_laser_to_analyse_power_end)*fit[0][-1],'--k')
		ax1.plot(np.sort(all_laser_to_analyse_power_end)*fit[0][-1],np.zeros_like(all_laser_to_analyse_power_end),'--k')
		ax2.plot(all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused']*fit[0][-1],sharpness_indicator[all_focus_status_end=='fully_defocused'],'o',color='g')
		ax2.plot(all_laser_to_analyse_power_end[all_focus_status_end=='partially_defocused']*fit[0][-1],sharpness_indicator[all_focus_status_end=='partially_defocused'],'v',color='g')
		ax2.plot(all_laser_to_analyse_power_end[all_focus_status_end=='focused']*fit[0][-1],sharpness_indicator[all_focus_status_end=='focused'],'+',color='g')
		ax2.plot(np.sort(all_laser_to_analyse_power_end)*fit[0][-1],np.ones_like(all_laser_to_analyse_power_end),'--g')
		ax1.set_ylabel('detected power [W]', color='k')
		ax2.set_ylabel('sharpness [au]', color='g')  # we already handled the x-label with ax1
		ax1.tick_params(axis='y', labelcolor='k')
		ax2.tick_params(axis='y', labelcolor='g')
		ax2.set_ylim(bottom=0)
		ax1.grid()
		figure_index+=1
		plt.savefig(path_where_to_save_everything_int + str(figure_index) +'.eps', bbox_inches='tight')
		plt.close()

		fig, ax1 = plt.subplots(figsize=(20, 10))
		ax1.set_title('found coefficients: thickness '+str(fitted_coefficients[0])+'m, emissivity '+str(fitted_coefficients[1])+'\nthermal diffusivity '+str(fitted_coefficients[2])+'m2/s, window trasmissivity ' +str(fitted_coefficients[3])+ ', defocused/focused factor ' +str(fitted_coefficients[4])+ ' \nin '+ str(np.unique(all_case_ID)) + ', spot location '+str([laser_location[0],laser_location[2]])+', max freq focused %.3gHz, defocused %.3gHz, camera FR%.3gHz, R2%.3g' %(max_laser_frequency_focused,max_laser_frequency_defocused,laser_framerate,R2))
		ax1.set_xlabel('Duration of ON phase [s]')
		ax1.set_xscale('log')
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		# plt.errorbar(all_laser_to_analyse_frequency_end[all_focus_status_end=='fully_defocused'],total_output[2::4][all_focus_status_end=='fully_defocused'],yerr=total_output[3::4][all_focus_status_end=='fully_defocused'],fmt='o',color='r')
		# plt.errorbar(all_laser_to_analyse_frequency_end[np.logical_not(all_focus_status_end=='fully_defocused')],total_output[2::4][np.logical_not(all_focus_status_end=='fully_defocused')],yerr=total_output[3::4][np.logical_not(all_focus_status_end=='fully_defocused')],fmt='+',color='r')
		# plt.errorbar(all_laser_to_analyse_frequency_end[all_focus_status_end=='fully_defocused'],total_output[0::4][all_focus_status_end=='fully_defocused'],yerr=total_output[1::4][all_focus_status_end=='fully_defocused'],fmt='o',color='b')
		# plt.errorbar(all_laser_to_analyse_frequency_end[np.logical_not(all_focus_status_end=='fully_defocused')],total_output[0::4][np.logical_not(all_focus_status_end=='fully_defocused')],yerr=total_output[1::4][np.logical_not(all_focus_status_end=='fully_defocused')],fmt='+',color='b')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='fully_defocused'],total_output[1::2][all_focus_status_end=='fully_defocused'],'o',color='r')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='partially_defocused'],total_output[1::2][all_focus_status_end=='partially_defocused'],'v',color='r')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='focused'],total_output[1::2][all_focus_status_end=='focused'],'+',color='r')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='fully_defocused'],total_output[0::2][all_focus_status_end=='fully_defocused'],'o',color='b')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='partially_defocused'],total_output[0::2][all_focus_status_end=='partially_defocused'],'v',color='b')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='focused'],total_output[0::2][all_focus_status_end=='focused'],'+',color='b')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='fully_defocused'],all_laser_to_analyse_power_end[all_focus_status_end=='fully_defocused']*fit[0][-1],'ok')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='partially_defocused'],all_laser_to_analyse_power_end[all_focus_status_end=='partially_defocused']*fit[0][-1],'vk')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='focused'],all_laser_to_analyse_power_end[all_focus_status_end=='focused']*fit[0][-1],'+k')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='fully_defocused'],np.zeros_like(all_laser_to_analyse_frequency_end[all_focus_status_end=='fully_defocused']),'ok')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='partially_defocused'],np.zeros_like(all_laser_to_analyse_frequency_end[all_focus_status_end=='partially_defocused']),'vk')
		ax1.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='focused'],np.zeros_like(all_laser_to_analyse_frequency_end[all_focus_status_end=='focused']),'+k')
		ax2.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='fully_defocused'],sharpness_indicator[all_focus_status_end=='fully_defocused'],'o',color='g')
		ax2.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='partially_defocused'],sharpness_indicator[all_focus_status_end=='partially_defocused'],'v',color='g')
		ax2.plot((all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end)[all_focus_status_end=='focused'],sharpness_indicator[all_focus_status_end=='focused'],'+',color='g')
		ax2.plot(np.sort(all_laser_to_analyse_duty_end/all_laser_to_analyse_frequency_end),np.ones_like(all_laser_to_analyse_frequency_end),'--g')
		ax1.set_ylabel('detected power [W]', color='k')
		ax2.set_ylabel('sharpness [au]', color='g')  # we already handled the x-label with ax1
		ax1.tick_params(axis='y', labelcolor='k')
		ax2.tick_params(axis='y', labelcolor='g')
		ax2.set_ylim(bottom=0)
		ax1.grid()
		figure_index+=1
		plt.savefig(path_where_to_save_everything_int + str(figure_index) +'.eps', bbox_inches='tight')
		plt.close()


	fig, ax1 = plt.subplots(figsize=(12, 7))
	fig.subplots_adjust(right=0.8)
	ax1.set_title(str(np.unique(all_case_ID)) + '\n, max freq focused %.3gHz, defocused %.3gHz, camera FR%.3gHz\nspot location ' %(max_laser_frequency_focused,max_laser_frequency_defocused,laser_framerate) +str([laser_location[0],laser_location[2]]) + ' pixels, '+"'--'=nominal specs, Pt")
	ax1.set_xlabel('Maximum laser frequency considered [Hz]')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3.spines["right"].set_position(("axes", 1.2))
	ax3.spines["right"].set_visible(True)
	ax4.spines["right"].set_position(("axes", 1.4))
	ax4.spines["right"].set_visible(True)
	ax1.plot(threshold_freq_list,np.ones_like(threshold_freq_list)*guess[0],'--b')
	a1, = ax1.plot(threshold_freq_list,np.array(coefficients)[:,0],'b')
	ax2.plot(threshold_freq_list,np.ones_like(threshold_freq_list)*guess[1],'--r')
	a2, = ax2.plot(threshold_freq_list,np.array(coefficients)[:,1],'r')
	ax3.plot(threshold_freq_list,np.ones_like(threshold_freq_list)*guess[2],'--g')
	a3, = ax3.plot(threshold_freq_list,np.array(coefficients)[:,2],'g')
	a4, = ax4.plot(threshold_freq_list,all_R2,'k')
	a4, = ax4.plot(threshold_freq_list,np.array(coefficients)[:,3],'xk')
	a4, = ax4.plot(threshold_freq_list,np.array(coefficients)[:,4],'ok')
	ax1.set_ylabel('fitted thickness [m]', color=a1.get_color())
	ax2.set_ylabel('fitted emissivity [au]', color=a2.get_color())  # we already handled the x-label with ax1
	ax3.set_ylabel('fitted diffusivity [m2/s]', color=a3.get_color())  # we already handled the x-label with ax1
	ax4.set_ylabel('R2 [au]', color=a4.get_color())  # we already handled the x-label with ax1
	ax1.tick_params(axis='y', labelcolor=a1.get_color())
	ax2.tick_params(axis='y', labelcolor=a2.get_color())
	ax3.tick_params(axis='y', labelcolor=a3.get_color())
	ax4.tick_params(axis='y', labelcolor=a4.get_color())
	# ax3.legend(loc='upper right', fontsize='x-small')
	# ax1.set_ylim(bottom=0)
	# ax2.set_ylim(bottom=0)
	# ax3.set_ylim(bottom=0)
	ax1.grid()
	figure_index+=1
	plt.savefig(path_where_to_save_everything_int + str(figure_index) +'.eps', bbox_inches='tight')
	plt.close()

	full_saved_file_dict = dict([])
	full_saved_file_dict['location'] = [laser_location[0],laser_location[2]]
	full_saved_file_dict['threshold_freq_list'] = threshold_freq_list
	full_saved_file_dict['coefficients'] = np.array(coefficients)
	full_saved_file_dict['all_R2'] = np.array(all_R2)
	np.savez_compressed(path_where_to_save_everything_int +'coefficients',**full_saved_file_dict)


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
		defocused_to_focesed_power = 1
		if focus_status=='fully_defocused':
			defocused_to_focesed_power = nominal_values(fitted_coefficients[4])
		BBrad = partial_BBrad * fitted_coefficients[1] * defocused_to_focesed_power
		BBrad_std = partial_BBrad_std * fitted_coefficients[1] * defocused_to_focesed_power
		diffusion = partial_diffusion * fitted_coefficients[0] * defocused_to_focesed_power
		diffusion_std = partial_diffusion_std * fitted_coefficients[0] * defocused_to_focesed_power
		timevariation = (1/fitted_coefficients[2])*fitted_coefficients[0] * partial_timevariation * defocused_to_focesed_power
		timevariation_std = (1/fitted_coefficients[2])*fitted_coefficients[0] * partial_timevariation_std * defocused_to_focesed_power
		powernoback = diffusion + timevariation + BBrad
		powernoback_std = (diffusion_std**2 + timevariation_std**2 + BBrad_std**2)**0.5

		frames_for_one_pulse = laser_framerate/experimental_laser_frequency
		# totalpower_filtered_1 = generic_filter(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])

		# part_powernoback = np.sort(nominal_values(powernoback)[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)])
		# # fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
		# fitted_power_2 = [np.median(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.std(part_powernoback[:-int(max(len(powernoback)*experimental_laser_duty*1,1))]),np.median(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):]),np.std(part_powernoback[-int(max(len(powernoback)*experimental_laser_duty*1,1)):])]

		time_axis = (time_of_experiment[1:-1]-time_of_experiment[1])/1e6
		totalpower_filtered_2_full = generic_filter(nominal_values(powernoback),np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty))])
		totalpower_filtered_2 = totalpower_filtered_2_full[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9))]
		time_axis_crop = time_axis[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9))]
		temp = max(1,int((len(totalpower_filtered_2)-len(totalpower_filtered_2)//frames_for_one_pulse*frames_for_one_pulse)/2))
		totalpower_filtered_2 = totalpower_filtered_2[temp:-temp]
		time_axis_crop = time_axis_crop[temp:-temp]
		peaks_loc = find_peaks(totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		peaks_loc = peaks_loc[np.logical_and(peaks_loc>5,peaks_loc<len(totalpower_filtered_2)-6)]
		peaks = np.mean(totalpower_filtered_2[peaks_loc])
		peaks_std = np.std(totalpower_filtered_2[peaks_loc])
		through_loc = find_peaks(-totalpower_filtered_2,distance=frames_for_one_pulse*0.95)[0]
		through_loc = through_loc[np.logical_and(through_loc>5,through_loc<len(totalpower_filtered_2)-6)]
		through = np.mean(totalpower_filtered_2[through_loc])
		through_std = np.std(totalpower_filtered_2[through_loc])
		fitted_power_2 = [through,through_std,peaks,peaks_std]

		totalpower_filtered_1_full = generic_filter(nominal_values(powernoback),np.mean,size=[max(1,int(frames_for_one_pulse*experimental_laser_duty/15//2*2+1))])
		totalpower_filtered_1 = totalpower_filtered_1_full[int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9):-max(1,int(max(1,int(frames_for_one_pulse*experimental_laser_duty))*0.9))]
		totalpower_filtered_1 = totalpower_filtered_1[temp:-temp]
		# noise_amplitude = min(max(np.std(nominal_values(powernoback) - totalpower_filtered_1_full)*2,(peaks-through)*0.1),(peaks-through)*0.4)
		noise_amplitude = (peaks-through)*0.05
		# sharpness_indicator = 1-np.sum(np.logical_and(totalpower_filtered_1 > (peaks-through)*0.1+through , totalpower_filtered_1 < (peaks-through)*0.9+through))/len(totalpower_filtered_1)
		sharpness_indicator = np.logical_and(totalpower_filtered_1 > through+noise_amplitude , totalpower_filtered_1 < peaks-noise_amplitude)
		sharpness_indicator = np.logical_or(sharpness_indicator,totalpower_filtered_1<through-noise_amplitude)
		sharpness_indicator = np.logical_or(sharpness_indicator,totalpower_filtered_1>peaks+noise_amplitude)
		sharpness_indicator = 1-np.sum(sharpness_indicator)/len(totalpower_filtered_1)


		plt.figure(figsize=(20, 10))
		plt.plot()
		plt.plot(time_axis,nominal_values(powernoback),label='totalpower')
		plt.plot(time_axis,nominal_values(BBrad),label='totalBBrad')
		plt.plot(time_axis,nominal_values(diffusion),label='totaldiffusion')
		plt.plot(time_axis,nominal_values(timevariation),label='totaltimevariation')
		plt.plot(time_axis,totalpower_filtered_1_full,linewidth=3,label='totalpower filtered')
		plt.plot(time_axis,totalpower_filtered_2_full,linewidth=3,label='totalpower super filtered')
		plt.plot(time_axis_crop[peaks_loc],totalpower_filtered_2[peaks_loc],'o',markersize=3,label='peaks')
		plt.plot(time_axis_crop[through_loc],totalpower_filtered_2[through_loc],'o',markersize=3,label='through')
		plt.errorbar([time_axis[0],time_axis[-1]],[fitted_power_2[0]]*2,yerr=[fitted_power_2[1]]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
		plt.errorbar([time_axis[0],time_axis[-1]],[fitted_power_2[2]]*2,yerr=[fitted_power_2[3]]*2,color='k',linestyle=':',linewidth=2)
		plt.plot([time_axis[0],time_axis[-1]],[through+noise_amplitude]*2,'--k',label='sharpness limits')
		plt.plot([time_axis[0],time_axis[-1]],[through-noise_amplitude]*2,'--k')
		plt.plot([time_axis[0],time_axis[-1]],[peaks-noise_amplitude]*2,'--k')
		plt.plot([time_axis[0],time_axis[-1]],[peaks+noise_amplitude]*2,'--k')
		plt.plot([time_axis[0],time_axis[-1]],[laser_to_analyse_power_end*nominal_values(fitted_coefficients[-1])]*2,'--r',label='power input')
		# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[0]]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
		# plt.plot([time_axis[0],time_axis[-1]],[fitted_power_2[1]]*2,color='k',linestyle=':',linewidth=2)
		plt.legend(loc='best', fontsize='small')
		plt.xlabel('time [s]')
		plt.ylabel('power [W]')
		plt.grid()
		plt.title(laser_to_analyse+' in '+case_ID+'\nInput '+focus_status+', spot location '+ str([int(laser_location[0]),int(laser_location[2])]) +' power %.3gW, freq %.3gHz, duty%.3g\nHigh Power=%.3g+/-%.3gW, Low Power=%.3g+/-%.3gW, sharpness=%.3g' %(laser_to_analyse_power_end*nominal_values(fitted_coefficients[-1]),experimental_laser_frequency,experimental_laser_duty,fitted_power_2[2],fitted_power_2[3],fitted_power_2[0],fitted_power_2[1],sharpness_indicator))
		# plt.pause(0.01)
		figure_index+=1
		plt.savefig(path_where_to_save_everything_int + 'example_' + str(figure_index) +'.eps', bbox_inches='tight')
		plt.close()
