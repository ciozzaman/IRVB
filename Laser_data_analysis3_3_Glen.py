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


all_cases_to_include = [['laserG01','laserG02','laserG03']]	# all cases in the same location
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




# 2023-07-16 new thing.
# I want to use as much info I can take from a single laser experiment for findint the foil properties

index = 1
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



elif False:	# here I try to estimate thickness_over_diffusivity from an analytic formula on the peak temperature, but I don't seems to have enough freedom to do the rest
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
	nuc_plate_emissivity_range = np.arange(0.1,3.15,0.3)
	# aggregated_emissivity_range = np.array(np.linspace(3.5,1.5,6).tolist() + np.linspace(1.2,0.3,10).tolist())
	from scipy.signal import savgol_filter

	collect_collect_thickness_over_diffusivity_from_peak_match = []
	collect_collect_real_thermaldiffusivity = []
	collect_small_error=[]
	collect_large_error=[]
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

				framerate = 1/np.median(np.diff(time_partial))
				filtration = max(1,int(np.round(0.015*framerate)))
				frames_for_one_pulse = framerate/experimental_laser_frequency
				partial_timevariation_small_filtered = generic_filter(partial_timevariation_small,np.mean,size=[filtration])
				partial_timevariation_filtered = generic_filter(partial_timevariation,np.mean,size=[filtration])
				# partial_timevariation_small_filtered = savgol_filter(partial_timevariation_small,7*1,1)
				# partial_timevariation_filtered = savgol_filter(partial_timevariation,7*3,2)
				BB_small_filtered = generic_filter(BB_small,np.mean,size=[filtration])
				BB_filtered = generic_filter(BB,np.mean,size=[filtration])
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

				if index in [0,1]:
					diff_small_filtered = generic_filter(diff_small,np.mean,size=[151])
					partial_diffusion_small_filtered = generic_filter(partial_diffusion_small,np.mean,size=[151])
					# diff_small_filtered = coleval.butter_lowpass_filter(diff_small,15,383/2,3)
					BB_small_filtered = generic_filter(BB_small,np.mean,size=[151])
					# timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)
				else:
					diff_small_filtered = generic_filter(diff_small,np.mean,size=[41])
					partial_diffusion_small_filtered = generic_filter(partial_diffusion_small,np.mean,size=[41])
					# diff_small_filtered = coleval.butter_lowpass_filter(diff_small,15,383/2,3)
					BB_small_filtered = generic_filter(BB_small,np.mean,size=[41])
					# timevariation_and_BB_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_small,15,383/2,3)

				temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
				# real_thermaldiffusivity = np.median(Ptthermaldiffusivity*((laser_to_analyse_power*emissivity-BB_small_filtered)/diff_small_filtered)[temp] )
				real_thermaldiffusivity = np.median(((laser_to_analyse_power*emissivity-BB_small_filtered)/partial_diffusion_small_filtered)[temp] ) / thickness_over_diffusivity

				timevariation = partial_timevariation * thickness_over_diffusivity
				timevariation_small = partial_timevariation_small * thickness_over_diffusivity
				BB = (partial_BBrad + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((20*dx)**2 *np.pi)) *emissivity
				BB_small = (partial_BBrad_small + 2*sigmaSB*((273.15+reference_temperature)**4-(273.15+reference_temperature+reference_temperature_correction)**4)*((9*dx)**2 *np.pi)) *emissivity
				diff = partial_diffusion * thickness_over_diffusivity * real_thermaldiffusivity
				diff_small = partial_diffusion_small * thickness_over_diffusivity * real_thermaldiffusivity
				timevariation_and_BB_diff = timevariation + BB + diff
				timevariation_and_BB_diff_small = timevariation_small + BB_small + diff_small

				filtration = max(1,int(np.round(0.015*framerate)))
				timevariation_small_filtered = generic_filter(timevariation_small,np.mean,size=[filtration])
				BB_small_filtered = generic_filter(BB_small,np.mean,size=[filtration])
				# filtration = int(np.round(0.025*framerate))
				# if index in [0,1]:
				# 	timevariation_and_BB_diff_small_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_diff_small,1,framerate,2)
				# 	timevariation_and_BB_diff_filtered = coleval.butter_lowpass_filter(timevariation_and_BB_diff,1,framerate,2)
				# else:
				timevariation_and_BB_diff_filtered = generic_filter(timevariation_and_BB_diff,np.mean,size=[filtration])
				# timevariation_and_BB_diff_filtered = generic_filter(timevariation_and_BB_diff_filtered,np.mean,size=[int(1.5*filtration)])
				# timevariation_and_BB_diff_filtered = generic_filter(timevariation_and_BB_diff_filtered,np.mean,size=[int(1.5*1.5*filtration)])
				# timevariation_and_BB_diff_filtered = generic_filter(timevariation_and_BB_diff_filtered,np.mean,size=[int(1.5*1.5*1.5*filtration)])
				timevariation_and_BB_diff_small_filtered = generic_filter(timevariation_and_BB_diff_small,np.mean,size=[filtration])

				# timevariation_and_BB_diff_small_filtered = generic_filter(timevariation_and_BB_diff_small_filtered,np.mean,size=[int(1.5*filtration)])
				# timevariation_and_BB_diff_small_filtered = generic_filter(timevariation_and_BB_diff_small_filtered,np.mean,size=[int(1.5*1.5*filtration)])
				# timevariation_and_BB_diff_small_filtered = generic_filter(timevariation_and_BB_diff_small_filtered,np.mean,size=[int(1.5*1.5*1.5*filtration)])

				# timevariation_and_BB_small_filtered = generic_filter(timevariation_and_BB_small,np.mean,size=[7])
				if False:
					spectra_orig = np.fft.fft(timevariation_and_BB_diff_small_filtered[start_loc[i]:end_loc[i]], axis=0)
					# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
					magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
					phase = np.angle(spectra_orig)
					freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)


				# peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
				# start_loc+=15
				start_loc = find_peaks(timevariation_small_filtered,distance=frames_for_one_pulse*0.95)[0]
				end_loc = find_peaks(-np.diff(BB_small_filtered),distance=frames_for_one_pulse*0.95)[0]
				end_loc-=1

				high_level_std = []
				high_level_std_small = []
				for i in range(len(start_loc)):
					high_level_std.append((np.sum((timevariation_and_BB_diff_filtered[start_loc[i]:end_loc[i]]-laser_to_analyse_power*emissivity)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
					high_level_std_small.append((np.sum((timevariation_and_BB_diff_small_filtered[start_loc[i]:end_loc[i]]-laser_to_analyse_power*emissivity)**2) / (end_loc[i]-start_loc[i]-1))**0.5 )
				high_level_std = np.median(high_level_std)
				high_level_std_small = np.median(high_level_std_small)

				temp = find_peaks(BB_small_filtered,distance=frames_for_one_pulse*0.9)[0]
				late_high_level_large = np.median(timevariation_and_BB_diff_filtered[temp] )

				# peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(-laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
				# modified only for Glen's version
				# peaks_loc,throughs_loc,frames_for_one_pulse,start_loc,end_loc = coleval.find_reliable_peaks_and_throughs(laser_temperature_minus_background_crop_max,time_partial,experimental_laser_frequency,experimental_laser_duty)
				start_loc,end_loc = np.array([0,end_loc[0]]),np.array([start_loc[0],len(timevariation_and_BB_diff_filtered)])
				start_loc+=2
				end_loc-=5

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
		# plt.close()

		plt.figure(figsize=(7, 6))
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=15,cmap='rainbow')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
		plt.colorbar().set_label('small_level_std')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=15,colors='k')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel(r'$\varepsilon$ [au]')
		# plt.xlim(right=1.5)
		plt.ylabel(r'$T_0 [K]$')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW\n' %(laser_to_analyse_power))
		# plt.savefig(path_where_to_save_everything + 'FIG_for_paper_4'  +'.eps')
		# plt.close()

		plt.figure()
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=20,cmap='rainbow')
		# plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
		plt.colorbar().set_label('full error [au]')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,colors='k')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c='k',marker='+')
		plt.axhline(y=27,color='k')
		plt.axvline(x=1,color='k')
		plt.xlabel('emissivity')
		# plt.xlim(right=1.5)
		plt.ylabel('reference_temperature')
		plt.title('nuc_plate_emissivity '+str(nuc_plate_emissivity) + '\nlaser power %.3gW' %(laser_to_analyse_power))
		# plt.close()

		plt.figure()
		# plt.scatter(collect_aggregated_emissivity[select],collect_reference_temperature[select],c=collect_peak_heating)
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		# plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std_small+collect_low_level_std_small)[select],levels=20,cmap='rainbow')
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=20,cmap='rainbow')
		# plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=(collect_high_level_std+collect_low_level_std+collect_high_level_std_small+collect_low_level_std_small)[select],cmap='rainbow')
		plt.colorbar().set_label('large_level_std')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],(collect_high_level_std+collect_low_level_std)[select],levels=20,colors='k')
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
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_thickness_over_diffusivity_from_peak_match[select]*Authermalconductivity,levels=15,cmap='rainbow')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=collect_thickness_over_diffusivity_from_peak_match[select]*Authermalconductivity,cmap='rainbow')
		plt.colorbar().set_label(r'$k t_f / \kappa$ $[W/m^2 s/K]$')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_thickness_over_diffusivity_from_peak_match[select]*Authermalconductivity,levels=15,colors='k')
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
		plt.tricontourf(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_real_thermaldiffusivity[select]*collect_thickness_over_diffusivity_from_peak_match[select]*Authermalconductivity,levels=15,cmap='rainbow')
		plt.scatter(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],c=collect_real_thermaldiffusivity[select]*collect_thickness_over_diffusivity_from_peak_match[select]*Authermalconductivity,cmap='rainbow')
		plt.colorbar().set_label(r'$k t_f$ $[W/K]$')
		plt.tricontour(collect_aggregated_emissivity[select]*nuc_plate_emissivity,collect_reference_temperature[select],collect_real_thermaldiffusivity[select]*collect_thickness_over_diffusivity_from_peak_match[select]*Authermalconductivity,levels=15,colors='k')
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
		collect_small_error.append(collect_high_level_std_small+collect_low_level_std_small)
		collect_large_error.append(collect_high_level_std+collect_low_level_std)
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

		if index in [0,1]:
			full_error_interpolator = CloughTocher2DInterpolator(list(zip(collect_aggregated_emissivity*nuc_plate_emissivity,collect_reference_temperature)),collect_small_error[i_])
		elif index in [2]:
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
	if False:	# 'case 3'
		diffusivity = 1.5e-06
		thickness_over_diffusivity = 0.11	# this is the real measurement I obtain
		thickness = thickness_over_diffusivity*diffusivity	# m
		emissivity = 0.8

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
	if False:
		index = 2
		nuc_plate_emissivity=2.5	# this is because I don't know it, and the search does not help
		reference_temperature_all = 24.1	# this is because I don't know it, and the search does not help, and also does not matter
		reference_temperature_correction=0

		diffusivity = 1.5e-06
		thickness_over_diffusivity = 0.11	# this is the real measurement I obtain
		thickness = thickness_over_diffusivity*diffusivity	# m
		emissivity = 0.8
	if False:
		index = 1
		nuc_plate_emissivity=1	# this is because I don't know it, and the search does not help
		reference_temperature_all = 32	# this is because I don't know it, and the search does not help, and also does not matter
		reference_temperature_correction=0

		diffusivity = 2.6e-06
		thickness_over_diffusivity = 2.3	# this is the real measurement I obtain
		thickness = thickness_over_diffusivity*diffusivity	# m
		emissivity = 0.6

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
		plt.plot(time_partial,timevariation+BB+diff,linestyle='--')
		a1, = plt.plot(time_partial,timevariation_small+BB_small+diff_small)
		plt.axhline(y=laser_to_analyse_power*emissivity,linestyle='--',color='k')
		plt.axhline(y=0,linestyle='--',color='k')
		plt.xlabel('time [s]')
		plt.ylabel('Power [W]')
		plt.grid()
		# plt.xlim(left=2.5,right=8.5)



			# I would prefer not to, but I really need to do this, otherwise I can't see anything with the known noise
			# timevariation_filtered = coleval.butter_lowpass_filter(timevariation,15,383/2,3)
			# timevariation_small_filtered = coleval.butter_lowpass_filter(timevariation_small,15,383/2,3)
			timevariation_filtered = coleval.butter_lowpass_filter(timevariation,15,framerate,3)
			timevariation_small_filtered = coleval.butter_lowpass_filter(timevariation_small,15,framerate,3)
			# timevariation_filtered = cp.deepcopy(timevariation)
			# timevariation_small_filtered = cp.deepcopy(timevariation_small)
			diff_filtered = coleval.butter_lowpass_filter(diff,15,framerate,3)
			diff_small_filtered = coleval.butter_lowpass_filter(diff_small,15,framerate,3)



			plt.figure(figsize=(10, 5))
			start = time_partial[569-7]
			end = time_partial[300+748-7]
			a1, = plt.plot(time_partial,timevariation_small_filtered,label=r'$P_{\delta T/\delta t}$')
			plt.plot(time_partial,timevariation_filtered,a1.get_color(),linestyle='--')
			a3, = plt.plot(time_partial,diff_small_filtered,label=r'$P_{\Delta T}$')
			plt.plot(time_partial,diff_filtered,a3.get_color(),linestyle='--')
			a2, = plt.plot(time_partial,BB_small,label=r'$P_{BB}$')
			plt.plot(time_partial,BB,a2.get_color(),linestyle='--')
			a4, = plt.plot(time_partial,BB_small+timevariation_small_filtered+diff_small_filtered,label=r'$P_{foil}$')
			plt.plot(time_partial,BB+timevariation_filtered+diff_filtered,a4.get_color(),linestyle='--')
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
			# plt.xlim(left=2.5,right=7.5)
			plt.legend(loc='best')
			# plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG_for_paper_2'+'.eps', bbox_inches='tight')
			plt.savefig(path_where_to_save_everything + 'FIG_for_paper_3'  +'.eps', bbox_inches='tight')


#
