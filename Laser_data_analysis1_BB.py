# Created 07/12/2022
# Fabio Federici
# the purpose of this is to change the temperature calibration from polynomial to a black body correlation
# I want also to include a proper laplacian (with diagonals) rather than the current aproach


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)
from scipy.signal import find_peaks, peak_prominences as get_proms

#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
from multiprocessing import Pool,cpu_count,current_process,set_start_method,get_context,Semaphore
number_cpu_available = 16
# number_cpu_available = cpu_count()


# degree of polynomial of choice
n=3
# reference frames
if False:	# manual collection of parameters
	all_path_reference_frames=[[vacuum3]*len(laser20),[vacuum3]*len(laser21)]
	# # Lares file I want to read
	all_laser_to_analyse = [laser20,laser21]
	all_laser_to_analyse_ROI = [laserROI20,laserROI21]
	all_laser_to_analyse_frequency = [freqlaser20,freqlaser21]
	all_laser_to_analyse_voltage = [voltlaser20,voltlaser21]
	all_laser_to_analyse_duty = [dutylaser20,dutylaser21]
	all_focus_status = [['focused']*len(laser20),['focused']*len(laser21)]
	all_power_interpolator = [[power_interpolator1]*len(laser20),[power_interpolator1]*len(laser21)]
	all_laser_to_analyse=coleval.flatten_full(all_laser_to_analyse)
	all_laser_to_analyse_frequency=coleval.flatten_full(all_laser_to_analyse_frequency)
	all_laser_to_analyse_voltage=coleval.flatten_full(all_laser_to_analyse_voltage)
	all_laser_to_analyse_duty=coleval.flatten_full(all_laser_to_analyse_duty)
	all_laser_to_analyse_ROI=coleval.flatten(all_laser_to_analyse_ROI)
	all_focus_status=coleval.flatten(all_focus_status)
	all_path_reference_frames=coleval.flatten(all_path_reference_frames)
else:	# automatic collection of parameters
	# cases_to_include = ['laser22','laser25','laser34','laser15','laser16','laser17','laser18','laser19','laser20','laser21','laser23','laser24','laser26','laser27','laser28','laser29','laser30','laser31','laser32','laser33','laser35','laser36','laser37','laser38','laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = ['laser20','laser21','laser23','laser24','laser26','laser27','laser28','laser29','laser30','laser31','laser32','laser33','laser35','laser36','laser37','laser38','laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = np.flip(cases_to_include,axis=0)
	# cases_to_include = ['laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = ['laser34','laser35','laser36','laser37','laser38','laser39']
	# cases_to_include = ['laser17','laser18','laser19','laser20','laser21','laser22','laser23','laser24','laser25','laser26','laser27','laser28','laser29','laser30','laser31','laser32','laser33','laser34','laser35','laser36','laser37','laser38','laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = ['laser30','laser31','laser32','laser33','laser34','laser35','laser36','laser37','laser38','laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = ['laser19','laser22','laser33']
	cases_to_include = ['laser22','laser33']
	# cases_to_include = ['laser33']
	# cases_to_include = ['laser41']
	# cases_to_include = np.flip(cases_to_include,axis=0)
	all_case_ID = []
	all_path_reference_frames = []
	all_laser_to_analyse = []
	all_laser_to_analyse_ROI = []
	all_laser_to_analyse_frequency = []
	all_laser_to_analyse_voltage = []
	all_laser_to_analyse_duty = []
	all_focus_status = []
	all_power_interpolator = []
	all_foil_position_dict = []
	for name in cases_to_include:
		# if collection_of_records[name]['focus_status'][0] == 'fully_defocused':
		# 	continue
		all_case_ID.extend([name] * len(collection_of_records[name]['path_files_laser']))
		all_path_reference_frames.extend(collection_of_records[name]['reference_clear'])
		all_laser_to_analyse.extend(collection_of_records[name]['path_files_laser'])
		all_laser_to_analyse_ROI.extend(collection_of_records[name]['laserROI'])
		all_laser_to_analyse_frequency.extend(collection_of_records[name]['freqlaser'])
		all_laser_to_analyse_voltage.extend(collection_of_records[name]['voltlaser'])
		all_laser_to_analyse_duty.extend(collection_of_records[name]['dutylaser'])
		all_focus_status.extend(collection_of_records[name]['focus_status'])
		all_power_interpolator.extend(collection_of_records[name]['power_interpolator'])
		all_foil_position_dict.extend(collection_of_records[name]['foil_position_dict'])
	path_to_save_figures_master = '/home/ffederic/work/irvb/laser/results/'

# limited_ROI = [[64,127],[0,319]]
# index = 0
# laser_to_analyse = laser19[index]
# experimental_laser_frequency = freqlaser19[index]
# # framerate of the IR camera in Hz
# framerate=383
# # integration time of the camera in ms
# int_time=1
# #filestype
# type='_stat.npy'
# # type='csv'

# folder of the parameters path
# pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
pathparams='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams):
	f.append(dirnames)
parameters_available = f[0]
parameters_available_int_time = []
parameters_available_framerate = []
for path in parameters_available:
	parameters_available_int_time.append(float(path[:path.find('ms')]))
	parameters_available_framerate.append(float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time = np.array(parameters_available_int_time)
parameters_available_framerate = np.array(parameters_available_framerate)

# folder of the parameters path for BB correlation
# pathparams_BB='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'	# original nuc measurements, calibration with wavewlength_top=5,wavelength_bottom=2.5
pathparams_BB='/home/ffederic/work/irvb/2022-12-07_multiple_search_for_parameters'	# original nuc measurements, calibration with wavewlength_top=5.1,wavelength_bottom=1.5
f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams_BB):
	f.append(dirnames)
parameters_available_BB = f[0]
parameters_available_int_time_BB = []
parameters_available_framerate_BB = []
for path in parameters_available_BB:
	parameters_available_int_time_BB.append(float(path[:path.find('ms')]))
	parameters_available_framerate_BB.append(float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time_BB = np.array(parameters_available_int_time_BB)
parameters_available_framerate_BB = np.array(parameters_available_framerate_BB)

# # Load data
# pathparams = pathparams+'/'+str(int_time)+'ms'+str(framerate)+'Hz/'+'numcoeff'+str(n)+'/average'
# fullpathparams=os.path.join(pathparams,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
# params_dict=np.load(fullpathparams)
# params = params_dict['coeff']
# errparams = params_dict['errcoeff']
#
# background_timestamps = [(np.nanmean(np.load(file+'.npz')['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
# background_temperatures = [(np.nanmean(np.load(file+'.npz')['SensorTemp_0'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
# background_counts = [(np.load(file+'.npz')['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
# background_counts_std = [(np.load(file+'.npz')['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
# relevant_background_files = [file for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]

# example good properties that I use to see how detected power changes with sampling area
sample_properties = dict([])
# sample_properties['thickness'] = 3.2*1e-6
# sample_properties['emissivity'] = 1
# sample_properties['diffusivity'] = 1.32*1e-5
sample_properties['thickness'] = 2.093616658223934e-06
sample_properties['emissivity'] = 1
sample_properties['diffusivity'] = 1.03*1e-5
# BBtreshold = 0.06
# BBtreshold = 0.13
# BBtreshold = 0.05 this is now defined dinamically

emissivity_range = np.array(np.linspace(3.5,1.5,6).tolist() + np.linspace(1.2,0.3,10).tolist() + np.arange(0.25,0.05,-0.05).tolist())
reference_temperature_range = np.linspace(16,34,10)
override = True

def function_a(index):
	from uncertainties.unumpy import nominal_values,std_devs
	from uncertainties import correlated_values,ufloat
	figure_index=0
	# index=0
	laser_to_analyse = all_laser_to_analyse[index]
	experimental_laser_frequency = all_laser_to_analyse_frequency[index]
	experimental_laser_voltage = all_laser_to_analyse_voltage[index]
	experimental_laser_duty = all_laser_to_analyse_duty[index]
	focus_status = all_focus_status[index]
	path_reference_frames = all_path_reference_frames[index]
	power_interpolator = all_power_interpolator[index]
	case_ID = all_case_ID[index]
	limited_ROI = all_laser_to_analyse_ROI[index]
	foil_position_dict = all_foil_position_dict[index]

	path_to_save_figures = path_to_save_figures_master + focus_status + '/' + case_ID + '/'
	if not(os.path.exists(path_to_save_figures)):
		os.makedirs(path_to_save_figures)
		print(path_to_save_figures+ ' created')
	path_to_save_figures2 = '_freq' + str(experimental_laser_frequency) + 'Hz_volt' + str(experimental_laser_voltage) + 'V_duty' + str(experimental_laser_duty) + '_'

	print('STARTING '+laser_to_analyse+' , '+case_ID)
	if not(os.path.exists(laser_to_analyse+'.npz')):
		print('missing .npz file, aborted')
		# return 0


	laser_to_analyse_power = power_interpolator(experimental_laser_voltage)

	try:
		laser_dict = coleval.read_IR_file(laser_to_analyse)
		laser_dict = np.load(laser_to_analyse+'.npz')
		laser_dict.allow_pickle=True
		laser_counts_filtered = laser_dict['laser_counts_filtered']
		full_saved_file_dict = dict(laser_dict)
	except:
		print('the file '+laser_to_analyse+'.npz'+' for some reason is bad, read failed, regenerated')
		laser_dict = coleval.read_IR_file(laser_to_analyse,force_regeneration=True)
		#this is the pre processing to remove the oscillation
		exec(open("/home/ffederic/work/analysis_scripts/scripts/Laser_data_preparation_only_one_file.py").read())
		laser_dict = np.load(laser_to_analyse+'.npz')
		laser_dict.allow_pickle=True
		full_saved_file_dict = dict(laser_dict)
		# full_saved_file_dict = dict([])
		# return 0

	# for type_of_calibration in ['NUC_plate','BB_source_w_window','BB_source_w/o_window']:
	for type_of_calibration in ['BB_source_w_window']:
		print(type_of_calibration)

		# full_saved_file_dict[type_of_calibration] = dict([])
		try:
			full_saved_file_dict[type_of_calibration] = full_saved_file_dict[type_of_calibration].all()
		except:
			pass

		# laser_counts, laser_digitizer_ID = coleval.separate_data_with_digitizer(laser_dict)
		time_of_experiment = cp.deepcopy(full_saved_file_dict['time_of_measurement'])	# microseconds
		mean_time_of_experiment = np.nanmean(time_of_experiment)
		laser_digitizer = cp.deepcopy(full_saved_file_dict['digitizer_ID'])
		if np.diff(time_of_experiment).max()>np.median(np.diff(time_of_experiment))*1.1:
			hole_pos = np.diff(time_of_experiment).argmax()
			if hole_pos<len(time_of_experiment)/2:
				time_of_experiment = time_of_experiment[hole_pos+1:]
				laser_digitizer = laser_digitizer[hole_pos+1:]
			else:
				time_of_experiment = time_of_experiment[:-(hole_pos+1)]
				laser_digitizer = laser_digitizer[:-(hole_pos+1)]
		time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)
		time_partial = (time_of_experiment-time_of_experiment[0])*1e-6
		laser_framerate = cp.deepcopy(full_saved_file_dict['FrameRate'])
		laser_int_time = cp.deepcopy(full_saved_file_dict['IntegrationTime'])

		preamble_4_prints_first = 'Case '+case_ID+', FR=%.3gHz, int time=%.3gms, ' %(laser_framerate,laser_int_time/1000) +'laser '+ focus_status +' , power=%.3gW, freq=%.3gHz, duty=%.3g\n' %(laser_to_analyse_power,experimental_laser_frequency,experimental_laser_duty)

		# Old way to calculate the temperature with the polynomial
		# temp = np.abs(parameters_available_int_time-laser_int_time/1000)<0.1
		# framerate = np.array(parameters_available_framerate)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
		# int_time = np.array(parameters_available_int_time)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
		# temp = np.array(parameters_available)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
		# print('parameters selected '+temp)
		#
		# # Load parameters
		# temp = pathparams+'/'+temp+'/numcoeff'+str(n)+'/average'
		# fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
		# params_dict=np.load(fullpathparams)
		# params = params_dict['coeff']
		# errparams = params_dict['errcoeff']
		# if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		# 	if limited_ROI == 'ff':
		# 		print('There is some issue with indexing. Data height,width is '+str([int(laser_dict['height']),int(laser_dict['width'])])+' but the indexing says it should be full frame ('+str([max_ROI[0][1]+1,max_ROI[1][1]+1])+')')
		# 		exit()
		# 	params = params[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1]
		# 	errparams = errparams[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1]

		# Selection of only the backgroud files with similar framerate and integration time
		# background_timestamps = [(np.nanmean(np.load(file+'.npz')['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		# background_temperatures = [(np.nanmean(np.load(file+'.npz')['SensorTemp_0'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		# background_counts = [(np.load(file+'.npz')['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		# background_counts_std = [(np.load(file+'.npz')['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		# relevant_background_files = [file for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		background_timestamps = [(np.nanmean(coleval.read_IR_file(file)['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		background_temperatures = [(np.nanmean(coleval.read_IR_file(file)['SensorTemp_0'])) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		background_counts = [(coleval.read_IR_file(file)['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		background_counts_std = [(coleval.read_IR_file(file)['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]
		relevant_background_files = [file for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]


		# Calculating the background image
		temp = np.sort(np.abs(background_timestamps-mean_time_of_experiment))[1]
		ref = np.arange(len(background_timestamps))[np.abs(background_timestamps-mean_time_of_experiment)<=temp]
		reference_background = (background_counts[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_counts[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
		reference_background_std = ((background_counts_std[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment))**2 + (background_counts_std[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))**2)**0.5/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
		reference_background_flat = np.nanmean(reference_background,axis=(1,2))
		reference_background_camera_temperature = (background_temperatures[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_temperatures[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
		reference_background_full_1,background_digitizer_ID = coleval.separate_data_with_digitizer(np.load(relevant_background_files[ref[0]]+'.npz'))
		reference_background_full_2,background_digitizer_ID = coleval.separate_data_with_digitizer(np.load(relevant_background_files[ref[1]]+'.npz'))
		if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
			flag_1 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_1]
			flag_2 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_2]
		else:
			flag_1 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_1]
			flag_2 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_2]
		bad_pixels_flag = np.add(flag_1,flag_2)

		# ani = coleval.movie_from_data(np.array([reference_background_full_2[0]]), laser_framerate, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]')
		# plt.pause(0.01)

		if False:	# I tried here to have an automatic recognition of the foil position and shape. I can't have a good enough contrast, so I abandon this route
			import cv2
			c=np.load(vacuum2[0]+'.npz')['data_time_avg_counts'][0]
			c-=c.min()
			c = c/c.max() * 255
			plt.figure();plt.imshow(c);plt.colorbar(),plt.pause(0.01)
			canny=cv2.Canny(np.uint8(c),20+c.min(),120+c.min())

			# cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
			#
			# for c in cnts:
			#	 cv2.drawContours(image,[c], 0, (0,255,0), 3)


			# d=cv2.Canny(np.uint8(c),30+c.min(),180+c.min())
			# plt.figure();plt.imshow(d);plt.colorbar()
			plt.figure();plt.imshow(canny);plt.colorbar(),plt.pause(0.01)

			# contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
			# cntrRect = []
			# for i in contours:
			# 		epsilon = 0.05*cv2.arcLength(i,True)
			# 		approx = cv2.approxPolyDP(i,epsilon,True)
			# 		# if len(approx) == 4:
			# 			# cv2.drawContours(roi,cntrRect,-1,(0,255,0),2)
			# 			# cv2.imshow('Roi Rect ONLY',roi)
			# 		cntrRect.append(approx)
			# 		plt.plot(approx[:,0],'--k',linewidth=4)

			e=cv2.HoughLinesP(canny, 1, np.pi/180, 70, np.array([]), 40, 10)
			for value in e:
				plt.plot(value[0][0::2],value[0][1::2],'--k',linewidth=4)
			plt.pause(0.01)
		else:	# I'm going to use the reference frames for foil position
			if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
				testrot = np.ones((max_ROI[0][1]+1,max_ROI[1][1]+1))*(-np.nanmean(reference_background[0]))
				testrot[limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = reference_background[0]
			else:
				testrot=reference_background[0]

			rotangle = foil_position_dict['angle'] #in degrees
			foilrot=rotangle*2*np.pi/360
			foilrotdeg=rotangle
			foilcenter = foil_position_dict['foilcenter']
			foilhorizw=0.09	# m
			foilvertw=0.07	# m
			foilhorizwpixel = foil_position_dict['foilhorizwpixel']
			foilvertwpixel=int((foilhorizwpixel*foilvertw)//foilhorizw)
			r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
			a=foilvertwpixel/np.cos(foilrot)
			tgalpha=np.tan(foilrot)
			delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
			foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
			foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
			foilxint=(np.rint(foilx)).astype('int')
			foilyint=(np.rint(foily)).astype('int')
			# plt.figure()
			# plt.imshow(testrot,'rainbow',origin='lower')
			# # plt.imshow(testrot,'rainbow',vmin=26., vmax=27.,origin='lower')
			# plt.plot(foilx,foily,'r')
			# plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
			# plt.grid()
			# plt.pause(0.01)

			plt.figure(figsize=(20, 10))
			plt.title(preamble_4_prints_first+'Laser spot location in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
			plt.imshow(testrot,'rainbow',origin='lower')
			plt.xlabel('Horizontal axis [pixles]')
			plt.ylabel('Vertical axis [pixles]')
			temp = np.sort(testrot[testrot>0])
			plt.clim(vmin=np.nanmean(temp[:max(20,int(len(temp)/20))]), vmax=np.nanmean(temp[-max(20,int(len(temp)/20)):]))
			# plt.clim(vmin=np.nanmin(testrot[testrot>0]), vmax=np.nanmax(testrot))
			plt.colorbar().set_label('counts [au]')
			plt.plot(foilxint,foilyint,'r')
			plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
			figure_index+=1
			plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
			plt.close('all')
			testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
			# plt.figure()
			# plt.imshow(testrotback,'rainbow',origin='lower')
			# plt.xlabel('Horizontal axis [pixles]')
			# plt.ylabel('Vertical axis [pixles]')
			# plt.clim(vmin=np.nanmin(testrot[testrot>0]), vmax=np.nanmax(testrot)) #this set the color limits
			precisionincrease=10
			dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
			dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
			dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
			dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
			dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
			dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
			dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
			foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype('int')
			foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype('int')
			foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype('int')
			# plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
			# plt.plot(foilxrot,foilyrot,'r')
			# plt.title('Foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
			# plt.colorbar().set_label('counts [au]')
			# plt.pause(0.01)

			foillx=min(foilxrot)
			foilrx=max(foilxrot)
			foilhorizwpixel=foilrx-foillx
			foildw=min(foilyrot)
			foilup=max(foilyrot)
			foilvertwpixel=foilup-foildw

			out_of_ROI_mask = np.ones_like(testrotback)
			out_of_ROI_mask[testrotback<np.nanmin(testrot[testrot>0])]=np.nan
			out_of_ROI_mask[testrotback>np.nanmax(testrot[testrot>0])]=np.nan
			a = generic_filter((testrotback),np.std,size=[19,19])
			out_of_ROI_mask[a>np.mean(a)]=np.nan



		# # before of the temperature conversion I'm better to remove the oscillation
		# # I try to use all the length of the record at first
		# laser_counts_filtered = [coleval.clear_oscillation_central2([counts],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_end=(len(counts)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False)[0] for counts in laser_counts]

		# here I should use the room temperature acquired by other means, but givend I didn't measured it I'll use the IR camera as a thermometer
		# reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_poly_multi_digitizer(reference_background,params,errparams,laser_digitizer_ID,number_cpu_available,counts_std=reference_background_std,report=0,parallelised=False)
		# ref_temperature = np.mean(reference_background_temperature)
		# ref_temperature_std = (np.sum(np.array(reference_background_temperature_std)**2)**0.5 / len(np.array(reference_background_temperature).flatten()))
		# I do it with the coefficients from the BB calibration

		# plt.figure()
		# plt.imshow(reference_background_temperature[0],'rainbow',origin='lower')
		# plt.colorbar().set_label('Temp [°C]')
		# plt.title('Reference frame for '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
		# plt.xlabel('Horizontal axis [pixles]')
		# plt.ylabel('Vertical axis [pixles]')
		# plt.pause(0.01)

		laser_counts_filtered = cp.deepcopy(full_saved_file_dict['laser_counts_filtered'])
		laser_digitizer = cp.deepcopy(full_saved_file_dict['digitizer_ID'])
		digitizer_ID = np.unique(laser_digitizer)

		try:
			test = full_saved_file_dict[type_of_calibration]
		except:
			full_saved_file_dict[type_of_calibration] = dict([])

		for emissivity in emissivity_range:
		# for emissivity in emissivity_range[7:]:

			# if not override:
			# 	try:
			# 		test = full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]
			# 		print(type_of_calibration + 'emissivity='+str(emissivity)+' skipped because data present')
			# 		continue
			# 	except:
			# 		pass

			try:
				test = full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]
			except:
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)] = dict([])


			if type_of_calibration == 'BB_source_w_window':
				temp = np.abs(parameters_available_int_time_BB-laser_int_time/1000)<0.1
				framerate = np.array(parameters_available_framerate_BB)[temp][np.abs(parameters_available_framerate_BB[temp]-laser_framerate).argmin()]
				int_time = np.array(parameters_available_int_time_BB)[temp][np.abs(parameters_available_framerate_BB[temp]-laser_framerate).argmin()]
				temp = np.array(parameters_available_BB)[temp][np.abs(parameters_available_framerate_BB[temp]-laser_framerate).argmin()]
				print('parameters selected '+temp)

				# Load parameters
				temp = pathparams_BB+'/'+temp+'/numcoeff'+str(n)
				fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
				params_dict=np.load(fullpathparams)
				params_dict.allow_pickle=True
				try:
					params_BB = params_dict['coeff2']
					errparams_BB = params_dict['errcoeff2']
				except:	# if not present it means that it is at a frequency for which I don't have the data to do the proper BB coefficients search
					print("not present it means that it is at a frequency for which I don't have the data to do the proper BB coefficients search. process aborted")
					return 0

				# emissivity correction
				params_BB[:,:,:,0] *= emissivity
				errparams_BB[:,:,:,0,:] *= emissivity
				errparams_BB[:,:,:,:,0] *= emissivity
			elif type_of_calibration == 'NUC_plate' or type_of_calibration == 'BB_source_w/o_window':
				temp = np.abs(parameters_available_int_time-laser_int_time/1000)<0.1
				framerate = np.array(parameters_available_framerate)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
				int_time = np.array(parameters_available_int_time)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
				temp = np.array(parameters_available)[temp][np.abs(parameters_available_framerate[temp]-laser_framerate).argmin()]
				print('parameters selected '+temp)

				# Load parameters
				temp = pathparams+'/'+temp+'/numcoeff'+str(n)
				fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
				params_dict=np.load(fullpathparams)
				params_dict.allow_pickle=True
				try:
					if type_of_calibration == 'NUC_plate':
						params = params_dict['coeff']
						errparams = params_dict['errcoeff']

						print('! ! ! WARNING ! ! !\nemissivity correction incompatible with polinomial fit\n! ! ! WARNING ! ! !')

						if emissivity!=1:
							continue
						# print(emissivity)
					elif type_of_calibration == 'BB_source_w/o_window':
						params_BB = params_dict['coeff4']
						errparams_BB = params_dict['errcoeff4']

						# emissivity correction
						params_BB[:,:,:,0] *= emissivity
						errparams_BB[:,:,:,0,:] *= emissivity
						errparams_BB[:,:,:,:,0] *= emissivity
				except:	# if not present it means that it is at a frequency for which I don't have the data to do the proper BB coefficients search
					print("not present it means that it is at a frequency for which I don't have the data to do the proper BB coefficients search. process aborted")
					return 0

			if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
				if limited_ROI == 'ff':
					print('There is some issue with indexing. Data height,width is '+str([int(full_saved_file_dict['height']),int(full_saved_file_dict['width'])])+' but the indexing says it should be full frame ('+str([max_ROI[0][1]+1,max_ROI[1][1]+1])+')')
					exit()
				params = params[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1]
				errparams = errparams[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1]

			# here I should use the room temperature acquired by other means, but givend I didn't measured it I'll use the IR camera as a thermometer
			if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
				reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_BB_multi_digitizer_no_reference(reference_background,reference_background_std,params_BB,errparams_BB,digitizer_ID,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
				if False:	# important: I need to fid first the emissivity to calculate the power!!
					emissivity_NUC = 1
					emissivity_FOIL = 0.6
					params_BB_with_emissivity = cp.deepcopy(params_BB)
					params_BB_with_emissivity[:,:,:,0] *= emissivity_FOIL/emissivity_NUC
					# reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_BB_multi_digitizer_no_reference(reference_background,reference_background_std,params_BB_with_emissivity,errparams_BB,digitizer_ID,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
					# np.mean(reference_background_temperature)
					ref_back_temp,ref_back_temp_std = coleval.count_to_temp_BB_multi_digitizer_no_reference(reference_background,reference_background_std,params_BB_with_emissivity,errparams_BB,digitizer_ID,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
					ref_back_temp = np.array([25.,25.])
					if False:	# just if needed
						laser_counts_filtered[0] = laser_counts_filtered[0][1:]
						laser_counts_filtered = np.array(laser_counts_filtered)
						laser_counts_filtered = np.array([laser_counts_filtered[0],laser_counts_filtered[1]])
					spot_location = np.unravel_index((laser_counts_filtered[0][np.max(laser_counts_filtered[0],axis=(1,2)).argmax()]).argmax(),laser_counts_filtered[0][0].shape)
					laser_temperature,laser_temperature_std = coleval.count_to_temp_BB_multi_digitizer(laser_counts_filtered[:,np.max(laser_counts_filtered[0],axis=(1,2)).argmax()-2:np.max(laser_counts_filtered[0],axis=(1,2)).argmax()],params_BB_with_emissivity,errparams_BB,digitizer_ID,reference_background=reference_background,reference_background_std=reference_background_std,ref_temperature=np.mean(ref_back_temp),ref_temperature_std=np.mean(ref_back_temp_std),wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
					dx=foilhorizw/(foilhorizwpixel-1)
					print(laser_temperature.max()-np.mean(ref_back_temp))
					# power = dx**2 *2*emissivity_FOIL*sigmaSB*np.sum((laser_temperature[0,-1,spot_location[0]-40:spot_location[0]+40,spot_location[1]-40:spot_location[1]+40]+273.15)**4 - (np.mean(ref_back_temp)+273.15)**4)
					power = dx**2 *2*emissivity_FOIL*sigmaSB*np.sum((laser_temperature[0,:,:]+273.15)**4 - (np.mean(ref_back_temp)+273.15)**4)
					print(power)
					while np.abs(power-laser_to_analyse_power)/laser_to_analyse_power>0.01:
						if power-laser_to_analyse_power>0:
							ref_back_temp += 0.1
						else:
							ref_back_temp -= 0.1
						laser_temperature,laser_temperature_std = coleval.count_to_temp_BB_multi_digitizer(laser_counts_filtered[:,np.max(laser_counts_filtered[0],axis=(1,2)).argmax()-2:np.max(laser_counts_filtered[0],axis=(1,2)).argmax()],params_BB_with_emissivity,errparams_BB,digitizer_ID,reference_background=reference_background,reference_background_std=reference_background_std,ref_temperature=np.mean(ref_back_temp),ref_temperature_std=np.mean(ref_back_temp_std),wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
						dx=foilhorizw/(foilhorizwpixel-1)
						# power = dx**2 *2*emissivity_FOIL*sigmaSB*np.sum((laser_temperature[0,-1,spot_location[0]-40:spot_location[0]+40,spot_location[1]-40:spot_location[1]+40]+273.15)**4 - (np.mean(ref_back_temp)+273.15)**4)
						power = dx**2 *2*emissivity_FOIL*sigmaSB*np.sum((laser_temperature[0,:,:]+273.15)**4 - (np.mean(ref_back_temp)+273.15)**4)
						print(power)
					print(np.mean(ref_back_temp))
					print(np.diff(laser_temperature[0],axis=0).max())
					# nope, it doesn't help, still cause unknown. the increase in dT due to low emissivity is not enough to compensate for the multimplyer of the powerBB
					# also accounting for the NUC emissivity doesn't help, as it actually reduces the temperature peak rather than increasing it
					# 2023/06/14 true. but the other 2 components are effected! let's see if it matter!
					# 2023/06/26 I calculated everything wrong until now. Tref was calculated using the camera, but this way it can become wild for emissivity far from 1 (already 0.7).
					# I need to fix reference temperature, otherwise it changes too much compared to temperatures that make sense (~20-25deg). for fixed emissivity, changing the background of 5 deg changes the power by ~10%
					# I will keep reference_background_temperature fixed at 22deg, so the real errer is likely +/-2.5%
				else:
					pass

			elif type_of_calibration == 'NUC_plate':
				reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_poly_multi_digitizer(reference_background,params,errparams,laser_digitizer_ID,number_cpu_available,counts_std=reference_background_std,report=0,parallelised=False)

			try:
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)] = full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)].all()
			except:
				pass

			for reference_temperature in reference_temperature_range:
			# for reference_temperature in reference_temperature_range[:2]:
				if not override:
					try:
						# test = np.load(laser_to_analyse+type_of_calibration + '_' + 'emissivity='+str(emissivity) +'_'+ 'T0='+str(reference_temperature)+'.npz')
						test = full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']
						print(type_of_calibration + 'emissivity='+str(emissivity) + ', T0='+str(reference_temperature)+' skipped because data present')
						continue
					except:
						pass


				# as per 2023/06/26 I will keep reference_background_temperature fixed at 22deg, so the real error on BB is likely +/-2.5%
				# 2023/07/04 no, I have to do a scan
				reference_background_temperature = np.ones_like(reference_background_temperature)*reference_temperature
				preamble_4_prints = preamble_4_prints_first +type_of_calibration + ' emissivity='+str(emissivity) + ' T0='+str(reference_temperature) +'\n'

				print(preamble_4_prints)

				reference_background_temperature_rot=rotate(reference_background_temperature,foilrotdeg,axes=(-1,-2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					reference_background_temperature_rot*=out_of_ROI_mask
					reference_background_temperature_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_full),reference_background_temperature_rot>np.nanmax(reference_background_temperature_full))]=0
				reference_background_temperature_crop=reference_background_temperature_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
				reference_background_temperature_std_rot=rotate(reference_background_temperature_std,foilrotdeg,axes=(-1,-2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					reference_background_temperature_std_rot*=out_of_ROI_mask
					reference_background_temperature_std_rot[np.logical_and(reference_background_temperature_std_rot<np.nanmin(reference_background_temperature_std_full),reference_background_temperature_std_rot>np.nanmax(reference_background_temperature_std_full))]=0
				reference_background_temperature_std_crop=reference_background_temperature_std_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
				if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
					ref_temperature = np.mean(reference_background_temperature_crop)
					ref_temperature_std = (np.sum(np.array(reference_background_temperature_std_crop)**2)**0.5 / len(np.array(reference_background_temperature_std_crop).flatten()))
				elif type_of_calibration == 'NUC_plate':
					ref_temperature = np.array(reference_background_temperature_crop)
					ref_temperature_std = np.array(reference_background_temperature_std_crop)

				# sanity check
				if np.sum([np.diff(time_of_experiment_digitizer_ID[i]).max()>np.median(np.diff(time_of_experiment_digitizer_ID[i]))*1.8 for i in range(len(laser_digitizer_ID))])>0:
					print('there is something wrong, there are holes in the time axis, process aborted')
					return 0

				# first I want to make sure that the frame for the 2 digitisers is the same
				len_dig0 = len(time_of_experiment_digitizer_ID[0])
				len_dig1 = len(time_of_experiment_digitizer_ID[1])
				if len_dig0>len_dig1:
					while len_dig0>len_dig1:
						print('digitiser 0 shortened of 1 frame')
						time_of_experiment_digitizer_ID[0] = time_of_experiment_digitizer_ID[0][:-1]
						laser_counts_filtered[0] = laser_counts_filtered[0][:-1]
						len_dig0 = len(time_of_experiment_digitizer_ID[0])
					laser_counts_filtered = np.stack(laser_counts_filtered)
				if len_dig0<len_dig1:
					while len_dig0<len_dig1:
						print('digitiser 1 shortened of 1 frame')
						time_of_experiment_digitizer_ID[1] = time_of_experiment_digitizer_ID[1][:-1]
						laser_counts_filtered[1] = laser_counts_filtered[1][:-1]
						len_dig1 = len(time_of_experiment_digitizer_ID[1])
					laser_counts_filtered = np.stack(laser_counts_filtered)
				laser_digitizer = [0 if time in time_of_experiment_digitizer_ID[0] else 1 for time in np.sort(np.array(time_of_experiment_digitizer_ID).flatten())]

				if False:	# I want to do a fiew tests to see the relationship between emissivity, Tref and the peal temperature profile from heat diffusion of a thin film
					import scipy as sp
					laser_counts_filtered_ = np.mean(laser_counts_filtered,axis=0)
					peak = np.unravel_index(laser_counts_filtered_[np.mean(laser_counts_filtered_,axis=(-1,-2)).argmax()].argmax(),laser_counts_filtered_[0].shape)
					counts = laser_counts_filtered_[:,peak[0],peak[1]].reshape((1,len(laser_counts_filtered_),1,1))
					emiss=0.9
					ref_temperature=20
					params = np.mean(params_BB,axis=0)[peak[0],peak[1]].reshape((1,1,1,4))*emiss
					errparams = np.mean(errparams_BB,axis=0)[peak[0],peak[1]].reshape((1,1,1,4,4))
					temp,trash = coleval.count_to_temp_BB_multi_digitizer(counts,params,errparams,[0],reference_background=np.mean(reference_background,axis=0)[peak[0],peak[1]].reshape((1,1,1)),reference_background_std=np.mean(reference_background_std,axis=0)[peak[0],peak[1]].reshape((1,1,1)),ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
					temp = temp[0,305:780,0,0] - ref_temperature
					def func_(t,t0,t_mult,csi,max):
						# print(t0,t_mult,csi,max)
						return max*(sp.special.expi(-csi*(1+4*t_mult*(t-t0))) - sp.special.expi(-csi))
					bds = [[-np.inf,0.,0.,0.],[0.,np.inf,np.inf,np.inf]]
					guess = [0.,1,0.1,1.]
					fit = curve_fit(func_, np.arange(len(temp))/(framerate/2),temp, p0=guess, bounds = bds, maxfev=100000000)
					# plt.figure()
					# plt.plot(np.arange(len(temp))/(framerate/2),temp)
					# # plt.plot(np.arange(len(temp))/(framerate/2),func_(np.arange(len(temp))/(framerate/2),*guess),':')
					# plt.plot(np.arange(len(temp))/(framerate/2),func_(np.arange(len(temp))/(framerate/2),*fit[0]),'--')
					# plt.pause(0.1)
					# print(fit[0])
					w0 = (2.5E-5/fit[0][1])**0.5
					hs_star = 2.5E-6/w0
					eta = 1*emiss*5.67E-8*4*((273+ref_temperature)**3)
					eta_star = eta*w0/71.6
					csi = eta_star/(2*hs_star)
					print([emiss,ref_temperature])
					print([csi,fit[0][2],csi/fit[0][2]])
					hs = eta/(2*Ptspecificheat*Ptdensity) * 1/fit[0][1] * 1/fit[0][2]
					print(hs)
					# it works! I need to use this
				else:
					pass

				if len(laser_counts_filtered[0])>500:	# this is to avoid going out of memory
					if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
						laser_temperature,laser_temperature_std = coleval.count_to_temp_BB_multi_digitizer(laser_counts_filtered[:,0*500:(0+1)*500],params_BB,errparams_BB,digitizer_ID,reference_background=reference_background,reference_background_std=reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
						for i in np.arange(1,np.ceil(len(laser_counts_filtered[0])/500)).astype(int):
							temp = coleval.count_to_temp_BB_multi_digitizer(laser_counts_filtered[:,i*500:(i+1)*500],params_BB,errparams_BB,digitizer_ID,reference_background=reference_background,reference_background_std=reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
							laser_temperature=np.append(laser_temperature,temp[0],axis=1)
							laser_temperature_std=np.append(laser_temperature_std,temp[1],axis=1)
					elif type_of_calibration == 'NUC_plate':
						reference_background_flat = np.mean(reference_background,axis=(1,2))
						laser_temperature,laser_temperature_std = coleval.count_to_temp_poly_multi_digitizer(laser_counts_filtered[:,0*500:(0+1)*500],params,errparams,laser_digitizer_ID,number_cpu_available,reference_background=reference_background,reference_background_std=reference_background_std,reference_background_flat=reference_background_flat,report=0)
						for i in np.arange(1,np.ceil(len(laser_counts_filtered[0])/500)).astype(int):
							temp = coleval.count_to_temp_poly_multi_digitizer(laser_counts_filtered[:,i*500:(i+1)*500],params,errparams,laser_digitizer_ID,number_cpu_available,reference_background=reference_background,reference_background_std=reference_background_std,reference_background_flat=reference_background_flat,report=0)
							laser_temperature=np.append(laser_temperature,temp[0],axis=1)
							laser_temperature_std=np.append(laser_temperature_std,temp[1],axis=1)
				else:
					if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
						laser_temperature,laser_temperature_std = coleval.count_to_temp_BB_multi_digitizer(laser_counts_filtered,params_BB,errparams_BB,digitizer_ID,reference_background=reference_background,reference_background_std=reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
					elif type_of_calibration == 'NUC_plate':
						laser_temperature,laser_temperature_std = coleval.count_to_temp_poly_multi_digitizer(laser_counts_filtered,params,errparams,laser_digitizer_ID,number_cpu_available,reference_background=reference_background,reference_background_std=reference_background_std,reference_background_flat=reference_background_flat,report=0)
				laser_counts_filtered_std = []
				for i in range(len(digitizer_ID)):
					laser_counts_filtered_std.append(coleval.estimate_counts_std(laser_counts_filtered[i],int_time=int_time))

				if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
					laser_temperature_minus_background = [laser_temperature[i]-ref_temperature for i in laser_digitizer_ID]
					laser_temperature_std_minus_background = [(laser_temperature_std[i]**2+ref_temperature_std**2)**0.5 for i in laser_digitizer_ID]
				elif type_of_calibration == 'NUC_plate':
					laser_temperature_minus_background = [laser_temperature[i]-reference_background_temperature[i] for i in laser_digitizer_ID]
					laser_temperature_std_minus_background = [(laser_temperature_std[i]**2+reference_background_temperature_std[i]**2)**0.5 for i in laser_digitizer_ID]
				if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
					BB_proportional,BB_proportional_std,constant_offset,constant_offset_std,photon_dict = coleval.calc_BB_coefficients_multi_digitizer(params_BB,errparams_BB,digitizer_ID,reference_background,reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=5.1,wavelength_bottom=1.5,inttime=int_time)
					photon_flux_over_temperature_interpolator = photon_dict['photon_flux_over_temperature_interpolator']

				# I can replace the dead pixels even after the transformation to temperature, given the flag comes from the background data
				# laser_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature)]
				laser_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_std)]
				laser_temperature_minus_background_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_minus_background)]
				laser_temperature_std_minus_background_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_std_minus_background)]
				# reference_background_temperature_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,reference_background_temperature)]
				# reference_background_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,reference_background_temperature_std)]
				laser_counts_filtered_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_counts_filtered_std)]
				if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
					BB_proportional_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,BB_proportional)]
					BB_proportional_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,BB_proportional_std)]
				reference_background_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,reference_background_std)]
				del laser_temperature,laser_temperature_std,laser_temperature_minus_background,laser_temperature_std_minus_background,laser_counts_filtered_std

				if False:	# this is wrong. I can t put together the temperatures. I need to calc the power independently and then sum the powers
					# I need to put together the data from the two digitizers
					# laser_temperature_full = [(laser_temperature_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_temperature_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)]
					laser_temperature_std_full = [(laser_temperature_std_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_temperature_std_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)]
					laser_temperature_minus_background_full = [(laser_temperature_minus_background_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if time in time_of_experiment_digitizer_ID[0] else (laser_temperature_minus_background_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time in time_of_experiment]
					laser_temperature_std_minus_background_full = [(laser_temperature_std_minus_background_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if time in time_of_experiment_digitizer_ID[0] else (laser_temperature_std_minus_background_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time in time_of_experiment]
					laser_counts_filtered_std_full = [(laser_counts_filtered_std_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if time in time_of_experiment_digitizer_ID[0] else (laser_counts_filtered_std_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time in time_of_experiment]
					del laser_temperature_minus_background_no_dead_pixels,laser_temperature_std_minus_background_no_dead_pixels,laser_counts_filtered_std_no_dead_pixels
				else:
					# I already made sure that the 2 digitisers have the same length of data, so I can treat them as an array
					laser_temperature_std_full = np.array(laser_temperature_std_no_dead_pixels)
					laser_temperature_minus_background_full = np.array(laser_temperature_minus_background_no_dead_pixels)
					laser_temperature_std_minus_background_full = np.array(laser_temperature_std_minus_background_no_dead_pixels)
					laser_counts_filtered_std_full = np.array(laser_counts_filtered_std_no_dead_pixels)
					time_of_experiment_digitizer_ID = np.array(time_of_experiment_digitizer_ID)
				time_partial = (time_of_experiment_digitizer_ID-np.min(time_of_experiment_digitizer_ID))*1e-6

				# rotation and crop
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					temp = np.ones((len(laser_temperature_minus_background_full[0]),len(laser_temperature_minus_background_full[1]),max_ROI[0][1]+1,max_ROI[1][1]+1))*(-np.nanmean(reference_background[0]))
					# temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_full
					# laser_temperature_full = cp.deepcopy(temp)
					temp[:,:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_std_full
					laser_temperature_std_full = cp.deepcopy(temp)
					temp[:,:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_minus_background_full
					laser_temperature_minus_background_full = cp.deepcopy(temp)
					temp[:,:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_std_minus_background_full
					laser_temperature_std_minus_background_full = cp.deepcopy(temp)
					temp[:,:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_counts_filtered_std_full
					laser_counts_filtered_std_full = cp.deepcopy(temp)
					temp = np.ones((len(laser_digitizer_ID),max_ROI[0][1]+1,max_ROI[1][1]+1))*(-np.nanmean(reference_background[0]))
					# temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = reference_background_temperature_no_dead_pixels
					# reference_background_temperature_no_dead_pixels = cp.deepcopy(temp)
					# temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = reference_background_temperature_std_no_dead_pixels
					# reference_background_temperature_std_no_dead_pixels = cp.deepcopy(temp)
					temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = BB_proportional_no_dead_pixels
					if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
						BB_proportional_no_dead_pixels = cp.deepcopy(temp)
						temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = BB_proportional_std_no_dead_pixels
						BB_proportional_std_no_dead_pixels = cp.deepcopy(temp)
						temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = reference_background_std_no_dead_pixels
					reference_background_std_no_dead_pixels = cp.deepcopy(temp)

				# laser_temperature_rot=rotate(laser_temperature_full,foilrotdeg,axes=(-1,-2))
				laser_temperature_std_rot=rotate(laser_temperature_std_full,foilrotdeg,axes=(-1,-2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					laser_temperature_std_rot*=out_of_ROI_mask
					laser_temperature_std_rot[np.logical_and(laser_temperature_std_rot<np.nanmin(laser_temperature_std_full),laser_temperature_std_rot>np.nanmax(laser_temperature_std_full))]=0
					# laser_temperature_rot*=out_of_ROI_mask
					# laser_temperature_rot[np.logical_and(laser_temperature_rot<np.nanmin(laser_temperature_full),laser_temperature_rot>np.nanmax(laser_temperature_full))]=np.nanmean(reference_background_temperature_no_dead_pixels)
				# laser_temperature_crop=laser_temperature_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
				laser_temperature_std_crop=laser_temperature_std_rot[:,:,foildw:foilup,foillx:foilrx].astype(np.float32)

				laser_temperature_minus_background_rot=rotate(laser_temperature_minus_background_full,foilrotdeg,axes=(-1,-2))
				laser_temperature_std_minus_background_rot=rotate(laser_temperature_std_minus_background_full,foilrotdeg,axes=(-1,-2))
				laser_counts_filtered_std_rot=rotate(laser_counts_filtered_std_full,foilrotdeg,axes=(-1,-2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					laser_temperature_std_minus_background_rot*=out_of_ROI_mask
					laser_temperature_std_minus_background_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
					laser_counts_filtered_std_rot*=out_of_ROI_mask
					laser_counts_filtered_std_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
					laser_temperature_minus_background_rot*=out_of_ROI_mask
					laser_temperature_minus_background_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
				laser_temperature_minus_background_crop=laser_temperature_minus_background_rot[:,:,foildw:foilup,foillx:foilrx].astype(np.float32)
				laser_temperature_std_minus_background_crop=laser_temperature_std_minus_background_rot[:,:,foildw:foilup,foillx:foilrx].astype(np.float32)
				laser_counts_filtered_std_crop=laser_counts_filtered_std_rot[:,:,foildw:foilup,foillx:foilrx].astype(np.float32)
				nan_ROI_mask = np.isfinite(np.nanmedian(laser_temperature_minus_background_crop[0,:10],axis=0))


				# reference_background_temperature_rot=rotate(reference_background_temperature_no_dead_pixels,foilrotdeg,axes=(-1,-2))
				# reference_background_temperature_std_rot=rotate(reference_background_temperature_std_no_dead_pixels,foilrotdeg,axes=(-1,-2))
				if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
					BB_proportional_rot=rotate(BB_proportional_no_dead_pixels,foilrotdeg,axes=(-1,-2))
					BB_proportional_std_rot=rotate(BB_proportional_std_no_dead_pixels,foilrotdeg,axes=(-1,-2))
				reference_background_std_rot=rotate(reference_background_std_no_dead_pixels,foilrotdeg,axes=(-1,-2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					# reference_background_temperature_std_rot*=out_of_ROI_mask
					# reference_background_temperature_std_rot[np.logical_and(reference_background_std_rot<np.nanmin(reference_background_std_rot),reference_background_std_rot>np.nanmax(reference_background_std_rot))]=0
					if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
						BB_proportional_rot*=out_of_ROI_mask
						BB_proportional_rot[np.logical_and(reference_background_std_rot<np.nanmin(reference_background_std_rot),reference_background_std_rot>np.nanmax(reference_background_std_rot))]=0
						BB_proportional_std_rot*=out_of_ROI_mask
						BB_proportional_std_rot[np.logical_and(reference_background_std_rot<np.nanmin(reference_background_std_rot),reference_background_std_rot>np.nanmax(reference_background_std_rot))]=0
					# reference_background_temperature_rot*=out_of_ROI_mask
					# reference_background_temperature_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=np.nanmean(reference_background_temperature_no_dead_pixels)
					reference_background_std_rot*=out_of_ROI_mask
					reference_background_std_rot[np.logical_and(reference_background_std_rot<np.nanmin(reference_background_std_rot),reference_background_std_rot>np.nanmax(reference_background_std_rot))]=0
				# reference_background_temperature_crop=reference_background_temperature_rot[:,foildw:foilup,foillx:foilrx]
				# reference_background_temperature_crop = np.nanmean(reference_background_temperature_crop,axis=0).astype(np.float32)
				# reference_background_temperature_std_crop=reference_background_temperature_std_rot[:,foildw:foilup,foillx:foilrx]
				# reference_background_temperature_std_crop = np.nanmean(reference_background_temperature_std_crop,axis=0).astype(np.float32)
				if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
					BB_proportional_crop=BB_proportional_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
					# BB_proportional_crop = np.nanmean(BB_proportional_crop,axis=0).astype(np.float32)
					BB_proportional_std_crop=BB_proportional_std_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
					# BB_proportional_std_crop = np.nanmean(BB_proportional_std_crop,axis=0).astype(np.float32)
				reference_background_std_crop=reference_background_std_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
				# reference_background_std_crop = np.nanmean(reference_background_std_crop,axis=0).astype(np.float32)
				del laser_temperature_minus_background_rot,laser_temperature_std_minus_background_rot

				# plt.figure()
				# plt.imshow(laser_temperature_crop[0],'rainbow',origin='lower')
				# plt.colorbar().set_label('Temp [°C]')
				# plt.title('Only foil in frame '+str(0)+' in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
				# plt.xlabel('Horizontal axis [pixles]')
				# plt.ylabel('Vertical axis [pixles]')
				# plt.pause(0.01)

				# FOIL PROPERTY ADJUSTMENT
				if True:	# spatially resolved foil properties from Japanese producer
					foilemissivityscaled=emissivity*resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]	# this is effectively corrected for the ratio of foil emissivity / nuc plate emissivity
					foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
					conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
					reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
				elif False:	# homogeneous foil properties from foil experiments
					foilemissivityscaled=1*np.ones((foilvertwpixel,foilhorizwpixel))
					foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel,foilhorizwpixel))
					conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel,foilhorizwpixel))
					reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel,foilhorizwpixel))

				# plt.figure()
				# plt.title('Foil emissivity scaled to camera pixels')
				# plt.imshow(foilemissivityscaled,'rainbow',origin='lower')
				# plt.xlabel('Foil reference axis [pixles]')
				# plt.ylabel('Foil reference axis [pixles]')
				# plt.colorbar().set_label('Emissivity [adimensional]')
				#
				# plt.figure()
				# plt.title('Foil thickness scaled to camera pixels')
				# plt.imshow(1000000*foilthicknessscaled,'rainbow',origin='lower')
				# plt.xlabel('Foil reference axis [pixles]')
				# plt.ylabel('Foil reference axis [pixles]')
				# plt.colorbar().set_label('Thickness [micrometer]')
				# plt.pause(0.01)

				laser_framerate = cp.deepcopy(full_saved_file_dict['FrameRate'])/2	# because of the 2 digitiser
				dt=1/laser_framerate
				dx=foilhorizw/(foilhorizwpixel-1)
				temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,1,2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					temp[np.isfinite(temp)] = [np.nanmedian(temp)]*2 + temp[np.isfinite(temp)][2:-2].tolist() + [np.nanmedian(temp)]*2
				temp1 = temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)
				horizontal_loc = (np.arange(len(temp))[temp1])[temp[temp1].argmax()]
				dhorizontal_loc = np.nansum(temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp))
				temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,1,3))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					temp[np.isfinite(temp)] = [np.nanmedian(temp)]*2 + temp[np.isfinite(temp)][2:-2].tolist() + [np.nanmedian(temp)]*2
				temp1 = temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)
				vertical_loc = (np.arange(len(temp))[temp1])[temp[temp1].argmax()]
				dvertical_loc = np.nansum(temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp))
				# dr = np.nanmean([dhorizontal_loc,dvertical_loc])
				if focus_status == 'focused':
					dr = np.nanmean([dhorizontal_loc,dvertical_loc])*0.4
				elif focus_status == 'partially_defocused':
					dr = np.nanmean([dhorizontal_loc,dvertical_loc])*0.5
				elif focus_status == 'fully_defocused':
					dr = np.nanmean([dhorizontal_loc,dvertical_loc])*0.6
				else:
					print('focus type '+focus_status+" unknown, used 'fully_defocused' settings")
					dr = np.nanmean([dhorizontal_loc,dvertical_loc])
				# dr_total_power = np.nanmean([dhorizontal_loc,dvertical_loc])*1.7
				horizontal_coord = np.arange(np.shape(laser_temperature_minus_background_crop)[3])
				vertical_coord = np.arange(np.shape(laser_temperature_minus_background_crop)[2])
				horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
				# select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr**2)[1:-1,1:-1]

				if False:	# I add this for the plots in the paper
					start_temp_rise = np.diff(np.max(np.mean(laser_temperature_minus_background_crop,axis=0),axis=(1,2))).argmax()+1-2
					plt.figure(figsize=(10, 5))
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc:,horizontal_loc])
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc:0:-1,horizontal_loc])
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc,horizontal_loc:])
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc,horizontal_loc:0:-1])

					for delta_time in [0,10,20,40,60,90,130,170,210]:
						delta_time = round(delta_time/1000*laser_framerate)
						temp = []
						len_min = np.inf
						temp.append(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise+delta_time][vertical_loc:,horizontal_loc])
						len_min = min(len_min,len(temp[-1]))
						temp.append(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise+delta_time][vertical_loc:0:-1,horizontal_loc])
						len_min = min(len_min,len(temp[-1]))
						temp.append(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise+delta_time][vertical_loc,horizontal_loc:])
						len_min = min(len_min,len(temp[-1]))
						temp.append(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise+delta_time][vertical_loc,horizontal_loc:0:-1])
						len_min = min(len_min,len(temp[-1]))
						temp[0] = temp[0][:len_min]
						temp[1] = temp[1][:len_min]
						temp[2] = temp[2][:len_min]
						temp[3] = temp[3][:len_min]
						temp = np.median(temp,axis=0)
						plt.plot(np.arange(len_min)*dx*1000,temp,label='pulse %+.0fms' %((delta_time-1)/laser_framerate*1000))
					plt.legend(loc='best', fontsize='small')
					plt.grid()
					plt.xlim(left=-0.1,right=8)
					plt.ylim(bottom=-0.1,top=2)
					plt.xlabel('distance from laser spot centre [mm]')
					plt.ylabel('temperature increase [K]')
					plt.axvline(x=0,color='k')
					plt.axvline(x=9*dx*1000,color='y',linestyle='--')
					plt.axvline(x=20*dx*1000,color='y',linestyle='--')
					plt.title(preamble_4_prints+'Laser spot location in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm prelim radious %.3gmm\n' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
					plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG_for_paper_1'+'.eps', bbox_inches='tight')






				plt.figure(figsize=(20, 10))
				temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,1,2))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					temp[np.isfinite(temp)] = [np.nanmedian(temp)]*2 + temp[np.isfinite(temp)][2:-2].tolist() + [np.nanmedian(temp)]*2
				plt.plot(np.arange(len(temp))-horizontal_loc,(temp-np.nanmin(temp))/(np.nanmax(temp)-np.nanmin(temp)),'b',label='horizontal')
				plt.plot((np.arange(len(temp))-horizontal_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)],[0.5]*np.nansum(temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)),'--b')
				plt.plot([(np.arange(len(temp))-horizontal_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)][0]]*2,[0,0.5],'--b')
				plt.plot([(np.arange(len(temp))-horizontal_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)][-1]]*2,[0,0.5],'--b')
				temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,1,3))
				if not (full_saved_file_dict['height']==max_ROI[0][1]+1 and full_saved_file_dict['width']==max_ROI[1][1]+1):
					temp[np.isfinite(temp)] = [np.nanmedian(temp)]*2 + temp[np.isfinite(temp)][2:-2].tolist() + [np.nanmedian(temp)]*2
				plt.plot(np.arange(len(temp))-vertical_loc,(temp-np.nanmin(temp))/(np.nanmax(temp)-np.nanmin(temp)),'r',label='vertical')
				plt.plot((np.arange(len(temp))-vertical_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)],[0.5]*np.nansum(temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)),'--r')
				plt.plot([(np.arange(len(temp))-vertical_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)][0]]*2,[0,0.5],'--r')
				plt.plot([(np.arange(len(temp))-vertical_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)][-1]]*2,[0,0.5],'--r')
				plt.plot([0]*2,[0,1],'--k')
				plt.plot([-dr,-dr,dr,dr],[0,1,1,0],'--k')
				plt.title(preamble_4_prints+'Laser spot location in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm prelim radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
				plt.xlabel('axis [pixels]')
				plt.ylabel('Temperature increase average in 1 direction [°C]')
				plt.legend(loc='best', fontsize='small')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')


				# successfull_plot = False
				# tries = 0
				# while (successfull_plot==True or tries>30):
				# 	try:
				h_coordinates,v_coordinates = np.meshgrid(np.arange(np.shape(laser_temperature_minus_background_crop)[-1]+1)*dx-dx/2,np.arange(np.shape(laser_temperature_minus_background_crop)[-2]+1)*dx-dx/2)
				plt.figure(figsize=(20, 10))
				plt.pcolor(h_coordinates,v_coordinates,np.nanmean(laser_temperature_minus_background_crop,axis=(0,1)),cmap='rainbow')
				plt.colorbar().set_label('Temp [°C]')
				plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--',label='guess for laser size')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--')
				# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
				# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--')
				plt.title(preamble_4_prints+'Average temperature increment in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm prelim radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
				plt.xlabel('Horizontal axis [m]')
				plt.ylabel('Vertical axis [m]')
				plt.legend(loc='best', fontsize='small')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')

				plt.figure(figsize=(20, 10))
				temp = np.nanmean(np.sort(np.mean(laser_temperature_minus_background_crop,axis=0),axis=0)[-int(len(laser_temperature_minus_background_crop[0])*experimental_laser_duty):],axis=0)
				plt.pcolor(h_coordinates,v_coordinates,temp,cmap='rainbow')
				plt.colorbar().set_label('Temp [°C]')
				plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'k--',label='guess for laser size')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'k--')
				# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
				# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--')
				plt.title(preamble_4_prints+'Hottest fraction of temperature increment in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm prelim radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
				plt.xlabel('Horizontal axis [pixles]')
				plt.ylabel('Vertical axis [pixles]')
				plt.legend(loc='best', fontsize='small')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')
					# 	# plt.pause(0.01)
					# 	successfull_plot = True
					# except:
					# 	tries+=1

				# this correction is necessay to avoid negative BB components when the temperature is even slightly lower then the reference.
				limit = 500
				test=0
				while True:
					select1 = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>(limit)**2)
					if np.nansum(nan_ROI_mask[select1])>200:
						break
					else:
						limit -= 10
				frames_for_one_pulse = laser_framerate/experimental_laser_frequency
				minimum_ON_period = 2	# seconds
				time_ON_after_SS = int(frames_for_one_pulse*experimental_laser_duty - minimum_ON_period*laser_framerate)
				ref_footprint = int(np.max([time_ON_after_SS,min(np.ceil(frames_for_one_pulse*(1-experimental_laser_duty)/3),4)]))
				footprint = np.concatenate([np.ones((ref_footprint)),np.zeros((ref_footprint))])
				peak_temp_filtered_1 = generic_filter(np.nanmax(laser_temperature_minus_background_crop,axis=(0,-1,-2)),np.mean,footprint=footprint)
				peaks = find_peaks(peak_temp_filtered_1,distance=max(1,frames_for_one_pulse*0.9))[0]
				peaks_n = find_peaks(-peak_temp_filtered_1,distance=max(1,frames_for_one_pulse*0.9))[0]
				best_peak_n = peaks_n[np.abs(peaks_n-len(laser_temperature_minus_background_crop[0])//2).argmin()]
				select_ref_time = np.logical_and(np.arange(len(laser_temperature_minus_background_crop[0]))<=best_peak_n,np.arange(len(laser_temperature_minus_background_crop[0]))>best_peak_n-ref_footprint)
				temp = np.mean(laser_temperature_minus_background_crop,axis=0)
				select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<10**2)
				laser_temperature_minus_background_crop_max = np.max(temp[:,select],axis=(-1))
				ref_laser_temperature_minus_background_crop = np.mean(temp[select_ref_time],axis=0)
				test = []
				for value in np.arange(5,limit-10):
					select1 = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>value**2)
					select2 = np.logical_and( (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=(value+10)**2) , select1 )
					test.append(np.nanmean(ref_laser_temperature_minus_background_crop[select2]))
				if np.argmin(test)==len(test)-1:
					print("offsett of the temperature skipped as the temperature is still dropping at the edge of the image, a clear minimum can't be found, so it is limited to above 0")
					rise_of_absolute_temperature_difference = max(0,-np.nanmin(test))
				else:
					rise_of_absolute_temperature_difference = -np.nanmin(test)
				plt.figure(figsize=(20, 10))
				plt.plot(np.arange(5,limit-10),test)
				plt.axhline(y=np.nanmin(test),linestyle='--',color='k')
				plt.grid()
				plt.title(preamble_4_prints+'search for the temperature factor to add or subtract to have positive BB power\n %.3gK found' %(rise_of_absolute_temperature_difference))
				plt.xlabel('pixels from the center of the laser spot [pixels]')
				plt.ylabel('average temperature increase\nin a ring 4 pixels wide [K]')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')
				laser_temperature_minus_background_crop += rise_of_absolute_temperature_difference
				laser_temperature_minus_background_crop = np.float32(laser_temperature_minus_background_crop)

				plt.figure(figsize=(20, 10))
				plt.plot(np.mean(time_partial,axis=0),laser_temperature_minus_background_crop_max)
				plt.grid()
				plt.title(preamble_4_prints+'peak foil temperature increase')
				plt.xlabel('time [s]')
				plt.ylabel('peak temperature increase [K]')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')

				if False:	# I try to add a little bit of smoothing, but only for the elements of the power balance
					laser_temperature_minus_background_crop_filtered = [generic_filter(value,np.mean,size=[3,3,3]) for value in laser_temperature_minus_background_crop]
				else:
					laser_temperature_minus_background_crop_filtered = cp.deepcopy(laser_temperature_minus_background_crop)

				# dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std = coleval.calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,laser_temperature_minus_background_crop,ref_temperature,time_partial,dx,laser_counts_filtered_std_crop,BB_proportional_crop,BB_proportional_std_crop,reference_background_std_crop,laser_temperature_std_crop,nan_ROI_mask)
				# temp_BBrad,temp_diffusion,temp_timevariation,temp_powernoback,temp_BBrad_std,temp_diffusion_std,temp_timevariation_std,temp_powernoback_std = coleval.calc_temp_to_power_BB_2(dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std,nan_ROI_mask,1,1,1,Ptthermalconductivity)
				output1 = []
				for i in range(len(laser_digitizer_ID)):
					if type_of_calibration == 'BB_source_w_window' or type_of_calibration == 'BB_source_w/o_window':
						if i==0:
							attempt = 0
							success = False
							while attempt<20 and success==False:	# this while is just to retry when I get the memory error
								attempt += 1
								try:
									output1.append(coleval.calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,laser_temperature_minus_background_crop_filtered[i],ref_temperature,time_partial[i],dx,laser_counts_filtered_std_crop[i],BB_proportional_crop[i],BB_proportional_std_crop[i],reference_background_std_crop[i],laser_temperature_std_crop[i],nan_ROI_mask,return_grid_laplacian=True))
									success = True
								except:
									success = False
							grid_laplacian = output1[0][-1]
							output1[0] = output1[0][:-1]
						else:
							output1.append(coleval.calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,laser_temperature_minus_background_crop_filtered[i],ref_temperature,time_partial[i],dx,laser_counts_filtered_std_crop[i],BB_proportional_crop[i],BB_proportional_std_crop[i],reference_background_std_crop[i],laser_temperature_std_crop[i],nan_ROI_mask,grid_laplacian=grid_laplacian))
					elif type_of_calibration == 'NUC_plate':
						averaged_params = np.mean(params,axis=(1,2))
						averaged_errparams = np.mean(errparams,axis=(1,2))
						output1.append(coleval.calc_temp_to_power_1(dx,dt,averaged_params[i],laser_counts_filtered_std_crop[i],averaged_errparams[i],1,1,laser_temperature_std_crop[i],laser_temperature_minus_background_crop_filteredqs[i],int_time*1e3,nan_ROI_mask,ref_temperature[i]))

				try:
					del grid_laplacian
				except:
					pass
				# output1 = [coleval.calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,laser_temperature_minus_background_crop[i],ref_temperature,time_partial[i],dx,laser_counts_filtered_std_crop[i],BB_proportional_crop[i],BB_proportional_std_crop[i],reference_background_std_crop[i],laser_temperature_std_crop[i],nan_ROI_mask) for i in range(len(laser_digitizer_ID))]
				dTdt = [[],[]]
				dTdt_std = [[],[]]
				d2Tdxy = [[],[]]
				d2Tdxy_std = [[],[]]
				negd2Tdxy = [[],[]]
				negd2Tdxy_std = [[],[]]
				T4_T04 = [[],[]]
				T4_T04_std = [[],[]]
				dTdt[0],dTdt_std[0],d2Tdxy[0],d2Tdxy_std[0],negd2Tdxy[0],negd2Tdxy_std[0],T4_T04[0],T4_T04_std[0] = output1[0]
				dTdt[1],dTdt_std[1],d2Tdxy[1],d2Tdxy_std[1],negd2Tdxy[1],negd2Tdxy_std[1],T4_T04[1],T4_T04_std[1] = output1[1]
				# output1 = [coleval.calc_temp_to_power_BB_2(dTdt[i],dTdt_std[i],d2Tdxy[i],d2Tdxy_std[i],negd2Tdxy[i],negd2Tdxy_std[i],T4_T04[i],T4_T04_std[i],nan_ROI_mask,foilemissivityscaled,foilthicknessscaled,reciprdiffusivityscaled,Ptthermalconductivity) for i in range(len(laser_digitizer_ID))]
				# output1 = [coleval.calc_temp_to_power_BB_2(dTdt[i],dTdt_std[i],d2Tdxy[i],d2Tdxy_std[i],negd2Tdxy[i],negd2Tdxy_std[i],T4_T04[i],T4_T04_std[i],nan_ROI_mask,sample_properties['emissivity'],sample_properties['thickness'],1/sample_properties['diffusivity'],Ptthermalconductivity) for i in range(len(laser_digitizer_ID))]
				output1 = [coleval.calc_temp_to_power_BB_2(dTdt[i],dTdt_std[i],d2Tdxy[i],d2Tdxy_std[i],negd2Tdxy[i],negd2Tdxy_std[i],T4_T04[i],T4_T04_std[i],nan_ROI_mask,1,1,1/1,Ptthermalconductivity) for i in range(len(laser_digitizer_ID))]
				temp_BBrad = [[],[]]
				temp_diffusion = [[],[]]
				temp_timevariation = [[],[]]
				temp_powernoback = [[],[]]
				temp_BBrad_std = [[],[]]
				temp_diffusion_std = [[],[]]
				temp_timevariation_std = [[],[]]
				temp_powernoback_std = [[],[]]
				temp_BBrad[0],temp_diffusion[0],temp_timevariation[0],temp_powernoback[0],temp_BBrad_std[0],temp_diffusion_std[0],temp_timevariation_std[0],temp_powernoback_std[0] = output1[0]
				temp_BBrad[1],temp_diffusion[1],temp_timevariation[1],temp_powernoback[1],temp_BBrad_std[1],temp_diffusion_std[1],temp_timevariation_std[1],temp_powernoback_std[1] = output1[1]

				# temp_powernoback = [(sample_properties['thickness']*temp_diffusion[i] + (1/sample_properties['diffusivity'])*sample_properties['thickness']*temp_timevariation[i] + sample_properties['emissivity']*temp_BBrad[i]).astype(np.float32) for i in range(len(laser_digitizer_ID))]

				temp_BBrad = np.mean(temp_BBrad,axis=0)	# this is already multiplied by 2
				temp_diffusion = np.mean(temp_diffusion,axis=0)
				temp_timevariation = np.mean(temp_timevariation,axis=0)
				temp_powernoback = np.mean(temp_powernoback,axis=0)
				temp_BBrad_std = 0.5*(np.sum(np.power(temp_BBrad_std,2),axis=0)**0.5)
				temp_diffusion_std = 0.5*(np.sum(np.power(temp_diffusion_std,2),axis=0)**0.5)
				temp_timevariation_std = 0.5*(np.sum(np.power(temp_timevariation_std,2),axis=0)**0.5)
				temp_powernoback_std = 0.5*(np.sum(np.power(temp_powernoback_std,2),axis=0)**0.5)
				time_partial = np.mean(time_partial,axis=0)[1:-1]

				temp_powernoback = sample_properties['thickness']*temp_diffusion + ((1/sample_properties['diffusivity'])*sample_properties['thickness'])*temp_timevariation + sample_properties['emissivity']*temp_BBrad

				# del dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std,temp_powernoback_std

				def power_vs_space_sampling_explorer():
					from uncertainties import ufloat
					area_multiplier = np.flip(np.array([4,3,2.5,2,1.7,1.5,1.3,1.15,1,0.9,0.8,0.7,0.6,0.55,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.08]),axis=0)
					number_of_pixels = []
					all_dr = []
					fitted_powers = []
					for dr in (np.nanmean([dhorizontal_loc,dvertical_loc])*area_multiplier):
						select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr**2)[1:-1,1:-1]
						number_of_pixels.append(np.nansum(select))
						totalpower=np.nansum(sample_properties['thickness']*temp_diffusion[:,select],axis=(-1))#*Ptthermalconductivity
						totalpower_filtered_1 = generic_filter(totalpower[:int(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/15//2*2+1))])
						fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
						fitted_powers.append(fitted_power_2)
						all_dr.append(dr)
					return dict([('number_of_pixels',np.array(number_of_pixels)),('fitted_powers',np.array(fitted_powers)),('all_dr',np.array(all_dr))])
				power_vs_space_sampling = power_vs_space_sampling_explorer()

				dr = (power_vs_space_sampling['all_dr'])[nominal_values(power_vs_space_sampling['fitted_powers'][:,1][power_vs_space_sampling['all_dr']<=dr]).argmax()]

				if False:	# other plots for the paper. i think i want to show that the area where the diffusion components cancels out grows in time
					start_temp_rise = np.diff(np.max(np.mean(laser_temperature_minus_background_crop,axis=0),axis=(1,2))).argmax()+1-2-1
					plt.figure(figsize=(10, 5))
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc:,horizontal_loc])
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc:0:-1,horizontal_loc])
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc,horizontal_loc:])
					# plt.plot(np.mean(laser_temperature_minus_background_crop,axis=0)[start_temp_rise][vertical_loc,horizontal_loc:0:-1])

					for delta_time in [0,10,20,40,60,90,130,170,210]:
						delta_time = round(delta_time/1000*laser_framerate)

						temp = []
						for spatial_range in np.arange(50):
							temp.append(np.sum(temp_diffusion[start_temp_rise+delta_time,((vertical_coord[1:-1,1:-1]-vertical_loc)**2 + (horizontal_coord[1:-1,1:-1]-horizontal_loc)**2)**0.5<=spatial_range]))
						plt.plot(np.arange(len(temp))*dx*1000,np.array(temp)*2.5/1000000*Ptthermalconductivity*(dx**2),label='pulse %+.0fms' %((delta_time-1)/laser_framerate*1000))
					plt.legend(loc='best', fontsize='small')
					plt.grid()
					plt.xlim(left=-0.1,right=8)
					# plt.ylim(bottom=-0.1,top=2)
					plt.xlabel('Radius of the integration area [mm]')
					plt.ylabel(r'integral of $P_{\Delta T}$ [W]')
					plt.axvline(x=0,color='k')
					plt.axvline(x=9*dx*1000,color='y',linestyle='--')
					plt.axvline(x=20*dx*1000,color='y',linestyle='--')
					plt.title(preamble_4_prints+'Laser spot location in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm prelim radious %.3gmm\n' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
					plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG_for_paper_2'+'.eps', bbox_inches='tight')


				if focus_status == 'focused':
					dr_total_power_minimum = 9	# this is slightly larger thanthe size of the pinhole
					# dr_total_power = 17
				elif focus_status == 'partially_defocused':
					dr_total_power_minimum = 9
					# dr_total_power = 20
				elif focus_status == 'fully_defocused':
					dr_total_power_minimum = 9
					# dr_total_power = 25
				else:
					print('focus type '+focus_status+" unknown, used 'fully_defocused' settings")
					# dr_total_power = 16
					dr_total_power_minimum = 9
				dr_total_power = 20	# fixed at 50 so it's the same for all


				limit = 210
				test=0
				while True:
					select1 = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>(limit)**2)[1:-1,1:-1]
					if np.nansum(nan_ROI_mask[1:-1,1:-1][select1])>300:
						break
					else:
						limit -= 10
				footprint = np.concatenate([np.ones((ref_footprint)),np.zeros((ref_footprint))])
				peak_temp_filtered_1 = generic_filter(np.nanmax(np.mean(laser_temperature_minus_background_crop,axis=0),axis=(-1,-2)),np.mean,footprint=footprint)
				peaks = find_peaks(peak_temp_filtered_1,distance=max(1,frames_for_one_pulse*0.9))[0]
				best_peak = peaks[np.abs(peaks-len(peak_temp_filtered_1)).argmin()]-1
				select_ref_time = np.logical_and(np.arange(len(temp_BBrad))<=best_peak,np.arange(len(temp_BBrad))>best_peak-ref_footprint)
				fig, ax = plt.subplots( 2,1,figsize=(10, 14), squeeze=False, sharex=True)
				time_averaged_BBrad_over_duty = np.nanmean(sample_properties['emissivity']*temp_BBrad[select_ref_time],axis=0)
				test = []
				for value in np.arange(2,limit-10):
					select2 = np.logical_and( (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=(value+5)**2)[1:-1,1:-1] , (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>value**2)[1:-1,1:-1] )
					test.append(np.nanmean(time_averaged_BBrad_over_duty[select2]))#-np.nanmean(time_averaged_BBrad_over_duty[select1]))
				BBtreshold = min(0.05,2*np.max(np.abs(np.array(test)[-5:])))
				# BBtreshold = min(0.05,3*np.max(np.abs(np.array(test)[-5:])))
				# BBtreshold = min(0.02,3*np.max(np.abs(np.array(test)[-5:])))	# this is how it was before
				dr_total_power_BB = np.arange(2,limit-10)[np.abs(np.array(test)-BBtreshold).argmin()]
				ax[0,0].plot(np.arange(2,limit-10)*dx,test)
				ax[0,0].axhline(y=BBtreshold,linestyle='--',color='k',label='treshold')
				ax[0,0].axvline(x=dr_total_power_BB*dx,linestyle='--',color='r',label='detected')
				ax[0,0].axvline(x=dr_total_power_minimum*dx,linestyle='--',color='b',label='minimum')
				ax[0,0].axvline(x=dr_total_power*dx,linestyle='--',color='y',label='maximum')
				ax[0,0].set_ylim(top=min(np.nanmax(test),BBtreshold*5),bottom=np.nanmin(test))
				ax[0,0].set_xlim(left=0)
				ax[0,0].set_ylabel('black body')
				ax[0,0].legend(loc='best', fontsize='x-small')
				ax[0,0].grid()
				# temp = np.nanmax(laser_temperature_minus_background_crop[1:-1],axis=(1,2))
				time_averaged_diffusion_over_duty = np.nanmean(sample_properties['thickness']*temp_diffusion[select_ref_time],axis=0)
				test = []
				for value in np.arange(2,limit-10):
					select1 = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>(value)**2)[1:-1,1:-1]
					select2 = np.logical_and( (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=(value+4)**2)[1:-1,1:-1] , (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>value**2)[1:-1,1:-1] )
					test.append(np.nanmean(time_averaged_diffusion_over_duty[select1]))
				dr_total_power_diff = np.arange(2,limit-10)[np.abs((test[:40]+2*np.max(np.abs(test[40:])))).argmin()]
				ax[1,0].plot(np.arange(2,limit-10)*dx,test)
				ax[1,0].axhline(y=-2*np.nanmax(np.abs(test[40:])),linestyle='--',color='k')
				ax[1,0].axvline(x=dr_total_power_diff*dx,linestyle='--',color='r')
				ax[1,0].axvline(x=dr_total_power_minimum*dx,linestyle='--',color='b')
				ax[1,0].axvline(x=dr_total_power*dx,linestyle='--',color='y',label='maximum')
				ax[1,0].set_ylim(bottom=-4*np.nanmax(np.abs(test[40:])),top=np.nanmax(test))
				ax[1,0].set_xlim(left=0)
				ax[1,0].set_ylabel('diffusion')
				ax[1,0].set_xlabel('radious of power sum [mm]')
				ax[1,0].grid()
				# dr_total_power = np.max([dr_total_power_diff,dr_total_power_minimum,dr_total_power_BB])
				# dr_total_power = 30	# fixed at 50 so it's the same for all
				# temp = np.logical_and(power_vs_space_sampling['all_dr']>dr*1.2,power_vs_space_sampling['all_dr']<np.max(power_vs_space_sampling['all_dr']))
				# dr_total_power = (power_vs_space_sampling['all_dr'][temp])[nominal_values(power_vs_space_sampling['fitted_powers'][:,1][temp]).argmin()]
				# dr_total_power = dr*2
				select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr_total_power**2)[1:-1,1:-1]
				select_small = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr_total_power_minimum**2)[1:-1,1:-1]
				fig.suptitle(preamble_4_prints+'Search for power sum size in '+laser_to_analyse+'\nfound %.3gmm with  %.3gK rise of absolute temperature difference' %(dr_total_power*dx,rise_of_absolute_temperature_difference))
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')

				def power_vs_time_sampling_explorer():
					from uncertainties import ufloat
					time_multiplier = np.arange(11)/10
					number_of_pulses = []
					fitted_powers = []
					totalpower=np.multiply(np.nansum(temp_powernoback[:,select],axis=(-1)),dx**2)
					for i_number_of_pulses in range(1,int(len(temp_powernoback)/frames_for_one_pulse)+1):
						number_of_pulses.append(i_number_of_pulses)
						totalpower_filtered_1 = generic_filter(totalpower[:int(frames_for_one_pulse*i_number_of_pulses)],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/15//2*2+1))])
						fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
						fitted_powers.append(fitted_power_2)
					return dict([('number_of_pulses',np.array(number_of_pulses)),('fitted_powers',np.array(fitted_powers))])
				power_vs_time_sampling = power_vs_time_sampling_explorer()
				del temp_powernoback,power_vs_space_sampling_explorer,power_vs_time_sampling_explorer

				partial_BBrad=np.multiply(np.nansum(temp_BBrad[:,select],axis=(-1)),dx**2)	# this is already multiplied by 2
				partial_BBrad_std=np.nansum(temp_BBrad_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				partial_diffusion=np.multiply(np.nansum(temp_diffusion[:,select],axis=(-1)),dx**2)
				partial_diffusion_std=np.nansum(temp_diffusion_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				partial_timevariation=np.multiply(np.nansum(temp_timevariation[:,select],axis=(-1)),dx**2)
				partial_timevariation_std=np.nansum(temp_timevariation_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				partial_BBrad_small=np.multiply(np.nansum(temp_BBrad[:,select_small],axis=(-1)),dx**2)
				partial_BBrad_std_small=np.nansum(temp_BBrad_std[:,select_small]**2,axis=(-1))**0.5*(dx**2)
				partial_diffusion_small=np.multiply(np.nansum(temp_diffusion[:,select_small],axis=(-1)),dx**2)
				partial_diffusion_std_small=np.nansum(temp_diffusion_std[:,select_small]**2,axis=(-1))**0.5*(dx**2)
				partial_timevariation_small=np.multiply(np.nansum(temp_timevariation[:,select_small],axis=(-1)),dx**2)
				partial_timevariation_std_small=np.nansum(temp_timevariation_std[:,select_small]**2,axis=(-1))**0.5*(dx**2)
				negative_partial_BBrad=temp_BBrad*(dx**2)
				negative_partial_BBrad_time_mean = np.nanmean(negative_partial_BBrad,axis=0)
				negative_partial_BBrad_time_std = np.nanstd(negative_partial_BBrad,axis=0)
				if np.nansum(np.logical_not(select))>0:
					negative_partial_BBrad = negative_partial_BBrad[:,np.logical_not(select)]
					negative_partial_BBrad_space_mean = np.nanmean(negative_partial_BBrad,axis=(-1))
					negative_partial_BBrad_space_std = np.nanstd(negative_partial_BBrad,axis=(-1))
				else:
					negative_partial_BBrad_space_mean = 0
					negative_partial_BBrad_space_std = 0
				del negative_partial_BBrad
				negative_partial_diffusion=temp_diffusion*(dx**2)
				negative_partial_diffusion_time_mean = np.nanmean(negative_partial_diffusion,axis=0)
				negative_partial_diffusion_time_std = np.nanstd(negative_partial_diffusion,axis=0)
				if np.nansum(np.logical_not(select))>0:
					negative_partial_diffusion = negative_partial_diffusion[:,np.logical_not(select)]
					negative_partial_diffusion_space_mean = np.nanmean(negative_partial_diffusion,axis=(-1))
					negative_partial_diffusion_space_std = np.nanstd(negative_partial_diffusion,axis=(-1))
				else:
					negative_partial_diffusion_space_mean = 0
					negative_partial_diffusion_space_std = 0
				del negative_partial_diffusion
				negative_partial_timevariation=temp_timevariation*(dx**2)
				negative_partial_timevariation_time_mean = np.nanmean(negative_partial_timevariation,axis=0)
				negative_partial_timevariation_time_std = np.nanstd(negative_partial_timevariation,axis=0)
				if np.nansum(np.logical_not(select))>0:
					negative_partial_timevariation = negative_partial_timevariation[:,np.logical_not(select)]
					negative_partial_timevariation_space_mean = np.nanmean(negative_partial_timevariation,axis=(-1))
					negative_partial_timevariation_space_std = np.nanstd(negative_partial_timevariation,axis=(-1))
				else:
					negative_partial_timevariation_space_mean = 0
					negative_partial_timevariation_space_std = 0
				del negative_partial_timevariation


				# output1 = [coleval.calc_temp_to_power_BB_2(dTdt[i],dTdt_std[i],d2Tdxy[i],d2Tdxy_std[i],negd2Tdxy[i],negd2Tdxy_std[i],T4_T04[i],T4_T04_std[i],nan_ROI_mask,foilemissivityscaled,foilthicknessscaled,reciprdiffusivityscaled,Ptthermalconductivity) for i in range(len(laser_digitizer_ID))]
				# BBrad = [[],[]]
				# diffusion = [[],[]]
				# timevariation = [[],[]]
				# powernoback = [[],[]]
				# BBrad_std = [[],[]]
				# diffusion_std = [[],[]]
				# timevariation_std = [[],[]]
				# powernoback_std = [[],[]]
				# BBrad[0],diffusion[0],timevariation[0],powernoback[0],BBrad_std[0],diffusion_std[0],timevariation_std[0],powernoback_std[0] = output1[0]
				# BBrad[1],diffusion[1],timevariation[1],powernoback[1],BBrad_std[1],diffusion_std[1],timevariation_std[1],powernoback_std[1] = output1[1]

				# BBrad = np.mean(BBrad,axis=0)
				# diffusion = np.mean(diffusion,axis=0)
				# timevariation = np.mean(timevariation,axis=0)
				# powernoback = np.mean(powernoback,axis=0)
				# BBrad_std = np.mean(BBrad_std,axis=0)
				# diffusion_std = np.mean(diffusion_std,axis=0)
				# timevariation_std = np.mean(timevariation_std,axis=0)
				# powernoback_std = np.mean(powernoback_std,axis=0)

				BBrad = temp_BBrad*foilemissivityscaled
				diffusion = temp_diffusion*foilthicknessscaled
				timevariation = temp_timevariation*foilthicknessscaled*reciprdiffusivityscaled
				powernoback = BBrad + diffusion + timevariation
				BBrad_std = temp_BBrad_std*foilemissivityscaled
				diffusion_std = temp_diffusion_std*foilthicknessscaled
				timevariation_std = temp_timevariation_std*foilthicknessscaled*reciprdiffusivityscaled
				powernoback_std = (BBrad_std**2 + diffusion_std**2 + timevariation_std**2)**0.5
				del dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std

				# BBrad = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
				# BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
				# BBrad_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
				# BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
				# diffusion = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
				# diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
				# diffusion_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
				# diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
				# timevariation = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
				# timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
				# timevariation_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
				# timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
				# del temp_BBrad_std,temp_diffusion_std,temp_timevariation,temp_timevariation_std
				# powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
				# powernoback_std = np.ones_like(powernoback)*np.nan
				# powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)

				totalpower=np.multiply(np.nansum(powernoback[:,select],axis=(-1)),dx**2)
				totalpower_std=np.nansum(powernoback_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				totalBBrad=np.multiply(np.nansum(BBrad[:,select],axis=(-1)),dx**2)
				totalBBrad_std=np.nansum(BBrad_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				totaldiffusion=np.multiply(np.nansum(diffusion[:,select],axis=(-1)),dx**2)
				totaldiffusion_std=np.nansum(diffusion_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				totaltimevariation=np.multiply(np.nansum(timevariation[:,select],axis=(-1)),dx**2)
				totaltimevariation_std=np.nansum(timevariation_std[:,select]**2,axis=(-1))**0.5*(dx**2)
				del BBrad,BBrad_std,diffusion,diffusion_std,timevariation,timevariation_std


				# ani = coleval.movie_from_data(np.array([powernoback]), laser_framerate, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]',extvmin=np.nanmin(powernoback),extvmax=np.nanmax(powernoback))
				# plt.pause(0.01)
				# ani = coleval.movie_from_data(np.array([powernoback]), framerate, integration=int_time,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]')
				# plt.pause(0.01)
				# ani = coleval.movie_from_data(np.array([d2Tdxy[::5]]), framerate, integration=int_time,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]')
				# plt.pause(0.01)

				h_coordinates,v_coordinates = np.meshgrid(np.arange(1,np.shape(laser_temperature_minus_background_crop)[-1])*dx-dx/2,np.arange(1,np.shape(laser_temperature_minus_background_crop)[-2])*dx-dx/2)
				plt.figure(figsize=(20, 10))
				time_averaged_power_over_duty = np.nanmean(powernoback[:int(np.ceil(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse))],axis=0)/experimental_laser_duty
				time_averaged_power_over_duty_std = np.nanstd(powernoback[:int(np.ceil(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse))],axis=0)/experimental_laser_duty
				plt.pcolor(h_coordinates,v_coordinates,time_averaged_power_over_duty,cmap='rainbow',vmin=np.nanmin(time_averaged_power_over_duty[select]),vmax=np.nanmax(time_averaged_power_over_duty[select]))
				plt.colorbar().set_label('Power [W/m2]')
				# plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--',label='found laser size')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--')
				plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
				plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--')
				plt.plot((horizontal_loc + np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10))*dx,(vertical_loc + np.abs(dr_total_power_minimum**2-np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10)**2)**0.5)*dx,'k',label='area accounted for sum small')
				plt.plot((horizontal_loc + np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10))*dx,(vertical_loc - np.abs(dr_total_power_minimum**2-np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10)**2)**0.5)*dx,'k')
				plt.title(preamble_4_prints+'Power source shape in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm laser radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
				plt.xlabel('Horizontal axis [mm]')
				plt.ylabel('Vertical axis [mm]')
				plt.legend(loc='best', fontsize='small')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')

				plt.figure(figsize=(20, 10))
				plt.pcolor(h_coordinates,v_coordinates,generic_filter(time_averaged_diffusion_over_duty,np.mean,size=[3,3]),cmap='rainbow',vmax=0)#vmin=np.nanmin(time_averaged_diff_over_duty[select]),vmax=np.nanmax(time_averaged_diff_over_duty[select]))
				plt.colorbar().set_label('Power [W/m2]')
				# plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--',label='found laser size')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--')
				plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
				plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--')
				plt.plot((horizontal_loc + np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10))*dx,(vertical_loc + np.abs(dr_total_power_minimum**2-np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10)**2)**0.5)*dx,'k',label='area accounted for sum small')
				plt.plot((horizontal_loc + np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10))*dx,(vertical_loc - np.abs(dr_total_power_minimum**2-np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10)**2)**0.5)*dx,'k')
				plt.title(preamble_4_prints+'Diffusion component shape in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm laser radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
				plt.xlabel('Horizontal axis [mm]')
				plt.ylabel('Vertical axis [mm]')
				plt.legend(loc='best', fontsize='small')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')
				del temp_diffusion

				plt.figure(figsize=(20, 10))
				plt.pcolor(h_coordinates,v_coordinates,time_averaged_BBrad_over_duty,cmap='rainbow',vmax=BBtreshold*5)
				plt.colorbar().set_label('Power [W/m2]')
				# plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--',label='found laser size')
				plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10/2,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10/2,dr/10)**2)**0.5)*dx,'r--')
				plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
				plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10/2,dr_total_power/10)**2)**0.5)*dx,'k--')
				plt.plot((horizontal_loc + np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10))*dx,(vertical_loc + np.abs(dr_total_power_minimum**2-np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10)**2)**0.5)*dx,'k',label='area accounted for sum small')
				plt.plot((horizontal_loc + np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10))*dx,(vertical_loc - np.abs(dr_total_power_minimum**2-np.arange(-dr_total_power_minimum,+dr_total_power_minimum+dr_total_power_minimum/10/2,dr_total_power_minimum/10)**2)**0.5)*dx,'k')
				plt.title(preamble_4_prints+'BB component shape in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm laser radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
				plt.xlabel('Horizontal axis [mm]')
				plt.ylabel('Vertical axis [mm]')
				plt.legend(loc='best', fontsize='small')
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')
				del temp_BBrad


				totalpower_filtered_1 = generic_filter(totalpower[:int(np.ceil(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse))],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/15//2*2+1))])
				totalpower_filtered_1_std = generic_filter((totalpower_std**2)/max(1,int(laser_framerate/experimental_laser_frequency/30//2*2+1)),np.nansum,size=[max(1,int(laser_framerate/experimental_laser_frequency/30//2*2+1))])**0.5
				totalpower_filtered_2 = generic_filter(totalpower[:int(np.ceil(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse))],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/1.5//2*2+1))])
				totalpower_filtered_2_std = generic_filter((totalpower_std**2)/max(1,int(laser_framerate/experimental_laser_frequency/3//2*2+1)),np.nansum,size=[max(1,int(laser_framerate/experimental_laser_frequency/3//2*2+1))])**0.5

				laser_power_detected_nominal_properties = np.nanmean(totalpower)/experimental_laser_duty

				if False:	# the fitting method is quite unreliabe and unnecessaty compared to the partial median
					def double_gaussian(x,A1,sig1,x01,A2,sig2,x02):
						temp1 = A1 * np.exp(-0.5*(((x-x01)/sig1)**2))
						temp2 = A2 * np.exp(-0.5*(((x-x02)/sig2)**2))
						return temp1+temp2

					fractions = 70
					temp1 = totalpower_filtered_1.max()
					temp = (totalpower_filtered_1/temp1*fractions).astype(int)
					counter = collections.Counter(temp)
					x=np.array(list(counter.keys()))/fractions*temp1
					y=np.array(list(counter.values()))
					y = np.array([y for _, y in sorted(zip(x, y))])
					x = np.sort(x)
					bds = [[0,0,totalpower_filtered_1.min(),0,0,totalpower_filtered_1.mean()],[np.inf,totalpower_filtered_1.std(),totalpower_filtered_1.mean(),np.inf,totalpower_filtered_1.std(),totalpower_filtered_1.max()]]
					guess=[fractions,temp1/10,totalpower_filtered_1.min(),fractions,temp1/10,totalpower_filtered_1.max()]
					x_scale=[fractions,temp1/100,totalpower_filtered_1.std()/20,fractions,temp1/100,totalpower_filtered_1.std()/20]
					fit = curve_fit(double_gaussian, x, y, p0=guess,x_scale=np.abs(x_scale),bounds=bds,maxfev=int(1e6),verbose=2,ftol=1e-14,xtol=1e-15,gtol=1e-14)
					fitted_power = [correlated_values(fit[0],fit[1])[2],correlated_values(fit[0],fit[1])[5]]

				fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]

				fig, ax = plt.subplots( 3,1,figsize=(20, 20), squeeze=False)
				fig.suptitle(preamble_4_prints+'Analysis of '+laser_to_analyse+'\nHigh Power=%.3g+/-%.3g, Low Power=%.3g+/-%.3g' %(nominal_values(fitted_power_2[1]),std_devs(fitted_power_2[1]),nominal_values(fitted_power_2[0]),std_devs(fitted_power_2[0])))
				# time_axis = (time_of_experiment[1:-1]-time_of_experiment[1])/1e6
				ax[0,0].plot(time_partial,totalpower,label='totalpower')
				ax[0,0].plot(time_partial,totaltimevariation,label='totaltimevariation')
				ax[0,0].plot(time_partial,totalBBrad,label='totalBBrad')
				ax[0,0].plot(time_partial,totaldiffusion,label='totaldiffusion')
				ax[0,0].plot(time_partial[:int(np.ceil(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse))],totalpower_filtered_1,linewidth=3,label='totalpower filtered')
				ax[0,0].plot(time_partial[:int(np.ceil(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse))],totalpower_filtered_2,linewidth=3,label='totalpower super filtered')
				ax[0,0].axhline(y=laser_power_detected_nominal_properties,color='k',linestyle='--',linewidth=2,label='power mean / duty cycle')
				ax[0,0].errorbar([time_partial[0],time_partial[-1]],[nominal_values(fitted_power_2[0])]*2,yerr=[std_devs(fitted_power_2[0])]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
				ax[0,0].errorbar([time_partial[0],time_partial[-1]],[nominal_values(fitted_power_2[1])]*2,yerr=[std_devs(fitted_power_2[1])]*2,color='k',linestyle=':',linewidth=2)
				# plt.errorbar([time_axis[0],time_axis[-1]],[nominal_values(fitted_power[0])]*2,yerr=[std_devs(fitted_power[0])]*2,color='silver',linestyle=':',linewidth=2,label='power found from fit')
				# plt.errorbar([time_axis[0],time_axis[-1]],[nominal_values(fitted_power[1])]*2,yerr=[std_devs(fitted_power[1])]*2,color='silver',linestyle=':',linewidth=2)
				# plt.plot([time_axis[0],time_axis[-1]],[laser_to_analyse_power]*2,color='silver',linestyle=':',linewidth=2,label='input power')
				ax[0,0].legend(loc='best', fontsize='small')
				ax[0,0].grid()
				ax[0,0].set_xlabel('time [s]')
				ax[0,0].set_ylabel('power [W]')
				ax[0,0].set_title('Using the nominal (given to us) foil properties')
				# plt.ylim(bottom=np.nanmin([totalpower,totalBBrad,totaldiffusion,totaltimevariation]),top=np.nanmax([totalpower,totalBBrad,totaldiffusion,totaltimevariation]))
				ax[1,0].plot([dr*dx*1e3]*2,[np.nanmax(nominal_values(power_vs_space_sampling['fitted_powers'][:,1])),np.nanmin(nominal_values(power_vs_space_sampling['fitted_powers'][:,0]))],'--r')
				ax[1,0].plot([dr_total_power*dx*1e3]*2,[np.nanmax(nominal_values(power_vs_space_sampling['fitted_powers'][:,1])),np.nanmin(nominal_values(power_vs_space_sampling['fitted_powers'][:,0]))],'--k')
				ax[1,0].errorbar(power_vs_space_sampling['all_dr']*dx*1e3,nominal_values(power_vs_space_sampling['fitted_powers'][:,1]),yerr=std_devs(power_vs_space_sampling['fitted_powers'][:,1]))
				ax[1,0].errorbar(power_vs_space_sampling['all_dr']*dx*1e3,nominal_values(power_vs_space_sampling['fitted_powers'][:,0]),yerr=std_devs(power_vs_space_sampling['fitted_powers'][:,0]))
				ax[1,0].grid()
				ax[1,0].set_xlabel('radious of sum [mm]')
				ax[1,0].set_ylabel('temperature laplacian [K/m2]')
				# ax[1,0].set_xscale('log')
				ax[1,0].set_title('search for laser spot size')
				ax[2,0].plot([power_vs_time_sampling['number_of_pulses'][-1]/experimental_laser_frequency]*2,[np.nanmax(nominal_values(power_vs_time_sampling['fitted_powers'][:,1])),np.nanmin(nominal_values(power_vs_time_sampling['fitted_powers'][:,0]))],'--k')
				ax[2,0].errorbar(power_vs_time_sampling['number_of_pulses']/experimental_laser_frequency,nominal_values(power_vs_time_sampling['fitted_powers'][:,1]),yerr=std_devs(power_vs_time_sampling['fitted_powers'][:,1]))
				ax[2,0].errorbar(power_vs_time_sampling['number_of_pulses']/experimental_laser_frequency,nominal_values(power_vs_time_sampling['fitted_powers'][:,0]),yerr=std_devs(power_vs_time_sampling['fitted_powers'][:,0]))
				ax[2,0].grid()
				ax[2,0].set_xlabel('time interval used [s]')
				ax[2,0].set_ylabel('power [W]')
				ax[2,0].set_title('manual foil properties: thickness=%.3gm, emissivity=%.3g, diffusivity=%.3gm2/s' %(sample_properties['thickness'],sample_properties['emissivity'],sample_properties['diffusivity']))
				figure_index+=1
				plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close('all')
				# plt.pause(0.01)


				if False:	# the fitting method is quite unreliabe and unnecessaty compared to the partial median
					plt.figure(figsize=(20, 10))
					plt.plot(x,y,'+')
					plt.plot(x,median_filter(y,size=[int(fractions/5-1)]))
					plt.plot(x,double_gaussian(x,*fit[0]),'--')
					# plt.plot(x,double_gaussian(x,*guess),'--')
					# plt.legend(loc='best', fontsize='small')
					plt.errorbar([nominal_values(fitted_power[0])]*2,[0,y.max()],xerr=[std_devs(fitted_power[0])]*2,linestyle=':',color='silver')
					plt.errorbar([nominal_values(fitted_power[1])]*2,[0,y.max()],xerr=[std_devs(fitted_power[1])]*2,linestyle=':',color='silver')
					plt.plot([laser_power_detected_nominal_properties]*2,[0,y.max()],'--',color='k')
					plt.errorbar([nominal_values(fitted_power_2[0])]*2,[0,y.max()],xerr=[std_devs(fitted_power_2[0])]*2,linestyle=':',color='k')
					plt.errorbar([nominal_values(fitted_power_2[1])]*2,[0,y.max()],xerr=[std_devs(fitted_power_2[1])]*2,linestyle=':',color='k')
					plt.grid()
					plt.xlabel('power [W]')
					plt.ylabel('frequency [au]')
					plt.title('Analysis of '+laser_to_analyse+'\nUsing the nominal (given to us) foil properties\nLow Power='+str(np.nanmin(fitted_power))+'W (theoretical %.3gW, freq %.3gHz), High Power=' %(laser_to_analyse_power,experimental_laser_frequency) +str(np.nanmax(fitted_power))+'W')
					figure_index+=1
					plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
					plt.close('all')
					# plt.pause(0.01)
					# spectra = np.fft.fft(totaldiffusion)
					# magnitude = 2 * np.abs(spectra) / len(spectra)
					# freq = np.fft.fftfreq(len(magnitude), d=1 / laser_framerate)
					# plt.figure();plt.plot(freq,magnitude);plt.semilogy();plt.pause(0.01)

				# if emissivity==1 and reference_temperature==21:
				full_saved_file_dict[type_of_calibration]['totalpower_nominal_properties']=totalpower
				full_saved_file_dict[type_of_calibration]['totalpower_std_nominal_properties']=totalpower_std
				full_saved_file_dict[type_of_calibration]['laser_power_detected_nominal_properties']=laser_power_detected_nominal_properties
				full_saved_file_dict[type_of_calibration]['laser_power_detected_std_nominal_properties']=np.nansum(totalpower_std**2/len(totalpower_std))**0.5
				full_saved_file_dict[type_of_calibration]['time_averaged_power_over_duty_nominal_properties']=time_averaged_power_over_duty
				full_saved_file_dict[type_of_calibration]['time_averaged_power_over_duty_std_nominal_properties']=time_averaged_power_over_duty_std
				full_saved_file_dict['laser_location']=[horizontal_loc,dhorizontal_loc,vertical_loc,dvertical_loc,dr]
				full_saved_file_dict['laser_input_power']=laser_to_analyse_power
				# full_saved_file_dict[type_of_calibration]['time_of_experiment'] = time_of_experiment
				full_saved_file_dict[type_of_calibration]['time_of_experiment'] = time_partial
				full_saved_file_dict[type_of_calibration]['emissivity_range'] = emissivity_range
				full_saved_file_dict[type_of_calibration]['reference_temperature_range'] = reference_temperature_range
				# np.savez_compressed(laser_to_analyse,**full_saved_file_dict)

				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)] = dict([])

				# full_saved_file_dict[type_of_calibration]['fitted_power']=fitted_power
				# single_full_saved_file_dict = dict([])
				# single_full_saved_file_dict[type_of_calibration] = dict([])
				# single_full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)] = dict([])
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)] = dict([])
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_BBrad']=partial_BBrad	# this is already multiplied by 2
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std']=partial_BBrad_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_diffusion']=partial_diffusion
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_std']=partial_diffusion_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_timevariation']=partial_timevariation
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std']=partial_timevariation_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_small']=partial_BBrad_small
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_BBrad_std_small']=partial_BBrad_std_small
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_small']=partial_diffusion_small
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_diffusion_std_small']=partial_diffusion_std_small
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_small']=partial_timevariation_small
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['partial_timevariation_std_small']=partial_timevariation_std_small
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_BBrad_time_mean']=negative_partial_BBrad_time_mean
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_BBrad_time_std']=negative_partial_BBrad_time_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_BBrad_space_mean']=negative_partial_BBrad_space_mean
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_BBrad_space_std']=negative_partial_BBrad_space_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_diffusion_time_mean']=negative_partial_diffusion_time_mean
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_diffusion_time_std']=negative_partial_diffusion_time_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_diffusion_space_mean']=negative_partial_diffusion_space_mean
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_diffusion_space_std']=negative_partial_diffusion_space_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_timevariation_time_mean']=negative_partial_timevariation_time_mean
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_timevariation_time_std']=negative_partial_timevariation_time_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_timevariation_space_mean']=negative_partial_timevariation_space_mean
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['negative_partial_timevariation_space_std']=negative_partial_timevariation_space_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['power_vs_space_sampling']=power_vs_space_sampling
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['power_vs_time_sampling']=power_vs_time_sampling
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['ref_temperature']=ref_temperature
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['ref_temperature_std']=ref_temperature_std
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['reference_background_temperature_crop']=reference_background_temperature_crop
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['reference_background_temperature_std_crop']=reference_background_temperature_std_crop
				full_saved_file_dict[type_of_calibration]['emissivity='+str(emissivity)]['T0='+str(reference_temperature)]['laser_temperature_minus_background_crop_max']=laser_temperature_minus_background_crop_max

				# np.savez_compressed(laser_to_analyse+type_of_calibration + '_' + 'emissivity='+str(emissivity) +'_'+ 'T0='+str(reference_temperature),**single_full_saved_file_dict)
				np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
				print('intermediate step saved')


	try:
		del full_saved_file_dict['totalpower_nominal_properties']
		del full_saved_file_dict['totalpower_std_nominal_properties']
		del full_saved_file_dict['laser_power_detected_nominal_properties']
		del full_saved_file_dict['laser_power_detected_std_nominal_properties']
		del full_saved_file_dict['time_averaged_power_over_duty_nominal_properties']
		del full_saved_file_dict['time_averaged_power_over_duty_std_nominal_properties']
		# del full_saved_file_dict['fitted_power']=fitted_power
		del full_saved_file_dict['partial_BBrad']
		del full_saved_file_dict['partial_BBrad_std']
		del full_saved_file_dict['partial_diffusion']
		del full_saved_file_dict['partial_diffusion_std']
		del full_saved_file_dict['partial_timevariation']
		del full_saved_file_dict['partial_timevariation_std']
		del full_saved_file_dict['partial_BBrad_small']
		del full_saved_file_dict['partial_BBrad_std_small']
		del full_saved_file_dict['partial_diffusion_small']
		del full_saved_file_dict['partial_diffusion_std_small']
		del full_saved_file_dict['partial_timevariation_small']
		del full_saved_file_dict['partial_timevariation_std_small']
		# del full_saved_file_dict['laser_location']
		del full_saved_file_dict['negative_partial_BBrad_time_mean']
		del full_saved_file_dict['negative_partial_BBrad_time_std']
		del full_saved_file_dict['negative_partial_BBrad_space_mean']
		del full_saved_file_dict['negative_partial_BBrad_space_std']
		del full_saved_file_dict['negative_partial_diffusion_time_mean']
		del full_saved_file_dict['negative_partial_diffusion_time_std']
		del full_saved_file_dict['negative_partial_diffusion_space_mean']
		del full_saved_file_dict['negative_partial_diffusion_space_std']
		del full_saved_file_dict['negative_partial_timevariation_time_mean']
		del full_saved_file_dict['negative_partial_timevariation_time_std']
		del full_saved_file_dict['negative_partial_timevariation_space_mean']
		del full_saved_file_dict['negative_partial_timevariation_space_std']
		del full_saved_file_dict['power_vs_space_sampling']
		del full_saved_file_dict['power_vs_time_sampling']
		# del full_saved_file_dict['laser_input_power']
		# del full_saved_file_dict['time_of_experiment'] = time_of_experiment
		del full_saved_file_dict['time_of_experiment']
	except:
		pass


	np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
	print('FINISHED '+laser_to_analyse)
	return 0



# import concurrent.futures as cf
# with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
	# executor.map(function_a,range(len(all_laser_to_analyse)))
	# # executor.map(function_a,range(2))
# with Pool(number_cpu_available,maxtasksperchild=1) as pool:
# 	temp = pool.map(function_a, range(len(all_laser_to_analyse)))
# 	pool.close()
# 	pool.join()
# 	pool.terminate()
# 	del pool
# for index in np.flip(np.arange(len(all_laser_to_analyse)),axis=0):
for index in np.arange(len(all_laser_to_analyse)):
# for index in [8]:
	# try:
	function_a(index)
	# except Exception as e:
	# 	print('index '+str(index)+' failed')
	# 	logging.exception('with error: ' + str(e))


#
