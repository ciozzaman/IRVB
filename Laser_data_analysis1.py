# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
from multiprocessing import Pool,cpu_count,current_process,set_start_method,get_context,Semaphore
number_cpu_available = 8
# number_cpu_available = cpu_count()


# degree of polynomial of choice
n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
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
	cases_to_include = ['laser15','laser16','laser17','laser18','laser19','laser20','laser21','laser22','laser23','laser24','laser25','laser26','laser27','laser28','laser29','laser30','laser31','laser32','laser33','laser34','laser35','laser36','laser37','laser38','laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = ['laser39','laser41','laser42','laser43','laser44','laser45','laser46','laser47']
	# cases_to_include = ['laser34','laser35','laser36','laser37','laser38','laser39']
	# cases_to_include = ['laser35']
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
BBtreshold = 0.13

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
		return 0

	laser_to_analyse_power = power_interpolator(experimental_laser_voltage)

	laser_dict = np.load(laser_to_analyse+'.npz')
	# laser_counts, laser_digitizer_ID = coleval.separate_data_with_digitizer(laser_dict)
	time_of_experiment = laser_dict['time_of_measurement']	# microseconds
	mean_time_of_experiment = np.nanmean(time_of_experiment)
	laser_digitizer = laser_dict['digitizer_ID']
	if np.diff(time_of_experiment).max()>np.median(np.diff(time_of_experiment))*1.1:
		hole_pos = np.diff(time_of_experiment).argmax()
		if hole_pos<len(time_of_experiment)/2:
			time_of_experiment = time_of_experiment[hole_pos+1:]
			laser_digitizer = laser_digitizer[hole_pos+1:]
		else:
			time_of_experiment = time_of_experiment[:-(hole_pos+1)]
			laser_digitizer = laser_digitizer[:-(hole_pos+1)]
	time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)
	laser_framerate = laser_dict['FrameRate']
	laser_int_time = laser_dict['IntegrationTime']

	preamble_4_prints = 'Case '+case_ID+', FR=%.3gHz, int time=%.3gms, ' %(laser_framerate,laser_int_time/1000) +'laser '+ focus_status +' , power=%.3gW, freq=%.3gHz, duty=%.3g\n' %(laser_to_analyse_power,experimental_laser_frequency,experimental_laser_duty)

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
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		if limited_ROI == 'ff':
			print('There is some issue with indexing. Data height,width is '+str([int(laser_dict['height']),int(laser_dict['width'])])+' but the indexing says it should be full frame ('+str([max_ROI[0][1]+1,max_ROI[1][1]+1])+')')
			exit()
		params = params[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1]
		errparams = errparams[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1]

	# Selection of only the backgroud files with similar framerate and integration time
	background_timestamps = [(np.nanmean(np.load(file+'.npz')['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
	background_temperatures = [(np.nanmean(np.load(file+'.npz')['SensorTemp_0'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
	background_counts = [(np.load(file+'.npz')['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
	background_counts_std = [(np.load(file+'.npz')['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
	relevant_background_files = [file for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]


	# Calculating the background image
	temp = np.sort(np.abs(background_timestamps-mean_time_of_experiment))[1]
	ref = np.arange(len(background_timestamps))[np.abs(background_timestamps-mean_time_of_experiment)<=temp]
	reference_background = (background_counts[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_counts[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
	reference_background_std = ((background_counts_std[ref[0]]**2)*(background_timestamps[ref[1]]-mean_time_of_experiment) + (background_counts_std[ref[1]]**2)*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
	reference_background_flat = np.nanmean(reference_background,axis=(1,2))
	reference_background_camera_temperature = (background_temperatures[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_temperatures[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
	reference_background_full_1,background_digitizer_ID = coleval.separate_data_with_digitizer(np.load(relevant_background_files[ref[0]]+'.npz'))
	reference_background_full_2,background_digitizer_ID = coleval.separate_data_with_digitizer(np.load(relevant_background_files[ref[1]]+'.npz'))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		flag_1 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_1]
		flag_2 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_2]
	else:
		flag_1 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_1]
		flag_2 = [coleval.find_dead_pixels([data],treshold_for_bad_difference=100) for data in reference_background_full_2]
	bad_pixels_flag = np.add(flag_1,flag_2)

	# ani = coleval.movie_from_data(np.array([reference_background_full_2[0]]), laser_framerate, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]')
	# plt.pause(0.01)

	if False:	# I tried here to have an automatic recognition of the foil position and shape. I can't have a good enough contrast,m so I abandon this route
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
		if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
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
		plt.title(preamble_4_prints+'Laser spot location in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
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

	if False:	# done with functions now
		from uncertainties import correlated_values,ufloat
		from uncertainties.unumpy import nominal_values,std_devs,uarray
		def function_a(arg):
			out = np.nansum(np.power(np.array([arg[0].tolist()]*arg[2]).T,np.arange(arg[2]-1,-1,-1))*arg[1],axis=1)
			out1 = nominal_values(out)
			out2 = std_devs(out)
			return [out1,out2]

		def function_b(arg):
			out = np.nansum(np.power(np.array([arg[0].tolist()]*arg[2]).T,np.arange(arg[2]-1,-1,-1))*arg[1],axis=0)
			out1 = nominal_values(out)
			out2 = std_devs(out)
			return [out1,out2]

		import time as tm
		import concurrent.futures as cf
		laser_temperature = []
		laser_temperature_std = []
		with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
			# executor = cf.ProcessPoolExecutor()#max_workers=number_cpu_available)
			for i in range(len(laser_digitizer_ID)):
				laser_counts_temp = np.array(laser_counts_filtered[i])
				temp1 = []
				temp2 = []
				for j in range(laser_counts_temp.shape[1]):
					start_time = tm.time()

					arg = []
					for k in range(laser_counts_temp.shape[2]):
						arg.append([laser_counts_temp[:,j,k]-ufloat(reference_background[i,j,k],reference_background_std[i,j,k])+reference_background_flat[i],correlated_values(params[i,j,k],errparams[i,j,k]),n])
					print(str(j) + ' , %.3gs' %(tm.time()-start_time))
					out = list(executor.map(function_a,arg))

					# print(str(j) + ' , %.3gs' %(tm.time()-start_time))
					# out = []
					# for k in range(laser_counts_temp.shape[2]):
					# 	out.append(function_a([laser_counts_temp[:,j,k],correlated_values(params[i,j,k],errparams[i,j,k]),n]))

					print(str(j) + ' , %.3gs' %(tm.time()-start_time))
					temp1.append([x for x,y in out])
					temp2.append([y for x,y in out])
					print(str(j) + ' , %.3gs' %(tm.time()-start_time))
				laser_temperature.append(np.transpose(temp1,(2,0,1)))
				laser_temperature_std.append(np.transpose(temp2,(2,0,1)))

		reference_background_temperature = []
		reference_background_temperature_std = []
		with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
			# executor = cf.ProcessPoolExecutor()#max_workers=number_cpu_available)
			for i in range(len(laser_digitizer_ID)):
				reference_background_temp = np.array(reference_background[i])
				reference_background_std_temp = np.array(reference_background_std[i])
				temp1 = []
				temp2 = []
				for j in range(reference_background_temp.shape[0]):
					start_time = tm.time()

					arg = []
					for k in range(reference_background_temp.shape[1]):
						arg.append([uarray(reference_background_temp[j,k],reference_background_std_temp[j,k]),correlated_values(params[i,j,k],errparams[i,j,k]),n])
					print(str(j) + ' , %.3gs' %(tm.time()-start_time))
					out = list(executor.map(function_b,arg))

					# print(str(j) + ' , %.3gs' %(tm.time()-start_time))
					# out = []
					# for k in range(laser_counts_temp.shape[2]):
					# 	out.append(function_a([laser_counts_temp[:,j,k],correlated_values(params[i,j,k],errparams[i,j,k]),n]))

					print(str(j) + ' , %.3gs' %(tm.time()-start_time))
					temp1.append([x for x,y in out])
					temp2.append([y for x,y in out])
					print(str(j) + ' , %.3gs' %(tm.time()-start_time))
				reference_background_temperature.append(np.array(temp1))
				reference_background_temperature_std.append(np.array(temp2))

	else:
		reference_background_temperature,reference_background_temperature_std = coleval.count_to_temp_poly_multi_digitizer(reference_background,params,errparams,laser_digitizer_ID,number_cpu_available,counts_std=reference_background_std,report=0,parallelised=False)

		# plt.figure()
		# plt.imshow(reference_background_temperature[0],'rainbow',origin='lower')
		# plt.colorbar().set_label('Temp [°C]')
		# plt.title('Reference frame for '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
		# plt.xlabel('Horizontal axis [pixles]')
		# plt.ylabel('Vertical axis [pixles]')
		# plt.pause(0.01)


	if False:	# I subtract the background to the counts, so this bit is not necessary
		# I want to subtract the reference temperature and add back 300K. in this way I whould remove all the disuniformities.
		# I should add a reference real temperature of the plate at the time of the reference frame recording, but I don't have it
		laser_temperature_relative = [(laser_temperature[i]-reference_background_temperature[i]+300) for i in np.arange(len(laser_digitizer_ID))]
		laser_temperature_std_relative = [(laser_temperature_std[i]**2+reference_background_temperature_std[i]**2)**0.5 for i in np.arange(len(laser_digitizer_ID))]
		# NO!! doing it like that the uncertainty is huge, I need to do it before to convert to temperature
		# I subtract the background to the counts and add back the average of the background counts
		# NO!! I cannot do this either, because the counts to temp coefficients are fuilt in with the disuniformity, and non't work if I remove it before hand.

	laser_dict.allow_pickle=True
	temp1 = laser_dict['laser_temperature_minus_background_median']	# this is NOT minus_background, that is a relic of what I did before, but I keep it not to recreate all .npz, same for the next3
	temp2 = laser_dict['laser_temperature_minus_background_minus_median_downgraded']
	temp3 = laser_dict['laser_temperature_minus_background_std_median']
	temp4 = laser_dict['laser_temperature_minus_background_std_minus_median_downgraded']
	laser_temperature = [(temp1[i] + temp2[i].astype(np.float32)) for i in laser_digitizer_ID]
	laser_temperature_std = [(temp3[i] + temp4[i].astype(np.float32)) for i in laser_digitizer_ID]
	del temp1,temp2,temp3,temp4	# modified to try to read the .npz as little as possible
	laser_temperature_minus_background = [laser_temperature[i]-reference_background_temperature[i] for i in laser_digitizer_ID]
	laser_temperature_std_minus_background = [(laser_temperature_std[i]**2+reference_background_temperature_std[i]**2)**0.5 for i in laser_digitizer_ID]


	# I can replace the dead pixels even after the transformation to temperature, given the flag comes from the background data
	laser_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature)]
	laser_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_std)]
	laser_temperature_minus_background_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_minus_background)]
	laser_temperature_std_minus_background_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_std_minus_background)]
	reference_background_temperature_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,reference_background_temperature)]
	reference_background_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([[data]],flag)[0][0] for flag,data in zip(bad_pixels_flag,reference_background_temperature_std)]
	del laser_temperature,laser_temperature_std,laser_temperature_minus_background,laser_temperature_std_minus_background


	# I need to put together the data from the two digitizers
	# laser_temperature_full = [(laser_temperature_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_temperature_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)]
	# laser_temperature_std_full = [(laser_temperature_std_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_temperature_std_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)]
	laser_temperature_minus_background_full = [(laser_temperature_minus_background_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if time in time_of_experiment_digitizer_ID[0] else (laser_temperature_minus_background_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time in time_of_experiment]
	laser_temperature_std_minus_background_full = [(laser_temperature_std_minus_background_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if time in time_of_experiment_digitizer_ID[0] else (laser_temperature_std_minus_background_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time in time_of_experiment]
	del laser_temperature_no_dead_pixels,laser_temperature_std_no_dead_pixels,laser_temperature_minus_background_no_dead_pixels,laser_temperature_std_minus_background_no_dead_pixels


	# rotation and crop
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		temp = np.ones((len(laser_temperature_minus_background_full),max_ROI[0][1]+1,max_ROI[1][1]+1))*(-np.nanmean(reference_background[0]))
		# temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_full
		# laser_temperature_full = cp.deepcopy(temp)
		# temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_std_full
		# laser_temperature_std_full = cp.deepcopy(temp)
		temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_minus_background_full
		laser_temperature_minus_background_full = cp.deepcopy(temp)
		temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = laser_temperature_std_minus_background_full
		laser_temperature_std_minus_background_full = cp.deepcopy(temp)
		temp = np.ones((len(laser_digitizer_ID),max_ROI[0][1]+1,max_ROI[1][1]+1))*(-np.nanmean(reference_background[0]))
		temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = reference_background_temperature_no_dead_pixels
		reference_background_temperature_no_dead_pixels = cp.deepcopy(temp)
		temp[:,limited_ROI[0][0]:limited_ROI[0][1]+1,limited_ROI[1][0]:limited_ROI[1][1]+1] = reference_background_temperature_std_no_dead_pixels
		reference_background_temperature_std_no_dead_pixels = cp.deepcopy(temp)

	# laser_temperature_rot=rotate(laser_temperature_full,foilrotdeg,axes=(-1,-2))
	# laser_temperature_std_rot=rotate(laser_temperature_std_full,foilrotdeg,axes=(-1,-2))
	# if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
	# 	laser_temperature_std_rot*=out_of_ROI_mask
	# 	laser_temperature_std_rot[np.logical_and(laser_temperature_rot<np.nanmin(laser_temperature_full),laser_temperature_rot>np.nanmax(laser_temperature_full))]=0
	# 	laser_temperature_rot*=out_of_ROI_mask
	# 	laser_temperature_rot[np.logical_and(laser_temperature_rot<np.nanmin(laser_temperature_full),laser_temperature_rot>np.nanmax(laser_temperature_full))]=np.nanmean(reference_background_temperature_no_dead_pixels)
	# laser_temperature_crop=laser_temperature_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
	# laser_temperature_std_crop=laser_temperature_std_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)

	laser_temperature_minus_background_rot=rotate(laser_temperature_minus_background_full,foilrotdeg,axes=(-1,-2))
	laser_temperature_std_minus_background_rot=rotate(laser_temperature_std_minus_background_full,foilrotdeg,axes=(-1,-2))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		laser_temperature_std_minus_background_rot*=out_of_ROI_mask
		laser_temperature_std_minus_background_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
		laser_temperature_minus_background_rot*=out_of_ROI_mask
		laser_temperature_minus_background_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
	laser_temperature_minus_background_crop=laser_temperature_minus_background_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
	laser_temperature_std_minus_background_crop=laser_temperature_std_minus_background_rot[:,foildw:foilup,foillx:foilrx].astype(np.float32)
	nan_ROI_mask = np.isfinite(np.nanmedian(laser_temperature_minus_background_crop[:10],axis=0))

	reference_background_temperature_rot=rotate(reference_background_temperature_no_dead_pixels,foilrotdeg,axes=(-1,-2))
	reference_background_temperature_std_rot=rotate(reference_background_temperature_std_no_dead_pixels,foilrotdeg,axes=(-1,-2))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		reference_background_temperature_std_rot*=out_of_ROI_mask
		reference_background_temperature_std_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=0
		reference_background_temperature_rot*=out_of_ROI_mask
		reference_background_temperature_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=np.nanmean(reference_background_temperature_no_dead_pixels)
	reference_background_temperature_crop=reference_background_temperature_rot[:,foildw:foilup,foillx:foilrx]
	reference_background_temperature_crop = np.nanmean(reference_background_temperature_crop,axis=0).astype(np.float32)
	reference_background_temperature_std_crop=reference_background_temperature_std_rot[:,foildw:foilup,foillx:foilrx]
	reference_background_temperature_std_crop = np.nanmean(reference_background_temperature_std_crop,axis=0).astype(np.float32)

	del laser_temperature_minus_background_rot,laser_temperature_std_minus_background_rot,reference_background_temperature_no_dead_pixels,reference_background_temperature_std_no_dead_pixels

	# plt.figure()
	# plt.imshow(laser_temperature_crop[0],'rainbow',origin='lower')
	# plt.colorbar().set_label('Temp [°C]')
	# plt.title('Only foil in frame '+str(0)+' in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
	# plt.xlabel('Horizontal axis [pixles]')
	# plt.ylabel('Vertical axis [pixles]')
	# plt.pause(0.01)

	# FOIL PROPERTY ADJUSTMENT
	if True:	# spatially resolved foil properties from Japanese producer
		foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
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

	dt=1/laser_framerate
	dx=foilhorizw/(foilhorizwpixel-1)
	temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,1))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		temp[np.isfinite(temp)] = [np.nanmedian(temp)]*2 + temp[np.isfinite(temp)][2:-2].tolist() + [np.nanmedian(temp)]*2
	temp1 = temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)
	horizontal_loc = (np.arange(len(temp))[temp1])[temp[temp1].argmax()]
	dhorizontal_loc = np.nansum(temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp))
	temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,2))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
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
	horizontal_coord = np.arange(np.shape(laser_temperature_minus_background_crop)[2])
	vertical_coord = np.arange(np.shape(laser_temperature_minus_background_crop)[1])
	horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
	# select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr**2)[1:-1,1:-1]

	plt.figure(figsize=(20, 10))
	temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,1))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
		temp[np.isfinite(temp)] = [np.nanmedian(temp)]*2 + temp[np.isfinite(temp)][2:-2].tolist() + [np.nanmedian(temp)]*2
	plt.plot(np.arange(len(temp))-horizontal_loc,(temp-np.nanmin(temp))/(np.nanmax(temp)-np.nanmin(temp)),'b',label='horizontal')
	plt.plot((np.arange(len(temp))-horizontal_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)],[0.5]*np.nansum(temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)),'--b')
	plt.plot([(np.arange(len(temp))-horizontal_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)][0]]*2,[0,0.5],'--b')
	plt.plot([(np.arange(len(temp))-horizontal_loc)[temp>=(np.nanmax(temp)-np.nanmin(temp))/2+np.nanmin(temp)][-1]]*2,[0,0.5],'--b')
	temp = np.nanmean(laser_temperature_minus_background_crop,axis=(0,2))
	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
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
	plt.pcolor(h_coordinates,v_coordinates,np.nanmean(laser_temperature_minus_background_crop,axis=0),cmap='rainbow')
	plt.colorbar().set_label('Temp [°C]')
	plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--',label='guess for laser size')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--')
	# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
	# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--')
	plt.title(preamble_4_prints+'Average temperature increment in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm prelim radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
	plt.xlabel('Horizontal axis [m]')
	plt.ylabel('Vertical axis [m]')
	plt.legend(loc='best', fontsize='small')
	figure_index+=1
	plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(20, 10))
	temp = np.nanmean(np.sort(laser_temperature_minus_background_crop,axis=0)[-int(len(laser_temperature_minus_background_crop)*experimental_laser_duty):],axis=0)
	plt.pcolor(h_coordinates,v_coordinates,temp,cmap='rainbow')
	plt.colorbar().set_label('Temp [°C]')
	plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'k--',label='guess for laser size')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'k--')
	# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
	# plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--')
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


	# basetemp=np.nanmean(datatempcrop[0,frame-7:frame+7,1:-1,1:-1],axis=0)
	if laser_framerate>500:
		smoothing_size = 1
	else:
		smoothing_size = 1
	if smoothing_size==1:
		dTdt=np.divide(laser_temperature_minus_background_crop[2:,1:-1,1:-1]-laser_temperature_minus_background_crop[:-2,1:-1,1:-1],2*dt).astype(np.float32)
		dTdt_std=np.divide((laser_temperature_std_minus_background_crop[2:,1:-1,1:-1]**2 + laser_temperature_std_minus_background_crop[:-2,1:-1,1:-1]**2)**0.5,2*dt).astype(np.float32)
	else:
		laser_temperature_minus_background_crop_time_median = generic_filter(laser_temperature_minus_background_crop,np.nanmedian,size=[1,smoothing_size,smoothing_size])
		laser_temperature_std_minus_background_crop_time_median = generic_filter((laser_temperature_std_minus_background_crop**2),np.nansum,size=[1,smoothing_size,smoothing_size])**0.5 /(smoothing_size**2)
		dTdt=np.divide(laser_temperature_minus_background_crop_time_median[2:,1:-1,1:-1]-laser_temperature_minus_background_crop_time_median[:-2,1:-1,1:-1],2*dt).astype(np.float32)
		dTdt_std=np.divide((laser_temperature_std_minus_background_crop_time_median[2:,1:-1,1:-1]**2 + laser_temperature_std_minus_background_crop_time_median[:-2,1:-1,1:-1]**2)**0.5,2*dt).astype(np.float32)
	# # I need to clean up further the time varying component from the 29Hz noise
	# dTdt = coleval.clear_oscillation_central2([dTdt],laser_framerate,oscillation_search_window_end=(len(dTdt)-1)/(laser_framerate),plot_conparison=False)[0]
	# dTdt = median_filter(dTdt,size=[5,1,1])
	# in order to increase precision I remove the background image
	if smoothing_size==1:
		d2Tdx2=np.divide(laser_temperature_minus_background_crop[1:-1,1:-1,2:]-np.multiply(2,laser_temperature_minus_background_crop[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop[1:-1,1:-1,:-2],dx**2).astype(np.float32)
		d2Tdx2_std=np.divide((laser_temperature_std_minus_background_crop[1:-1,1:-1,2:]**2+np.multiply(2,laser_temperature_std_minus_background_crop[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop[1:-1,1:-1,:-2]**2)**0.5,dx**2).astype(np.float32)
		d2Tdy2=np.divide(laser_temperature_minus_background_crop[1:-1,2:,1:-1]-np.multiply(2,laser_temperature_minus_background_crop[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop[1:-1,:-2,1:-1],dx**2).astype(np.float32)
		d2Tdy2_std=np.divide((laser_temperature_std_minus_background_crop[1:-1,2:,1:-1]**2+np.multiply(2,laser_temperature_std_minus_background_crop[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop[1:-1,:-2,1:-1]**2)**0.5,dx**2).astype(np.float32)
	else:
		laser_temperature_minus_background_crop_space_median_1 = generic_filter(laser_temperature_minus_background_crop,np.nanmedian,size=[smoothing_size,smoothing_size,1])
		laser_temperature_std_minus_background_crop_space_median_1 = generic_filter((laser_temperature_std_minus_background_crop**2),np.nansum,size=[smoothing_size,smoothing_size,1])**0.5 /(smoothing_size**2)
		laser_temperature_minus_background_crop_space_median_2 = generic_filter(laser_temperature_minus_background_crop,np.nanmedian,size=[smoothing_size,1,smoothing_size])
		laser_temperature_std_minus_background_crop_space_median_2 = generic_filter((laser_temperature_std_minus_background_crop**2),np.nansum,size=[smoothing_size,1,smoothing_size])**0.5 /(smoothing_size**2)
		d2Tdx2=np.divide(laser_temperature_minus_background_crop_space_median_1[1:-1,1:-1,2:]-np.multiply(2,laser_temperature_minus_background_crop_space_median_1[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop_space_median_1[1:-1,1:-1,:-2],dx**2).astype(np.float32)
		d2Tdx2_std=np.divide((laser_temperature_std_minus_background_crop_space_median_1[1:-1,1:-1,2:]**2+np.multiply(2,laser_temperature_std_minus_background_crop_space_median_1[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop_space_median_1[1:-1,1:-1,:-2]**2)**0.5,dx**2).astype(np.float32)
		d2Tdy2=np.divide(laser_temperature_minus_background_crop_space_median_2[1:-1,2:,1:-1]-np.multiply(2,laser_temperature_minus_background_crop_space_median_2[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop_space_median_2[1:-1,:-2,1:-1],dx**2).astype(np.float32)
		d2Tdy2_std=np.divide((laser_temperature_std_minus_background_crop_space_median_2[1:-1,2:,1:-1]**2+np.multiply(2,laser_temperature_std_minus_background_crop_space_median_2[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop_space_median_2[1:-1,:-2,1:-1]**2)**0.5,dx**2).astype(np.float32)
	d2Tdxy = np.ones_like(dTdt).astype(np.float32)*np.nan
	d2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2[:,nan_ROI_mask[1:-1,1:-1]],d2Tdy2[:,nan_ROI_mask[1:-1,1:-1]])
	del d2Tdx2,d2Tdy2
	d2Tdxy_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	d2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2_std[:,nan_ROI_mask[1:-1,1:-1]]**2,d2Tdy2_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5
	del d2Tdx2_std,d2Tdy2_std
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	negd2Tdxy_std=d2Tdxy_std
	T4=(laser_temperature_minus_background_crop[1:-1,1:-1,1:-1]+np.nanmean(reference_background_temperature_crop)+zeroC)**4
	T4_std=T4**(3/4) *4 *laser_temperature_std_minus_background_crop[1:-1,1:-1,1:-1]	# the error resulting from doing the average on the whole ROI is completely negligible
	T04=(np.nanmean(reference_background_temperature_crop)+zeroC)**4 *np.ones_like(laser_temperature_minus_background_crop[1:-1,1:-1,1:-1])
	T04_std=0
	T4_T04 = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04[:,nan_ROI_mask[1:-1,1:-1]] = (T4[:,nan_ROI_mask[1:-1,1:-1]]-T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	T4_T04_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]] = ((T4_std[:,nan_ROI_mask[1:-1,1:-1]]**2+T04_std**2)**0.5).astype(np.float32)

	# ktf=(np.multiply(conductivityscaled,foilthicknessscaled)).astype(np.float32)
	# BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
	# BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]*T4_T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	# BBrad_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	# BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]*T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	# diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
	# diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (ktf[nan_ROI_mask[1:-1,1:-1]]*negd2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	# diffusion_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	# diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (ktf[nan_ROI_mask[1:-1,1:-1]]*negd2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	# timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
	# timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (ktf[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]*dTdt[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	# timevariation_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	# timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (ktf[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]*dTdt_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	#
	# powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
	# powernoback_std = np.ones_like(dTdt)*np.nan
	# powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)

	temp_BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	temp_BBrad_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	temp_diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	temp_diffusion_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	temp_timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	temp_timevariation_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	temp_powernoback = (sample_properties['thickness']*temp_diffusion + (1/sample_properties['diffusivity'])*sample_properties['thickness']*temp_timevariation + sample_properties['emissivity']*temp_BBrad).astype(np.float32)
	del dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4,T4_std,T04,T04_std

	frames_for_one_pulse = laser_framerate/experimental_laser_frequency
	def power_vs_space_sampling_explorer():
		from uncertainties import ufloat
		area_multiplier = np.flip(np.array([4,3,2.5,2,1.7,1.5,1.3,1.15,1,0.9,0.8,0.7,0.6,0.55,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.08]),axis=0)
		number_of_pixels = []
		all_dr = []
		fitted_powers = []
		for dr in (np.nanmean([dhorizontal_loc,dvertical_loc])*area_multiplier):
			select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr**2)[1:-1,1:-1]
			number_of_pixels.append(np.nansum(select))
			totalpower=np.nansum(temp_diffusion[:,select],axis=(-1))/Ptthermalconductivity
			totalpower_filtered_1 = generic_filter(totalpower[:int(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/15//2*2+1))])
			fitted_power_2 = [ufloat(np.median(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)]),np.std(np.sort(totalpower_filtered_1)[:-int(len(totalpower_filtered_1)*experimental_laser_duty)])),ufloat(np.median(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]),np.std(np.sort(totalpower_filtered_1)[-int(len(totalpower_filtered_1)*experimental_laser_duty):]))]
			fitted_powers.append(fitted_power_2)
			all_dr.append(dr)
		return dict([('number_of_pixels',np.array(number_of_pixels)),('fitted_powers',np.array(fitted_powers)),('all_dr',np.array(all_dr))])
	power_vs_space_sampling = power_vs_space_sampling_explorer()

	dr = (power_vs_space_sampling['all_dr'])[nominal_values(power_vs_space_sampling['fitted_powers'][:,1][power_vs_space_sampling['all_dr']<=dr]).argmax()]

	if focus_status == 'focused':
		dr_total_power_minimum = 5
		# dr_total_power = 17
	elif focus_status == 'partially_defocused':
		dr_total_power_minimum = 7
		# dr_total_power = 20
	elif focus_status == 'fully_defocused':
		dr_total_power_minimum = 10
		# dr_total_power = 25
	else:
		print('focus type '+focus_status+" unknown, used 'fully_defocused' settings")
		# dr_total_power = 16
		dr_total_power_minimum = 25

	limit = 110
	test=0
	while True:
		select1 = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>(limit)**2)[1:-1,1:-1]
		if np.nansum(nan_ROI_mask[1:-1,1:-1][select1])>300:
			break
		else:
			limit -= 10
	fig, ax = plt.subplots( 2,1,figsize=(10, 14), squeeze=False, sharex=True)
	time_averaged_BBrad_over_duty = np.nanmean(temp_BBrad[:int(len(temp_BBrad)//frames_for_one_pulse*frames_for_one_pulse)],axis=0)
	test = []
	for value in np.arange(5,limit-10):
		select2 = np.logical_and( (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=(value+2)**2)[1:-1,1:-1] , (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>value**2)[1:-1,1:-1] )
		test.append(np.nanmean(time_averaged_BBrad_over_duty[select2])-np.nanmean(time_averaged_BBrad_over_duty[select1]))
	dr_total_power_BB = np.arange(5,limit-10)[np.abs(np.array(test)-BBtreshold).argmin()]
	ax[0,0].plot(np.arange(5,limit-10)*dx,test)
	ax[0,0].axhline(y=BBtreshold,linestyle='--',color='k',label='treshold')
	ax[0,0].axvline(x=dr_total_power_BB*dx,linestyle='--',color='r',label='detected')
	ax[0,0].axvline(x=dr_total_power_minimum*dx,linestyle='--',color='b',label='minimum')
	ax[0,0].set_ylim(top=min(np.nanmax(test),BBtreshold*5),bottom=np.nanmin(test))
	ax[0,0].set_xlim(left=0)
	ax[0,0].set_ylabel('black body')
	ax[0,0].legend(loc='best', fontsize='x-small')
	ax[0,0].grid()
	temp = np.nanmax(laser_temperature_minus_background_crop[1:-1],axis=(1,2))
	time_averaged_diffusion_over_duty = np.nanmean(temp_diffusion[temp>np.sort(temp)[-int(len(temp)*experimental_laser_duty)]],axis=0)
	test = []
	for value in np.arange(5,limit-10):
		select1 = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>(value)**2)[1:-1,1:-1]
		select2 = np.logical_and( (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=(value+4)**2)[1:-1,1:-1] , (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)>value**2)[1:-1,1:-1] )
		test.append(np.nanmean(time_averaged_diffusion_over_duty[select1]))
	dr_total_power_diff = np.arange(5,limit-10)[np.abs((test[:40]+2*np.max(np.abs(test[40:])))).argmin()]
	ax[1,0].plot(np.arange(5,limit-10)*dx,test)
	ax[1,0].axhline(y=-2*np.nanmax(np.abs(test[40:])),linestyle='--',color='k')
	ax[1,0].axvline(x=dr_total_power_diff*dx,linestyle='--',color='r')
	ax[1,0].axvline(x=dr_total_power_minimum*dx,linestyle='--',color='b')
	ax[1,0].set_ylim(bottom=-4*np.nanmax(np.abs(test[40:])),top=np.nanmax(test))
	ax[1,0].set_xlim(left=0)
	ax[1,0].set_ylabel('diffusion')
	ax[1,0].set_xlabel('radious of power sum [mm]')
	ax[1,0].grid()
	dr_total_power = np.max([dr_total_power_diff,dr_total_power_minimum,dr_total_power_BB])
	# temp = np.logical_and(power_vs_space_sampling['all_dr']>dr*1.2,power_vs_space_sampling['all_dr']<np.max(power_vs_space_sampling['all_dr']))
	# dr_total_power = (power_vs_space_sampling['all_dr'][temp])[nominal_values(power_vs_space_sampling['fitted_powers'][:,1][temp]).argmin()]
	# dr_total_power = dr*2
	select = (((horizontal_coord-horizontal_loc)**2 + (vertical_coord-vertical_loc)**2)<=dr_total_power**2)[1:-1,1:-1]
	fig.suptitle(preamble_4_prints+'Search for power sum size in '+laser_to_analyse+'\nfound %.3gmm' %(dr_total_power*dx))
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

	partial_BBrad=np.multiply(np.nansum(temp_BBrad[:,select],axis=(-1)),dx**2)
	partial_BBrad_std=np.nansum(temp_BBrad_std[:,select]**2,axis=(-1))**0.5*(dx**2)
	partial_diffusion=np.multiply(np.nansum(temp_diffusion[:,select],axis=(-1)),dx**2)
	partial_diffusion_std=np.nansum(temp_diffusion_std[:,select]**2,axis=(-1))**0.5*(dx**2)
	partial_timevariation=np.multiply(np.nansum(temp_timevariation[:,select],axis=(-1)),dx**2)
	partial_timevariation_std=np.nansum(temp_timevariation_std[:,select]**2,axis=(-1))**0.5*(dx**2)
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

	BBrad = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
	BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	BBrad_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
	BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
	diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
	diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
	timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
	timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	del temp_BBrad_std,temp_diffusion_std,temp_timevariation,temp_timevariation_std
	powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
	powernoback_std = np.ones_like(powernoback)*np.nan
	powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)

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
	time_averaged_power_over_duty = np.nanmean(powernoback[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],axis=0)/experimental_laser_duty
	time_averaged_power_over_duty_std = np.nanstd(powernoback[:int(len(powernoback)//frames_for_one_pulse*frames_for_one_pulse)],axis=0)/experimental_laser_duty
	plt.pcolor(h_coordinates,v_coordinates,time_averaged_power_over_duty,cmap='rainbow',vmin=np.nanmin(time_averaged_power_over_duty[select]),vmax=np.nanmax(time_averaged_power_over_duty[select]))
	plt.colorbar().set_label('Power [W/m2]')
	# plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--',label='found laser size')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--')
	plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
	plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--')
	plt.title(preamble_4_prints+'Power source shape in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm laser radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
	plt.xlabel('Horizontal axis [mm]')
	plt.ylabel('Vertical axis [mm]')
	plt.legend(loc='best', fontsize='small')
	figure_index+=1
	plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(20, 10))
	time_averaged_diffusion_over_duty = np.nanmean(temp_diffusion[temp>np.sort(temp)[-int(len(temp)*experimental_laser_duty)]],axis=0)
	plt.pcolor(h_coordinates,v_coordinates,generic_filter(time_averaged_diffusion_over_duty,np.mean,size=[3,3]),cmap='rainbow',vmax=0)#vmin=np.nanmin(time_averaged_diff_over_duty[select]),vmax=np.nanmax(time_averaged_diff_over_duty[select]))
	plt.colorbar().set_label('Power [W/m2]')
	# plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--',label='found laser size')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--')
	plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
	plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--')
	plt.title(preamble_4_prints+'Diffusion component shape in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm laser radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
	plt.xlabel('Horizontal axis [mm]')
	plt.ylabel('Vertical axis [mm]')
	plt.legend(loc='best', fontsize='small')
	figure_index+=1
	plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')
	del temp_diffusion

	plt.figure(figsize=(20, 10))
	time_averaged_BBrad_over_duty = np.nanmean(temp_BBrad[:int(len(temp_BBrad)//frames_for_one_pulse*frames_for_one_pulse)],axis=0)
	plt.pcolor(h_coordinates,v_coordinates,time_averaged_BBrad_over_duty,cmap='rainbow',vmax=BBtreshold*5)
	plt.colorbar().set_label('Power [W/m2]')
	# plt.errorbar(horizontal_loc*dx,vertical_loc*dx,xerr=dhorizontal_loc/2*dx,yerr=dvertical_loc/2*dx,color='k',linestyle='--')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc + np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--',label='found laser size')
	plt.plot((horizontal_loc + np.arange(-dr,+dr+dr/10,dr/10))*dx,(vertical_loc - np.abs(dr**2-np.arange(-dr,+dr+dr/10,dr/10)**2)**0.5)*dx,'r--')
	plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc + np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--',label='area accounted for sum')
	plt.plot((horizontal_loc + np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10))*dx,(vertical_loc - np.abs(dr_total_power**2-np.arange(-dr_total_power,+dr_total_power+dr_total_power/10,dr_total_power/10)**2)**0.5)*dx,'k--')
	plt.title(preamble_4_prints+'BB component shape in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixels, [%.3g,%.3g]mm\n laser located at [%.3g,%.3g]mm laser radious %.3gmm' %(foilhorizwpixel*dx*1e3,foilvertwpixel*dx*1e3,horizontal_loc*dx*1e3,vertical_loc*dx*1e3,dr*dx*1e3))
	plt.xlabel('Horizontal axis [mm]')
	plt.ylabel('Vertical axis [mm]')
	plt.legend(loc='best', fontsize='small')
	figure_index+=1
	plt.savefig(path_to_save_figures+laser_to_analyse[-6:] + path_to_save_figures2 + 'FIG'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')
	del temp_BBrad


	totalpower_filtered_1 = generic_filter(totalpower[:int(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/15//2*2+1))])
	totalpower_filtered_1_std = generic_filter((totalpower_std**2)/max(1,int(laser_framerate/experimental_laser_frequency/30//2*2+1)),np.nansum,size=[max(1,int(laser_framerate/experimental_laser_frequency/30//2*2+1))])**0.5
	totalpower_filtered_2 = generic_filter(totalpower[:int(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse)],np.mean,size=[max(1,int(laser_framerate/experimental_laser_frequency*experimental_laser_duty/1.5//2*2+1))])
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
	time_axis = (time_of_experiment[1:-1]-time_of_experiment[1])/1e6
	ax[0,0].plot(time_axis,totalpower,label='totalpower')
	ax[0,0].plot(time_axis,totaltimevariation,label='totaltimevariation')
	ax[0,0].plot(time_axis,totalBBrad,label='totalBBrad')
	ax[0,0].plot(time_axis,totaldiffusion,label='totaldiffusion')
	ax[0,0].plot(time_axis[:int(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse)],totalpower_filtered_1,linewidth=3,label='totalpower filtered')
	ax[0,0].plot(time_axis[:int(len(totalpower)//frames_for_one_pulse*frames_for_one_pulse)],totalpower_filtered_2,linewidth=3,label='totalpower super filtered')
	ax[0,0].plot([time_axis[0],time_axis[-1]],[laser_power_detected_nominal_properties]*2,color='k',linestyle='--',linewidth=2,label='power mean / duty cycle')
	ax[0,0].errorbar([time_axis[0],time_axis[-1]],[nominal_values(fitted_power_2[0])]*2,yerr=[std_devs(fitted_power_2[0])]*2,color='k',linestyle=':',linewidth=2,label='power upper/lower median')
	ax[0,0].errorbar([time_axis[0],time_axis[-1]],[nominal_values(fitted_power_2[1])]*2,yerr=[std_devs(fitted_power_2[1])]*2,color='k',linestyle=':',linewidth=2)
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

	full_saved_file_dict = dict(laser_dict)
	full_saved_file_dict['totalpower_nominal_properties']=totalpower
	full_saved_file_dict['totalpower_std_nominal_properties']=totalpower_std
	full_saved_file_dict['laser_power_detected_nominal_properties']=laser_power_detected_nominal_properties
	full_saved_file_dict['laser_power_detected_std_nominal_properties']=np.nansum(totalpower_std**2/len(totalpower_std))**0.5
	full_saved_file_dict['time_averaged_power_over_duty_nominal_properties']=time_averaged_power_over_duty
	full_saved_file_dict['time_averaged_power_over_duty_std_nominal_properties']=time_averaged_power_over_duty_std
	# full_saved_file_dict['fitted_power']=fitted_power
	full_saved_file_dict['partial_BBrad']=partial_BBrad
	full_saved_file_dict['partial_BBrad_std']=partial_BBrad_std
	full_saved_file_dict['partial_diffusion']=partial_diffusion
	full_saved_file_dict['partial_diffusion_std']=partial_diffusion_std
	full_saved_file_dict['partial_timevariation']=partial_timevariation
	full_saved_file_dict['partial_timevariation_std']=partial_timevariation_std
	full_saved_file_dict['laser_location']=[horizontal_loc,dhorizontal_loc,vertical_loc,dvertical_loc,dr]
	full_saved_file_dict['negative_partial_BBrad_time_mean']=negative_partial_BBrad_time_mean
	full_saved_file_dict['negative_partial_BBrad_time_std']=negative_partial_BBrad_time_std
	full_saved_file_dict['negative_partial_BBrad_space_mean']=negative_partial_BBrad_space_mean
	full_saved_file_dict['negative_partial_BBrad_space_std']=negative_partial_BBrad_space_std
	full_saved_file_dict['negative_partial_diffusion_time_mean']=negative_partial_diffusion_time_mean
	full_saved_file_dict['negative_partial_diffusion_time_std']=negative_partial_diffusion_time_std
	full_saved_file_dict['negative_partial_diffusion_space_mean']=negative_partial_diffusion_space_mean
	full_saved_file_dict['negative_partial_diffusion_space_std']=negative_partial_diffusion_space_std
	full_saved_file_dict['negative_partial_timevariation_time_mean']=negative_partial_timevariation_time_mean
	full_saved_file_dict['negative_partial_timevariation_time_std']=negative_partial_timevariation_time_std
	full_saved_file_dict['negative_partial_timevariation_space_mean']=negative_partial_timevariation_space_mean
	full_saved_file_dict['negative_partial_timevariation_space_std']=negative_partial_timevariation_space_std
	full_saved_file_dict['power_vs_space_sampling']=power_vs_space_sampling
	full_saved_file_dict['power_vs_time_sampling']=power_vs_time_sampling
	full_saved_file_dict['laser_input_power']=laser_to_analyse_power
	full_saved_file_dict['time_of_experiment'] = time_of_experiment
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
for index in range(len(all_laser_to_analyse)):
	try:
		function_a(index)
	except Exception as e:
		print('index '+str(index)+' failed')
		logging.exception('with error: ' + str(e))


#
