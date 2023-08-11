
print('STARTING pre processing'+laser_to_analyse)

try:
	laser_dict = coleval.read_IR_file(laser_to_analyse)
	# laser_dict = np.load(laser_to_analyse+'.npz')
except:
	print('missing .npz file. rigenerated')
	# full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse+'.ats')
	# np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
	# laser_dict = np.load(laser_to_analyse+'.npz')
	laser_dict = coleval.read_IR_file(laser_to_analyse,force_regeneration=True)

path_reference_frames = all_path_reference_frames[index]
limited_ROI = all_laser_to_analyse_ROI[index]

# laser_dict = np.load(laser_to_analyse+'.npz')
# laser_counts, laser_digitizer_ID = coleval.separate_data_with_digitizer(laser_dict)
laser_counts = laser_dict['data']
try:
	laser_counts += laser_dict['data_median']
except:
	pass
time_of_experiment = laser_dict['time_of_measurement']
mean_time_of_experiment = np.mean(time_of_experiment)
laser_digitizer = laser_dict['digitizer_ID']
if np.diff(time_of_experiment).max()>np.median(np.diff(time_of_experiment))*1.1:
	hole_pos = np.diff(time_of_experiment).argmax()
	if hole_pos<len(time_of_experiment)/2:
		time_of_experiment = time_of_experiment[hole_pos+1:]
		laser_digitizer = laser_digitizer[hole_pos+1:]
		laser_counts = laser_counts[hole_pos+1:]
	else:
		time_of_experiment = time_of_experiment[:-(hole_pos+1)]
		laser_digitizer = laser_digitizer[:-(hole_pos+1)]
		laser_counts = laser_counts[:-(hole_pos+1)]
laser_counts, laser_digitizer_ID = coleval.generic_separate_with_digitizer(laser_counts,laser_digitizer)
time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)
laser_framerate = cp.deepcopy(laser_dict['FrameRate'])	# full, not divided by digitizer
laser_int_time = cp.deepcopy(laser_dict['IntegrationTime'])
print('laser_framerate')
print(laser_framerate)
print('laser_int_time')
print(laser_int_time)

# # !!!!! T E M P O R A R Y !!!!!!
# # for now I have to skip stuff that is not full frame, so I have to do this
# if laser_framerate>383*1.2:
# 	continue

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
# background_timestamps = [(np.mean(np.load(file+'.npz')['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
# background_counts = [(np.load(file+'.npz')['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
# background_counts_std = [(np.load(file+'.npz')['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']-laser_int_time)<laser_int_time/100)]
background_timestamps = []
background_counts=[]
background_counts_std=[]
for file in path_reference_frames:
	if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_framerate)<laser_framerate/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100):
		background_timestamps.append((np.mean(coleval.read_IR_file(file)['time_of_measurement'])))
		background_counts.append((coleval.read_IR_file(file)['data_time_avg_counts']))
		background_counts_std.append((coleval.read_IR_file(file)['data_time_avg_counts_std']))
# background_timestamps = [(np.mean(coleval.read_IR_file(file)['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_dict['FrameRate'])<laser_dict['FrameRate']/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]
# background_counts = [(coleval.read_IR_file(file)['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_dict['FrameRate'])<laser_dict['FrameRate']/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]
# background_counts_std = [(coleval.read_IR_file(file)['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(coleval.read_IR_file(file)['FrameRate']-laser_dict['FrameRate'])<laser_dict['FrameRate']/100,np.abs(coleval.read_IR_file(file)['IntegrationTime']-laser_int_time)<laser_int_time/100)]

# Calculating the background image
temp = np.sort(np.abs(background_timestamps-mean_time_of_experiment))[1]
ref = np.arange(len(background_timestamps))[np.abs(background_timestamps-mean_time_of_experiment)<=temp]
reference_background = (background_counts[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_counts[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
reference_background_std = ((background_counts_std[ref[0]]**2)*(background_timestamps[ref[1]]-mean_time_of_experiment) + (background_counts_std[ref[1]]**2)*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
reference_background_flat = np.mean(reference_background,axis=(1,2))

# before of the temperature conversion I'm better to remove the oscillation
# I try to use all the length of the record at first
# I only do this for frequencies lower than 30Hz, otherwise it trashes good signal
if False:
	laser_counts_filtered = [coleval.clear_oscillation_central2([counts.astype(np.float32)],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_end=(len(counts)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False)[0] for counts in laser_counts]
else:
	laser_counts_filtered = [np.array(counts.astype(np.float32)) for counts in laser_counts]


if False:	# I want to check if this filters the oscillation
	plt.figure()
	dpixel = 5
	# spectra_orig = np.fft.fft(laser_counts[0][:,226,93], axis=0)
	spectra_orig = np.fft.fft(np.mean(laser_counts[0][:,93-dpixel:93+dpixel+1,226-dpixel:226+dpixel+1],axis=(1,2)), axis=0)
	magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
	freq = np.fft.fftfreq(len(magnitude), d=1 / (laser_framerate/len(laser_digitizer_ID)))
	plt.plot(freq,magnitude)
	# spectra_orig = np.fft.fft(laser_counts_filtered[0][:,100,100], axis=0)
	spectra_orig = np.fft.fft(np.mean(laser_counts_filtered[0][:,93-dpixel:93+dpixel+1,226-dpixel:226+dpixel+1],axis=(1,2)), axis=0)
	magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
	freq = np.fft.fftfreq(len(magnitude), d=1 / (laser_framerate/len(laser_digitizer_ID)))
	plt.plot(freq,magnitude,'--')
	plt.semilogy()
	plt.pause(0.01)

if False:	# done with functions now
	from uncertainties import correlated_values,ufloat
	from uncertainties.unumpy import nominal_values,std_devs,uarray
	def function_a(arg):
		out = np.sum(np.power(np.array([arg[0].tolist()]*arg[2]).T,np.arange(arg[2]-1,-1,-1))*arg[1],axis=1)
		out1 = nominal_values(out)
		out2 = std_devs(out)
		return [out1,out2]

	def function_b(arg):
		out = np.sum(np.power(np.array([arg[0].tolist()]*arg[2]).T,np.arange(arg[2]-1,-1,-1))*arg[1],axis=0)
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
	laser_temperature,laser_temperature_std = coleval.count_to_temp_poly_multi_digitizer(laser_counts_filtered,params,errparams,laser_digitizer_ID,number_cpu_available,reference_background=reference_background,reference_background_std=reference_background_std,reference_background_flat=reference_background_flat,report=1)

# plt.figure()
# plt.imshow(laser_temperature[0][0],'rainbow',origin='lower')
# plt.colorbar().set_label('Temp [°C]')
# plt.title('Only foil in frame '+str(0)+' in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
# plt.xlabel('Horizontal axis [pixles]')
# plt.ylabel('Vertical axis [pixles]')
# plt.pause(0.01)


# I don't downgrade the temperature because I'll do the difference and I don't want to loose resolution
laser_temperature_minus_background_median = [np.median(data) for data in laser_temperature]
laser_temperature_minus_background_minus_median_downgraded = [np.float16(data-median) for data,median in zip(laser_temperature,laser_temperature_minus_background_median)]
laser_temperature_minus_background_std_median = [np.median(data) for data in laser_temperature_std]
laser_temperature_minus_background_std_minus_median_downgraded = [np.float16(data-median) for data,median in zip(laser_temperature_std,laser_temperature_minus_background_std_median)]

try:
	laser_dict.allow_pickle=True
except:
	pass
full_saved_file_dict = dict(laser_dict)
full_saved_file_dict['laser_temperature_minus_background_median']=laser_temperature_minus_background_median	# this is NOT minus_background, that is a relic of what I did before, but I keep it not to recreate all .npz, same for the next3
full_saved_file_dict['laser_temperature_minus_background_minus_median_downgraded']=laser_temperature_minus_background_minus_median_downgraded
full_saved_file_dict['laser_temperature_minus_background_std_median']=laser_temperature_minus_background_std_median
full_saved_file_dict['laser_temperature_minus_background_std_minus_median_downgraded']=laser_temperature_minus_background_std_minus_median_downgraded
full_saved_file_dict['laser_counts_filtered']=laser_counts_filtered	# I need this for the BB calculations
np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
print('FINISHED '+laser_to_analyse)


#
