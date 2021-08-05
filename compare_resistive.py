# Created 27/07/2021
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

import netCDF4

path = '/common/uda-scratch/jlovell/abm044308.nc'
f = netCDF4.Dataset(path)

core_res_bolometer = f['abm']['core']['brightness'][:].data
time = f['abm']['core']['time'][:].data

plt.figure()
plt.imshow(core_res_bolometer[np.logical_and(time>0,time<1)],'rainbow',extent=[1,core_res_bolometer.shape[1],0,1],vmin=0,vmax=2e5)
plt.axes().set_aspect(10)
plt.colorbar()
plt.pause(0.01)

plt.figure()
plt.plot(time,core_res_bolometer[:,7])
plt.plot(time,core_res_bolometer[:,8])
plt.pause(0.01)


midplane = core_res_bolometer[:,8]

EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
print('reading '+EFIT_path_default+'/epm044308.nc')
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm044377.nc')

efit_reconstruction.mag_axis_r


core_tangential_common_point = [-0.2165,1.734,0]	# x,y,z	17 to 28
core_tangential_arrival = []
core_tangential_arrival.append([-1.99,0.196,0])
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([-1.903,-0.615,0])
core_tangential_arrival.append([-1.806,-0.861,0])
core_tangential_arrival.append([-1.665,-1.108,0])
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([np.nan]*3)
core_tangential_arrival.append([-1.064,-1.694,0])
core_tangential_arrival.append([-0.808,-1.829,0])
core_tangential_arrival.append([-0.533,-1.928,0])
core_tangential_arrival.append([-0.23,0.123,0])
core_tangential_arrival = np.array(core_tangential_arrival)

resolution = 10000
core_tangential_location_on_foil = []
for i in range(len(core_tangential_arrival)):
	point_location = np.array([np.linspace(core_tangential_arrival[i][0],core_tangential_common_point[0],resolution),np.linspace(core_tangential_arrival[i][1],core_tangential_common_point[1],resolution),[core_tangential_arrival[i][2]]*resolution]).T
	point_location = coleval.find_location_on_foil(point_location)
	core_tangential_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))

core_poloidal_common_point = [1.755,0,90]	# r,z,teta	01 to 16
core_poloidal_arrival = []
core_poloidal_arrival.append([0.417,1.424,90])
core_poloidal_arrival.append([0.335,1.311,90])
core_poloidal_arrival.append([0.335,1.093,90])
core_poloidal_arrival.append([0.311,0.921,90])
core_poloidal_arrival.append([0.285,0.703,90])
core_poloidal_arrival.append([0.261,0.493,90])
core_poloidal_arrival.append([0.261,0.285,90])
core_poloidal_arrival.append([0.261,0.09,90])
core_poloidal_arrival.append([0.261,-0.09,90])
core_poloidal_arrival.append([0.261,-0.285,90])
core_poloidal_arrival.append([0.261,-0.493,90])
core_poloidal_arrival.append([0.285,-0.703,90])
core_poloidal_arrival.append([0.311,-0.921,90])
core_poloidal_arrival.append([0.333,-1.095,90])
core_poloidal_arrival.append([0.335,-1.311,90])
core_poloidal_arrival.append([0.417,-1.424,90])
core_poloidal_arrival = np.array(core_poloidal_arrival)

resolution = 10000
core_poloidal_location_on_foil = []
for i in range(len(core_poloidal_arrival)):
	point_location = np.array([np.linspace(core_poloidal_arrival[i][0],core_poloidal_common_point[0],resolution),np.linspace(core_poloidal_arrival[i][1],core_poloidal_common_point[1],resolution),[core_poloidal_arrival[i][2]]*resolution]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	core_poloidal_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))

divertor_poloidal_common_point = [1.846,-1.564,90]	# r,z,teta	01 to 16
divertor_poloidal_arrival = []
divertor_poloidal_arrival.append([1.32,-2.066,90])
divertor_poloidal_arrival.append([1.27,-2.066,90])
divertor_poloidal_arrival.append([1.213,-2.066,90])
divertor_poloidal_arrival.append([1.115,-2.066,90])
divertor_poloidal_arrival.append([1.09,-2.066,90])
divertor_poloidal_arrival.append([1.05,-2.03,90])
divertor_poloidal_arrival.append([1.02,-2,90])
divertor_poloidal_arrival.append([0.985,-1.965,90])
divertor_poloidal_arrival.append([0.95,-1.93,90])
divertor_poloidal_arrival.append([0.92,-1.9,90])
divertor_poloidal_arrival.append([0.88,-1.86,90])
divertor_poloidal_arrival.append([0.855,-1.82,90])
divertor_poloidal_arrival.append([0.8,-1.78,90])
divertor_poloidal_arrival.append([0.76,-1.74,90])
divertor_poloidal_arrival.append([0.715,-1.69,90])
divertor_poloidal_arrival.append([0.665,-1.645,90])
divertor_poloidal_arrival = np.array(divertor_poloidal_arrival)

resolution = 10000
divertor_poloidal_location_on_foil = []
for i in range(len(divertor_poloidal_arrival)):
	point_location = np.array([np.linspace(divertor_poloidal_arrival[i][0],divertor_poloidal_common_point[0],resolution),np.linspace(divertor_poloidal_arrival[i][1],divertor_poloidal_common_point[1],resolution),[divertor_poloidal_arrival[i][2]]*resolution]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	divertor_poloidal_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))



plt.figure()
for i in range(len(core_tangential_location_on_foil)):
	if np.sum(np.isfinite(np.max(core_tangential_location_on_foil[i],axis=1)))!=0:
		plt.plot(core_tangential_location_on_foil[i][:,0],core_tangential_location_on_foil[i][:,1],label=str(i+17))

for i in range(len(core_poloidal_location_on_foil)):
	if np.sum(np.isfinite(np.max(core_poloidal_location_on_foil[i],axis=1)))!=0:
		plt.plot(core_poloidal_location_on_foil[i][:,0],core_poloidal_location_on_foil[i][:,1],label=str(i+1))

for i in range(len(divertor_poloidal_location_on_foil)):
	if np.sum(np.isfinite(np.max(divertor_poloidal_location_on_foil[i],axis=1)))!=0:
		plt.plot(divertor_poloidal_location_on_foil[i][:,0],divertor_poloidal_location_on_foil[i][:,1],'--',label=str(i+1))

plt.legend(loc='best',fontsize='xx-small')
plt.plot([0,0.07,0.07,0,0],[0,0,0.09,0.09,0],'k')
plt.pause(0.01)









exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())
structure_point_location_on_foil = []
for time in range(len(stucture_r)):
	point_location = np.array([stucture_r[time],stucture_z[time],stucture_t[time]]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	structure_point_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))
fueling_point_location_on_foil = []
for time in range(len(fueling_r)):
	point_location = np.array([fueling_r[time],fueling_z[time],fueling_t[time]]).T
	point_location = coleval.point_toroidal_to_cartesian(point_location)
	point_location = coleval.find_location_on_foil(point_location)
	fueling_point_location_on_foil.append(coleval.absolute_position_on_foil_to_foil_coord(point_location))
foil_size = [0.07,0.09]

foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
flat_foil_properties = dict([])
flat_foil_properties['thickness'] = 2.093616658223934e-06
flat_foil_properties['emissivity'] = 1
flat_foil_properties['diffusivity'] = 1.03*1e-5

laser_to_analyse = '/home/ffederic/work/irvb/MAST-U/2021-07-15/IRVB-MASTU_shot-44492.npz'
print('starting power analysis' + laser_to_analyse)

laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
laser_dict.allow_pickle=True
# full_saved_file_dict = dict(laser_dict)

time_full = laser_dict['full_frame'].all()['time_full']
laser_framerate = 1/np.mean(np.sort(np.diff(time_full))[2:-2])
laser_int_time = laser_dict['IntegrationTime']

shot_number = int(laser_to_analyse[-9:-4])
path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
if not os.path.exists(path_power_output):
	os.makedirs(path_power_output)

# saved_file_dict_short = np.load(laser_to_analyse[:-4]+'_short.npz')
# saved_file_dict_short.allow_pickle=True
# saved_file_dict_short = dict(saved_file_dict_short)

# try:
# 	try:
# 		foilemissivityscaled = saved_file_dict_short['power_data'].all()['foil_properties']['emissivity']['dfsgdfg']	# I want them all
# 		foilthicknessscaled = saved_file_dict_short['power_data'].all()['foil_properties']['thickness']
# 		reciprdiffusivityscaled = saved_file_dict_short['power_data'].all()['foil_properties']['diffusivity']
# 		nan_ROI_mask = saved_file_dict_short['other'].all()['nan_ROI_mask']
# 		dx = full_saved_file_dict['power_data'].all()['dx']
#
# 		temp_BBrad = laser_dict['power_data'].all()['temp_BBrad']
# 		temp_BBrad_std = laser_dict['power_data'].all()['temp_BBrad_std']
# 		temp_diffusion = laser_dict['power_data'].all()['temp_diffusion']
# 		temp_diffusion_std = laser_dict['power_data'].all()['temp_diffusion_std']
# 		temp_timevariation = laser_dict['power_data'].all()['temp_timevariation']
# 		temp_timevariation_std = laser_dict['power_data'].all()['temp_timevariation_std']
#
# 		BBrad = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
# 		BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
# 		BBrad_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
# 		BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
# 		diffusion = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
# 		diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
# 		diffusion_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
# 		diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
# 		timevariation = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
# 		timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
# 		timevariation_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
# 		timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
#
# 		powernoback = saved_file_dict_short['power_data'].all()['powernoback']
# 		powernoback_std = saved_file_dict_short['power_data'].all()['powernoback_std']
#
# 		foilhorizw=0.09	# m
# 		foilvertw=0.07	# m
#
# 	except:
print('missing '+laser_to_analyse[:-4]+'_short.npz'+' file. rigenerated')
# full_saved_file_dict.allow_pickle=True
# full_saved_file_dict = dict(laser_dict)
saved_file_dict_short = dict([])
foilhorizw= foil_position_dict['foilhorizw']
foilvertw=foil_position_dict['foilvertw']
foilhorizwpixel = foil_position_dict['foilhorizwpixel']

laser_digitizer_ID = laser_dict['uniques_digitizer_ID']
time_partial = []
laser_temperature_no_dead_pixels_crop = []
laser_temperature_std_no_dead_pixels_crop = []
reference_background_temperature_crop = []
reference_background_temperature_std_crop = []
for i in range(len(laser_digitizer_ID)):
	laser_temperature_no_dead_pixels_crop.append(( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_crop_median'] + laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_crop_minus_median'].astype(np.float32).T ).T )
	laser_temperature_std_no_dead_pixels_crop.append(( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_crop_median'] + laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_crop_minus_median'].astype(np.float32).T ).T )
	time_partial.append(laser_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['time'])
	reference_background_temperature_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['reference_background_temperature_crop'] )
	reference_background_temperature_std_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['reference_background_temperature_std_crop'] )
nan_ROI_mask = laser_dict['only_foil'].all()['nan_ROI_mask']
time_full = laser_dict['full_frame'].all()['time_full']

shrink_factor_t = 3
shrink_factor_x = 1
binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
print('working on binning \n'+binning_type)
seconds_for_reference_frame = 1	# s
BBrad_full = []
BBrad_std_full = []
diffusion_full = []
diffusion_std_full = []
timevariation_full = []
timevariation_std_full = []
powernoback_full = []
powernoback_std_full = []
time_full_binned = []
laser_temperature_crop_binned_full = []
laser_temperature_minus_background_crop_binned_full = []
timesteps = np.inf
laser_framerate_binned = laser_framerate/shrink_factor_t/len(laser_digitizer_ID)	# I add laser_framerate_binned/len(laser_digitizer_ID) because I keep the data from the 2 digitizers always separated and add them later, so the time resolution will always be divided by the number of them
plt.figure(10,figsize=(20, 10))
plt.title('Oscillation after rotation\nbinning' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x))
for i in range(len(laser_digitizer_ID)):
	laser_temp_filtered,nan_ROI_mask = coleval.proper_homo_binning_t_2D(laser_temperature_no_dead_pixels_crop[i],shrink_factor_t,shrink_factor_x)
	laser_temp_ref = coleval.proper_homo_binning_2D(reference_background_temperature_crop[i],shrink_factor_x)
	full_average = np.mean(laser_temperature_no_dead_pixels_crop[i],axis=(-1,-2))
	full_spectra = np.fft.fft(full_average)
	full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
	full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (laser_framerate/len(laser_digitizer_ID)))
	plt.plot(full_freq,full_magnitude*(100**i),color=color[i],label='full frame dig '+str(laser_digitizer_ID[i]))
	full_average = np.mean(laser_temp_filtered,axis=(-1,-2))
	full_spectra = np.fft.fft(full_average)
	full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
	full_freq = np.fft.fftfreq(len(full_magnitude), d=1/laser_framerate_binned)
	plt.plot(full_freq,full_magnitude*(100**i),'--k')
	time_partial_binned = coleval.proper_homo_binning_t(time_partial[i],shrink_factor_t)
	# select_time = np.logical_and(time_partial_binned>0,time_partial_binned<1.5)	# this is done before rotating, now
	# time_partial_binned = time_partial_binned[select_time]
	# initial_mean = np.nanmean(laser_temp_filtered[select_time],axis=(1,2))
	# plt.plot(time_partial_binned,initial_mean,color=color[i],label='initial, dig '+str(laser_digitizer_ID[i]))
	laser_temp_filtered_std = 1/shrink_factor_x*coleval.proper_homo_binning_t_2D(laser_temperature_std_no_dead_pixels_crop[i]**2,shrink_factor_t,shrink_factor_x,type='sum')[0]**0.5
	laser_temp_ref_std = 1/shrink_factor_x*coleval.proper_homo_binning_2D(reference_background_temperature_std_crop[i]**2,shrink_factor_x,type='sum')[0]**0.5
	laser_temperature_crop_binned = cp.deepcopy(laser_temp_filtered)
	laser_temperature_std_crop_binned = cp.deepcopy(laser_temp_filtered_std)
	reference_background_temperature_crop_binned = cp.deepcopy(laser_temp_ref)
	reference_background_temperature_std_crop_binned = cp.deepcopy(laser_temp_ref_std)
	laser_temperature_minus_background_crop_binned = laser_temperature_crop_binned-reference_background_temperature_crop_binned
	laser_temperature_std_minus_background_crop_binned = (laser_temperature_std_crop_binned**2+reference_background_temperature_std_crop_binned**2)**0.5
	laser_temperature_crop_binned_full.append(laser_temperature_crop_binned[1:-1,1:-1,1:-1])
	laser_temperature_minus_background_crop_binned_full.append(laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])
	if len(np.unique(np.diff(time_partial_binned)))<5:
		temp = np.polyval(np.polyfit(np.arange(len(time_partial_binned)),time_partial_binned,1),np.arange(len(time_partial_binned)))
		time_partial_binned = temp - (temp[0]-time_partial_binned[0])


	# FOIL PROPERTY ADJUSTMENT
	if False:	# spatially resolved foil properties from Japanese producer
		foilemissivityscaled=resize(foilemissivity,reference_background_temperature_crop_binned.shape,order=0)[1:-1,1:-1]
		foilthicknessscaled=resize(foilthickness,reference_background_temperature_crop_binned.shape,order=0)[1:-1,1:-1]
		conductivityscaled=np.multiply(Ptthermalconductivity,np.ones(np.array(reference_background_temperature_crop_binned.shape)-2))
		reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones(np.array(reference_background_temperature_crop_binned.shape)-2))
	elif True:	# homogeneous foil properties from foil experiments
		foilemissivityscaled=flat_foil_properties['emissivity']*np.ones(np.array(reference_background_temperature_crop_binned.shape)-2)
		foilthicknessscaled=flat_foil_properties['thickness']*np.ones(np.array(reference_background_temperature_crop_binned.shape)-2)
		conductivityscaled=Ptthermalconductivity*np.ones(np.array(reference_background_temperature_crop_binned.shape)-2)
		reciprdiffusivityscaled=(1/flat_foil_properties['diffusivity'])*np.ones(np.array(reference_background_temperature_crop_binned.shape)-2)

	# ani = coleval.movie_from_data(np.array([laser_temperature_minus_background_crop]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=-start_time_of_pulse,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]')
	# dt=1/laser_framerate/shrink_factor_t
	dt = time_partial_binned[2:]-time_partial_binned[:-2]
	dx=foilhorizw/foilhorizwpixel*shrink_factor_x
	dTdt=np.divide((laser_temperature_crop_binned[2:,1:-1,1:-1]-laser_temperature_crop_binned[:-2,1:-1,1:-1]).T,2*dt).T.astype(np.float32)
	dTdt_std=np.divide((laser_temperature_std_crop_binned[2:,1:-1,1:-1]**2 + laser_temperature_std_crop_binned[:-2,1:-1,1:-1]**2).T**0.5,2*dt).T.astype(np.float32)
	d2Tdx2=np.divide(laser_temperature_minus_background_crop_binned[1:-1,1:-1,2:]-np.multiply(2,laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop_binned[1:-1,1:-1,:-2],dx**2).astype(np.float32)
	d2Tdx2_std=np.divide((laser_temperature_std_minus_background_crop_binned[1:-1,1:-1,2:]**2+np.multiply(2,laser_temperature_std_minus_background_crop_binned[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop_binned[1:-1,1:-1,:-2]**2)**0.5,dx**2).astype(np.float32)
	d2Tdy2=np.divide(laser_temperature_minus_background_crop_binned[1:-1,2:,1:-1]-np.multiply(2,laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop_binned[1:-1,:-2,1:-1],dx**2).astype(np.float32)
	d2Tdy2_std=np.divide((laser_temperature_std_minus_background_crop_binned[1:-1,2:,1:-1]**2+np.multiply(2,laser_temperature_std_minus_background_crop_binned[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop_binned[1:-1,:-2,1:-1]**2)**0.5,dx**2).astype(np.float32)
	d2Tdxy = np.ones_like(dTdt).astype(np.float32)*np.nan
	d2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2[:,nan_ROI_mask[1:-1,1:-1]],d2Tdy2[:,nan_ROI_mask[1:-1,1:-1]])
	del d2Tdx2,d2Tdy2
	d2Tdxy_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	d2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2_std[:,nan_ROI_mask[1:-1,1:-1]]**2,d2Tdy2_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5
	del d2Tdx2_std,d2Tdy2_std
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	negd2Tdxy_std=d2Tdxy_std
	T4=(laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1]+np.nanmean(reference_background_temperature_crop_binned)+zeroC)**4
	T4_std=T4**(3/4) *4 *laser_temperature_std_minus_background_crop_binned[1:-1,1:-1,1:-1]	# the error resulting from doing the average on the whole ROI is completely negligible
	T04=(np.nanmean(reference_background_temperature_crop_binned)+zeroC)**4 *np.ones_like(laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])
	T04_std=0
	T4_T04 = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04[:,nan_ROI_mask[1:-1,1:-1]] = (T4[:,nan_ROI_mask[1:-1,1:-1]]-T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	T4_T04_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]] = ((T4_std[:,nan_ROI_mask[1:-1,1:-1]]**2+T04_std**2)**0.5).astype(np.float32)

	BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	BBrad_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	diffusion_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	timevariation_std = np.ones_like(dTdt).astype(np.float32)*np.nan
	timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
	del dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4,T4_std,T04,T04_std
	powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
	powernoback_std = np.ones_like(powernoback)*np.nan
	powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)

	BBrad_full.append(BBrad)
	BBrad_std_full.append(BBrad_std)
	diffusion_full.append(diffusion)
	diffusion_std_full.append(diffusion_std)
	timevariation_full.append(timevariation)
	timevariation_std_full.append(timevariation_std)
	powernoback_full.append(powernoback)
	powernoback_std_full.append(powernoback_std)
	time_full_binned.append(time_partial_binned[1:-1])
	timesteps = min(timesteps,len(time_partial_binned[1:-1]))
plt.close('all')

for i in range(len(laser_digitizer_ID)):
	BBrad_full[i] = BBrad_full[i][:timesteps]
	BBrad_std_full[i] = BBrad_std_full[i][:timesteps]
	diffusion_full[i] = diffusion_full[i][:timesteps]
	diffusion_std_full[i] = diffusion_std_full[i][:timesteps]
	timevariation_full[i] = timevariation_full[i][:timesteps]
	timevariation_std_full[i] = timevariation_std_full[i][:timesteps]
	powernoback_full[i] = powernoback_full[i][:timesteps]
	powernoback_std_full[i] = powernoback_std_full[i][:timesteps]
	time_full_binned[i] = time_full_binned[i][:timesteps]
	laser_temperature_crop_binned_full[i] = laser_temperature_crop_binned_full[i][:timesteps]
	laser_temperature_minus_background_crop_binned_full[i] = laser_temperature_minus_background_crop_binned_full[i][:timesteps]
BBrad_full = np.nanmean(BBrad_full,axis=0)
BBrad_std_full = 0.5*np.nansum(np.array(BBrad_std_full)**2,axis=0)**0.5
diffusion_full = np.nanmean(diffusion_full,axis=0)
diffusion_std_full = 0.5*np.nansum(np.array(diffusion_std_full)**2,axis=0)**0.5
timevariation_full = np.nanmean(timevariation_full,axis=0)
timevariation_std_full = 0.5*np.nansum(np.array(timevariation_std_full)**2,axis=0)**0.5
powernoback_full = np.nanmean(powernoback_full,axis=0)
powernoback_std_full = 0.5*np.nansum(np.array(powernoback_std_full)**2,axis=0)**0.5
time_full_binned = np.nanmean(time_full_binned,axis=0)
laser_temperature_crop_binned_full = np.nanmean(laser_temperature_crop_binned_full,axis=0)
laser_temperature_minus_background_crop_binned_full = np.nanmean(laser_temperature_minus_background_crop_binned_full,axis=0)

horizontal_coord = np.arange(np.shape(powernoback_full[0])[1])
vertical_coord = np.arange(np.shape(powernoback_full[0])[0])
horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
horizontal_coord = (horizontal_coord+1+0.5)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
vertical_coord = (vertical_coord+1+0.5)*dx
horizontal_coord -= foilhorizw*0.5+0.0198
vertical_coord -= foilvertw*0.5-0.0198
distance_from_vertical = (horizontal_coord**2+vertical_coord**2)**0.5
pinhole_to_foil_vertical = 0.008 + 0.003 + 0.002 + 0.045	# pinhole holder, washer, foil holder, standoff
pinhole_to_pixel_distance = (pinhole_to_foil_vertical**2 + distance_from_vertical**2)**0.5

etendue = np.ones_like(powernoback_full[0]) * (np.pi*(0.002**2)) / (pinhole_to_pixel_distance**2)	# I should include also the area of the pixel, but that is already in the w/m2 power
etendue *= (pinhole_to_foil_vertical/pinhole_to_pixel_distance)**2	 # cos(a)*cos(b). for pixels not directly under the pinhole both pinhole and pixel are tilted respect to the vertical, with same angle.

brightness_full = 4*np.pi*powernoback_full/etendue


# I want to avoid weirdness close to the frame of the foil
border_to_neglect = 10	# pixels
brightness_full_crop = brightness_full[:,border_to_neglect:-border_to_neglect,border_to_neglect:-border_to_neglect]

horizontal_coord = np.arange(np.shape(brightness_full_crop[0])[1])
vertical_coord = np.arange(np.shape(brightness_full_crop[0])[0])
horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
horizontal_coord = (horizontal_coord+1+0.5+border_to_neglect)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
vertical_coord = (vertical_coord+1+0.5+border_to_neglect)*dx
horizontal_coord -= foilhorizw*0.5	# later I will use the centre of the foil as reference, so I don't have to consider the close point of the pinhole here
vertical_coord -= foilvertw*0.5

peak_horizontal = np.unravel_index(np.abs(horizontal_coord-0.0198).argmin(), horizontal_coord.shape)[1]

binning_factor_x = 20
binning_factor_x +=1
brightness_full_horizontal = brightness_full_crop[:,:,peak_horizontal-(binning_factor_x-1)//2:peak_horizontal+(binning_factor_x-1)//2+1]
brightness_full_horizontal = np.mean(brightness_full_horizontal,axis=-1)

plt.figure()
plt.imshow(brightness_full_horizontal)
plt.pause(0.01)


# functions to draw the x-point on images or movies
Rf=1.54967	# m
plane_equation = np.array([1,-1,0,2**0.5 * Rf])
pinhole_location = np.array([-1.04087,1.068856,-0.7198])
centre_of_foil = np.array([-1.095782166, 1.095782166, -0.7])
foil_size = [0.07,0.09]
R_centre_column = 0.261	# m

pixels_location = np.array([centre_of_foil]*len(brightness_full_horizontal[0]))
pixels_location[:,2] -= horizontal_coord[:,peak_horizontal].min()
pixels_location[:,0] -= vertical_coord[:,0]/(2**0.5)
pixels_location[:,1] -= vertical_coord[:,0]/(2**0.5)

B = 2*pixels_location*(pinhole_location-pixels_location)
B[:,2]=0
B = np.sum(B,axis=1)
C = (pinhole_location-pixels_location)**2
C[:,2]=0
C = np.sum(C,axis=1)
t = -B/(2*C)
radial_distance = (pixels_location+((pinhole_location-pixels_location).T*t).T)**2
radial_distance[:,2]=0
radial_distance = np.sum(radial_distance,axis=1)**0.5

# plt.figure()
# plt.plot(radial_distance)
# plt.pause(0.01)
# plt.figure()
# plt.plot(radial_distance,brightness_full_horizontal[150])
# plt.pause(0.01)
# # technique from A New Method for Numerical Abel-Inversion, Oliver, J., 2013

n = np.linspace(0,6,6+1)
r_fine = np.linspace(0,np.max(radial_distance)+0.001,100000)
dr = np.mean(np.diff(r_fine))
fn_r = (1-((-1)**n)*np.cos(n*np.pi*np.array([r_fine-R_centre_column]*len(n)).T/np.max(radial_distance))).T
fn_r[0] = 1
fn_r[:,r_fine<R_centre_column] = 0
hn_y = []
for y in radial_distance:
	temp = fn_r*r_fine/((r_fine**2-y**2)**0.5)*dr
	if y<R_centre_column:
		hn_y.append(np.sum(temp[:,r_fine>=y],axis=1)/2)	# I don't see the half of the emission on the back of the centre column
	else:
		hn_y.append(np.sum(temp[:,r_fine>=y],axis=1))
hn_y = np.array(hn_y)

plt.figure()
for i in range(len(n)):
	plt.plot(hn_y[:,i])
plt.pause(0.01)

def residual_ext(i_time,brightness_full_horizontal=brightness_full_horizontal,hn_y=hn_y):
	def residual(amplitudes):
		return (brightness_full_horizontal[i_time] - np.sum(hn_y*amplitudes,axis=1))**2
	return residual
emissivity = []
for i_time in range(len(brightness_full_horizontal)):
	guess = np.ones_like(n)
	sol = scipy.optimize.least_squares(residual_ext(i_time), guess, verbose=0,ftol=1e-11)
	emissivity.append(np.sum(fn_r.T*sol.x,axis=1))

	# plt.figure()
	# plt.plot(radial_distance,brightness_full_horizontal[i_time])
	# plt.plot(radial_distance,np.sum(hn_y*sol.x,axis=1),'x')
	# plt.pause(0.01)
	#
	# plt.figure()
	# plt.plot(r_fine,np.sum(fn_r.T*sol.x,axis=1))
	# plt.pause(0.01)

plt.figure()
plt.imshow(emissivity,'rainbow',vmin=0,extent=[0,r_fine.max(),time_full_binned.min(),time_full_binned.max()])
plt.colorbar()
plt.pause(0.01)
