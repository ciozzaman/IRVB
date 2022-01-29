from PIL import Image
MASTU_wireframe = Image.open("/home/ffederic/work/irvb/MAST-U/Calcam_theoretical_view.png")
MASTU_wireframe_resize = np.array(np.asarray(MASTU_wireframe)[:,:,3].tolist()).astype(np.float64)
MASTU_wireframe_resize[MASTU_wireframe_resize>0] = 1
MASTU_wireframe_resize = np.flip(MASTU_wireframe_resize,axis=0)
masked = np.ma.masked_where(MASTU_wireframe_resize == 0, MASTU_wireframe_resize)

efit_reconstruction = None

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

# foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix
flat_foil_properties = dict([])
# flat_foil_properties['thickness'] = 1.4214e-06
# flat_foil_properties['emissivity'] = 0.89
# flat_foil_properties['diffusivity'] = 1.03e-5
# # changed 30/07/2021
# flat_foil_properties['thickness'] = 1.4e-6
# flat_foil_properties['emissivity'] = 0.9309305250670584
# flat_foil_properties['diffusivity'] = 9.999999999999999e-06
# # changed 30/07/2021
# flat_foil_properties['thickness'] = 1.4859095354482858e-06
# flat_foil_properties['emissivity'] = 0.9884061389741369
# flat_foil_properties['diffusivity'] = 1.045900223180454e-05
# # directly and blindly copied from Matt
# flat_foil_properties['thickness'] = 1.94e-6
# flat_foil_properties['emissivity'] = 0.85
# flat_foil_properties['diffusivity'] = 10.9e-6
# from 2021/09/17, Laser_data_analysis3_3.py
flat_foil_properties['thickness'] = 2.0531473351462095e-06
flat_foil_properties['emissivity'] = 0.9999999999999
flat_foil_properties['diffusivity'] = 1.0283685197530968e-05

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
temp_counts_no_dead_pixels_crop = []
temp_counts_std_no_dead_pixels_crop = []
temp_ref_counts_no_dead_pixels_crop = []
temp_ref_counts_std_no_dead_pixels_crop = []
BB_proportional_no_dead_pixels_crop = []
BB_proportional_std_no_dead_pixels_crop = []
for i in range(len(laser_digitizer_ID)):
	laser_temperature_no_dead_pixels_crop.append(( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_crop_median'] + laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_crop_minus_median'].astype(np.float32).T ).T )
	laser_temperature_std_no_dead_pixels_crop.append(( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_crop_median'] + laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_crop_minus_median'].astype(np.float32).T ).T )
	time_partial.append(laser_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['time'])
	reference_background_temperature_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['reference_background_temperature_crop'] )
	reference_background_temperature_std_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['reference_background_temperature_std_crop'] )
	BB_proportional_no_dead_pixels_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['BB_proportional_no_dead_pixels_crop'] )
	BB_proportional_std_no_dead_pixels_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['BB_proportional_std_no_dead_pixels_crop'] )
	temp_counts_no_dead_pixels_crop.append(( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['temp_counts_no_dead_pixels_crop_median'] + laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['temp_counts_no_dead_pixels_crop_minus_median'].astype(np.float32).T ).T )
	temp_counts_std_no_dead_pixels_crop.append(( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['temp_counts_std_no_dead_pixels_crop_median'] + laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['temp_counts_std_no_dead_pixels_crop_minus_median'].astype(np.float32).T ).T )
	temp_ref_counts_no_dead_pixels_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['temp_ref_counts_no_dead_pixels_crop'] )
	temp_ref_counts_std_no_dead_pixels_crop.append( laser_dict['only_foil'].all()[str(laser_digitizer_ID[i])]['temp_ref_counts_std_no_dead_pixels_crop'] )
nan_ROI_mask = laser_dict['only_foil'].all()['nan_ROI_mask']
time_full = laser_dict['full_frame'].all()['time_full']
ref_temperature = laser_dict['ref_temperature']
ref_temperature_std = laser_dict['ref_temperature_std']
photon_flux_over_temperature_interpolator = laser_dict['photon_flux_over_temperature_interpolator'].tolist()
laser_temperature_no_dead_pixels_minus_background_crop = []
laser_temperature_std_no_dead_pixels_minus_background_crop = []
for i in range(len(laser_digitizer_ID)):
	laser_temperature_no_dead_pixels_minus_background_crop.append( laser_temperature_no_dead_pixels_crop[i] - ref_temperature )
	laser_temperature_std_no_dead_pixels_minus_background_crop.append( (laser_temperature_std_no_dead_pixels_crop[i]**2 + ref_temperature**2)**0.5 )


# problem: apparently I have a significant amount of pixels with negative counts.
# I need to fix this to calculate properly the BB component
for i in range(len(laser_digitizer_ID)):
	plt.figure(figsize=(20, 10))
	if flag_use_of_first_frames_as_reference:
		plt.title('NOT USED\nMinimum relative temperature correction in '+laser_to_analyse+' dig '+str(laser_digitizer_ID[i]))
	else:
		plt.title('Minimum relative temperature correction in '+laser_to_analyse+' dig '+str(laser_digitizer_ID[i]))
	correction = np.min(generic_filter(laser_temperature_no_dead_pixels_crop[i][time_partial[i]<0.2]-ref_temperature,np.mean,size=[6,1,1]),axis=0)	# I need some smoothing so I do this
	correction_counts = np.min(generic_filter(temp_counts_no_dead_pixels_crop[i][time_partial[i]<0.2]-temp_ref_counts_no_dead_pixels_crop[i],np.mean,size=[6,1,1]),axis=0)	# I need some smoothing so I do this
	plt.imshow(np.flip(np.transpose(correction,(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
	plt.colorbar().set_label('temp increase [K]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	# plt.clim(vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
	plt.savefig(path_power_output + '/min_temp_correction_dig_'+ str(laser_digitizer_ID[i]) +'.eps', bbox_inches='tight')
	plt.close('all')
	if not flag_use_of_first_frames_as_reference:
		laser_temperature_no_dead_pixels_crop[i] -= correction
		laser_temperature_no_dead_pixels_minus_background_crop[i] -= correction
		temp_counts_no_dead_pixels_crop[i] -= correction_counts

	if False:
		plt.figure(figsize=(20, 10))
		plt.imshow(np.flip(np.transpose(laser_temperature_no_dead_pixels_crop[i][0]-ref_temperature,(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
		plt.colorbar().set_label('temp increase [K]')
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		plt.pause(0.01)

		plt.figure(figsize=(20, 10))
		plt.imshow(np.flip(np.transpose(laser_temperature_no_dead_pixels_crop[i][0]-correction-ref_temperature,(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
		plt.colorbar().set_label('temp increase [K]')
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		plt.pause(0.01)

for shrink_factor_x in [1,2,3,5,10]:
	grid_laplacian_binned = []
	for shrink_factor_t in [1,2,3,5]:
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
			laser_temperature_minus_background_crop_binned,nan_ROI_mask = coleval.proper_homo_binning_t_2D(laser_temperature_no_dead_pixels_minus_background_crop[i],shrink_factor_t,shrink_factor_x)
			BB_proportional_crop_binned = coleval.proper_homo_binning_2D(BB_proportional_no_dead_pixels_crop[i],shrink_factor_x)
			reference_background_temperature_crop_binned = coleval.proper_homo_binning_2D(reference_background_temperature_crop[i],shrink_factor_x)
			full_average = np.mean(laser_temperature_no_dead_pixels_minus_background_crop[i],axis=(-1,-2))
			full_spectra = np.fft.fft(full_average)
			full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
			full_freq = np.fft.fftfreq(len(full_magnitude), d=1 / (laser_framerate/len(laser_digitizer_ID)))
			plt.plot(full_freq,full_magnitude*(100**i),color=color[i],label='full frame dig '+str(laser_digitizer_ID[i]))
			full_average = np.mean(laser_temperature_minus_background_crop_binned,axis=(-1,-2))
			full_spectra = np.fft.fft(full_average)
			full_magnitude = 2 * np.abs(full_spectra) / len(full_spectra)
			full_freq = np.fft.fftfreq(len(full_magnitude), d=1/laser_framerate_binned)
			plt.plot(full_freq,full_magnitude*(100**i),'--k')
			time_partial_binned = coleval.proper_homo_binning_t(time_partial[i],shrink_factor_t)
			# select_time = np.logical_and(time_partial_binned>0,time_partial_binned<1.5)	# this is done before rotating, now
			# time_partial_binned = time_partial_binned[select_time]
			# initial_mean = np.nanmean(laser_temp_filtered[select_time],axis=(1,2))
			# plt.plot(time_partial_binned,initial_mean,color=color[i],label='initial, dig '+str(laser_digitizer_ID[i]))

			laser_temperature_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*coleval.proper_homo_binning_t_2D(laser_temperature_std_no_dead_pixels_crop[i]**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5
			# laser_temperature_std_minus_background_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*coleval.proper_homo_binning_t_2D(laser_temperature_std_minus_background_no_dead_pixels_crop[i]**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5
			temp_counts_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*coleval.proper_homo_binning_t_2D(temp_counts_std_no_dead_pixels_crop[i]**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5
			BB_proportional_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(coleval.proper_homo_binning_2D(BB_proportional_std_no_dead_pixels_crop[i]**2,shrink_factor_x,type='np.nansum')**0.5)
			reference_background_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(coleval.proper_homo_binning_2D(temp_ref_counts_std_no_dead_pixels_crop[i]**2,shrink_factor_x,type='np.nansum')**0.5)

			laser_temperature_crop_binned_full.append(laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1]+ref_temperature)
			laser_temperature_minus_background_crop_binned_full.append(laser_temperature_minus_background_crop_binned[1:-1,1:-1,1:-1])
			if len(np.unique(np.diff(time_partial_binned)))<5:
				temp = np.polyval(np.polyfit(np.arange(len(time_partial_binned)),time_partial_binned,1),np.arange(len(time_partial_binned)))
				time_partial_binned = temp - (temp[0]-time_partial_binned[0])

			plt.figure(figsize=(20, 10))
			temp = np.max(laser_temperature_minus_background_crop_binned,axis=(-1,-2)).argmax()
			plt.title('Foil search in '+laser_to_analyse+'\nhottest frame at %.4gsec' %(time_partial_binned[temp]) + ' binned ' +str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) +' with mask from Calcam'  + '\ngas injection points at R=0.261m, Z=-0.264m, '+r'$theta$'+'=105°(LX), 195°(DX)')
			plt.imshow(np.flip(np.transpose(laser_temperature_minus_background_crop_binned[temp],(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			# MASTU_wireframe_resize = MASTU_wireframe.resize(np.shape(laser_temperature_minus_background_crop_binned[temp]))
			# MASTU_wireframe_resize[MASTU_wireframe_resize==0] = np.nan
			plt.colorbar().set_label('temp increase [K]')
			# plt.imshow(masked, 'gray', interpolation='none', alpha=0.4,origin='lower',extent = [0,np.shape(laser_temperature_minus_background_crop_binned[temp])[0]-1,0,np.shape(laser_temperature_minus_background_crop_binned[temp])[1]-1])
			temp2 = laser_temperature_minus_background_crop_binned[0]
			for __i in range(len(fueling_point_location_on_foil)):
				plt.plot(np.array(fueling_point_location_on_foil[__i][:,0])*(np.shape(temp2)[1]-1)/foil_size[1],np.array(fueling_point_location_on_foil[__i][:,1])*(np.shape(temp2)[0]-1)/foil_size[0],'+k',markersize=40,alpha=0.5)
				plt.plot(np.array(fueling_point_location_on_foil[__i][:,0])*(np.shape(temp2)[1]-1)/foil_size[1],np.array(fueling_point_location_on_foil[__i][:,1])*(np.shape(temp2)[0]-1)/foil_size[0],'ok',markersize=5,alpha=0.5)
			for __i in range(len(structure_point_location_on_foil)):
				plt.plot(np.array(structure_point_location_on_foil[__i][:,0])*(np.shape(temp2)[1]-1)/foil_size[1],np.array(structure_point_location_on_foil[__i][:,1])*(np.shape(temp2)[0]-1)/foil_size[0],'--k',alpha=0.5)
			plt.xlabel('Horizontal axis [pixles]')
			plt.ylabel('Vertical axis [pixles]')
			# plt.clim(vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			plt.savefig(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_dig_'+ str(laser_digitizer_ID[i]) +'_temp_overlay.png', bbox_inches='tight')
			plt.close()
			# dropped to save memory
			# plt.figure(figsize=(20, 10))
			# plt.title('Foil search in '+laser_to_analyse+'\nhottest frame at %.4gsec' %(time_partial_binned[temp]) +' binned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]))
			# plt.imshow(np.flip(np.transpose(laser_temperature_minus_background_crop_binned[temp],(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			# plt.colorbar().set_label('temp increase [K]')
			# plt.xlabel('Horizontal axis [pixles]')
			# plt.ylabel('Vertical axis [pixles]')
			# # plt.clim(vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			# plt.savefig(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_dig_'+ str(laser_digitizer_ID[i]) +'_temp.eps', bbox_inches='tight')
			# plt.close()
			plt.figure(figsize=(20, 10))
			plt.title('Foil search in '+laser_to_analyse+'\nreference background binned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\nNOTE: NOT USED ANYMORE')
			plt.imshow(np.flip(np.transpose(reference_background_temperature_crop_binned,(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			plt.colorbar().set_label('temp increase [K]')
			plt.xlabel('Horizontal axis [pixles]')
			plt.ylabel('Vertical axis [pixles]')
			# plt.clim(vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			plt.savefig(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_dig_'+ str(laser_digitizer_ID[i]) +'_reference.eps', bbox_inches='tight')
			plt.close()

			# FOIL PROPERTY ADJUSTMENT
			if False:	# spatially resolved foil properties from Japanese producer
				foilemissivityscaled=resize(foilemissivity,BB_proportional_crop_binned.shape,order=0)[1:-1,1:-1]
				foilthicknessscaled=resize(foilthickness,BB_proportional_crop_binned.shape,order=0)[1:-1,1:-1]
				conductivityscaled=np.multiply(Ptthermalconductivity,np.ones(np.array(BB_proportional_crop_binned.shape)-2))
				reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones(np.array(BB_proportional_crop_binned.shape)-2))
			elif True:	# homogeneous foil properties from foil experiments
				foilemissivityscaled=flat_foil_properties['emissivity']*np.ones(np.array(BB_proportional_crop_binned.shape)-2)
				foilthicknessscaled=flat_foil_properties['thickness']*np.ones(np.array(BB_proportional_crop_binned.shape)-2)
				conductivityscaled=Ptthermalconductivity*np.ones(np.array(BB_proportional_crop_binned.shape)-2)
				reciprdiffusivityscaled=(1/flat_foil_properties['diffusivity'])*np.ones(np.array(BB_proportional_crop_binned.shape)-2)


			# ani = coleval.movie_from_data(np.array([laser_temperature_minus_background_crop]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=-start_time_of_pulse,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]')
			# dt=1/laser_framerate/shrink_factor_t
			dx=foilhorizw/foilhorizwpixel*shrink_factor_x
			if len(grid_laplacian_binned)==0:
				horizontal_coord = np.arange(np.shape(laser_temperature_minus_background_crop_binned)[2])
				vertical_coord = np.arange(np.shape(laser_temperature_minus_background_crop_binned)[1])
				horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
				grid_binned = np.array([[horizontal_coord.flatten()]*4,[vertical_coord.flatten()]*4]).T
				grid_laplacian_binned = -coleval.build_laplacian(grid_binned,diagonal_factor=0.5) / (dx**2) / 2	# the /2 comes from the fact that including the diagonals amounts to double counting, so i do a mean by summing half of it
			dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std = coleval.calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,laser_temperature_minus_background_crop_binned,ref_temperature,time_partial_binned,dx,temp_counts_std_crop_binned,BB_proportional_crop_binned,BB_proportional_std_crop_binned,reference_background_std_crop_binned,laser_temperature_std_crop_binned,nan_ROI_mask,grid_laplacian=grid_laplacian_binned)
			BBrad,diffusion,timevariation,powernoback,BBrad_std,diffusion_std,timevariation_std,powernoback_std = coleval.calc_temp_to_power_BB_2(dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std,nan_ROI_mask,foilemissivityscaled,foilthicknessscaled,reciprdiffusivityscaled,Ptthermalconductivity)

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
		plt.figure(10)
		plt.semilogy()
		plt.xlabel('freq [Hz]')
		plt.ylabel('amplitude [au]')
		plt.legend(loc='best', fontsize='x-small')
		plt.grid()
		plt.xlim(left=0)
		plt.savefig(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_after_rot_oscillation.eps', bbox_inches='tight')
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

		binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		saved_file_dict_short[binning_type] = dict([])
		saved_file_dict_short[binning_type]['foil_properties'] = dict([])
		saved_file_dict_short[binning_type]['foil_properties']['emissivity'] = np.float32(foilemissivityscaled)
		saved_file_dict_short[binning_type]['foil_properties']['thickness'] = np.float32(foilthicknessscaled)
		saved_file_dict_short[binning_type]['foil_properties']['diffusivity'] = np.float32(reciprdiffusivityscaled)
		saved_file_dict_short[binning_type]['laser_temperature_crop_binned_full'] = np.float32(laser_temperature_crop_binned_full)
		saved_file_dict_short[binning_type]['laser_temperature_minus_background_crop_binned_full'] = np.float32(laser_temperature_minus_background_crop_binned_full)
		saved_file_dict_short[binning_type]['BBrad_full'] = np.float32(BBrad_full)
		saved_file_dict_short[binning_type]['BBrad_std_full'] = np.float32(BBrad_std_full)
		saved_file_dict_short[binning_type]['diffusion_full'] = np.float32(diffusion_full)
		saved_file_dict_short[binning_type]['diffusion_std_full'] = np.float32(diffusion_std_full)
		saved_file_dict_short[binning_type]['timevariation_full'] = np.float32(timevariation_full)
		saved_file_dict_short[binning_type]['timevariation_std_full'] = np.float32(timevariation_std_full)
		saved_file_dict_short[binning_type]['powernoback_full'] = np.float32(powernoback_full)
		saved_file_dict_short[binning_type]['powernoback_std_full'] = np.float32(powernoback_std_full)
		saved_file_dict_short[binning_type]['time_full_binned'] = time_full_binned
		saved_file_dict_short[binning_type]['nan_ROI_mask'] = nan_ROI_mask
		saved_file_dict_short[binning_type]['laser_framerate_binned'] = laser_framerate_binned
		try:
			del saved_file_dict_short[binning_type]['powernoback']
			del saved_file_dict_short[binning_type]['powernoback_std']
		except:
			print('no legacy')

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
		peak_etendue = np.unravel_index(etendue.argmax(), etendue.shape)
		saved_file_dict_short[binning_type]['etendue'] = etendue

		plt.figure(figsize=(8, 5))
		plt.imshow(np.flip(np.transpose(etendue,(1,0)),axis=1)*(dx**2),'rainbow',origin='lower')
		plt.colorbar().set_label('Etendue [m2]')
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		plt.title('Etendue map')
		plt.savefig(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'etendue'+'.eps', bbox_inches='tight')
		plt.close('all')

		temp = powernoback_full[:,:,:int(np.shape(powernoback_full)[2]*0.75)]
		temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
		ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(powernoback_full,(0,2,1)),axis=2)]), laser_framerate_binned,timesteps=time_full_binned,integration=laser_int_time/1000,time_offset=time_full_binned[0],xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
		ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_power.mp4', fps=5*laser_framerate_binned/383, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

		brightness_full = 4*np.pi*powernoback_full/etendue
		brightness_std_full = 4*np.pi*powernoback_std_full/etendue
		saved_file_dict_short[binning_type]['brightness_full'] = np.float32(brightness_full)
		saved_file_dict_short[binning_type]['brightness_std_full'] = np.float32(brightness_std_full)
		temp = brightness_full[:,:,:int(np.shape(brightness_full)[2]*0.75)]
		temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
		ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(brightness_full,(0,2,1)),axis=2)]), laser_framerate_binned,timesteps=time_full_binned,integration=laser_int_time/1000,time_offset=time_full_binned[0],xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
		ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_brightness.mp4', fps=5*laser_framerate_binned/383, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

		if False:	# I speed up the process by not doing these. i have the GUI is I want to take a closer look
			temp = BBrad_full[:,:,:int(np.shape(BBrad_full)[2]*0.75)]
			temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
			ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(BBrad_full,(0,2,1)),axis=2)]), laser_framerate_binned,integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Power BB on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
			ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_power_BB.mp4', fps=5*laser_framerate_binned/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')

			temp = diffusion_full[:,:,:int(np.shape(diffusion_full)[2]*0.75)]
			temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
			ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(diffusion_full,(0,2,1)),axis=2)]), laser_framerate_binned,timesteps=time_full_binned,integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
			ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_power_diff.mp4', fps=5*laser_framerate_binned/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')

			temp = timevariation_full[:,:,:int(np.shape(timevariation_full)[2]*0.75)]
			temp = np.sort(temp[np.max(temp,axis=(1,2)).argmax()].flatten())
			ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(timevariation_full,(0,2,1)),axis=2)]), laser_framerate_binned,timesteps=time_full_binned,integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmin=0,extvmax=np.nanmean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Power time deriv. diffusion on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
			ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_power_dt.mp4', fps=5*laser_framerate_binned/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')
np.savez_compressed(laser_to_analyse[:-4]+'_short',**saved_file_dict_short)
# 	shrink_factor = 20
# 	temp = generic_filter(powernoback,np.nanmean,size=[1,shrink_factor,shrink_factor])
# 	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t,mask=masked, integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed spatially '+str([shrink_factor,shrink_factor]) + '\n', include_EFIT=True, pulse_ID=laser_to_analyse[-9:-4], overlay_x_point=True, overlay_mag_axis=True)
# 	ani.save(laser_to_analyse[:-4]+ '_power_smooth1.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
# 	plt.close('all')
# 	saved_file_dict_short['power_data']['smooth_'+str(1) + 'x' + str(shrink_factor) + 'x' + str(shrink_factor)] = temp
# 	temp = generic_filter(4*np.pi*powernoback/etendue,np.nanmean,size=[1,shrink_factor,shrink_factor])
# 	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed spatially '+str([shrink_factor,shrink_factor]) + '\n')
# 	ani.save(laser_to_analyse[:-4]+ '_brightness_smooth1.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
# 	plt.close('all')
# 	shrink_factor_t = 13
# 	shrink_factor_x = 10
# 	temp = generic_filter(powernoback,np.nanmean,size=[shrink_factor_t,shrink_factor_x,shrink_factor_x])
# 	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t,mask=masked, integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n', include_EFIT=True, pulse_ID=laser_to_analyse[-9:-4], overlay_x_point=True, overlay_mag_axis=True)
# 	ani.save(laser_to_analyse[:-4]+ '_power_smooth2.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
# 	plt.close('all')
# 	saved_file_dict_short['power_data']['smooth_'+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)] = temp
#
# 	plt.figure(figsize=(20, 10))
# 	temp_2 = np.max(temp,axis=(-1,-2)).argmax()
# 	plt.title('Foil search in '+laser_to_analyse+'\nhighest power frame at %.4gsec with mask from Calcam' %(time_full_binned[temp_2]) +' smooth '+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)+ '\ngas injection points at R=0.261m, Z=-0.264m, '+r'$\theta$'+'=105°(LX), 195°(DX)')
# 	plt.imshow(np.flip(np.transpose(temp[temp_2],(1,0)),axis=1),'rainbow',interpolation='none')#,origin='lower',vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp])
# 	# MASTU_wireframe_resize = MASTU_wireframe.resize(np.shape(temp[temp_2]))
# 	# MASTU_wireframe_resize[MASTU_wireframe_resize==0] = np.nan
# 	plt.colorbar().set_label('Power on foil [W/m2]')
# 	plt.imshow(masked, 'gray', interpolation='none', alpha=0.4,origin='lower',extent = [0,np.shape(temp[temp_2])[0]-1,0,np.shape(temp[temp_2])[1]-1])
# 	plt.xlabel('Horizontal axis [pixles]')
# 	plt.ylabel('Vertical axis [pixles]')
# 	# plt.clim(vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp_2])
# 	plt.savefig(laser_to_analyse[:-4]+'_power_overlay.png', bbox_inches='tight')
# 	plt.close('all')
# 	plt.figure(figsize=(20, 10))
# 	plt.title('Foil search in '+laser_to_analyse+'\nhighest power frame at %.4gsec' %(time_full_binned[temp_2]) +' smooth '+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x))
# 	plt.imshow(np.flip(np.transpose(temp[temp_2],(1,0)),axis=1),'rainbow',interpolation='none')#,origin='lower',vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp])
# 	plt.colorbar().set_label('Power on foil [W/m2]')
# 	plt.xlabel('Horizontal axis [pixles]')
# 	plt.ylabel('Vertical axis [pixles]')
# 	# plt.clim(vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp_2])
# 	plt.savefig(laser_to_analyse[:-4]+'_power.eps', bbox_inches='tight')
# 	plt.close('all')
#
# 	temp = generic_filter(4*np.pi*powernoback/etendue,np.nanmean,size=[shrink_factor_t,shrink_factor_x,shrink_factor_x])
# 	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full_binned[0],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
# 	ani.save(laser_to_analyse[:-4]+ '_brightness_smooth2.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
# 	plt.close('all')
#
#
# 	def bin_ndarray(ndarray, new_shape, operation='sum'):
# 	    """
# 	    Bins an ndarray in all axes based on the target shape, by summing or
# 	        averaging.
#
# 	    Number of output dimensions must match number of input dimensions and
# 	        new axes must divide old ones.
#
# 	    Example
# 	    -------
# 	    >>> m = np.arange(0,100,1).reshape((10,10))
# 	    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
# 	    >>> print(n)
#
# 	    [[ 22  30  38  46  54]
# 	     [102 110 118 126 134]
# 	     [182 190 198 206 214]
# 	     [262 270 278 286 294]
# 	     [342 350 358 366 374]]
#
# 	    """
# 	    operation = operation.lower()
# 	    if not operation in ['sum', 'mean']:
# 	        raise ValueError("Operation not supported.")
# 	    if ndarray.ndim != len(new_shape):
# 	        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
# 	                                                           new_shape))
# 	    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
# 	                                                  ndarray.shape)]
# 	    flattened = [l for p in compression_pairs for l in p]
# 	    ndarray = ndarray.reshape(flattened)
# 	    for i in range(len(new_shape)):
# 	        op = getattr(ndarray, operation)
# 	        ndarray = op(-1*(i+1))
# 	    return ndarray
#
# 	old_shape = np.array(np.shape(powernoback))
# 	for shrink_factor_t in [13,7]:
# 		shrink_factor_x = 8
# 		# new_shape=np.array([np.array(powernoback.shape)[0]//shrink_factor_t,np.array(powernoback.shape)[1]//shrink_factor_x,np.array(powernoback.shape)[2]//shrink_factor_x]).astype(int)
# 		# powernoback_short = powernoback[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
# 		new_shape=np.array([int(np.ceil(np.array(powernoback.shape)[0]/shrink_factor_t)),int(np.ceil(np.array(powernoback.shape)[1]/shrink_factor_x)),int(np.ceil(np.array(powernoback.shape)[2]/shrink_factor_x))]).astype(int)
# 		to_pad=np.array([shrink_factor_t-np.array(powernoback.shape)[0]%shrink_factor_t,shrink_factor_x-np.array(powernoback.shape)[1]%shrink_factor_x,shrink_factor_x-np.array(powernoback.shape)[2]%shrink_factor_x]).astype(int)
# 		to_pad_right = to_pad//2
# 		to_pad_left = to_pad - to_pad_right
# 		to_pad = np.array([to_pad_left,to_pad_right]).T
# 		a = np.pad(powernoback,to_pad,mode='mean',stat_length=((shrink_factor_t,shrink_factor_t),(shrink_factor_x,shrink_factor_x),(shrink_factor_x,shrink_factor_x)))
# 		a=bin_ndarray(a, new_shape=new_shape, operation='mean')
# 		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
# 		# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]',extvmax=a.max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
# 		ani.save(laser_to_analyse[:-4]+ '_power_bin_' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
# 		plt.close('all')
# 		saved_file_dict_short['power_data']['bin_'+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)] = a
#
# 		timevariation_short = timevariation[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
# 		a=bin_ndarray(timevariation_short, new_shape=new_shape, operation='mean')
# 		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
# 		ani.save(laser_to_analyse[:-4]+ '_dT_dt_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
# 		plt.close('all')
#
# 		diffusion_short = diffusion[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
# 		a=bin_ndarray(diffusion_short, new_shape=new_shape, operation='mean')
# 		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
# 		ani.save(laser_to_analyse[:-4]+ '_diffus_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
# 		plt.close('all')
#
# 		BBrad_short = BBrad[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
# 		a=bin_ndarray(BBrad_short, new_shape=new_shape, operation='mean')
# 		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
# 		ani.save(laser_to_analyse[:-4]+ '_BB_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
# 		plt.close('all')
#
# 	np.savez_compressed(laser_to_analyse[:-4]+'_short',**saved_file_dict_short)
# 	print('completed ' + laser_to_analyse)
#
# except Exception as e:
# 	print('FAILED ' + laser_to_analyse)
# 	logging.exception('with error: ' + str(e))
