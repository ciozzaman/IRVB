
print('starting power to emissivity conversion'+laser_to_analyse)

shot_number = int(laser_to_analyse[-9:-4])
path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
path_for_plots = path_power_output + '/invertions_log'
if not os.path.exists(path_for_plots):
	os.makedirs(path_for_plots)

laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
laser_int_time = laser_dict['IntegrationTime']

saved_file_dict_short = np.load(laser_to_analyse[:-4]+'_short.npz')
saved_file_dict_short.allow_pickle=True
saved_file_dict_short = dict(saved_file_dict_short)

all_binning_type = list(saved_file_dict_short.keys())
all_shrink_factor_x = []
all_shrink_factor_t = []
for binning_type in all_binning_type:
	all_shrink_factor_t.append(int(binning_type[binning_type.find('bin')+len('bin'):binning_type.find('x')]))
	all_shrink_factor_x.append(int(binning_type[binning_type.find('x')+len('x'):binning_type[binning_type.find('x')+len('x'):].find('x')+binning_type.find('x')+len('x')]))
all_shrink_factor_t = np.unique(all_shrink_factor_t)
all_shrink_factor_x = np.unique(all_shrink_factor_x)


# EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
# efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
# all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)


development_plots = False
inverted_dict = dict([])
try:
	inverted_dict['efit_reconstruction'] = efit_reconstruction
except:
	EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
	efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
	inverted_dict['efit_reconstruction'] = efit_reconstruction


# for grid_resolution in [8, 4, 2]:	# 8cm resolution is way not enough
# for grid_resolution in [4, 2]:
for grid_resolution in [2]:
	inverted_dict[str(grid_resolution)] = dict([])
	# grid_resolution = 8  # in cm
	foil_resolution = '187'

	foil_res = '_foil_pixel_h_' + str(foil_resolution)

	grid_type = 'core_res_' + str(grid_resolution) + 'cm'
	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'
	try:
		sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())
	except:
		sensitivities = np.load(path_sensitivity + '/sensitivity.npy')

	filenames = coleval.all_file_names(path_sensitivity, '.csv')[0]
	with open(os.path.join(path_sensitivity, filenames)) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			if row[0] == 'foil vertical pixels ':
				pixel_v = int(row[1])
			if row[0] == 'foil horizontal pixels ':
				pixel_h = int(row[1])
			if row[0] == 'pipeline type ':
				pipeline = row[1]
			if row[0] == 'type of volume grid ':
				grid_type = row[1]
		# print(row)

	directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
	grid_file = os.path.join(directory,'{}_rectilinear_grid.pickle'.format(grid_type))
	with open(grid_file, 'rb') as f:
		grid_data_all = pickle.load(f)
	grid_laplacian = grid_data_all['laplacian']
	grid_mask = grid_data_all['mask']
	grid_data = grid_data_all['voxels']
	grid_index_2D_to_1D_map = grid_data_all['index_2D_to_1D_map']
	grid_index_1D_to_2D_map = grid_data_all['index_1D_to_2D_map']

	sensitivities_reshaped = sensitivities.reshape((pixel_v,pixel_h,len(grid_laplacian)))
	sensitivities_reshaped = np.transpose(sensitivities_reshaped , (1,0,2))

	if grid_resolution==8:
		# temp=1e-3
		temp=1e-7
		temp2=0
	elif grid_resolution==2:
		temp=1e-4
		temp2=1e-4
	elif grid_resolution==4:
		temp=0
		temp2=0
	sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = coleval.reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)

	if development_plots:
		plt.figure()
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities_reshaped,axis=(0,1)),marker='s')
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.std(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
		plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.sum(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		# plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked,axis=(0,1)),marker='s')
		# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.std(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.max(grid_Z_derivate_masked,axis=(1)),marker='s')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.max(grid_R_derivate_masked,axis=(1)),marker='s')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar()
		plt.pause(0.01)

	# this step is to adapt the matrix to the size of the foil I measure, that can be slightly different
	binning_type = 'bin' + str(all_shrink_factor_t[0]) + 'x' + str(all_shrink_factor_x[0]) + 'x' + str(all_shrink_factor_x[0])
	shape = list(np.array(saved_file_dict_short[binning_type].all()['powernoback_full'].shape[1:])+2)	# +2 spatially because I remove +/- 1 pixel when I calculate the laplacian of the temperature
	if shape!=list(sensitivities_reshaped_masked.shape[:-1]):
		shape.extend([len(grid_laplacian_masked)])
		def mapping(output_coords):
			return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
		sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)
	else:
		sensitivities_reshaped_masked2 = cp.deepcopy(sensitivities_reshaped_masked)

	if development_plots:
		plt.figure()
		plt.imshow(np.sum(sensitivities_reshaped_masked2,axis=-1),'rainbow',origin='lower')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked2,axis=(0,1)),marker='s')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar()
		plt.pause(0.01)

		# plt.figure()
		# plt.imshow(sensitivities_reshaped_masked2[:,:,335],'rainbow',origin='lower')
		# plt.colorbar()
		# plt.pause(0.01)



	class calc_stuff_output:
		def __init__(self, exp, alpha, score_x, score_y):
			self.exp = exp
			self.alpha = alpha
			self.score_x = score_x
			self.score_y = score_y

	# for shrink_factor_x in np.flip(all_shrink_factor_x,axis=0):
	# for shrink_factor_x in all_shrink_factor_x:
	# for shrink_factor_x in [3,2,1]:
	for shrink_factor_x in [2]:
		inverted_dict[str(grid_resolution)][str(shrink_factor_x)] = dict([])
		sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
		sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
		sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

		if development_plots:
			plt.figure()
			plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned,axis=(0,1)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

		# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
		# ROI = np.array([[0.2,0.85],[0.1,0.9]])
		# ROI = np.array([[0.05,0.95],[0.05,0.95]])
		# ROI = np.array([[0.2,0.95],[0.1,1]])
		ROI1 = np.array([[0.03,0.80],[0.03,0.85]])
		ROI2 = np.array([[0.03,0.7],[0.03,0.91]])
		ROI_beams = np.array([[0.,0.3],[0.5,1]])
		sensitivities_binned_crop,selected_ROI = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse)

		plt.figure(figsize=(10,6))
		plt.imshow(np.flip(np.transpose(selected_ROI,(1,0)),axis=1),'rainbow',origin='lower')
		plt.plot([ROI1[0,0]-0.5,ROI1[0,1]-0.5,ROI1[0,1]-0.5,ROI1[0,0]-0.5,ROI1[0,0]-0.5],[ROI1[1,0]-0.5,ROI1[1,0]-0.5,ROI1[1,1]-0.5,ROI1[1,1]-0.5,ROI1[1,0]-0.5],'k')
		plt.plot([ROI2[0,0]-0.5,ROI2[0,1]-0.5,ROI2[0,1]-0.5,ROI2[0,0]-0.5,ROI2[0,0]-0.5],[ROI2[1,0]-0.5,ROI2[1,0]-0.5,ROI2[1,1]-0.5,ROI2[1,1]-0.5,ROI2[1,0]-0.5],'--k')
		plt.plot([ROI_beams[0,0]-0.5,ROI_beams[0,1]-0.5,ROI_beams[0,1]-0.5,ROI_beams[0,0]-0.5,ROI_beams[0,0]-0.5],[ROI_beams[1,0]-0.5,ROI_beams[1,0]-0.5,ROI_beams[1,1]-0.5,ROI_beams[1,1]-0.5,ROI_beams[1,0]-0.5],'-.k')
		# plt.colorbar()
		plt.title('Parts of the foil considered')
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_foil_area_considered.eps')
		# plt.pause(0.01)

		if development_plots:
			plt.figure()
			plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
			plt.plot([ROI1[0,0]-0.5,ROI1[0,1]-0.5,ROI1[0,1]-0.5,ROI1[0,0]-0.5,ROI1[0,0]-0.5],[ROI1[1,0]-0.5,ROI1[1,0]-0.5,ROI1[1,1]-0.5,ROI1[1,1]-0.5,ROI1[1,0]-0.5],'k')
			plt.plot([ROI2[0,0]-0.5,ROI2[0,1]-0.5,ROI2[0,1]-0.5,ROI2[0,0]-0.5,ROI2[0,0]-0.5],[ROI2[1,0]-0.5,ROI2[1,0]-0.5,ROI2[1,1]-0.5,ROI2[1,1]-0.5,ROI2[1,0]-0.5],'--k')
			plt.colorbar()
			plt.pause(0.01)

		if True:	# setting zero to the sensitivities I want to exclude
			sensitivities_binned_crop = cp.deepcopy(sensitivities_binned)
			sensitivities_binned_crop[np.logical_not(selected_ROI),:] = 0
		else:	# cutting sensitivity out of ROI
			sensitivities_binned_crop = sensitivities_binned[sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]

		if development_plots:
			plt.figure()
			plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned_crop,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
			# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
			plt.colorbar()
			plt.pause(0.01)

		select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-3)
		plt.figure(figsize=(10,6))
		# plt.imshow(np.flip(np.transpose(1*selected_ROI + 1*select_foil_region_with_plasma,(1,0)),axis=1),'rainbow',origin='lower')
		plt.imshow(np.flip(1*selected_ROI + 1*select_foil_region_with_plasma,axis=0),'rainbow',origin='lower')
		# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		# plt.colorbar()
		plt.title('blue: excluded area\nred: plasma area')
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_plasma_region.eps')
		plt.pause(0.01)
		select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()

		if development_plots:
			plt.figure()
			plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s',norm=LogNorm())
			plt.colorbar()
			plt.pause(0.01)

			plt.figure()
			plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s')
			plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			plt.colorbar()
			plt.pause(0.01)

		if grid_resolution==8:
			# temp=1e-3
			temp=1e-7
			temp2=0
		elif grid_resolution==2:
			temp=1e-4
			temp2=np.sum(sensitivities_binned_crop,axis=(0,1)).max()*1e-3
		elif grid_resolution==4:
			temp=0
			temp2=0

		sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp)

		if development_plots:
			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.mean(sensitivities_binned_crop,axis=(0,1)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.mean(sensitivities_binned_crop,axis=(0,1)),cmap='rainbow',marker='s',norm=LogNorm(),vmin=2e-8)
			# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			plt.plot(structure_radial_profile[1][:, 0], structure_radial_profile[1][:, 1], 'k')
			ax = plt.gca() #you first need to get the axis handle
			ax.set_aspect(1) #sets the height to width ratio to 1.5.
			temp = np.abs(efit_reconstruction.time-0.7).argmin()
			for i in range(len(all_time_sep_r[temp])):
				plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
			plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
			plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
			plt.plot(efit_reconstruction.lower_xpoint_r[temp],-efit_reconstruction.lower_xpoint_z[temp],'xr')
			plt.plot(efit_reconstruction.strikepointR[temp],-efit_reconstruction.strikepointZ[temp],'xr')
			plt.colorbar()
			plt.xlim(left=0.2,right=2.05)
			plt.ylim(top=2.2,bottom=-2.2)
			plt.pause(0.01)

		# # I want to disconnect the 2 voxels with highest sensitivity from the others, because they have an almost homogeneous effect on the foil
		# # if there is unaccounted and uniform light all over the foil this is reflected in this voxels, so it doesn't have much sense to have them connected via the laplacian to all the rest
		# voxels_centre = np.mean(grid_data_masked_crop,axis=1)
		# dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
		# dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
		# voxel_to_disconnect = np.mean(sensitivities_binned_crop,axis=(0,1)).argmax()
		# voxels_to_disconnect = np.logical_and( np.abs(np.mean(grid_data_masked_crop,axis=1)[:,0]-np.mean(grid_data_masked_crop,axis=1)[:,0][voxel_to_disconnect])<2*dr , np.abs(np.mean(grid_data_masked_crop,axis=1)[:,1]-np.mean(grid_data_masked_crop,axis=1)[:,1][voxel_to_disconnect])<2*dz)
		# grid_laplacian_masked_crop[voxels_to_disconnect] = 0
		# grid_laplacian_masked_crop[:,voxels_to_disconnect] = 0

		if development_plots:
			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.max(grid_laplacian_masked_crop,axis=(0)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.max(grid_Z_derivate_masked_crop,axis=(1)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

		selected_super_x_cells = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.85,np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.65)

		x1 = [1.55,0.25]	# r,z
		x2 = [1.1,-0.15]
		interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
		select = np.mean(grid_data_masked_crop,axis=1)[:,1]>interp(np.mean(grid_data_masked_crop,axis=1)[:,0])
		selected_central_border_cells = np.logical_and(select,np.logical_and(np.max(grid_Z_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,1]>-0.5))
		selected_central_border_cells = np.dot(grid_laplacian_masked_crop,selected_central_border_cells*np.random.random(selected_central_border_cells.shape))!=0

		selected_central_column_border_cells = np.logical_and(np.logical_and(np.max(grid_R_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)
		selected_central_column_border_cells = np.logical_and(np.logical_and(np.dot(grid_laplacian_masked_crop,selected_central_column_border_cells*np.random.random(selected_central_column_border_cells.shape))!=0,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)

		selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))
		selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))

		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
		if grid_resolution<8:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		if grid_resolution<4:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells,marker='s')
		plt.title('ede region with emissivity\nrequired to be negligible')
		plt.colorbar()
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region1.eps')
		# plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells_for_laplacian,marker='s')
		plt.title('ede region with\nlaplacian of emissivity\nrequired to be low')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region2.eps')
		plt.colorbar()
		# plt.pause(0.01)

		if development_plots:
			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_central_border_cells,marker='s')
			plt.colorbar()
			plt.pause(0.01)

			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_central_column_border_cells,marker='s')
			plt.colorbar()
			plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.title('super-x region with\nlaplacian of emissivity\nless restricted')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_super_x_cells,marker='s')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_super-x.eps')
		plt.colorbar()
		plt.pause(0.01)


		# plt.figure()
		# plt.imshow(np.flip(np.transpose(powernoback_full[40],(1,0)),axis=1),'rainbow',origin='lower')
		# plt.plot([ROI[1,0],ROI[1,0],ROI[1,1],ROI[1,1],ROI[1,0]],[ROI[0,0],ROI[0,1],ROI[0,1],ROI[0,0],ROI[0,0]],'k')
		# plt.colorbar()
		# plt.pause(0.01)
		sensitivities_binned_crop_shape = sensitivities_binned_crop.shape
		sensitivities_binned_crop = sensitivities_binned_crop.reshape((sensitivities_binned_crop.shape[0]*sensitivities_binned_crop.shape[1],sensitivities_binned_crop.shape[2]))

		if shrink_factor_x > 1:
			foil_resolution = str(shrink_factor_x) + 'x' + str(shrink_factor_x)
		else:
			foil_resolution = str(shape[0])

		foil_res = '_foil_pixel_h_' + str(foil_resolution)
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
		path_sensitivity_original = cp.deepcopy(path_sensitivity)

		# for shrink_factor_t in np.flip(all_shrink_factor_t,axis=0):
		# for shrink_factor_t in [7,5,3,2,1]:
		for shrink_factor_t in [7]:
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)] = dict([])
			binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
			print('starting '+binning_type)
			# powernoback_full = saved_file_dict_short[binning_type].all()['powernoback_full']
			# powernoback_std_full = saved_file_dict_short[binning_type].all()['powernoback_std_full']

			# sigma_emissivity = 0.1
			# sigma_thickness = 0.15
			# sigma_rec_diffusivity = 0.1
			sigma_emissivity = np.std(foilemissivity)/np.mean(foilemissivity)	# I use the varition on the japanese data as reference
			sigma_thickness = np.std(foilthickness)/np.mean(foilthickness)
			sigma_rec_diffusivity = 0	# this was not consodered as variable

			tend = coleval.get_tend(laser_to_analyse[-9:-4])+0.01	 # I add 10ms just for safety and to catch disruptions

			time_full_binned = saved_file_dict_short[binning_type].all()['time_full_binned']
			BBrad_full = saved_file_dict_short[binning_type].all()['BBrad_full']
			BBrad_std_full = saved_file_dict_short[binning_type].all()['BBrad_std_full']
			diffusion_full = saved_file_dict_short[binning_type].all()['diffusion_full']
			diffusion_std_full = saved_file_dict_short[binning_type].all()['diffusion_std_full']
			timevariation_full = saved_file_dict_short[binning_type].all()['timevariation_full']
			timevariation_std_full = saved_file_dict_short[binning_type].all()['timevariation_std_full']
			BBrad_full_crop = BBrad_full[time_full_binned<tend]
			BBrad_full_crop[:,np.logical_not(selected_ROI)] = 0
			BBrad_std_full_crop = BBrad_std_full[time_full_binned<tend]
			BBrad_std_full_crop[:,np.logical_not(selected_ROI)] = 0
			diffusion_full_crop = diffusion_full[time_full_binned<tend]
			diffusion_full_crop[:,np.logical_not(selected_ROI)] = 0
			diffusion_std_full_crop = diffusion_std_full[time_full_binned<tend]
			diffusion_std_full_crop[:,np.logical_not(selected_ROI)] = 0
			timevariation_full_crop = timevariation_full[time_full_binned<tend]
			timevariation_full_crop[:,np.logical_not(selected_ROI)] = 0
			timevariation_std_full_crop = timevariation_std_full[time_full_binned<tend]
			timevariation_std_full_crop[:,np.logical_not(selected_ROI)] = 0
			time_full_binned_crop = time_full_binned[time_full_binned<tend]

			powernoback_full_orig = diffusion_full_crop + timevariation_full_crop + BBrad_full_crop
			sigma_powernoback_full = ( (diffusion_full_crop**2)*((diffusion_std_full_crop/diffusion_full_crop)**2+sigma_thickness**2) + (timevariation_full_crop**2)*((timevariation_std_full_crop/timevariation_full_crop)**2+sigma_thickness**2+sigma_rec_diffusivity**2) + (BBrad_full_crop**2)*((BBrad_std_full_crop/BBrad_full_crop)**2+sigma_emissivity**2) )**0.5

			plt.close('all')

			grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
			grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
			grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
			reference_sigma_powernoback_all = np.nanmedian(sigma_powernoback_full[:,selected_ROI],axis=1)
			number_cells_ROI = np.sum(selected_ROI)
			number_cells_plasma = np.sum(select_foil_region_with_plasma)
			not_selected_super_x_cells = np.logical_not(selected_super_x_cells)

			sigma_emissivity = 1e6	# this is completely arbitrary
			sigma_emissivity_2 = sigma_emissivity**2
			r_int = np.mean(grid_data_masked_crop,axis=1)[:,0]
			r_int_2 = r_int**2

			sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1e10
			selected_ROI_internal = selected_ROI.flatten()
			inverted_data = []
			inverted_data_sigma = []
			inverted_data_likelihood = []
			inverted_data_info = []
			inverted_data_plasma_region_offset = []
			inverted_data_homogeneous_offset = []
			fitted_foil_power = []
			foil_power = []
			foil_power_residuals = []
			foil_power_std = []
			fit_error = []
			chi_square_all = []
			regolarisation_coeff_all = []
			outer_leg_tot_rad_power_all = []
			inner_leg_tot_rad_power_all = []
			core_tot_rad_power_all = []
			x_point_tot_rad_power_all = []
			time_per_iteration = []
			score_x_all = []
			score_y_all = []
			Lcurve_curvature_all = []
			regolarisation_coeff_range_all = []
			x_optimal_ext = []

			plt.figure(10,figsize=(20, 10))
			plt.title('L-curve evolution\nlight=early, dark=late')
			plt.figure(11,figsize=(20, 10))
			plt.title('L-curve curvature evolution\nlight=early, dark=late')
			path_for_plots = path_power_output + '/invertions_log/'+binning_type
			if not os.path.exists(path_for_plots):
				os.makedirs(path_for_plots)
			# x_optimal_all_guess = []
			first_guess = []
			regolarisation_coeff_upper_limit = 10**-0.2
			for i_t in range(len(time_full_binned_crop)):
				time_start = tm.time()
				print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))

				powernoback = powernoback_full_orig[i_t].flatten()
				sigma_powernoback = sigma_powernoback_full[i_t].flatten()
				reference_sigma_powernoback = reference_sigma_powernoback_all[i_t]
				# sigma_powernoback = np.ones_like(powernoback)*10
				sigma_powernoback_2 = sigma_powernoback**2
				homogeneous_scaling=1e-4

				guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
				if len(first_guess) != 0:
					guess = cp.deepcopy(first_guess)

				target_chi_square = sensitivities_binned_crop.shape[1]	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
				target_chi_square_sigma = 200	# this should be tight, because for such a high number of degrees of freedom things should average very well

				# regolarisation_coeff_edge = 10
				# regolarisation_coeff_edge_multiplier = 100
				regolarisation_coeff_central_border_Z_derivate_multiplier = 0
				regolarisation_coeff_central_column_border_R_derivate_multiplier = 0
				# regolarisation_coeff_edge_laplacian_multiplier = 1e1
				regolarisation_coeff_divertor_multiplier = 1
				regolarisation_coeff_non_negativity_multiplier = 40
				regolarisation_coeff_offsets_multiplier = 1e-10
				regolarisation_coeff_edge_laplacian = 0.01
				regolarisation_coeff_edge = 100

				def prob_and_gradient(emissivity_plus,*args):
					# time_start = tm.time()
					# emissivity_plus = emissivity_plus
					powernoback = args[0]
					sigma_powernoback = args[1]
					sigma_emissivity = args[2]
					regolarisation_coeff = args[3]
					sigma_powernoback_2 = args[4]
					sigma_emissivity_2 = args[5]
					homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
					homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
					regolarisation_coeff_divertor = regolarisation_coeff*regolarisation_coeff_divertor_multiplier
					regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*regolarisation_coeff_central_column_border_R_derivate_multiplier
					# regolarisation_coeff_edge_laplacian = regolarisation_coeff*regolarisation_coeff_edge_laplacian_multiplier
					# regolarisation_coeff_edge = regolarisation_coeff*regolarisation_coeff_edge_multiplier
					# print(homogeneous_offset,homogeneous_offset_plasma)
					emissivity = emissivity_plus[:-2]
					# emissivity[emissivity==0] = 1e-10
					# foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
					foil_power_error = powernoback - (np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma)
					emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
					emissivity_laplacian_not_selected_super_x_cells = emissivity_laplacian*not_selected_super_x_cells
					emissivity_laplacian_selected_super_x_cells = emissivity_laplacian*selected_super_x_cells
					emissivity_laplacian_selected_edge_cells_for_laplacian = emissivity_laplacian*selected_edge_cells_for_laplacian
					if regolarisation_coeff_central_column_border_R_derivate!=0:
						R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)
						R_derivate_selected_central_column_border_cells = R_derivate*selected_central_column_border_cells
					# print(tm.time()-time_start)
					# time_start = tm.time()

					likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
					likelihood_emissivity_pos = (regolarisation_coeff_non_negativity_multiplier**2)*np.sum((np.minimum(0.,emissivity)*r_int/sigma_emissivity*1)**2)	# I added a weight on the redious, becaus the power increase with radious and a negative voxel at high r is more important that one at low r
					likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian_not_selected_super_x_cells /sigma_emissivity)**2))
					likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian_selected_super_x_cells /sigma_emissivity)**2))
					likelihood_emissivity_edge_laplacian = (regolarisation_coeff_edge_laplacian**2)* np.sum(((emissivity_laplacian_selected_edge_cells_for_laplacian /sigma_emissivity)**2))
					likelihood_emissivity_edge = (regolarisation_coeff_edge**2)*np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
					if regolarisation_coeff_central_column_border_R_derivate==0:
						likelihood_emissivity_central_column_border_R_derivate = 0
					else:
						likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate_selected_central_column_border_cells/sigma_emissivity)**2)
					likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge + likelihood_emissivity_laplacian_superx + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_edge_laplacian
					likelihood_homogeneous_offset = number_cells_ROI*(homogeneous_offset/reference_sigma_powernoback)**2
					likelihood_homogeneous_offset_plasma = number_cells_plasma*(homogeneous_offset_plasma/reference_sigma_powernoback)**2
					likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma
					# print(tm.time()-time_start)
					# time_start = tm.time()

					temp = foil_power_error/sigma_powernoback_2
					likelihood_power_fit_derivate = np.concatenate((-2*np.dot(temp,sensitivities_binned_crop),[-2*np.sum(temp*select_foil_region_with_plasma)*homogeneous_scaling,-2*np.sum(temp*selected_ROI_internal)*homogeneous_scaling]))
					likelihood_emissivity_pos_derivate = 2*(regolarisation_coeff_non_negativity_multiplier**2)*np.minimum(0.,emissivity)*r_int_2/sigma_emissivity_2*1

					# likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian_not_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					# likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					# likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge_laplacian**2) * np.dot(emissivity_laplacian_selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_laplacian_derivate_all = 2* np.dot( (regolarisation_coeff**2)*emissivity_laplacian_not_selected_super_x_cells + (regolarisation_coeff_edge_laplacian**2)*emissivity_laplacian_selected_edge_cells_for_laplacian + (regolarisation_coeff_divertor**2)*emissivity_laplacian_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)

					likelihood_emissivity_edge_derivate = 2*(regolarisation_coeff_edge**2)*emissivity*selected_edge_cells/sigma_emissivity_2
					if regolarisation_coeff_central_column_border_R_derivate==0:
						likelihood_emissivity_central_column_border_R_derivate_derivate = 0
					else:
						likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate_selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
					likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate_all + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate
					likelihood_homogeneous_offset_derivate = 2*number_cells_ROI*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_homogeneous_offset_plasma_derivate = 2*number_cells_plasma*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate
					# print(tm.time()-time_start)
					# time_start = tm.time()
					return likelihood,likelihood_derivate

				def calc_hessian(emissivity_plus,*args):
					# time_start = tm.time()
					# emissivity_plus = emissivity_plus
					powernoback = args[0]
					sigma_powernoback = args[1]
					sigma_emissivity = args[2]
					regolarisation_coeff = args[3]
					sigma_powernoback_2 = args[4]
					sigma_emissivity_2 = args[5]
					homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
					homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
					regolarisation_coeff_divertor = regolarisation_coeff*regolarisation_coeff_divertor_multiplier
					regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*regolarisation_coeff_central_column_border_R_derivate_multiplier
					# regolarisation_coeff_edge_laplacian = regolarisation_coeff*regolarisation_coeff_edge_laplacian_multiplier
					# regolarisation_coeff_edge = regolarisation_coeff*regolarisation_coeff_edge_multiplier
					# print(homogeneous_offset,homogeneous_offset_plasma)
					emissivity = emissivity_plus[:-2]
					# emissivity[emissivity==0] = 1e-10
					# foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
					foil_power_error = powernoback - (np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma)
					emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
					emissivity_laplacian_not_selected_super_x_cells = emissivity_laplacian*not_selected_super_x_cells
					emissivity_laplacian_selected_super_x_cells = emissivity_laplacian*selected_super_x_cells
					emissivity_laplacian_selected_edge_cells_for_laplacian = emissivity_laplacian*selected_edge_cells_for_laplacian
					if regolarisation_coeff_central_column_border_R_derivate!=0:
						R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)
						R_derivate_selected_central_column_border_cells = R_derivate*selected_central_column_border_cells
					# print(tm.time()-time_start)
					# time_start = tm.time()

					# based on https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DAVIES1/rd_bhatt_cvonline/node9.html#SECTION00041000000000000000
					likelihood_power_fit_derivate = np.dot(sensitivities_binned_crop.T*sigma_powernoback_2,sensitivities_binned_crop)
					temp = np.zeros((np.shape(sensitivities_binned_crop)[1]+2,np.shape(sensitivities_binned_crop)[1]+2))
					temp[:-2,:-2] = likelihood_power_fit_derivate
					temp[-2:,:-2] = np.array([np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*select_foil_region_with_plasma).T,axis=0)*homogeneous_scaling,np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*selected_ROI_internal).T,axis=0)*homogeneous_scaling])
					temp[:-2,-2:] = np.array([np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*select_foil_region_with_plasma).T,axis=0)*homogeneous_scaling,np.sum(-(sensitivities_binned_crop.T/sigma_powernoback_2*selected_ROI_internal).T,axis=0)*homogeneous_scaling]).T
					temp[-2,-2] = -np.sum(select_foil_region_with_plasma/sigma_powernoback_2*select_foil_region_with_plasma)*homogeneous_scaling
					temp[-1,-1] = -np.sum(selected_ROI_internal/sigma_powernoback_2*selected_ROI_internal)*homogeneous_scaling
					temp[-1,-2] = -np.sum(selected_ROI_internal/sigma_powernoback_2*select_foil_region_with_plasma)*homogeneous_scaling
					temp[-2,-1] = -np.sum(selected_ROI_internal/sigma_powernoback_2*select_foil_region_with_plasma)*homogeneous_scaling
					likelihood_power_fit_derivate = cp.deepcopy(temp)

					likelihood_emissivity_pos_derivate = (regolarisation_coeff_non_negativity_multiplier**2)*np.diag((emissivity<0)*np.logical_not(selected_edge_cells)*r_int_2/sigma_emissivity_2*1)

					# likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian_not_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					# likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian_selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					# likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge_laplacian**2) * np.dot(emissivity_laplacian_selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_laplacian_derivate_all = np.dot(grid_laplacian_masked_crop_scaled*( (regolarisation_coeff**2)*not_selected_super_x_cells + (regolarisation_coeff_edge_laplacian**2)*selected_edge_cells_for_laplacian + (regolarisation_coeff_divertor**2)*selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)

					likelihood_emissivity_edge_derivate = (regolarisation_coeff_edge**2)*np.diag(selected_edge_cells*r_int_2/sigma_emissivity_2*1)
					if regolarisation_coeff_central_column_border_R_derivate==0:
						likelihood_emissivity_central_column_border_R_derivate_derivate = 0
					else:
						likelihood_emissivity_central_column_border_R_derivate_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)*np.dot( grid_R_derivate_masked_crop_scaled*selected_central_column_border_cells ,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
					likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate_all + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate
					likelihood_homogeneous_offset_derivate = regolarisation_coeff_offsets_multiplier*number_cells_ROI*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_homogeneous_offset_plasma_derivate = regolarisation_coeff_offsets_multiplier*number_cells_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_power_fit_derivate[:-2,:-2]+=likelihood_derivate
					likelihood_power_fit_derivate[-1,-1] += likelihood_homogeneous_offset_derivate
					likelihood_power_fit_derivate[-2,-2] += likelihood_homogeneous_offset_plasma_derivate
					return likelihood_power_fit_derivate

				if False:	# only for testinf the prob_and_gradient function
					target = -2
					# guess[target] = -10
					temp1 = prob_and_gradient(guess,*powernoback)
					guess[target] +=1e-7
					temp2 = prob_and_gradient(guess,*powernoback)
					guess[target] += -2e-7
					temp3 = prob_and_gradient(guess,*powernoback)
					print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/2e-7)))

				# regolarisation_coeff_range = 10**np.linspace(1,-6,num=120)
				# regolarisation_coeff_range = 10**np.linspace(1,-5,num=102)
				regolarisation_coeff_range = 10**np.linspace(0.5,-5,num=80)
				x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,voxels_centre = coleval.loop_fit_over_regularisation(prob_and_gradient,regolarisation_coeff_range,guess,grid_data_masked_crop,powernoback,sigma_powernoback,sigma_emissivity,factr=1e8)
				# x_optimal_all_guess = cp.deepcopy(x_optimal_all)
				first_guess = x_optimal_all[0]

				regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
				x_optimal_all = np.flip(x_optimal_all,axis=0)
				recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)
				y_opt_all = np.flip(y_opt_all,axis=0)
				opt_info_all = np.flip(opt_info_all,axis=0)

				score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
				score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,np.array(x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)
				score_x_all.append(score_x)
				score_y_all.append(score_y)
				regolarisation_coeff_range_all.append(regolarisation_coeff_range)

				plt.figure(10)
				plt.plot(np.log(score_x),np.log(score_y),'--',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))

				score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks,best_index = coleval.find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,regolarisation_coeff_upper_limit=regolarisation_coeff_upper_limit)

				# plt.figure(10)
				plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(score_x,score_y,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(score_x[best_index],score_y[best_index],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(score_x[peaks],score_y[peaks],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
				plt.xlabel('log ||Gm-d||2')
				plt.ylabel('log ||Laplacian(m)||2')
				plt.title('L-curve evolution\nlight=early, dark=late\ncurvature_range = '+str(curvature_range))
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_evolution.eps')
				plt.figure(11)
				plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(regolarisation_coeff_range[best_index],Lcurve_curvature[best_index-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
				plt.plot(regolarisation_coeff_range[peaks],Lcurve_curvature[peaks-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
				plt.axvline(x=regolarisation_coeff_upper_limit,color='r')
				plt.semilogx()
				plt.xlabel('regularisation coeff')
				plt.ylabel('L-curve turvature')
				plt.title('L-curve curvature evolution\nlight=early, dark=late\ncurvature_range = '+str(curvature_range)+'\ncurvature_range_left_all = '+str(curvature_range_left_all)+'\ncurvature_range_right_all = '+str(curvature_range_right_all))
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution.eps')


				foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling
				foil_power_error = powernoback - foil_power_guess
				chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
				print('chi_square '+str(chi_square))


				if False:	# just for visualisation

					plt.figure(figsize=(12,13))
					# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
					plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
					plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
					temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
					for i in range(len(all_time_sep_r[temp])):
						plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
					plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
					plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
					plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
					plt.colorbar().set_label('emissivity [W/m3]')
					plt.ylim(top=0.5)
					plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example1.eps')
					plt.pause(0.01)

					temp = x_optimal
					temp[1226]=0
					plt.figure(figsize=(6,12))
					plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
					plt.pause(0.01)

					plt.figure(figsize=(15,12))
					plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nplasma region offset %.3g, whole foil offset %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,x_optimal[-2],x_optimal[-1]))
					plt.imshow((np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
					plt.colorbar().set_label('power [W/m2]')
					plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example2.eps')
					plt.pause(0.01)

					plt.figure(figsize=(15,12))
					plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
					plt.imshow(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
					plt.colorbar().set_label('power error [W/m2]')
					plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example3.eps')
					plt.pause(0.01)

					plt.figure()
					plt.imshow(powernoback_full_orig[i_t])
					plt.colorbar().set_label('power [W/m2]')
					plt.title('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
					plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example0.eps')
					plt.pause(0.01)

					plt.figure()
					# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=likelihood_emissivity_central_border_Z_derivate,marker='s')
					plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,marker='s')
					plt.colorbar()
					plt.pause(0.01)

				else:
					pass

				inverted_data.append(recompose_voxel_emissivity)
				inverted_data_likelihood.append(y_opt)
				inverted_data_plasma_region_offset.append(x_optimal[-2]*homogeneous_scaling)
				inverted_data_homogeneous_offset.append(x_optimal[-1]*homogeneous_scaling)
				x_optimal_ext.append(x_optimal)
				inverted_data_info.append(opt_info)
				fitted_foil_power.append((np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig[i_t].shape))
				foil_power.append(powernoback_full_orig[i_t])
				foil_power_std.append(sigma_powernoback_full[i_t])
				foil_power_residuals.append(powernoback_full_orig[i_t]-fitted_foil_power[-1])
				fit_error.append(np.sum(((powernoback_full_orig[i_t][selected_ROI]-fitted_foil_power[-1][[selected_ROI]]))**2)**0.5/np.sum(selected_ROI))
				chi_square_all.append(chi_square)
				regolarisation_coeff_all.append(regolarisation_coeff)
				time_per_iteration.append(tm.time()-time_start)
				for value in points_removed:
					Lcurve_curvature = np.concatenate([Lcurve_curvature[:value],[np.nan],Lcurve_curvature[value:]])
				Lcurve_curvature_all.append(Lcurve_curvature)

				args = [powernoback,sigma_powernoback,sigma_emissivity,regolarisation_coeff,sigma_powernoback**2,sigma_emissivity**2]
				hessian=calc_hessian(x_optimal,*args)
				covariance = np.linalg.inv(hessian)
				trash,recompose_voxel_sigma = translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.diag(covariance)**0.5,np.mean(grid_data_masked_crop,axis=1))
				inverted_data_sigma.append(recompose_voxel_sigma)


			inverted_data = np.array(inverted_data)
			inverted_data_sigma = np.array(inverted_data_sigma)
			inverted_data_likelihood = -np.array(inverted_data_likelihood)
			inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
			inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
			x_optimal_ext = np.array(x_optimal_ext)
			fitted_foil_power = np.array(fitted_foil_power)
			foil_power = np.array(foil_power)
			foil_power_std = np.array(foil_power_std)
			foil_power_residuals = np.array(foil_power_residuals)
			fit_error = np.array(fit_error)
			chi_square_all = np.array(chi_square_all)
			plt.close('all')

			plt.figure(figsize=(20, 10))
			plt.plot(time_full_binned_crop,inverted_data_likelihood)
			# plt.semilogy()
			plt.title('Fit log likelihood')
			plt.xlabel('time [s]')
			plt.ylabel('log likelihoog [au]')
			plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_likelihood.eps')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(time_full_binned_crop,chi_square_all)
			plt.plot(time_full_binned_crop,np.ones_like(time_full_binned_crop)*target_chi_square,'--k')
			# plt.semilogy()
			plt.title('chi square obtained vs requested\nfixed regularisation of '+str(regolarisation_coeff))
			plt.xlabel('time [s]')
			plt.ylabel('chi square [au]')
			plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_chi_square.eps')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(time_full_binned_crop,regolarisation_coeff_all)
			# plt.semilogy()
			plt.title('regolarisation coefficient obtained')
			plt.semilogy()
			plt.xlabel('time [s]')
			plt.ylabel('regolarisation coefficient [au]')
			plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_regolarisation_coeff.eps')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(time_full_binned_crop,fit_error)
			# plt.semilogy()
			plt.title('Fit error ( sum((image-fit)^2)^0.5/num pixels )')
			plt.xlabel('time [s]')
			plt.ylabel('average fit error [W/m2]')
			plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_fit_error.eps')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(time_full_binned_crop,inverted_data_plasma_region_offset,label='plasma region')
			plt.plot(time_full_binned_crop,inverted_data_homogeneous_offset,label='whole foil')
			plt.title('Offsets to match foil power')
			plt.legend(loc='best', fontsize='x-small')
			plt.xlabel('time [s]')
			plt.ylabel('power [W/m2]')
			plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_offsets.eps')
			plt.close()

			extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			additional_each_frame_label_description = ['reg coeff=']*len(inverted_data)
			additional_each_frame_label_number = np.array(regolarisation_coeff_all)
			ani,trash = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate_multiplier %.3g\nregolarisation_coeff_central_column_border_R_derivate_multiplier %.3g\nregolarisation_coeff_edge_laplacian %.3g\nregolarisation_coeff_divertor_multiplier %.3g\nregolarisation_coeff_non_negativity_multiplier %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_edge_laplacian,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_non_negativity_multiplier,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,additional_each_frame_label_description=additional_each_frame_label_description,additional_each_frame_label_number=additional_each_frame_label_number)#,extvmin=0,extvmax=4e4)
			ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
			plt.close()

			ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(fitted_foil_power,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))) ,timesteps=time_full_binned_crop,integration=laser_int_time/1000,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Fitted power on foil [W/m2]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate_multiplier %.3g\nregolarisation_coeff_central_column_border_R_derivate_multiplier %.3g\nregolarisation_coeff_edge_laplacian %.3g\nregolarisation_coeff_divertor_multiplier %.3g\nregolarisation_coeff_non_negativity_multiplier %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_edge_laplacian,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_non_negativity_multiplier,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
			ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_fitted_foil_power_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')

			if efit_reconstruction!=None:
				all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
				all_time_strike_points_location = coleval.return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
				outer_leg_tot_rad_power_all = []
				inner_leg_tot_rad_power_all = []
				core_tot_rad_power_all = []
				sxd_tot_rad_power_all = []
				x_point_tot_rad_power_all = []
				outer_leg_tot_rad_power_sigma_all = []
				inner_leg_tot_rad_power_sigma_all = []
				core_tot_rad_power_sigma_all = []
				sxd_tot_rad_power_sigma_all = []
				x_point_tot_rad_power_sigma_all = []
				for i_t in range(len(time_full_binned_crop)):
					temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
					xpoint_r = efit_reconstruction.lower_xpoint_r[temp]
					xpoint_z = efit_reconstruction.lower_xpoint_z[temp]
					z_,r_ = np.meshgrid(np.unique(voxels_centre[:,1]),np.unique(voxels_centre[:,0]))
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_>xpoint_z] = 0
					temp[r_<xpoint_r] = 0
					temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
					temp_sigma[z_>xpoint_z] = 0
					temp_sigma[r_<xpoint_r] = 0
					outer_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					outer_leg_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_>xpoint_z] = 0
					temp[r_>xpoint_r] = 0
					temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
					temp_sigma[z_>xpoint_z] = 0
					temp_sigma[r_>xpoint_r] = 0
					inner_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					inner_leg_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_<xpoint_z] = 0
					temp[z_>0] = 0
					temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
					temp_sigma[z_<xpoint_z] = 0
					temp_sigma[z_>0] = 0
					core_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					core_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_>-1.5] = 0
					temp[r_<0.8] = 0
					temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
					temp_sigma[z_>-1.5] = 0
					temp_sigma[r_<0.8] = 0
					sxd_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					sxd_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
					temp = cp.deepcopy(inverted_data[i_t])
					temp[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.10] = 0
					temp_sigma = cp.deepcopy(inverted_data_sigma[i_t])
					temp_sigma[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.20] = 0
					x_point_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					x_point_tot_rad_power_sigma = np.nansum((temp_sigma*2*np.pi*r_*((grid_resolution*0.01)**2))**2)**0.5
					outer_leg_tot_rad_power_all.append(outer_leg_tot_rad_power)
					inner_leg_tot_rad_power_all.append(inner_leg_tot_rad_power)
					core_tot_rad_power_all.append(core_tot_rad_power)
					sxd_tot_rad_power_all.append(sxd_tot_rad_power)
					x_point_tot_rad_power_all.append(x_point_tot_rad_power)
					outer_leg_tot_rad_power_sigma_all.append(outer_leg_tot_rad_power_sigma)
					inner_leg_tot_rad_power_sigma_all.append(inner_leg_tot_rad_power_sigma)
					core_tot_rad_power_sigma_all.append(core_tot_rad_power_sigma)
					sxd_tot_rad_power_sigma_all.append(sxd_tot_rad_power_sigma)
					x_point_tot_rad_power_sigma_all.append(x_point_tot_rad_power_sigma)
				outer_leg_tot_rad_power_all = np.array(outer_leg_tot_rad_power_all)
				inner_leg_tot_rad_power_all = np.array(inner_leg_tot_rad_power_all)
				core_tot_rad_power_all = np.array(core_tot_rad_power_all)
				sxd_tot_rad_power_all = np.array(sxd_tot_rad_power_all)
				x_point_tot_rad_power_all = np.array(x_point_tot_rad_power_all)
				outer_leg_tot_rad_power_sigma_all = np.array(outer_leg_tot_rad_power_sigma_all)
				inner_leg_tot_rad_power_sigma_all = np.array(inner_leg_tot_rad_power_sigma_all)
				core_tot_rad_power_sigma_all = np.array(core_tot_rad_power_sigma_all)
				sxd_tot_rad_power_sigma_all = np.array(sxd_tot_rad_power_sigma_all)
				x_point_tot_rad_power_sigma_all = np.array(x_point_tot_rad_power_sigma_all)

				plt.figure(figsize=(20, 15))
				plt.errorbar(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,yerr=outer_leg_tot_rad_power_sigma_all/1e3,label='outer_leg',capsize=5)
				plt.errorbar(time_full_binned_crop,sxd_tot_rad_power_all/1e3,yerr=sxd_tot_rad_power_sigma_all/1e3,label='sxd',capsize=5)
				plt.errorbar(time_full_binned_crop,inner_leg_tot_rad_power_all/1e3,yerr=inner_leg_tot_rad_power_sigma_all/1e3,label='inner_leg',capsize=5)
				plt.errorbar(time_full_binned_crop,core_tot_rad_power_all/1e3,yerr=core_tot_rad_power_sigma_all/1e3,label='core',capsize=5)
				plt.errorbar(time_full_binned_crop,x_point_tot_rad_power_all/1e3,yerr=x_point_tot_rad_power_sigma_all/1e3,label='x_point (dist<20cm)',capsize=5)
				plt.errorbar(time_full_binned_crop,(outer_leg_tot_rad_power_all+inner_leg_tot_rad_power_all+core_tot_rad_power_all)/1e3,yerr=((outer_leg_tot_rad_power_sigma_all**2+inner_leg_tot_rad_power_sigma_all**2+core_tot_rad_power_sigma_all**2)**0.5)/1e3,label='tot',capsize=5)
				plt.title('radiated power in the lower half of the machine')
				plt.legend(loc='best', fontsize='x-small')
				plt.xlabel('time [s]')
				plt.ylabel('power [kW]')
				plt.grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_tot_rad_power.eps')
				plt.close()

				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['outer_leg_tot_rad_power_all'] = outer_leg_tot_rad_power_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inner_leg_tot_rad_power_all'] = inner_leg_tot_rad_power_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['core_tot_rad_power_all'] = core_tot_rad_power_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['sxd_tot_rad_power_all'] = sxd_tot_rad_power_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['x_point_tot_rad_power_all'] = x_point_tot_rad_power_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['outer_leg_tot_rad_power_sigma_all'] = outer_leg_tot_rad_power_sigma_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inner_leg_tot_rad_power_sigma_all'] = inner_leg_tot_rad_power_sigma_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['core_tot_rad_power_sigma_all'] = core_tot_rad_power_sigma_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['sxd_tot_rad_power_sigma_all'] = sxd_tot_rad_power_sigma_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['x_point_tot_rad_power_sigma_all'] = x_point_tot_rad_power_sigma_all

				additional_points_dict,radiator_xpoint_distance_all,radiator_above_xpoint_all,radiator_magnetic_radious_all,radiator_baricentre_magnetic_radious_all,radiator_baricentre_above_xpoint_all = coleval.find_radiator_location(inverted_data,np.unique(voxels_centre[:,0]),np.unique(voxels_centre[:,1]),time_full_binned_crop,efit_reconstruction)

				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_location_all'] = additional_points_dict['0']
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_xpoint_distance_all'] = radiator_xpoint_distance_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_above_xpoint_all'] = radiator_above_xpoint_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_magnetic_radious_all'] = radiator_magnetic_radious_all

				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				ani,efit_reconstruction = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate_multiplier %.3g\nregolarisation_coeff_central_column_border_R_derivate_multiplier %.3g\nregolarisation_coeff_edge_laplacian %.3g\nregolarisation_coeff_divertor_multiplier %.3g\nregolarisation_coeff_non_negativity_multiplier %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_edge_laplacian,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_non_negativity_multiplier,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,additional_points_dict=additional_points_dict)#,extvmin=0,extvmax=4e4)
				ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
				plt.close()

				fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False,sharex=True)
				ax[0,0].plot(time_full_binned_crop,radiator_magnetic_radious_all)
				ax[0,0].plot(time_full_binned_crop,radiator_baricentre_magnetic_radious_all,'--')
				ax[0,0].set_ylim(top=min(np.nanmax(radiator_magnetic_radious_all),1.1),bottom=max(np.nanmin(radiator_magnetic_radious_all),0.9))
				ax[1,0].plot(time_full_binned_crop,radiator_above_xpoint_all)
				ax[1,0].plot(time_full_binned_crop,radiator_baricentre_above_xpoint_all,'--')
				fig.suptitle('Location of the x-point radiator\n"--"=baricentre r=20cm around x-point')
				ax[0,0].set_ylabel('normalised psi [au]')
				ax[0,0].grid()
				ax[1,0].set_xlabel('time [s]')
				ax[1,0].set_ylabel('position above x-point [m]')
				ax[1,0].grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_x_point_location.eps')
				plt.close('all')


			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data'] = inverted_data
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_sigma'] = inverted_data_sigma
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_likelihood'] = inverted_data_likelihood
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_info'] = inverted_data_info
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['select_foil_region_with_plasma'] = select_foil_region_with_plasma
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_plasma_region_offset'] = inverted_data_plasma_region_offset
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_homogeneous_offset'] = inverted_data_homogeneous_offset
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['brightness_full'] = brightness_full
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop'] = time_full_binned_crop
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['fitted_foil_power'] = fitted_foil_power
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['foil_power'] = foil_power
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['foil_power_residuals'] = foil_power_residuals
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['fit_error'] = fit_error
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['chi_square_all'] = chi_square_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry'] = dict([])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['R'] = np.unique(voxels_centre[:,0])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['Z'] = np.unique(voxels_centre[:,1])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['x_optimal_ext'] = x_optimal_ext
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['foil_power_std'] = foil_power_std
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['score_x_all'] = score_x_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['score_y_all'] = score_y_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['regolarisation_coeff_range_all'] = regolarisation_coeff_range_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['Lcurve_curvature_all'] = Lcurve_curvature_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['sensitivities_binned_crop'] = sensitivities_binned_crop
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['regolarisation_coeff_all'] = regolarisation_coeff_all


			np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian',**inverted_dict)

exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_power_to_emissivity2.py").read())
