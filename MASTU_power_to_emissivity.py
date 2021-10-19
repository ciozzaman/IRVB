
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
for grid_resolution in [4]:
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

	def build_laplacian(grid):
		# Try making grid laplacian matrix for spatial regularisation
		num_cells = len(grid)
		grid_laplacian = np.zeros((num_cells, num_cells))
		unique_x = np.unique(np.mean(grid,axis=1)[:,0])
		unique_y = np.unique(np.mean(grid,axis=1)[:,1])

		for ith_cell in range(num_cells):

			# get the 2D mesh coordinates of this cell
			ix = np.abs(unique_x - np.mean(grid,axis=1)[ith_cell,0]).argmin()	# radious
			iy = np.abs(unique_y - np.mean(grid,axis=1)[ith_cell,1]).argmin()	# Z

			neighbours = 0

			if ix>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 1 left
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except KeyError:
					pass

			if ix>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 2 top left
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except:
					pass

			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 3 top
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except:
				pass

			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 4 top right
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except:
				pass

			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 5 right
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = -1
					neighbours += 1
			except:
				pass

			if iy>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 6 down right
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except:
					pass

			if iy>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 7 down
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except:
					pass

			if ix>0 and iy>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 8 down left
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except:
					pass

			grid_laplacian[ith_cell, ith_cell] = neighbours
		return grid_laplacian

	def build_Z_derivate(grid):
		# Try making grid Z direction derivate matrix for spatial regularisation
		num_cells = len(grid)
		grid_laplacian = np.zeros((num_cells, num_cells))
		unique_x = np.unique(np.mean(grid,axis=1)[:,0])
		unique_y = np.unique(np.mean(grid,axis=1)[:,1])

		for ith_cell in range(num_cells):

			# get the 2D mesh coordinates of this cell
			ix = np.abs(unique_x - np.mean(grid,axis=1)[ith_cell,0]).argmin()	# radious
			iy = np.abs(unique_y - np.mean(grid,axis=1)[ith_cell,1]).argmin()	# Z

			neighbours = 0

			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy+1])  # neighbour 3 top
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = 1
					neighbours += 1
			except:
				pass

			if iy>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix],np.mean(grid,axis=1)[:,1]==unique_y[iy-1])  # neighbour 7 down
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except:
					pass

			if neighbours==2:
				grid_laplacian[ith_cell, ith_cell] = 0
				grid_laplacian[ith_cell] /=2
			elif neighbours==1:
				grid_laplacian[ith_cell, ith_cell] = 1
			else:
				grid_laplacian[ith_cell, ith_cell] = 0

		return grid_laplacian

	def build_R_derivate(grid):
		# Try making grid R direction derivate matrix for spatial regularisation
		num_cells = len(grid)
		grid_laplacian = np.zeros((num_cells, num_cells))
		unique_x = np.unique(np.mean(grid,axis=1)[:,0])
		unique_y = np.unique(np.mean(grid,axis=1)[:,1])

		for ith_cell in range(num_cells):

			# get the 2D mesh coordinates of this cell
			ix = np.abs(unique_x - np.mean(grid,axis=1)[ith_cell,0]).argmin()	# radious
			iy = np.abs(unique_y - np.mean(grid,axis=1)[ith_cell,1]).argmin()	# Z

			neighbours = 0

			if ix>0:
				try:
					select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix-1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 1 left
					if np.sum(select)>0:
						n1 = select.argmax()
						grid_laplacian[ith_cell, n1] = -1
						neighbours += 1
				except KeyError:
					pass

			try:
				select = np.logical_and(np.mean(grid,axis=1)[:,0]==unique_x[ix+1],np.mean(grid,axis=1)[:,1]==unique_y[iy])  # neighbour 5 right
				if np.sum(select)>0:
					n1 = select.argmax()
					grid_laplacian[ith_cell, n1] = 1
					neighbours += 1
			except:
				pass

			if neighbours==2:
				grid_laplacian[ith_cell, ith_cell] = 0
				grid_laplacian[ith_cell] /=2
			elif neighbours==1:
				grid_laplacian[ith_cell, ith_cell] = 1
			else:
				grid_laplacian[ith_cell, ith_cell] = 0
		return grid_laplacian


	if grid_resolution==8:
		# temp=1e-3
		temp=1e-7
	elif grid_resolution==2:
		temp=1e-4
	elif grid_resolution==4:
		temp=0

	def reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,std_treshold = temp, sum_treshold = 0.000, core_radious_treshold = 1.9, divertor_radious_treshold = 1.9,chop_top_corner = False, extra_chop_top_corner = False , chop_corner_close_to_baffle = False):
		# masking the voxels whose emission does not reach the foil
		# select = np.sum(sensitivities_reshaped,axis=(0,1))>0.05
		select = np.logical_and(np.std(sensitivities_reshaped,axis=(0,1))>std_treshold*(np.std(sensitivities_reshaped,axis=(0,1)).max()),np.sum(sensitivities_reshaped,axis=(0,1))>sum_treshold)
		grid_data_masked = grid_data[select]
		sensitivities_reshaped_masked = sensitivities_reshaped[:,:,select]

		# this is not enough because the voxels close to the pinhole have a too large influence on it and the inversion is weird
		# select = np.median(sensitivities_reshaped_masked,axis=(0,1))<5*np.mean(np.median(sensitivities_reshaped_masked,axis=(0,1)))
		select = np.logical_or(np.mean(grid_data_masked,axis=1)[:,0]<core_radious_treshold,np.mean(grid_data_masked,axis=1)[:,1]<-1.3)
		if chop_top_corner:
			x1 = [1.1,0.6]	# r,z
			x2 = [1.6,0.0]
			interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
			select = np.logical_and(select,np.mean(grid_data_masked,axis=1)[:,1]<interp(np.mean(grid_data_masked,axis=1)[:,0]))
		if extra_chop_top_corner:
			x1 = [1.1,0.3]	# r,z
			x2 = [1.4,-0.05]
			interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
			select = np.logical_and(select,np.mean(grid_data_masked,axis=1)[:,1]<interp(np.mean(grid_data_masked,axis=1)[:,0]))
		if chop_corner_close_to_baffle:
			x1 = [1.33,-0.9]	# r,z
			x2 = [1.5,-1.03]	# r,z
			x3 = [1.2,-1.07]	# r,z
			interp1 = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
			interp2 = interp1d([x1[0],x3[0]],[x1[1],x3[1]],fill_value="extrapolate",bounds_error=False)
			# select2 = np.logical_or(np.logical_or(np.mean(grid_data_masked,axis=1)[:,1]<-1.1,np.mean(grid_data_masked,axis=1)[:,1]>interp1(np.mean(grid_data_masked,axis=1)[:,0])),np.mean(grid_data_masked,axis=1)[:,1]>interp2(np.mean(grid_data_masked,axis=1)[:,0]))
			x4 = [1.4,-0.55]	# r,z
			interp3 = interp1d([x4[0],x3[0]],[x4[1],x3[1]],fill_value="extrapolate",bounds_error=False)
			select2 = np.logical_or(np.mean(grid_data_masked,axis=1)[:,1]<-1.1,np.mean(grid_data_masked,axis=1)[:,1]>interp3(np.mean(grid_data_masked,axis=1)[:,0]))
			select = np.logical_and(select,select2)
		grid_data_masked = grid_data_masked[select]
		sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
		select = np.logical_or(np.mean(grid_data_masked,axis=1)[:,0]<divertor_radious_treshold,np.mean(grid_data_masked,axis=1)[:,1]>-1.3)
		grid_data_masked = grid_data_masked[select]
		sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
		grid_laplacian_masked = build_laplacian(grid_data_masked)
		grid_Z_derivate_masked = build_Z_derivate(grid_data_masked)
		grid_R_derivate_masked = build_R_derivate(grid_data_masked)


		return sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked

	sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)

	if development_plots:
		plt.figure()
		plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities_reshaped,axis=(0,1)),marker='s')
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.std(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.sum(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked,axis=(0,1)),marker='s')
		# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.std(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
		# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
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
	for shrink_factor_x in [5,3]:
	# for shrink_factor_x in [3]:
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
		foil_shape = np.shape(sensitivities_binned)[:-1]
		# ROI = np.array([[0.2,0.85],[0.1,0.9]])
		# ROI = np.array([[0.05,0.95],[0.05,0.95]])
		ROI1 = np.array([[0.03,0.85],[0.03,0.95]])
		ROI2 = np.array([[0.03,0.6],[0.03,1-0.03]])
		# ROI = np.array([[0.2,0.95],[0.1,1]])
		ROI1 = np.round((ROI1.T*foil_shape).T).astype(int)
		ROI2 = np.round((ROI2.T*foil_shape).T).astype(int)
		a,b = np.meshgrid(np.arange(foil_shape[1]),np.arange(foil_shape[0]))
		selected_ROI = np.logical_and(np.logical_and(a>=ROI1[1,0],a<ROI1[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI1[0,1],b<sensitivities_binned.shape[0]-ROI1[0,0]))
		selected_ROI = np.logical_or(selected_ROI,np.logical_and(np.logical_and(a>=ROI2[1,0],a<ROI2[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI2[0,1],b<sensitivities_binned.shape[0]-ROI2[0,0])))

		if development_plots:
			plt.figure()
			plt.imshow(np.flip(np.transpose(selected_ROI,(1,0)),axis=1),'rainbow',origin='lower')
			plt.plot([ROI1[0,0]-0.5,ROI1[0,1]-0.5,ROI1[0,1]-0.5,ROI1[0,0]-0.5,ROI1[0,0]-0.5],[ROI1[1,0]-0.5,ROI1[1,0]-0.5,ROI1[1,1]-0.5,ROI1[1,1]-0.5,ROI1[1,0]-0.5],'k')
			plt.plot([ROI2[0,0]-0.5,ROI2[0,1]-0.5,ROI2[0,1]-0.5,ROI2[0,0]-0.5,ROI2[0,0]-0.5],[ROI2[1,0]-0.5,ROI2[1,0]-0.5,ROI2[1,1]-0.5,ROI2[1,1]-0.5,ROI2[1,0]-0.5],'--k')
			# plt.colorbar()
			plt.pause(0.01)

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
			plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

			plt.figure()
			plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s')
			plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			plt.colorbar()
			plt.pause(0.01)

		sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked)

		if development_plots:
			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.mean(sensitivities_binned_crop,axis=(0,1)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

			plt.figure()
			plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.std(sensitivities_binned_crop,axis=(0,1)),marker='s',norm=LogNorm())
			plt.colorbar()
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

		selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=6,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1)
		selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=6,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.5))

		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
		if grid_resolution<8:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells,marker='s')
		plt.title('ede region with emissivity\nrequired to be negligible')
		plt.colorbar()
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region1.eps')
		plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells_for_laplacian,marker='s')
		plt.title('ede region with\nlaplacian of emissivity\nrequired to be low')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region2.eps')
		plt.colorbar()
		plt.pause(0.01)

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
		for shrink_factor_t in [5,3,2]:
		# for shrink_factor_t in [3]:
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)] = dict([])
			binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
			print('starting '+binning_type)
			# powernoback_full = saved_file_dict_short[binning_type].all()['powernoback_full']
			# powernoback_std_full = saved_file_dict_short[binning_type].all()['powernoback_std_full']

			# from here I make the new method.
			# I consider the nominal properties as central value, with:
			# emissivity -10% (from Japanese properties i have std of ~5%, but my nominal value is ~1 and emissivity cannot be >1 so I double the interval down)
			# thickness +/-15% (from Japanese properties i have std of ~15%)
			# diffusivity -10% (this is missing from the Japanese data, so I guess std ~10%)

			emissivity_steps = 3
			thickness_steps = 7
			rec_diffusivity_steps = 7
			sigma_emissivity = 0.1
			sigma_thickness = 0.15
			sigma_rec_diffusivity = 0.1
			emissivity = np.unique(saved_file_dict_short[binning_type].all()['foil_properties']['emissivity'])[0]
			emissivity_array = np.linspace(1-sigma_emissivity*3,1,num=emissivity_steps)
			emissivity_log_prob =  -(0.5*(((1-emissivity_array)/sigma_emissivity)**2))**1	# super gaussian order 1, probability assigned linearly
			emissivity_log_prob = emissivity_log_prob -np.log(np.trapz(np.exp(emissivity_log_prob),x=emissivity_array))	# normalisation for logarithmic probabilities
			thickness = np.unique(saved_file_dict_short[binning_type].all()['foil_properties']['thickness'])[0]
			thickness_array = np.linspace(1-sigma_thickness*3,1+sigma_thickness*3,num=thickness_steps)
			thickness_log_prob =  -(0.5*(((1-thickness_array)/sigma_thickness)**2))**1	# super gaussian order 1, probability assigned linearly
			thickness_log_prob = thickness_log_prob -np.log(np.trapz(np.exp(thickness_log_prob),x=thickness_array))	# normalisation for logarithmic probabilities
			rec_diffusivity = np.unique(saved_file_dict_short[binning_type].all()['foil_properties']['diffusivity'])[0]
			rec_diffusivity_array = np.linspace(1-sigma_rec_diffusivity*3,1+sigma_rec_diffusivity*3,num=rec_diffusivity_steps)
			rec_diffusivity_log_prob =  -(0.5*(((1-rec_diffusivity_array)/sigma_rec_diffusivity)**2))**1	# super gaussian order 1, probability assigned linearly
			rec_diffusivity_log_prob = rec_diffusivity_log_prob -np.log(np.trapz(np.exp(rec_diffusivity_log_prob),x=rec_diffusivity_array))	# normalisation for logarithmic probabilities

			from pycpf import pycpf
			try:
				tend = pycpf.query(['tend'], filters=['exp_number = '+laser_to_analyse[-9:-4]])['tend'][0]+0.05	 # I add 50ms just for safety and to catch disruptions
			except:
				tend = 1

			time_full_binned = saved_file_dict_short[binning_type].all()['time_full_binned']
			BBrad_full = saved_file_dict_short[binning_type].all()['BBrad_full']
			BBrad_std_full = saved_file_dict_short[binning_type].all()['BBrad_std_full']
			diffusion_full = saved_file_dict_short[binning_type].all()['diffusion_full']
			diffusion_std_full = saved_file_dict_short[binning_type].all()['diffusion_std_full']
			timevariation_full = saved_file_dict_short[binning_type].all()['timevariation_full']
			timevariation_std_full = saved_file_dict_short[binning_type].all()['timevariation_std_full']
			if False:	# cutting sensitivity out of ROI
				BBrad_full_crop = BBrad_full[time_full_binned<tend,sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
				BBrad_std_full_crop = BBrad_std_full[time_full_binned<tend,sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
				diffusion_full_crop = diffusion_full[time_full_binned<tend,sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
				diffusion_std_full_crop = diffusion_std_full[time_full_binned<tend,sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
				timevariation_full_crop = timevariation_full[time_full_binned<tend,sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
				timevariation_std_full_crop = timevariation_std_full[time_full_binned<tend,sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]
			else:	# setting zero to the sensitivities I want to exclude
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

			# alternative method using a function that takes the likelihood based on the emissivity and finds the peak of that
			# I think that using the log prob or prob is the same, from the point of view of finding the peak, so I use the log prob because is softer than the prob itself

			# I start with the previous method just to get a decent guess
			powernoback_full = (np.array([[[BBrad_full_crop[0].tolist()]*rec_diffusivity_steps]*thickness_steps]*emissivity_steps).T*emissivity_array).T	# emissivity, thickness, rec_diffusivity
			powernoback_full += (np.array([(np.array([[diffusion_full_crop[0].tolist()]*rec_diffusivity_steps]*thickness_steps).T*thickness_array).T.tolist()]*emissivity_steps))
			powernoback_full += (np.array([(np.array([(np.array([timevariation_full_crop[0].tolist()]*rec_diffusivity_steps).T*rec_diffusivity_array).T.tolist()]*thickness_steps).T*thickness_array).T.tolist()]*emissivity_steps))

			alpha = 1e-4
			A_ = np.dot(sensitivities_binned_crop.T, sensitivities_binned_crop) + (alpha**2) * np.dot(grid_laplacian_masked_crop.T, grid_laplacian_masked_crop)
			d=powernoback_full.reshape(emissivity_steps*thickness_steps*rec_diffusivity_steps,powernoback_full.shape[-2]*powernoback_full.shape[-1])
			b_ = np.dot(sensitivities_binned_crop.T,d.T)

			U, s, Vh = np.linalg.svd(A_)
			sigma = np.diag(s)
			inv_sigma = np.diag(1 / s)
			a1 = np.dot(U, np.dot(sigma, Vh))
			a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
			m = np.dot(a1_inv, b_).T
			neg_m_penalty = np.zeros_like(m)
			neg_m_penalty[m<0] = m[m<0]
			# neg_d_penalty = np.dot(sensitivities_binned_crop,neg_m_penalty.T).T
			neg_m_penalty = -20*neg_m_penalty/np.median(np.flip(np.sort(m[m<0]),axis=0)[-np.sum(m<0)//10:])
			neg_m_penalty = neg_m_penalty.reshape((*powernoback_full.shape[:-2],neg_m_penalty.shape[-1]))

			edge_penalty = np.zeros_like(m)
			edge_penalty[:,selected_edge_cells] = np.max(m[:,selected_edge_cells],0)
			edge_penalty = -50*edge_penalty/np.median(np.sort(edge_penalty[edge_penalty>0])[-np.sum(edge_penalty>0)//10:])
			edge_penalty = edge_penalty.reshape((*powernoback_full.shape[:-2],edge_penalty.shape[-1]))

			if True:	# if I want to bypass this penalty
				neg_powernoback_full_penalty = np.zeros_like(powernoback_full)	# emissivity, thickness, rec_diffusivity
				neg_powernoback_full_penalty[powernoback_full<0] = powernoback_full[powernoback_full<0]
				neg_powernoback_full_penalty = neg_powernoback_full_penalty.reshape((emissivity_steps*thickness_steps*rec_diffusivity_steps,neg_powernoback_full_penalty.shape[-2]*neg_powernoback_full_penalty.shape[-1]))
				neg_powernoback_full_penalty = np.dot(a1_inv, np.dot(sensitivities_binned_crop.T,neg_powernoback_full_penalty.T)).T
				neg_powernoback_full_penalty -= np.max(neg_powernoback_full_penalty)
				# neg_powernoback_full_penalty[neg_powernoback_full_penalty<0] = -10*neg_powernoback_full_penalty[neg_powernoback_full_penalty<0]/np.min(neg_powernoback_full_penalty[neg_powernoback_full_penalty<0])
				if neg_powernoback_full_penalty.min()<0:
					neg_powernoback_full_penalty = -20*neg_powernoback_full_penalty/np.median(np.flip(np.sort(neg_powernoback_full_penalty[neg_powernoback_full_penalty<0]),axis=0)[-np.sum(neg_powernoback_full_penalty<0)//10:])
				neg_powernoback_full_penalty = neg_powernoback_full_penalty.reshape(neg_m_penalty.shape)
			else:
				neg_powernoback_full_penalty = np.zeros_like(neg_m_penalty)
			# neg_powernoback_full_penalty = np.zeros_like(powernoback_full[:,:,:,0])	# emissivity, thickness, rec_diffusivity,coord_1,coord_2
			likelihood = np.transpose(np.transpose(neg_powernoback_full_penalty, (1,2,3,0)) + emissivity_log_prob, (3,0,1,2))
			likelihood = np.transpose(np.transpose(likelihood, (0,2,3,1)) + thickness_log_prob, (0,3,1,2))
			likelihood = np.transpose(np.transpose(likelihood, (0,1,3,2)) + rec_diffusivity_log_prob, (0,1,3,2))
			likelihood += neg_m_penalty
			likelihood += edge_penalty
			# likelihood = np.sum(likelihood, axis=-3)
			likelihood = likelihood -np.log(np.trapz(np.trapz(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0))	# normalisation for logarithmic probabilities
			total_volume = np.trapz(np.trapz(np.trapz(np.ones((emissivity_steps,thickness_steps,rec_diffusivity_steps)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)
			final_emissivity = np.trapz(np.trapz(np.trapz(np.exp(likelihood)*(m.reshape(neg_m_penalty.shape)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)

			powernoback_full_orig = diffusion_full_crop + timevariation_full_crop + BBrad_full_crop
			sigma_powernoback_full = ( (diffusion_full_crop**2)*((diffusion_std_full_crop/diffusion_full_crop)**2+sigma_thickness**2) + (timevariation_full_crop**2)*((timevariation_std_full_crop/timevariation_full_crop)**2+sigma_thickness**2+sigma_rec_diffusivity**2) + (BBrad_full_crop**2)*((BBrad_std_full_crop/BBrad_full_crop)**2+sigma_emissivity**2) )**0.5

			grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
			grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
			grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
			reference_sigma_powernoback = np.nanmedian(sigma_powernoback_full)
			if grid_resolution==4:
				regolarisation_coeff = 1e-3
			elif grid_resolution==2:
				regolarisation_coeff = 5e-4
			# regolarisation_coeff = 1e-5 / ((reference_sigma_powernoback/78.18681)**0.5)
			if True:	# these are the settings that seem to work with regolarisation_coeff = 1e-5
				sigma_emissivity = 1e4	# this is completely arbitrary
				# sigma_emissivity = 1e4 * ((np.median(sigma_powernoback_full)/78)**0.5)	# I think it must go hand in hand with the uncertanty in the pixels
				regolarisation_coeff_edge = 1e-9	# I raiset it artificially from 1e-3 to engage regolarisation_coeff_central_border_Z_derivate and regolarisation_coeff_central_column_border_R_derivate
				regolarisation_coeff_central_border_Z_derivate = 1e-20
				regolarisation_coeff_central_column_border_R_derivate = 1e-20
				regolarisation_coeff_divertor = regolarisation_coeff/2e0
			elif False:	# I want now to rescale all forcing parameters to regolarisation_coeff
				sigma_emissivity = 1e4	# this is completely arbitrary
				# sigma_emissivity = 1e4 * ((np.median(sigma_powernoback_full)/78)**0.5)	# I think it must go hand in hand with the uncertanty in the pixels
				# regolarisation_coeff_edge = regolarisation_coeff*1e2
				regolarisation_coeff_edge = regolarisation_coeff*1e2
				regolarisation_coeff_central_border_Z_derivate = regolarisation_coeff*1e1
				regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*1e2
				regolarisation_coeff_divertor = regolarisation_coeff*1e-5	# in the super-x things seem anways more smooth, so I want to allow more freedom there
			elif False:
				sigma_emissivity = 1e20
				regolarisation_coeff = 1e-20
				regolarisation_coeff_edge = 1e-20
				regolarisation_coeff_central_border_Z_derivate = 1e-20
				regolarisation_coeff_central_column_border_R_derivate = 1e-20
				regolarisation_coeff_divertor = 1e-20

			sigma_emissivity_2 = sigma_emissivity**2

			sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1
			selected_ROI_internal = selected_ROI.flatten()
			inverted_data = []
			inverted_data_likelihood = []
			inverted_data_info = []
			inverted_data_plasma_region_offset = []
			inverted_data_homogeneous_offset = []
			fitted_foil_power = []
			foil_power = []
			foil_power_residuals = []
			fit_error = []
			chi_square_all = []
			for i_t in range(len(time_full_binned_crop)):

				powernoback = powernoback_full_orig[i_t].flatten()
				sigma_powernoback = sigma_powernoback_full[i_t].flatten()
				# sigma_powernoback = np.ones_like(powernoback)*10
				sigma_powernoback_2 = sigma_powernoback**2
				homogeneous_scaling=1e-4
				if time_full_binned_crop[i_t]<0.2:
					if True:	# I can start from a random sample no problem
						guess = final_emissivity+(np.random.random(final_emissivity.shape)-0.5)*1e3	# the 1e3 is to make things harder for the solver, but also to go away a bit from the best fit of other functions
						guess[selected_edge_cells] = guess[selected_edge_cells]*1e-4
						guess = np.concatenate((guess,[0.1/homogeneous_scaling,0.1/homogeneous_scaling]))
					else:
						guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
				else:
					guess = cp.deepcopy(x_optimal)
					guess[:-2] += (np.random.random(x_optimal.shape[0]-2)-0.5)*1e3	# I still add a bit of scramble to give it the freedom to find the best configuration

				target_chi_square = sensitivities_binned_crop.shape[1]	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
				target_chi_square_sigma = 200	# this should be tight, because for such a high number of degrees of freedom things should average very well

				def prob_and_gradient(emissivity_plus,*powernoback):
					homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
					homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
					# print(homogeneous_offset,homogeneous_offset_plasma)
					emissivity = emissivity_plus[:-2]
					emissivity[emissivity==0] = 1e-10
					foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
					foil_power_error = powernoback - foil_power_guess
					emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
					Z_derivate = np.dot(grid_Z_derivate_masked_crop_scaled,emissivity)
					R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)

					likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
					likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
					likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian*np.logical_not(selected_super_x_cells) /sigma_emissivity)**2))
					likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian*selected_super_x_cells /sigma_emissivity)**2))
					likelihood_emissivity_edge_laplacian = (regolarisation_coeff_edge**2)* np.sum(((emissivity_laplacian*selected_edge_cells_for_laplacian /sigma_emissivity)**2))
					likelihood_emissivity_edge = (regolarisation_coeff_edge>1e-10)*np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
					likelihood_emissivity_central_border_Z_derivate = (regolarisation_coeff_central_border_Z_derivate**2)* np.sum((Z_derivate*selected_central_border_cells/sigma_emissivity)**2)
					likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate*selected_central_column_border_cells/sigma_emissivity)**2)
					likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge_laplacian + likelihood_emissivity_edge + likelihood_emissivity_central_border_Z_derivate + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_laplacian_superx
					likelihood_homogeneous_offset = (homogeneous_offset/reference_sigma_powernoback)**2
					likelihood_homogeneous_offset_plasma = (homogeneous_offset_plasma/reference_sigma_powernoback)**2
					likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma

					likelihood_power_fit_derivate = np.concatenate((-2*np.dot((foil_power_error/sigma_powernoback_2),sensitivities_binned_crop),[-2*np.sum(foil_power_error*select_foil_region_with_plasma/sigma_powernoback_2)*homogeneous_scaling,-2*np.sum(foil_power_error*selected_ROI_internal/sigma_powernoback_2)*homogeneous_scaling]))
					likelihood_emissivity_pos_derivate = 2*(np.minimum(0.,emissivity)**2)/emissivity/sigma_emissivity_2
					likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*np.logical_not(selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian*selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge**2) * np.dot(emissivity_laplacian*selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_edge_derivate = 2*(regolarisation_coeff_edge>1e-10)*emissivity*selected_edge_cells/sigma_emissivity_2
					likelihood_emissivity_central_border_Z_derivate_derivate = 2*(regolarisation_coeff_central_border_Z_derivate**2)*np.dot(Z_derivate*selected_central_border_cells,grid_Z_derivate_masked_crop_scaled)/sigma_emissivity_2
					likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate*selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
					likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_edge_laplacian_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_border_Z_derivate_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate + likelihood_emissivity_laplacian_derivate_superx
					likelihood_homogeneous_offset_derivate = 2*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_homogeneous_offset_plasma_derivate = 2*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate
					# likelihood_derivate = likelihood_emissivity_central_border_derivate
					# print([likelihood,likelihood_derivate.max(),likelihood_derivate.min()])
					return likelihood,likelihood_derivate


				if False:	# only for testinf the prob_and_gradient function
					target = -2
					# guess[target] = -10
					temp1 = prob_and_gradient(guess,*powernoback)
					guess[target] +=1e-7
					temp2 = prob_and_gradient(guess,*powernoback)
					guess[target] += -2e-7
					temp3 = prob_and_gradient(guess,*powernoback)
					print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/2e-7)))



				# x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, disp=1, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-7, maxiter=5000)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				if opt_info['warnflag']>0 and False:	# inhibited
					print('incomplete fit so restarted')
					x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=x_optimal, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-7, maxiter=5000)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				x_optimal[-2:] *= homogeneous_scaling

				foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal
				foil_power_error = powernoback - foil_power_guess
				chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
				print('chi_square '+str(chi_square))

				voxels_centre = np.mean(grid_data_masked_crop,axis=1)
				dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
				dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
				dist_mean = (dz**2 + dr**2)/2
				recompose_voxel_emissivity = np.zeros((len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
				for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
					for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
						dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
						if dist.min()<dist_mean/2:
							index = np.abs(dist).argmin()
							# recompose_voxel_emissivity[i_r,i_z] = guess[index]
							# recompose_voxel_emissivity[i_r,i_z] = (x_optimal-guess)[index]
							recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
							# recompose_voxel_emissivity[i_r,i_z] = likelihood_emissivity_laplacian[index]
				recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi


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

				inverted_data.append(recompose_voxel_emissivity)
				inverted_data_likelihood.append(y_opt)
				inverted_data_plasma_region_offset.append(x_optimal[-2])
				inverted_data_homogeneous_offset.append(x_optimal[-1])
				inverted_data_info.append(opt_info)
				fitted_foil_power.append((np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
				foil_power.append(powernoback_full_orig[i_t])
				foil_power_residuals.append(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
				fit_error.append(np.sum(((powernoback_full_orig[i_t][selected_ROI]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape)[[selected_ROI]]))**2)**0.5/np.sum(selected_ROI))
				chi_square_all.append(chi_square)


			inverted_data = np.array(inverted_data)
			inverted_data_likelihood = -np.array(inverted_data_likelihood)
			inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
			inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
			fitted_foil_power = np.array(fitted_foil_power)
			foil_power = np.array(foil_power)
			foil_power_residuals = np.array(foil_power_residuals)
			fit_error = np.array(fit_error)
			chi_square_all = np.array(chi_square_all)

			path_for_plots = path_power_output + '/invertions_log/'+binning_type
			if not os.path.exists(path_for_plots):
				os.makedirs(path_for_plots)


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

			# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
			extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
			ani,efit_reconstruction = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
			ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
			plt.close()

			ani,efit_reconstruction = coleval.movie_from_data(np.array([np.flip(np.transpose(fitted_foil_power,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))) ,timesteps=time_full_binned_crop,integration=laser_int_time/1000,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Fitted power on foil [W/m2]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,grid_resolution),overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
			ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_fitted_foil_power_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
			plt.close('all')

			if efit_reconstruction!=None:
				all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
				all_time_strike_points_location = coleval.return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
				outer_leg_tot_rad_power_all = []
				inner_leg_tot_rad_power_all = []
				core_tot_rad_power_all = []
				x_point_tot_rad_power_all = []
				for i_t in range(len(time_full_binned_crop)):
					temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
					xpoint_r = efit_reconstruction.lower_xpoint_r[temp]
					xpoint_z = efit_reconstruction.lower_xpoint_z[temp]
					z_,r_ = np.meshgrid(np.unique(voxels_centre[:,1]),np.unique(voxels_centre[:,0]))
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_>xpoint_z] = 0
					temp[r_<xpoint_r] = 0
					outer_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_>xpoint_z] = 0
					temp[r_>xpoint_r] = 0
					inner_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					temp = cp.deepcopy(inverted_data[i_t])
					temp[z_<xpoint_z] = 0
					temp[z_>0] = 0
					core_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					temp = cp.deepcopy(inverted_data[i_t])
					temp[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.10] = 0
					x_point_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
					outer_leg_tot_rad_power_all.append(outer_leg_tot_rad_power)
					inner_leg_tot_rad_power_all.append(inner_leg_tot_rad_power)
					core_tot_rad_power_all.append(core_tot_rad_power)
					x_point_tot_rad_power_all.append(x_point_tot_rad_power)
				outer_leg_tot_rad_power_all = np.array(outer_leg_tot_rad_power_all)
				inner_leg_tot_rad_power_all = np.array(inner_leg_tot_rad_power_all)
				core_tot_rad_power_all = np.array(core_tot_rad_power_all)
				x_point_tot_rad_power_all = np.array(x_point_tot_rad_power_all)

				plt.figure(figsize=(20, 10))
				plt.plot(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,label='outer_leg')
				plt.plot(time_full_binned_crop,inner_leg_tot_rad_power_all/1e3,label='inner_leg')
				plt.plot(time_full_binned_crop,core_tot_rad_power_all/1e3,label='core')
				plt.plot(time_full_binned_crop,x_point_tot_rad_power_all/1e3,label='x_point')
				plt.plot(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3+inner_leg_tot_rad_power_all/1e3+core_tot_rad_power_all/1e3,label='tot')
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
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['x_point_tot_rad_power_all'] = x_point_tot_rad_power_all

				additional_points_dict,radiator_xpoint_distance_all,radiator_above_xpoint_all,radiator_magnetic_radious_all = coleval.find_radiator_location(inverted_data,np.unique(voxels_centre[:,0]),np.unique(voxels_centre[:,1]),time_full_binned_crop,efit_reconstruction)

				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_location_all'] = additional_points_dict['0']
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_xpoint_distance_all'] = radiator_xpoint_distance_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_above_xpoint_all'] = radiator_above_xpoint_all
				inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['radiator_magnetic_radious_all'] = radiator_magnetic_radious_all

				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				ani,efit_reconstruction = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,grid_resolution) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,additional_points_dict=additional_points_dict)#,extvmin=0,extvmax=4e4)
				ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
				plt.close()

				fig, ax = plt.subplots( 2,1,figsize=(8, 12), squeeze=False,sharex=True)
				ax[0,0].plot(time_full_binned_crop,radiator_magnetic_radious_all)
				ax[0,0].set_ylim(top=min(np.nanmax(radiator_magnetic_radious_all),1.1),bottom=max(np.nanmin(radiator_magnetic_radious_all),0.9))
				ax[1,0].plot(time_full_binned_crop,radiator_above_xpoint_all)
				fig.suptitle('Location of the x-point radiator')
				ax[0,0].set_ylabel('normalised psi [au]')
				ax[0,0].grid()
				ax[1,0].set_xlabel('time [s]')
				ax[1,0].set_ylabel('position above x-point [m]')
				ax[1,0].grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_x_point_location.eps')
				plt.close('all')


			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data'] = inverted_data
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_likelihood'] = inverted_data_likelihood
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_info'] = inverted_data_info
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['select_foil_region_with_plasma'] = select_foil_region_with_plasma
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_plasma_region_offset'] = inverted_data_plasma_region_offset
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data_homogeneous_offset'] = inverted_data_homogeneous_offset
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop'] = time_full_binned_crop
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['fitted_foil_power'] = fitted_foil_power
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['foil_power'] = foil_power
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['foil_power_residuals'] = foil_power_residuals
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['fit_error'] = fit_error
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['chi_square_all'] = chi_square_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry'] = dict([])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['R'] = np.unique(voxels_centre[:,0])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['Z'] = np.unique(voxels_centre[:,1])


			np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian',**inverted_dict)

exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_power_to_emissivity2.py").read())
