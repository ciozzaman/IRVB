

# spatial_averaging_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for spatial_averaging in np.flip(spatial_averaging_all,0):

	spatial_averaging_old = 1
	if (is_this_extra and spatial_averaging_old == 1):
		continue

	if spatial_averaging_old > 1:
		foil_resolution = str(spatial_averaging_old) + 'x' + str(spatial_averaging_old)
	else:
		foil_resolution = '187'


	# data analysis


	if with_noise:
		# time_averaging_tries = [3,4,5,6]
		time_averaging_tries=cp.deepcopy(time_averaging_all)
	else:
		time_averaging_tries=[1]
		# time_averaging_tries=cp.deepcopy(time_averaging_all)


	# noise_on_temporal_comp = 190	#this is increased  from 76 to account for the correction of diffusivity from laser measurements
	# noise_on_laplacian_comp = 223
	# noise_on_bb_comp = 0.46

	for time_averaging in time_averaging_tries:
		if not is_this_extra:
			# noise_on_power = np.sqrt( (noise_on_temporal_comp/(np.sqrt(time_averaging)))**2 + (noise_on_laplacian_comp/int(np.around(foil_resolution_max/foil_resolution)))**2 + noise_on_bb_comp**2 )
			# noise_on_power = return_power_noise_level(int(np.around(foil_resolution_max/foil_resolution)),time_averaging)
			noise_on_power = return_power_noise_level(spatial_averaging_old,time_averaging)
		else:
			# noise_on_power = np.sqrt((noise_on_temporal_comp / (np.sqrt(time_averaging))) ** 2 + (noise_on_laplacian_comp) ** 2 + noise_on_bb_comp ** 2)
			noise_on_power = return_power_noise_level(1,time_averaging)


		if is_this_extra:
			foil_res = '_foil_pixel_h_'+str(foil_resolution)+'_extra'
		else:
			foil_res = '_foil_pixel_h_'+str(foil_resolution)
		grid_type = 'core_res_'+str(grid_resolution)+'cm'
		path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
		path_sensitivity_original = cp.deepcopy(path_sensitivity)
		# path_for_plots = cp.deepcopy(path_sensitivity)
		path_for_plots = '/home/ffederic/work/analysis_scripts'+'/SOLPS'+str(ref_number)+'/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
		if not os.path.exists(path_sensitivity + '/sensitivity.npz'):
			sensitivities = np.load(path_sensitivity + '/sensitivity.npy')
			scipy.sparse.save_npz(path_sensitivity + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities))
		else:
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity.npz')).todense())

		if (os.path.exists(path_sensitivity + '/mask2_on_camera_pixels.npz') and os.path.exists(path_sensitivity + '/sensitivity_masked2.npz') and enable_mask2):
			mask2_on_camera_pixels = sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/mask2_on_camera_pixels.npz')).todense())[0]
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity_masked2.npz')).todense())
			flag_mask2_present = True
			flag_mask1_present = True
		elif (os.path.exists(path_sensitivity + '/sensitivity_masked1.npz') and enable_mask1):
			sensitivities = np.array((scipy.sparse.load_npz(path_sensitivity + '/sensitivity_masked1.npz')).todense())
			flag_mask2_present = False
			flag_mask1_present = True
		else:
			flag_mask2_present = False
			flag_mask1_present = False

		type='.csv'
		filenames = coleval.all_file_names(path_sensitivity, type)[0]
		with open(os.path.join(path_sensitivity,filenames)) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if row[0]=='foil horizontal pixels ':
					pixel_h=int(row[1])
				if row[0] == 'foil vertical pixels ':
					pixel_v = int(row[1])
				if row[0]=='pipeline type ':
					pipeline =row[1]
				if row[0]=='type of volume grid ':
					grid_type =row[1]
				# print(row)

		grid_type_not_masked = cp.deepcopy(grid_type)
		if flag_mask1_present:
			grid_type = grid_type+'_masked1'
		# with open(path_sensitivity+'/svd_decomposition.pickle', 'rb') as f:
		# 	decomposition = pickle.load(f)

		from raysect.optical import World, ConstantSF
		# world = World()
		from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid
		from cherab.tools.inversions import ToroidalVoxelGrid
		core_voxel_grid = load_standard_voxel_grid(grid_type,parent=world)
		if (flag_mask1_present and is_this_extra):
			core_voxel_grid_not_masked = load_standard_voxel_grid(grid_type_not_masked, parent=world)
		# sensitivities = np.random.rand(500,155)
		# print(scipy.sparse.issparse(scipy.sparse.csr_matrix(sensitivities)))
		# import scipy.sparse
		# scipy.sparse.save_npz(path_sensitivity+'/sensitivity.npz',scipy.sparse.csr_matrix(sensitivities))
		# sensitivities_sparse=scipy.sparse.load_npz(path_sensitivity+'/sensitivity.npz')
		# print(np.allclose(sensitivities, sensitivities_sparse.todense()))

		directory = '/home/ffederic/work/cherab/cherab_mastu/cherab/mastu/bolometry/grid_construction'
		grid_file = os.path.join(directory,'{}_rectilinear_grid.pickle'.format(grid_type))
		with open(grid_file, 'rb') as f:
			grid_data = pickle.load(f)
		laplacian = grid_data['laplacian']

		# check=[]
		# alpha_record=[]
		# for alpha in np.linspace(0.000001,0.0000001,num=20):
		# 	# print(repr(sensitivities))
		# 	# shape=np.shape(sensitivities)
		# 	# trashold=tre
		# 	test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
		#
		# 	# U, s, Vh = np.linalg.svd(sensitivities)
		# 	U, s, Vh = np.linalg.svd(test_matrix)
		# 	# s, Vh = scipy.sparse.linalg.eigs(sensitivities,k=min(np.shape(sensitivities))-4)
		# 	sgna = np.linalg.cond(test_matrix)
		# 	check.append(sgna)
		# 	alpha_record.append(alpha)
		# 	# zero=[]
		# 	print('alpha='+str(alpha))
		# 	print('max s='+str(s[0]))
		# 	print('conditioning='+str(sgna))
		# 	# for i in range(min(shape)):
		# 	# 	if s[i]<trashold:
		# 	# 		zero.append(i)
		# 	# # if zero==[]:
		# 	# # 	zero.append(min(shape))
		# 	# U=U[:,:zero[0]]
		# 	# Vh=Vh[:zero[0]]
		# 	# sigma = np.zeros((zero[0],zero[0]))
		# 	# for i in range(zero[0]):
		# 	# 	if s[i]<trashold:
		# 	# 		sigma[i, i]=0
		# 	# 	else:
		# 	# 		sigma[i, i] = s[i]
		# 	# a1 = np.dot(U, np.dot(sigma, Vh))
		# 	# # np.allclose(sensitivities, a1)
		# 	# inv_sigma = np.zeros((zero[0],zero[0]))
		# 	# for i in range(zero[0]):
		# 	# 	inv_sigma[i, i] = 1/s[i]
		# 	# # SGNAPPO = np.dot(Vh.T, np.dot(inv_sigma, U.T))
		# 	#
		# 	# print('tre='+str(trashold))
		# 	# print(str((np.dot(sensitivities,np.linspace(0,155-1,155)) - np.dot(a1,np.linspace(0,155-1,155))).max()))
		# 	# # # del a
		# 	# BLUE=0
		# 	# # del SGNAPPO
		# 	# del sigma
		# 	# del U,Vh,s,zero,shape,inv_sigma
		# 	del U, Vh, s
		# print(coleval.find_nearest_index(check,0))
		# alpha=alpha_record[coleval.find_nearest_index(check,0)[0]]


		#Here there are different types of radiator

		if False:
			test_voxel=43
			power_on_voxels = np.zeros((core_voxel_grid.count))
			power_on_voxels[test_voxel] = 2
		elif False:
			power_on_voxels=np.linspace(1E+08,1,np.shape( sensitivities)[1])
		elif True:
			power_on_voxels = np.zeros((core_voxel_grid.count))
			num_voxels = core_voxel_grid.count

			# i = np.linspace(0, num_voxels - 1, num_voxels, dtype=int)
			for index in range(num_voxels):
				p1, p2, p3, p4 = core_voxel_grid._voxels[index].vertices
				voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
				power_on_voxels[index] = radiation_function(voxel_centre.x, 0, voxel_centre.y)
			if (flag_mask1_present and is_this_extra):
				power_on_voxels_no_mask = np.zeros((core_voxel_grid_not_masked.count))
				num_voxels_not_masked = core_voxel_grid_not_masked.count

				# i = np.linspace(0, num_voxels - 1, num_voxels, dtype=int)
				for index in range(num_voxels_not_masked):
					p1, p2, p3, p4 = core_voxel_grid_not_masked._voxels[index].vertices
					voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
					power_on_voxels_no_mask[index] = radiation_function(voxel_centre.x, 0, voxel_centre.y)
			else:
				power_on_voxels_no_mask=power_on_voxels


		if is_this_extra:
			d = np.load(path_sensitivity + '/foil_power.npy')
		else:
			d = np.dot(sensitivities, power_on_voxels)
		d_original = cp.deepcopy(d)

		if flag_mask2_present:
			path_for_plots = path_for_plots + '/mask2'
			print('Sensitivity matrix filtered for voxels that do not see the foil and pixels that are not shone by any plasma')
			mod = '_mask2'
		elif (flag_mask1_present and not flag_mask2_present):
			path_for_plots = path_for_plots + '/mask1'
			print('Sensitivity matrix filtered for voxels that do not see the foil')
			mod = '_mask1'
		else:
			path_for_plots = path_for_plots + '/unmasked'
			print('Sensitivity matrix not filtered')
			mod = ''

		# if treshold_method_try_to_search:
		# 	path_for_plots = path_for_plots + '/SVG_search_best_cut'
		# 	print('Cut of the eigenvalues defined based minimizing the residuals')
		# else:
		# 	if eigenvalue_cut_vertical:
		# 		path_for_plots = path_for_plots + '/SVG_vertical_cut'
		# 		print('Cut of the eigenvalues in the vertical direction')
		# 	else:
		# 		path_for_plots = path_for_plots + '/SVG_horizontal_cut'
		# 		print('Cut of the eigenvalues in the horizontal direction')
		path_for_plots = path_for_plots + '/only_Tikhonov'
		print('Only Tikhonov regularization performed')

		if weigthed_best_search:
			path_for_plots = path_for_plots + '/weighted_best_alpha_search'
			print('Estimation of the residuals weighted on the expected signals')
		else:
			path_for_plots = path_for_plots + '/no_weighted_best_alpha_search'
			print('Estimation of the residuals not weighted')

		if residuals_on_power_on_voxels:
			path_for_plots = path_for_plots + '/residuals_on_power_on_voxels'
			print('Estimation of the residuals on the expected versus inverted power on voxels')
		else:
			path_for_plots = path_for_plots + '/residuals_on_power_on_foil_pixels'
			print('Estimation of the residuals on the expected versus inverted power on foil pixels')

		if spatial_averaging>1:
			path_for_plots = path_for_plots + '/pixel_binning_of_'+str(spatial_averaging)+'x'+str(spatial_averaging)
			print('Binning of foil pixels of '+str(spatial_averaging)+'x'+str(spatial_averaging))
		else:
			print('No pixel binning')


		if not os.path.exists(path_for_plots):
			os.makedirs(path_for_plots)

		if with_noise:
			path_for_plots = path_for_plots + '/with_noise_time_averaging_' + str(time_averaging)
			if not os.path.exists(path_for_plots):
				os.makedirs(path_for_plots)
			print('with noise, power std='+str(noise_on_power))
			if is_this_extra:
				foil_res_max = '_foil_pixel_h_' + str(foil_resolution_max)
				path_sensitivity_max_res = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type_not_masked[5:] + foil_res_max + '_power'
				if not os.path.exists(path_sensitivity_max_res + '/sensitivity.npz'):
					sensitivities_max_res = np.load(path_sensitivity_max_res + '/sensitivity.npy')
					scipy.sparse.save_npz(path_sensitivity_max_res + '/sensitivity.npz', scipy.sparse.csr_matrix(sensitivities_max_res))
				else:
					sensitivities_max_res = np.array((scipy.sparse.load_npz(path_sensitivity_max_res + '/sensitivity.npz')).todense())
				type = '.csv'
				filenames = coleval.all_file_names(path_sensitivity_max_res, type)[0]
				with open(os.path.join(path_sensitivity_max_res, filenames)) as csv_file:
					csv_reader = csv.reader(csv_file, delimiter=',')
					for row in csv_reader:
						if row[0] == 'foil horizontal pixels ':
							pixel_h_max_res = int(row[1])
				d_high_res = np.dot(sensitivities_max_res, power_on_voxels_no_mask)
				d_original = coleval.foil_measurement_averaging_foil_pixels_extra_loseless(d_high_res,pixel_h_max_res,spatial_averaging_old,spatial_averaging_old)
				d_high_res = d_high_res + np.random.normal(0, noise_on_power, len(d_high_res))
				d = coleval.foil_measurement_averaging_foil_pixels_extra_loseless(d_high_res, pixel_h_max_res, spatial_averaging_old,spatial_averaging_old)

				if flag_mask2_present:
					d_temp = []
					d_original_temp = []
					for index, masked in enumerate(mask2_on_camera_pixels):
						if not masked:
							d_temp.append(d[index])
							d_original_temp.append(d_original[index])
					d=np.array(d_temp)
					d_original=np.array(d_original_temp)

			else:
				d_original = cp.deepcopy(d)
				d = d_original + np.random.normal(0, noise_on_power, len(d_original))
		else:
			print('without noise')

			# for index,value in enumerate(d):
			# 	if value<=0:
			# 		d[index]=0


		# if flag_mask2_present:
		# 	temp1 = cp.deepcopy(d)
		# 	temp2 = cp.deepcopy(d_original)
		# 	d = []
		# 	d_original = []
		# 	for index,mask in enumerate(mask2_on_camera_pixels):
		# 		if not mask:
		# 			d.append(temp1[index])
		# 			d_original.append(temp2[index])
		# 	d = np.array(d)
		# 	d_original = np.array(d_original)

		# Here there are different weighting functions
		if False:
			weight = np.ones(len(power_on_voxels))
		elif False:
			weight = power_on_voxels / (power_on_voxels).max()

		if weigthed_best_search:
			weight1 = d_original / (d_original).max()
			weight2 = power_on_voxels / (power_on_voxels).max()
		else:
			weight1 = np.ones(len(d_original))
			weight2 = np.ones(len(power_on_voxels))


		# treshold=3E-22	# this for 2cm 7 h pixels foil resolution
		# treshold=2E-21	# this for 2cm 19 h pixels foil resolution
		# treshold=1E-18	# this for 2cm full foil resolution
		# check=[]
		# s_record = []
		# alpha_record=[]
		# U_record = []
		# Vh_record = []
		# score_x_record=[]
		# score_y_record=[]
		# treshold_record = []
		# treshold_horizontal_record = []

		# find_minimum = min(np.shape(sensitivities))
		# flag_for_extrapolated_treshold = False
		# initial_treshold_s = len(power_on_voxels)-1-300
		# first_treshold_s_search = True



		# 04/10/2019 bit for foil binning
		if spatial_averaging >1:

			fig, ax = plt.subplots()
			cmap = plt.cm.rainbow
			cmap.set_under(color='white')
			plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap)
			plt.title('Power density on the foil via sensitivity matrix')
			plt.colorbar().set_label('Power density on the foil [W/m^2]')
			plt.xlabel('Horizontal axis [pixles]')
			plt.ylabel('Vertical axis [pixles]')
			plt.savefig(path_for_plots + '/power_distribution_on_foil_not_binned.eps')
			plt.close()

		if with_noise:

			fig, ax = plt.subplots()
			cmap = plt.cm.rainbow
			cmap.set_under(color='white')
			plt.imshow(coleval.split_fixed_length(d_original, pixel_h), origin='lower', cmap=cmap)
			plt.title('Power density on the foil via sensitivity matrix')
			plt.colorbar().set_label('Power density on the foil [W/m^2]')
			plt.xlabel('Horizontal axis [pixles]')
			plt.ylabel('Vertical axis [pixles]')
			plt.savefig(path_for_plots + '/power_distribution_on_foil_no_noise.eps')
			plt.close()

		if spatial_averaging > 1:

			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) )
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) )
			dh = 0.07 / pixel_h
			dv = 0.09 / pixel_v
			pixels_centre_location = np.array([(np.ones((pixel_v, pixel_h)) * binning_h*dh+dh/2).flatten(),((np.ones((pixel_v, pixel_h)).T * binning_v*dv+dv/2).T).flatten()])

			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) '+str(pixel_h_new)+'x'+str(pixel_v_new)+' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging)
			binning_h[binning_h>pixel_h_new-1] = pixel_h_new-1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging)
			binning_v[binning_v>pixel_v_new-1] = pixel_v_new-1
			pixels_to_bin = ((np.ones((pixel_v,pixel_h))*binning_h).T+binning_v*pixel_h_new).T.astype('int').flatten()
			bin_centre_location = np.zeros((np.max(pixels_to_bin)+1,2))
			dh = 0.07 / pixel_h
			dv = 0.09 / pixel_v

			for bin_index in range(np.max(pixels_to_bin)+1):	# coordinates as (horizontal,vertical)
				bin_centre_location[bin_index] = np.mean(pixels_centre_location[:,pixels_to_bin==bin_index] ,axis=-1)


			temp = np.zeros((np.max(pixels_to_bin)+1,np.shape(sensitivities)[-1]))
			for bin_index in range(np.max(pixels_to_bin)+1):
				temp[bin_index] = np.mean(sensitivities[pixels_to_bin==bin_index],axis=0)
			sensitivities_not_binned = cp.deepcopy(sensitivities)
			sensitivities = cp.deepcopy(temp)

			temp = np.zeros((np.max(pixels_to_bin)+1))
			for bin_index in range(np.max(pixels_to_bin)+1):
				temp[bin_index] = np.mean(d[pixels_to_bin==bin_index])
			d_not_binned = cp.deepcopy(d)
			d = cp.deepcopy(temp)

			temp = np.zeros((np.max(pixels_to_bin)+1))
			for bin_index in range(np.max(pixels_to_bin)+1):
				temp[bin_index] = np.mean(weight1[pixels_to_bin==bin_index])
			weight1_not_binned = cp.deepcopy(weight1)
			weight1 = cp.deepcopy(temp)

			temp = np.zeros((np.max(pixels_to_bin)+1))
			for bin_index in range(np.max(pixels_to_bin)+1):
				temp[bin_index] = np.sum(d_original[pixels_to_bin==bin_index])
			d_original_not_binned = cp.deepcopy(d_original)
			d_original = cp.deepcopy(temp)

			pixel_h = cp.deepcopy(pixel_h_new)
			pixel_v = cp.deepcopy(pixel_v_new)

		elif False:

			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) )
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) )
			dh = 0.07 / pixel_h
			dv = 0.09 / pixel_v
			pixels_centre_location = np.array([(np.ones((pixel_v, pixel_h)) * binning_h*dh+dh/2),((np.ones((pixel_v, pixel_h)).T * binning_v*dv+dv/2).T)])

			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) '+str(pixel_h_new)+'x'+str(pixel_v_new)+' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging)
			binning_h[binning_h>pixel_h_new-1] = pixel_h_new-1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging)
			binning_v[binning_v>pixel_v_new-1] = pixel_v_new-1
			pixels_to_bin = ((np.ones((pixel_v,pixel_h))*binning_h).T+binning_v*pixel_h_new).T.astype('int')
			bin_centre_location = np.zeros((np.max(pixels_to_bin)+1,2))
			dh = 0.07 / pixel_h
			dv = 0.09 / pixel_v

			for bin_index in range(np.max(pixels_to_bin)+1):	# coordinates as (horizontal,vertical)
				bin_centre_location[bin_index] = np.mean(pixels_centre_location[:,pixels_to_bin==bin_index] ,axis=-1)


			temp = np.zeros((np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[bin_index] = np.mean(d[pixels_to_bin.flatten() == bin_index])

			bin_centre_location = bin_centre_location.reshape((pixel_v_new,pixel_h_new,2))
			temp = temp.reshape((pixel_v_new,pixel_h_new))

			# pixels_centre_location = np.array([(np.ones((pixel_v, pixel_h)) * binning_h*dh+dh/2).flatten(),((np.ones((pixel_v, pixel_h)).T * binning_v*dv+dv/2).T).flatten()])

			fig, ax = plt.subplots()
			cmap = plt.cm.rainbow
			cmap.set_under(color='white')
			# plt.pcolor(bin_centre_location[:,:,0],bin_centre_location[:,:,1],temp, cmap=cmap)
			plt.imshow(temp, origin='lower', cmap=cmap)
			plt.title('Power density on the foil via sensitivity matrix')
			plt.colorbar().set_label('Power density on the foil [W/m^2]')
			plt.xlabel('Horizontal axis [pixles]')
			plt.ylabel('Vertical axis [pixles]')
			plt.axis('equal')
			plt.pause(0.01)

			# # this works but introduces strong smoothing in order to work properly
			# foil_power_interpolator = bisplrep(bin_centre_location[:,:,1], bin_centre_location[:,:,0], temp)
			# # d_interpolated = bisplev(pixels_centre_location[1], pixels_centre_location[0], foil_power_interpolator)
			# d_interpolated = bisplev(pixels_centre_location[1,:,0],pixels_centre_location[0,0,:], foil_power_interpolator)

			# this introduces blurring but I think no smoothing
			d_interpolated = griddata(np.array([bin_centre_location[:,:,1].flatten(), bin_centre_location[:,:,0].flatten()]).T , temp.flatten(), (pixels_centre_location[1,:,:],pixels_centre_location[0,:,:]), method='cubic')


			# fig, ax = plt.subplots()
			# cmap = plt.cm.rainbow
			# cmap.set_under(color='white')
			# # plt.pcolor(pixels_centre_location[0,:,:],pixels_centre_location[1,:,:],d_interpolated, cmap=cmap)
			# plt.imshow(d_interpolated, origin='lower', cmap=cmap)
			# plt.title('Power density on the foil via sensitivity matrix')
			# plt.colorbar().set_label('Power density on the foil [W/m^2]')
			# plt.xlabel('Horizontal axis [pixles]')
			# plt.ylabel('Vertical axis [pixles]')
			# plt.axis('equal')
			# plt.pause(0.01)
			# plt.savefig(path_for_plots + '/power_distribution_on_foil_no_noise.eps')
			# plt.close()




		class calc_stuff_output:
			def __init__(self, exp, alpha, score_x, score_y):
				self.exp = exp
				self.alpha = alpha
				self.score_x = score_x
				self.score_y = score_y


		def calc_stuff(arg, sensitivities=sensitivities, laplacian=laplacian, path_sensitivity_original=path_sensitivity_original,path_for_plots=path_for_plots,mod=mod,residuals_on_power_on_voxels=residuals_on_power_on_voxels ,d=d,  power_on_voxels=power_on_voxels,spatial_averaging=spatial_averaging, weight1=weight1, weight2=weight2,spatial_averaging_old=spatial_averaging_old):
			index_ext = arg[0]
			exp = arg[1]
			alpha=1*10**(exp)
			if exp==-50:
				alpha=0
			# test_matrix = np.dot(sensitivities.T, sensitivities) + (alpha**2) * laplacian
			test_matrix = np.dot(sensitivities.T, sensitivities) + (alpha**2) * np.dot(laplacian.T, laplacian)

			if False:	# modified 21/10/2019
				if spatial_averaging==1:
					if not os.path.exists(path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha)+mod+'.npz'):
						try:
							# np.linalg.cond(test_matrix)
							U, s, Vh = np.linalg.svd(test_matrix)
						except:
							print('the SVG decomposition of alpha = ' + str(alpha) + ' did not converge')
							# continue
							output = calc_stuff_output(exp, alpha, np.inf, np.inf)
							return output
						# U, s, Vh = np.linalg.svd(test_matrix)
						np.savez_compressed(path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha)+mod,U=U, s=s, Vh=Vh)
					else:
						try:
							U = np.load(path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha)+mod+'.npz')['U']
							s = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) +mod+ '.npz')['s']
							Vh = np.load(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha) +mod+ '.npz')['Vh']
						except:
							try:
								print('Trying to recover SVG decomposition ' + path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha)+mod)
								U, s, Vh = np.linalg.svd(test_matrix)
								if spatial_averaging == 1:
									np.savez_compressed(path_sensitivity_original + '/svg_decomp_alpha_' + str(alpha)+mod, U=U, s=s, Vh=Vh)
								print('stored SVG decomposition '+path_sensitivity_original+'/svg_decomp_alpha_'+str(alpha)+mod+' recovered')
							except:
								print('the SVG decomposition of alpha = ' + str(alpha) + ' did not converge')
								# continue
								output = calc_stuff_output(exp, alpha, np.inf, np.inf)
								return output
				else:
					# np.linalg.cond(test_matrix)
					try:
						U, s, Vh = np.linalg.svd(test_matrix)
						print('the SVG decomposition of alpha = ' + str(alpha) + ', spatial averaging = '+str(spatial_averaging)+', time averaging = '+str(time_averaging)+' converged')
						convergence_test = np.linalg.cond(test_matrix)
						print('with convergence parameter np.linalg.cond(test_matrix) = '+str(convergence_test))

					except:
						print('the SVG decomposition of alpha = ' + str(alpha) + ', spatial averaging = '+str(spatial_averaging)+', time averaging = '+str(time_averaging)+' did not converge')
						try:
							convergence_test = np.linalg.cond(test_matrix)
							print('with convergence parameter np.linalg.cond(test_matrix) = '+str(convergence_test))
						except:
							print('I cannot calculate the convergence parameter...')
						output = calc_stuff_output(exp, alpha, np.nan, np.nan)
						return output

				# s_record.append(s)
				# U_record.append(U)
				# Vh_record.append(Vh)

				sigma = np.diag(s)
				inv_sigma = np.diag(1/s)

				a1 = np.dot(U, np.dot(sigma, Vh))
				a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))


			else:
				if not os.path.exists(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod + '.npz'):
					try:
						# np.linalg.cond(test_matrix)
						U, s, Vh = np.linalg.svd(test_matrix)
						print('the SVG decomposition of alpha = ' + str(alpha) + ' is generated')
					except:
						print('the SVG decomposition of alpha = ' + str(alpha) + ' did not converge')
						# continue
						output = calc_stuff_output(exp, alpha, np.nan, np.nan)
						return output
					# U, s, Vh = np.linalg.svd(test_matrix)

					sigma = np.diag(s)
					inv_sigma = np.diag(1 / s)
					a1 = np.dot(U, np.dot(sigma, Vh))
					a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

					np.savez_compressed(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod, a1_inv=a1_inv,s=s)

				else:
					try:
						a1_inv = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod + '.npz')['a1_inv']
						s = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod + '.npz')['s']
						print('the SVG decomposition of alpha = ' + str(alpha) + ' is imported')
					except:
						try:
							print('Trying to recover SVG decomposition ' + path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod)
							U, s, Vh = np.linalg.svd(test_matrix)
							sigma = np.diag(s)
							inv_sigma = np.diag(1 / s)
							a1 = np.dot(U, np.dot(sigma, Vh))
							a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

							np.savez_compressed(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod, a1_inv=a1_inv,s=s)
							print('stored SVG decomposition ' + path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod + ' recovered')
						except:
							print('the SVG decomposition of alpha = ' + str(alpha) + ' did not converge')
							# continue
							output = calc_stuff_output(exp, alpha, np.nan, np.nan)
							return output


			m = np.dot(a1_inv, np.dot(sensitivities.T, d))


			'''
			if not treshold_method_try_to_search:
				flag_vertical_treshold_found = False
				if False:
					treshold = 1.5*s[50+coleval.find_nearest_index(s[100:]-s[:-100],0)]	#	the 1.5 is just to make sure to catch all of the negligible terms
				else:
					s_scaling = np.log(np.array(s)) - min(np.log(np.array(s)) )
					s_scaling = s_scaling * 1 * (len(s) / max(s_scaling))
					s_scaling = s_scaling * np.linspace(int(min(s_scaling)), int(min(s_scaling)) + len(s), len(s))
					# where1 = coleval.find_nearest_index(s_scaling, 1E20)
					search_failed = True
					for index in range(101, len(s_scaling) - 1, 1):
						# if ( np.mean(s_scaling[index-100:index-10])>s_scaling[index] and np.mean(s_scaling[index+10:index+100])>s_scaling[index] ):
						if (np.mean(s_scaling[index - 100:index - 10]) > s_scaling[index] and np.mean(s_scaling[index + 10:index + 100]) > s_scaling[index]):
							find_minimum = index +coleval.find_nearest_index(s_scaling[index - 100:index + 101],0)-100
							search_failed = False
							break
					if flag_for_extrapolated_treshold:
						treshold = np.exp(line(np.log(alpha),*coeff_fitting))
						treshold_not_estimated = s[find_minimum]
						search_failed = False
					else:
						treshold = s[find_minimum]
					if search_failed:
						find_minimum = coleval.find_nearest_index(s_scaling,max(s_scaling))
						treshold = s[find_minimum]
					treshold_horizontal_record.append(treshold)
					if eigenvalue_cut_vertical:
						for index in range(find_minimum, 100, -1):
							# if ( np.mean(s_scaling[index-100:index-10])>s_scaling[index] and np.mean(s_scaling[index+10:index+100])>s_scaling[index] ):
							if (np.mean(s_scaling[index - 100:index - 5]) < s_scaling[index] and np.mean(s_scaling[index + 5:index + 100]) < s_scaling[index]):
								find_maximum = index + coleval.find_nearest_index(s_scaling[index - 100:index + 101],1E10) - 100
								flag_vertical_treshold_found = True
								treshold = s[find_maximum]
								break
				treshold_s = coleval.find_nearest_index(s,treshold)

				shape = np.shape(test_matrix)
				zero = []
				for i in range(treshold_s,shape[0],1):
					zero.append(i)
				if zero == []:
					zero.append(shape[0])
				sigma = np.zeros((zero[0], zero[0]))
				U = U[:, :zero[0]]
				Vh = Vh[:zero[0]]
				for i in range(zero[0]):
					if s[i] <= treshold:
						sigma[i, i] = 0
					else:
						sigma[i, i] = s[i]
				a1 = np.dot(U, np.dot(sigma, Vh))
				inv_sigma = np.zeros((zero[0], zero[0]))
				for i in range(zero[0]):
					inv_sigma[i, i] = 1 / s[i]
				a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
				m = np.dot(a1_inv, np.dot(sensitivities.T, d))
				# for index,value in enumerate(m):
				# 	if value<=0:
				# 		m[index]=0

			else:
				if first_treshold_s_search:
					# This is just to have a decent value of the treshold from which start from
					s_scaling = np.log(np.array(s)) - min(np.log(np.array(s)))
					s_scaling = s_scaling * 1 * (len(s) / max(s_scaling))
					s_scaling = s_scaling * np.linspace(int(min(s_scaling)), int(min(s_scaling)) + len(s), len(s))
					# where1 = coleval.find_nearest_index(s_scaling, 1E20)
					for index in range(101, len(s_scaling) - 1, 1):
						# if ( np.mean(s_scaling[index-100:index-10])>s_scaling[index] and np.mean(s_scaling[index+10:index+100])>s_scaling[index] ):
						if (np.mean(s_scaling[index - 100:index - 10]) > s_scaling[index] and np.mean(
								s_scaling[index + 10:index + 100]) > s_scaling[index]):
							initial_treshold_s = index + coleval.find_nearest_index(s_scaling[index - 100:index + 101], 0) - 100
							break

				residual_error_1=np.inf
				residual_error_2 = np.inf
				residual_error_3 = np.inf
				treshold_s = initial_treshold_s
				treshold_s_1 = len(power_on_voxels)
				treshold_s_2 = len(power_on_voxels)
				treshold_s_3 = len(power_on_voxels)
				tries_max = 100
				treshold_s_jump=50

				U_stored = cp.deepcopy(U)
				Vh_stored = cp.deepcopy(Vh)
				shape = np.shape(test_matrix)
				if first_treshold_s_search:
					actual_tries_max = int(1.1*len(s)/treshold_s_jump)
					first_treshold_s_search = False
				else:
					actual_tries_max = tries_max
				for iteration_index in range(actual_tries_max):

					zero = []
					for i in range(treshold_s, shape[0], 1):
						zero.append(i)
					if zero == []:
						zero.append(shape[0])
					sigma = np.zeros((zero[0], zero[0]))
					U = U_stored[:, :zero[0]]
					Vh = Vh_stored[:zero[0]]
					np.fill_diagonal(sigma[:treshold_s,:treshold_s],s[:treshold_s],wrap=False)
					# for i in range(zero[0]):
					# 	if s[i] <= s[treshold_s]:
					# 		sigma[i, i] = 0
					# 	else:
					# 		sigma[i, i] = s[i]
					a1 = np.dot(U, np.dot(sigma, Vh))
					inv_sigma = np.zeros((zero[0], zero[0]))
					for i in range(zero[0]):
						inv_sigma[i, i] = 1 / s[i]
					a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
					m = np.dot(a1_inv, np.dot(sensitivities.T, d))
					# for index, value in enumerate(m):
					# 	if value <= 0:
					# 		m[index] = 0

					if residuals_on_power_on_voxels:
						residual_error = np.sum(((power_on_voxels - m) ** 2) * weight2)
					else:
						residual_error = np.sum(((np.dot(sensitivities, m) - d) ** 2) * weight1)


					print('treshold_s,residual_error,treshold_s_1,residual_error_1,treshold_s_2,residual_error_2,treshold_s_3,residual_error_3')
					print(treshold_s,residual_error,treshold_s_1,residual_error_1,treshold_s_2,residual_error_2,treshold_s_3,residual_error_3)

					if iteration_index==0:
						treshold_s_3 = treshold_s_2
						treshold_s_2 = treshold_s_1
						treshold_s_1 = treshold_s
						residual_error_3 = residual_error_2
						residual_error_2 = residual_error_1
						residual_error_1 = residual_error
						treshold_s -= treshold_s_jump
					else:
						if residual_error<residual_error_1:
							if (treshold_s == treshold_s_2 and residual_error<residual_error_3 and residual_error_3!=np.inf):
								treshold = s[treshold_s]
								break
							residual_error_3 = residual_error_2
							residual_error_2 = residual_error_1
							residual_error_1 = residual_error
							if treshold_s<treshold_s_1:
								treshold_s_3 = treshold_s_2
								treshold_s_2 = treshold_s_1
								treshold_s_1 = treshold_s
								treshold_s-=treshold_s_jump
							elif treshold_s>treshold_s_1:
								treshold_s_3 = treshold_s_2
								treshold_s_2 = treshold_s_1
								treshold_s_1 = treshold_s
								treshold_s += treshold_s_jump

						else:
							residual_error_3 = residual_error_2
							residual_error_2 = residual_error_1
							residual_error_1 = residual_error
							if treshold_s < treshold_s_1:
								treshold_s_3 = treshold_s_2
								treshold_s_2 = treshold_s_1
								treshold_s_1 = treshold_s
								treshold_s += treshold_s_jump
							elif treshold_s > treshold_s_1:
								treshold_s_3 = treshold_s_2
								treshold_s_2 = treshold_s_1
								treshold_s_1 = treshold_s
								treshold_s -= treshold_s_jump
						if treshold_s<=0:
							treshold_s = 50
							treshold = s[treshold_s]
							break
						if (treshold_s>(len(s)-1)):
							treshold_s = len(s)-1
							treshold = s[treshold_s]
							break

				initial_treshold_s=treshold_s


			treshold_record.append(treshold)
			'''
			# alpha_record.append(alpha)

			print('alpha='+str(alpha))
			# print('treshold='+str(treshold))
			# print('zero='+str(zero[0]))
			if False:
				score=np.sum(((power_on_voxels-m)**2)*weight)
				print('check=' + str(score))
				check.append(score)

			if residuals_on_power_on_voxels:
				score_x = np.sum(((power_on_voxels - m) ** 2) * weight2)
				print('||input m - estimated m||2=' + str(score_x))
			else:
				score_x = np.sum(((np.dot(sensitivities, m) - d) ** 2) * weight1)
				print('||Gm-d||2=' + str(score_x))
			score_y = np.sum(((np.dot(laplacian,m)) ** 2) * weight2)
			print('||Laplacian(m)||2=' + str(score_y))
			# score_x_record.append(score_x)
			# score_y_record.append(score_y)


			# print('conditioning=' + str(np.linalg.cond(test_matrix)))
			if (spatial_averaging==1 and time_averaging==1 and with_noise==False):
				plt.figure()
				plt.figure(alpha)
				plt.title('Eigenvalues for foil averaged on '+str(spatial_averaging**2)+' pixels, alpha=' + str(alpha))
				plt.plot(s, 'o')
				plt.plot(s, label='eigenvalues')
				'''
				if not treshold_method_try_to_search:
					if eigenvalue_cut_vertical:
						if flag_vertical_treshold_found:
							plt.plot([find_maximum,find_maximum],[min(s), max(s)], label='treshold')
						else:
							plt.plot([0,len(s)-1],[treshold, treshold],label='treshold_estimated')
					else:
						plt.plot([0, len(s) - 1], [treshold, treshold], label='treshold')
					if flag_for_extrapolated_treshold:
						if not eigenvalue_cut_vertical:
							plt.plot([0, len(s) - 1], [treshold_not_estimated, treshold_not_estimated],'--', label='not extrapolated treshold')
	
					if (alpha > 10**(-14.001) and ( not flag_for_extrapolated_treshold)):
						start = coleval.find_nearest_index(alpha_record, 1E-20)
						coeff_fitting, trash = curve_fit(line, np.log(alpha_record[start:]), np.log(treshold_horizontal_record[start:]), p0=[1, 1],maxfev=100000000)
						flag_for_extrapolated_treshold = True
				else:
					plt.plot([treshold_s], [treshold],'x', label='treshold')
				'''
				plt.yscale('log')
				plt.xlabel('i')
				plt.ylabel('eigenvalues')
				plt.grid()
				plt.legend(loc='best')
				# print(path_for_plots + '/eigenvalues_for_alpha_'+ str('%.3g' % alpha) +'.eps')
				try:
					plt.savefig(path_for_plots + '/eigenvalues_for_alpha_'+ str('%.3g' % alpha) +'.eps' )
				except:
					print("for some reason it didn't want to save "+path_for_plots + '/eigenvalues_for_alpha_'+ str('%.3g' % alpha) +'.eps')
				plt.close()




			output = calc_stuff_output(exp, alpha, score_x, score_y)

			return output


		pool = Pool(number_cpu_available)
		# pool = Pool(1)

		composed_array = [*pool.map(calc_stuff, enumerate(alpha_exponents_to_test))]
		# composed_array = set(composed_array)
		print('np.shape(composed_array)' + str(np.shape(composed_array)))

		pool.close()
		pool.join()

		composed_array = list(composed_array)

		'''temp = []
		for i in range(len(composed_array)):
			if (not np.isnan(composed_array[i].score_x)):
				temp.append(composed_array[i])
		composed_array = np.array(temp)

		exp_indexes = []
		for i in range(len(composed_array)):
			exp_indexes.append(composed_array[i].exp)
		composed_array = np.array([to_be_ordered for _, to_be_ordered in sorted(zip(exp_indexes, composed_array))])

		temp = []
		for i in range(len(composed_array)):
			temp.append(composed_array[i].alpha)
		alpha_record = np.array(temp)
		temp = []
		for i in range(len(composed_array)):
			temp.append(composed_array[i].score_x)
		score_x_record = np.array(temp)
		temp = []
		for i in range(len(composed_array)):
			temp.append(composed_array[i].score_y)
		score_y_record = np.array(temp)'''

		exp_indexes = []
		alpha_record = []
		score_x_record = []
		score_y_record = []
		for i in range(len(composed_array)):
			if (not np.isnan(composed_array[i].score_x)):
				exp_indexes.append(composed_array[i].exp)
				alpha_record.append(composed_array[i].alpha)
				score_x_record.append(composed_array[i].score_x)
				score_y_record.append(composed_array[i].score_y)
		exp_indexes = np.array(exp_indexes)
		alpha_record = np.array([alpha_record for _, alpha_record in sorted(zip(exp_indexes, alpha_record))])
		score_x_record = np.array([score_x_record for _, score_x_record in sorted(zip(exp_indexes, score_x_record))])
		score_y_record = np.array([score_y_record for _, score_y_record in sorted(zip(exp_indexes, score_y_record))])

		if residuals_on_power_on_voxels or (not with_noise):
			# index_worst_fit=np.array(score_x_record).argmax()
			index_best_fit = np.array(score_x_record).argmin()
			if np.sum(np.diff(score_y_record)>0)!=0:
				print('check here')
				print((np.diff(score_y_record)>0)*(np.linspace(1,len(score_x_record),len(score_x_record)-1).astype('int')))
				index_worst_fit = np.max((np.diff(score_y_record)>0)*(np.linspace(1,len(score_x_record),len(score_x_record)-1).astype('int')))
				print(index_worst_fit)
				index_best_fit=np.array(score_x_record[index_worst_fit:]).argmin() + index_worst_fit

			# if np.sum(np.logical_and(score_x_record>score_x_record[-1],alpha_record<1000*alpha_record[index_best_fit]))!=0:
			# 	print('check here')
			# 	print((np.logical_and(score_x_record>score_x_record[-1],alpha_record<1000*alpha_record[index_best_fit]))*(np.linspace(0,len(score_x_record)-1,len(score_x_record)).astype('int')))
			# 	index_worst_fit = np.max((np.logical_and(score_x_record>score_x_record[-1],alpha_record<1000*alpha_record[index_best_fit]))*(np.linspace(0,len(score_x_record)-1,len(score_x_record)).astype('int')))
			# 	if index_worst_fit<index_best_fit:
			# 		index_best_fit=np.array(score_x_record[index_worst_fit:]).argmin() + index_worst_fit
			best_alpha = alpha_record[index_best_fit]
		else:
			# score_y_record_function = interp1d(alpha_record,score_y_record,fill_value='extrapolate')
			# score_y_record_derivative = misc.derivative(score_y_record_function,alpha_record)

			# Lcurve_curvature = ( ((alpha_record**2) * score_y_record * (score_x_record**2)) + 2*(alpha_record * (score_y_record**2) * (score_x_record**2) / score_y_record_derivative) + ((alpha_record**4) * (score_y_record**2) * score_x_record) ) / (( ((alpha_record**4) * (score_y_record**2)) + (score_x_record**2) )**(3/2))
			# plt.figure()
			# plt.plot(alpha_record, Lcurve_curvature)
			# plt.plot(alpha_record, Lcurve_curvature, 'x')

			print('Alpha that made possible SVG decomposition ')
			print(str(alpha_record.tolist()))
			print('record of the residuals on solution ')
			print(str(score_x_record.tolist()))
			print('record of the residuals on smoothing ')
			print(str(score_y_record.tolist()))

			score_y_record_derivative = (score_y_record[2:] - score_y_record[:-2]) / (alpha_record[2:] - alpha_record[:-2])

			# score_y_record_function = interp1d(alpha_record[1:-1],score_y_record_derivative,fill_value='extrapolate')
			# score_y_record_derivative = misc.derivative(score_y_record_function,alpha_record)

			# score_y_record_function = splrep(alpha_record,score_y_record,k=1,s=0)
			# score_y_record_derivative = splev(alpha_record,score_y_record_function,der=1)[1:-1]

			# score_y_record_derivative = coleval.log_log_fit_derivative(alpha_record,score_y_record)[1:-1]

			Lcurve_curvature = -(((alpha_record[1:-1] ** 2) * score_y_record[1:-1] * (score_x_record[1:-1] ** 2)) + 2 * (alpha_record[1:-1] * (score_y_record[1:-1] ** 2) * (score_x_record[1:-1] ** 2) / score_y_record_derivative) + ((alpha_record[1:-1] ** 4) * (score_y_record[1:-1] ** 2) * score_x_record[1:-1])) / ((((alpha_record[1:-1] ** 4) * (score_y_record[1:-1] ** 2)) + (score_x_record[1:-1] ** 2)) ** (3 / 2))

			'''
			score_y_record_hat = np.log(score_y_record)
			score_x_record_hat = np.log(score_x_record)
			score_y_record_hat_derivative = (score_y_record_hat[2:] - score_y_record_hat[:-2]) / (alpha_record[2:] - alpha_record[:-2])
			score_x_record_hat_derivative = (score_x_record_hat[2:] - score_x_record_hat[:-2]) / (alpha_record[2:] - alpha_record[:-2])
			score_y_record_hat_derivative_2 = (score_y_record_hat[2:] - 2 * score_y_record_hat[1:-1] + score_y_record_hat[:-2]) / ((alpha_record[2:] - alpha_record[:-2]) ** 2)
			score_x_record_hat_derivative_2 = (score_x_record_hat[2:] - 2 * score_x_record_hat[1:-1] + score_x_record_hat[:-2]) / ((alpha_record[2:] - alpha_record[:-2]) ** 2)
			Lcurve_curvature_hat = (score_y_record_hat_derivative * score_x_record_hat_derivative_2 - score_y_record_hat_derivative_2 * score_x_record_hat_derivative) / (((score_y_record_hat_derivative ** 2) + (score_x_record_hat_derivative ** 2)) ** (3 / 2))
			'''

			plt.figure()
			# plt.plot(alpha_record[1:-1], Lcurve_curvature)
			# plt.plot(alpha_record[1:-1], Lcurve_curvature, 'x')



			if True:	# made True 16/12/3019 because it is better to analyse solps max seed case
				fine_alpha = np.logspace(np.log10((alpha_record[:-2][Lcurve_curvature > 0]).min()),np.log10((alpha_record[:-2][Lcurve_curvature > 0]).max()), 1000,endpoint=True)
				# Lcurve_curvature_interp = interp1d(alpha_record, Lcurve_curvature,fill_value='extrapolate', kind = 'cubic')
				# Lcurve_curvature_interp = splrep(np.log10(alpha_record[Lcurve_curvature>0]), np.log10(Lcurve_curvature[Lcurve_curvature>0]), k=3)
				Lcurve_curvature_interp = splrep(np.log10(alpha_record[:-2][Lcurve_curvature > 0]),np.log10(Lcurve_curvature[Lcurve_curvature > 0]), k=3)
				Lcurve_curvature_interp = splev(np.log10(fine_alpha), Lcurve_curvature_interp)
				plt.plot(fine_alpha, 10 ** Lcurve_curvature_interp, '--')
				plt.plot(alpha_record[:-2], Lcurve_curvature, 'x')
				plt.plot(alpha_record[:-2], -Lcurve_curvature, 'y.')
				plt.yscale('log')
				plt.xscale('log')

				index_best_fit = np.array(Lcurve_curvature).argmax()

				# 21/10/2019 lines added to deal with an anbiguous detection of the laximum curvature
				temp = cp.deepcopy(Lcurve_curvature)
				temp[temp<0]=0
				peaks = scipy.signal.find_peaks(temp)[0]
				proms = scipy.signal.peak_prominences(temp, peaks)[0]
				peaks = np.array([peaks for _, peaks in sorted(zip(proms, peaks))])
				proms = np.log10(np.sort(proms))
				if ((proms[-2] - proms[-1]) / proms[-1] < 0.2 and peaks[-2] < peaks[-1]):
					plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit ], 'r+', markersize=9)
					index_best_fit = peaks[-2]
					plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit ], 'gx')
					most_lickely_alpha = fine_alpha[np.logical_and(fine_alpha < alpha_record[index_best_fit + 1],fine_alpha > alpha_record[index_best_fit -1])][(10**Lcurve_curvature_interp[np.logical_and(fine_alpha<alpha_record[index_best_fit+1],fine_alpha>alpha_record[index_best_fit-1])]).argmax()]
				else:
					plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit ], 'gx')
					most_lickely_alpha = fine_alpha[(10**Lcurve_curvature_interp).argmax()]
				# plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit-1], 'gx')

			else:
				temp = cp.deepcopy(Lcurve_curvature)
				minimum_alpha_index = np.argmax(np.logical_and(alpha_record[1:-1]<10**-9,np.logical_and(temp<10**-3,temp!=0))*np.linspace(1,len(temp),len(temp)))

				fine_alpha = np.logspace(np.log10((alpha_record[1:-1][np.logical_and(Lcurve_curvature > 0,alpha_record[1:-1]>alpha_record[minimum_alpha_index])]).min()),np.log10((alpha_record[1:-1][np.logical_and(Lcurve_curvature > 0,alpha_record[1:-1]>alpha_record[minimum_alpha_index])]).max()), 1000,endpoint=True)
				# Lcurve_curvature_interp = interp1d(alpha_record, Lcurve_curvature,fill_value='extrapolate', kind = 'cubic')
				# Lcurve_curvature_interp = splrep(np.log10(alpha_record[Lcurve_curvature>0]), np.log10(Lcurve_curvature[Lcurve_curvature>0]), k=3)
				Lcurve_curvature_interp = splrep(np.log(alpha_record[1:-1][np.logical_and(Lcurve_curvature > 0,alpha_record[1:-1]>alpha_record[minimum_alpha_index])]),np.log(Lcurve_curvature[np.logical_and(Lcurve_curvature > 0,alpha_record[1:-1]>alpha_record[minimum_alpha_index])]), k=3)
				Lcurve_curvature_interp = splev(np.log(fine_alpha), Lcurve_curvature_interp)
				plt.plot(fine_alpha, np.exp(Lcurve_curvature_interp), '--')
				plt.plot(alpha_record[1:-1], Lcurve_curvature, 'x')
				plt.plot(alpha_record[1:-1], -Lcurve_curvature, 'y.')
				plt.yscale('log')
				plt.xscale('log')


				temp[temp<0]=10**-6
				temp = np.log(temp[minimum_alpha_index:])
				temp = temp-np.min(temp)
				peaks = scipy.signal.find_peaks(temp)[0]
				proms = scipy.signal.peak_prominences(temp, peaks)[0]
				peaks = np.array([peaks for _, peaks in sorted(zip(proms, peaks))])
				proms = np.sort(proms)
				if False:
					try:
						index_best_fit = np.min(peaks[proms>1]) + minimum_alpha_index+1
					except:
						print('lowered the requirements for finding the peak of the curve')
						index_best_fit = peaks[-1] + minimum_alpha_index+1
				else:
					index_best_fit = peaks[(np.argmax(Lcurve_curvature[(np.array(peaks) + minimum_alpha_index).astype('int')])).astype('int')-1] + minimum_alpha_index+1
					plt.plot(alpha_record[(np.array(peaks) + minimum_alpha_index + 1).astype('int')],Lcurve_curvature[(np.array(peaks) + minimum_alpha_index-1).astype('int')],'cv')
				plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit - 1], 'gx')
				most_lickely_alpha = fine_alpha[np.logical_and(fine_alpha < alpha_record[index_best_fit + 1],fine_alpha > alpha_record[index_best_fit - 1])][(10 ** Lcurve_curvature_interp[np.logical_and(fine_alpha < alpha_record[index_best_fit + 1],fine_alpha > alpha_record[index_best_fit - 1])]).argmax()]

			best_alpha = alpha_record[index_best_fit]
			plt.title('Best tested regularization coefficient is alpha=' + '%.3g' % best_alpha +'\neven if the best is likely ~'+ '%.3g' % most_lickely_alpha )
			plt.xlabel('Regularisation parameter')
			plt.ylabel('L-curve curvature')
			plt.grid()
			plt.savefig(path_for_plots + '/L_curve_curvature.eps')
			plt.close()

		plt.figure()
		plt.plot(score_x_record, score_y_record)
		plt.plot(score_x_record,score_y_record, 'x')
		plt.plot(score_x_record[0],score_y_record[0], 'kx', label = 'reg param '+ '%.3g' % alpha_record[0])
		plt.plot(score_x_record[-1],score_y_record[-1], 'rx', label = 'reg param '+ '%.3g' % alpha_record[-1])
		plt.plot(score_x_record[index_best_fit], score_y_record[index_best_fit], 'gx', label='reg param ' + '%.3g' % alpha_record[index_best_fit])
		plt.legend(loc = 'best')
		plt.yscale('log')
		plt.xscale('log')
		if residuals_on_power_on_voxels:
			plt.title('Best regularization coefficient is alpha=' + str(best_alpha) + '\nwith residuals ||input m - estimated m||2=' + str(score_x_record[index_best_fit]))
			plt.xlabel('||input m - estimated m||2')
		else:
			plt.title('Best regularization coefficient is alpha=' + str(best_alpha) + '\nwith residuals ||Gm-d||2=' + str(score_x_record[index_best_fit]))
			plt.xlabel('||Gm-d||2')
		plt.ylabel('||Laplacian(m)||2')
		plt.grid()
		plt.savefig(path_for_plots + '/L_curve.eps')
		plt.close()


		# if spatial_averaging >1:
		#
		# 	binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) )
		# 	binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) )
		# 	dh = 0.07 / pixel_h
		# 	dv = 0.09 / pixel_v
		# 	pixels_centre_location = np.array([(np.ones((pixel_v, pixel_h)) * binning_h*dh+dh/2).flatten(),((np.ones((pixel_v, pixel_h)).T * binning_v*dv+dv/2).T).flatten()])
		#
		# 	pixel_h_new = np.floor(pixel_h / spatial_averaging)
		# 	pixel_v_new = np.floor(pixel_v / spatial_averaging)
		# 	binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging)
		# 	binning_h[binning_h>pixel_h_new-1] = pixel_h_new-1
		# 	binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging)
		# 	binning_v[binning_v>pixel_v_new-1] = pixel_v_new-1
		# 	pixels_to_bin = ((np.ones((pixel_v,pixel_h))*binning_h).T+binning_v*pixel_h_new).T.astype('int').flatten()
		# 	bin_centre_location = np.zeros((len(pixels_to_bin),2))
		# 	dh = 0.07 / pixel_h
		# 	dv = 0.09 / pixel_v
		#
		# 	for bin_index in range(np.max(pixels_to_bin)+1):	# coordinates as (horizontal,vertical)
		# 		bin_centre_location[bin_index] = np.mean(pixels_centre_location[:,pixels_to_bin==bin_index] ,axis=-1)
		#
		#
		# 	temp = np.zeros((np.max(pixels_to_bin)+1,np.shape(sensitivities)[-1]))
		# 	for bin_index in range(np.max(pixels_to_bin)+1):
		# 		temp[bin_index] = np.sum(sensitivities[pixels_to_bin==bin_index],axis=0)
		# 	sensitivities_not_binned = cp.deepcopy(sensitivities)
		# 	sensitivities = cp.deepcopy(temp)
		#
		# 	temp = np.zeros((np.max(pixels_to_bin)+1))
		# 	for bin_index in range(np.max(pixels_to_bin)+1):
		# 		temp[bin_index] = np.sum(d[pixels_to_bin==bin_index])
		# 	d_not_binned = cp.deepcopy(d)
		# 	d = cp.deepcopy(temp)
		#
		# 	temp = np.zeros_like((np.max(pixels_to_bin)+1))
		# 	for bin_index in range(np.max(pixels_to_bin)+1):
		# 		temp[bin_index] = np.mean(weight1[pixels_to_bin==bin_index])
		# 	weight1_not_binned = cp.deepcopy(weight1)
		# 	weight1 = cp.deepcopy(temp)



		# print('alpha')
		# print(alpha_record)
		# print('check')
		# print(check)
		if False:
			alpha=alpha_record[coleval.find_nearest_index(check,0)[0]]
		elif False:
			alpha=0
		else:
			alpha = alpha_record[index_best_fit]
			# treshold = treshold_record[index_best_fit]

		print('Best alpha='+str(alpha))
		# test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
		test_matrix = np.dot(sensitivities.T, sensitivities) + (alpha ** 2) * np.dot(laplacian.T, laplacian)
		# print(np.dot(sensitivities.T, sensitivities))

		if False:
			# check that SVD decomposition is consistent
			for i in range(10):
				U, s, Vh = np.linalg.svd(test_matrix)
				print('The next number should not change')
				print(s[0])
		elif False:
			U, s, Vh = np.linalg.svd(test_matrix)

		try:
			a1_inv = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod + '.npz')['a1_inv']
			s = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(spatial_averaging) + '_alpha_' + str(alpha) + mod + '.npz')['s']
			print('the SVG decomposition of alpha = ' + str(alpha) + ' is imported')
			sigma = np.diag(s)
			inv_sigma = np.diag(1 / s)
		except:
			print('attempt to import SVG decomposition failed\nthis should not have happened')
			U, s, Vh = np.linalg.svd(test_matrix)
			sigma = np.diag(s)
			inv_sigma = np.diag(1 / s)
			a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))


		print(s)

		shape = np.shape(test_matrix)

		# sigma = np.diag(s)
		# inv_sigma = np.diag(1 / s)

		# a1 = np.dot(U, np.dot(sigma, Vh))
		# a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))


		# zero = []
		# for i in range(shape[0]):
		# 	if s[i] <= treshold:
		# 		zero.append(i)
		# if zero == []:
		# 	zero.append(shape[0])
		# print(zero)
		# sigma = np.zeros((zero[0], zero[0]))
		# U = U[:, :zero[0]]
		# Vh = Vh[:zero[0]]
		# for i in range(zero[0]):
		# 	if s[i] <= treshold:
		# 		sigma[i, i] = 0
		# 	else:
		# 		sigma[i, i] = s[i]
		# a1 = np.dot(U, np.dot(sigma, Vh))
		# print(np.allclose(test_matrix, a1))
		#
		# inv_sigma = np.zeros((zero[0], zero[0]))
		# for i in range(zero[0]):
		# 	inv_sigma[i, i] = 1 / s[i]
		# a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

		# scipy.sparse.save_npz(path_for_plots + '/inverse_sensitivity_alpha' + str(alpha) + '.npz',scipy.sparse.csr_matrix(a1_inv))

		# print(np.trace(np.dot(a1,a1_inv)))
		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).max())
		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).min())

		to_print =  cp.deepcopy(power_on_voxels)
		maximum_original = max(power_on_voxels)
		minimum_original = min([x for x in power_on_voxels if x != 0])
		if (int(maximum_original)==int(minimum_original)):
			to_print[-1] = 1
		core_voxel_grid.plot(voxel_values=to_print, colorbar=['rainbow', 'Emissivity [W/m3]', 'log'])
		# print(path_sensitivity+'/gnappo.eps')
		plt.title('Input emission prifle')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.savefig(path_for_plots + '/input_emission_profile.eps')
		plt.close()

		if not is_this_extra:
			if flag_mask2_present:
				d_for_the_plot = []
				pixels_added = 0
				for index,masked in enumerate(mask2_on_camera_pixels):
					index_relative = index - pixels_added
					if not masked:
						d_for_the_plot.append(d[index_relative])
					else:
						d_for_the_plot.append(0)
						pixels_added+=1
				if with_noise:
					plt.figure()
					# fig, ax = plt.subplots()
					cmap = plt.cm.rainbow
					cmap.set_under(color='white')
					# vmin = 0.0000000000000000000000001
					# plt.imshow(coleval.split_fixed_length(d_for_the_plot, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
					plt.imshow(coleval.split_fixed_length(d_for_the_plot, pixel_h), origin='lower', cmap=cmap)
					plt.title('Power density on the foil via sensitivity matrix\nnoise std of ' + str(noise_on_power))
					# plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
				else:
					plt.figure()
					# fig, ax = plt.subplots()
					cmap = plt.cm.rainbow
					cmap.set_under(color='white')
					# vmin = 0.0000000000000000000000001
					# plt.imshow(coleval.split_fixed_length(d_for_the_plot, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
					plt.imshow(coleval.split_fixed_length(d_for_the_plot, pixel_h), origin='lower', cmap=cmap)
					plt.title('Power density on the foil via sensitivity matrix')
					# plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
			else:
				if with_noise:
					plt.figure()
					# fig, ax = plt.subplots()
					cmap = plt.cm.rainbow
					cmap.set_under(color='white')
					# vmin = 0.0000000000000000000000001
					# plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
					# print('np.shape(coleval.split_fixed_length(d, pixel_h))')
					# print(np.shape(coleval.split_fixed_length(d, pixel_h)))
					# print(coleval.split_fixed_length(d, pixel_h))
					plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap)
					plt.title('Power density on the foil via sensitivity matrix\nnoise std of ' + str(noise_on_power))
					# plt.colorbar().set_label('Power density on the foil [W/m^2]')#, cut-off ' + str(vmin) + 'W/m^2')
				else:
					plt.figure()
					# fig, ax = plt.subplots()
					cmap = plt.cm.rainbow
					cmap.set_under(color='white')
					# vmin = 0.0000000000000000000000001
					# plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
					plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap)
					plt.title('Power density on the foil via sensitivity matrix')
					# plt.colorbar().set_label('Power density on the foil [W/m^2])')#, cut-off ' + str(vmin) + 'W/m^2')
		else:
			if with_noise:
				plt.figure()
				# fig, ax = plt.subplots()
				cmap = plt.cm.rainbow
				cmap.set_under(color='white')
				# vmin = 0.0000000000000000000000001
				# plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
				plt.imshow(coleval.split_fixed_length(d_high_res, foil_resolution_max), origin='lower', cmap=cmap)
				plt.title('Power density on the foil via sensitivity matrix\nnoise std of ' + str(noise_on_power))
				# plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
			else:
				plt.figure()
				# fig, ax = plt.subplots()
				cmap = plt.cm.rainbow
				cmap.set_under(color='white')
				# vmin = 0.0000000000000000000000001
				# plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
				plt.imshow(coleval.split_fixed_length(d_high_res, foil_resolution_max), origin='lower', cmap=cmap)
				plt.title('Power density on the foil via sensitivity matrix')
				# plt.colorbar().set_label('Power density on the foil [W/m^2], cut-off ' + str(vmin) + 'W/m^2')
			if flag_mask2_present:
				fig.subplots_adjust(top=0.85)
				fig.suptitle('Beware this does not show the foil masked as it is')
		plt.colorbar().set_label('Power density on the foil [W/m^2]')
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		plt.savefig(path_for_plots + '/power_distribution_on_foil.eps')
		plt.close()

		# def l_curve_sample(x,*params):
		# 	import numpy as np
		# 	treshold=params[0]
		# 	max=1
		# 	if len(x)==():
		# 		if x>treshold:
		# 			return max
		# 		else:
		# 			return treshold
		# 	else:
		# 		out=[]
		# 		for i in range(len(x)):
		# 			if x[i] > treshold:
		# 				out.append(max)
		# 			else:
		# 				out.append(treshold)
		# 		return np.array(out)
		#
		# guess=np.array([1E-10])
		# curve_fit(l_curve_sample, np.linspace(0,len(s)-1,len(s)), s, p0=guess, maxfev=100000000)

		m = np.dot(a1_inv, np.dot(sensitivities.T, d))
		np.save(path_for_plots + '/inverted_emissivity',m)
		if (spatial_averaging==1 and time_averaging==1 and with_noise==False):
			np.save(path_for_plots + '/input_emissivity', power_on_voxels)
		# scipy.sparse.save_npz(path_for_plots + '/inverse_sensitivity_alpha' + str(alpha) + '.npz',scipy.sparse.csr_matrix(m))

		# for index, value in enumerate(m):
		# 	if value <= 0:
		# 		m[index] = 0
		print('np.allclose(power_on_voxels, m) = ' + str(np.allclose(power_on_voxels, m)))
		if False:
			score = np.sum(((power_on_voxels - m) ** 2) * weight)
			print('check=' + str(score))

		if residuals_on_power_on_voxels:
			score_x = np.sum(((power_on_voxels - m) ** 2) * weight2)
			print('||input m - estimated m||2=' + str(score_x))
		else:
			score_x = np.sum(((np.dot(sensitivities, m) - d) ** 2) * weight1)
			print('||Gm-d||2=' + str(score_x))
		score_y = np.sum(((np.dot(laplacian, m)) ** 2) * weight2)
		print('||Laplacian(m)||2=' + str(score_y))

		# print('m')
		# print(m)
		# print('power_on_voxels')
		# print(power_on_voxels)

		plt.close()
		plt.figure()
		plt.title('Input vs estimated emission profile ||Gm-d||2=' + '%.3g' % score_x + '||Laplacian(m)||2=' + '%.3g' % score_y)
		plt.plot(m, label='estimation')
		plt.plot(power_on_voxels, label='input')
		plt.legend(loc='best')
		plt.savefig(path_for_plots + '/emission_profile_compare.eps')
		plt.close()

		plt.figure()
		cmap = plt.cm.rainbow
		cmap.set_under(color='white')
		# vmin = 0.0000000000000000000000001
		# plt.imshow(coleval.split_fixed_length(d, pixel_h), origin='lower', cmap=cmap, vmin=vmin)
		plt.imshow(coleval.split_fixed_length(np.dot(sensitivities, m), pixel_h), origin='lower', cmap=cmap)
		plt.title('Power density on the foil using the estimated solution')
		plt.colorbar().set_label('Power density on the foil [W/m^2]')
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		plt.savefig(path_for_plots + '/inverted_power_distribution_on_foil.eps')
		plt.close()


		maximum_original = max(power_on_voxels)
		minimum_original = min([x for x in power_on_voxels if x != 0])
		if (int(maximum_original)==int(minimum_original)):
			minimum_original = 1
		to_print = np.zeros(np.shape(m))
		maximum_new_record = 0
		for index, value in enumerate(m):
			if (value >= minimum_original and value <= maximum_original):
				to_print[index] = value
			elif (value > maximum_original):
				to_print[index] = maximum_original
				if value > maximum_new_record:
					maximum_new_record = value
			elif (value < minimum_original and value != 0):
				to_print[index] = minimum_original
		to_print[-1] = maximum_original
		core_voxel_grid.plot(voxel_values=to_print, colorbar=['rainbow', 'Emissivity [W/m3]', 'log'])
		if maximum_new_record > 0:
			plt.title('Estimated emissivity\nmaximum of the inverted emissivity = ' + str(
				maximum_new_record) + '\n instead of ' + str(maximum_original))
		else:
			plt.title('Estimated emissivity\n')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.savefig(path_for_plots + '/estimated_emission.eps')
		plt.close()

		core_voxel_grid.plot(voxel_values=m, colorbar=['rainbow', 'Emissivity [W/m3]', 'log'])
		plt.title('Estimated emissivity\n')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.savefig(path_for_plots + '/estimated_emission_not_limited.eps')
		plt.close()

		difference = m - power_on_voxels
		core_voxel_grid.plot(voxel_values=difference, colorbar=['rainbow', 'Emissivity [W/m3]'])
		plt.title('Difference of emission profile')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.savefig(path_for_plots + '/emission_difference.eps')
		plt.close()

		# min_greater_zero = min([x for x in m if x > 0])
		# max_below_zero = max([x for x in m if x < 0])
		# reference_min=min(min_greater_zero,-max_below_zero)
		reference_min = 1
		max_greater_zero = max([x for x in m if x > 0])
		# min_below_zero = min([x for x in m if x < 0])
		# reference_max = max(max_greater_zero, -min_below_zero)
		reference_max = max_greater_zero
		to_print = np.zeros(np.shape(difference))
		for index, value in enumerate(difference):
			if value > reference_min:
				to_print[index] = np.log(value / reference_min)
			elif value < -reference_min:
				to_print[index] = -np.log(-value / reference_min)
		core_voxel_grid.plot(voxel_values=to_print, colorbar=['rainbow',
															  'Logaritm of the difference with sign scaled on ' + str(
																  reference_min) + 'W/m3'])
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.savefig(path_for_plots + '/logaritmic_estimated_difference.eps')
		plt.close()

		compare = np.zeros(np.shape(difference))
		for index, value in enumerate(difference):
			if power_on_voxels[index] > 0:
				compare[index] = value / power_on_voxels[index]
		if max(compare) > 0:
			core_voxel_grid.plot(voxel_values=compare, colorbar=['rainbow', 'Relative emissivity','log'])
			plt.title('Logaritmic relative difference of emission profile')
			plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			plt.savefig(path_for_plots + '/relative_estimated_difference.eps')
			plt.close()

		# Here I camculate the total radiated power of the inversion

		total_input_radiated_power = 0
		total_estimated_radiated_power = 0
		for index in range(num_voxels):
			p1, p2, p3, p4 = core_voxel_grid._voxels[index].vertices
			r_maj = max([p1.x, p2.x, p3.x, p4.x])
			r_min = min([p1.x, p2.x, p3.x, p4.x])
			thickness = max([p1.y, p2.y, p3.y, p4.y]) - min([p1.y, p2.y, p3.y, p4.y])
			voxel_centre = Point2D((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4)
			volume = np.pi * thickness * (r_maj ** 2 - r_min ** 2)
			total_estimated_radiated_power += m[index] * volume
			total_input_radiated_power += power_on_voxels[index] * volume

		# I do this bit to evaluate what is the standard deviation on the power on the voxels with noise

		if with_noise:
			number_of_tries = 1000
			m_all = []
			for i in range(number_of_tries):
				if spatial_averaging==1:
					d = d_original + np.random.normal(0, noise_on_power, len(d_original))
				else:
					d = np.zeros((np.max(pixels_to_bin) + 1))
					temp = d_original_not_binned + np.random.normal(0, noise_on_power, len(d_original_not_binned))
					for bin_index in range(np.max(pixels_to_bin) + 1):
						d[bin_index] = np.sum(temp[pixels_to_bin == bin_index])
				# for index, value in enumerate(d):
				# 	if value <= 0:
				# 		d[index] = 0
				m = np.dot(a1_inv, np.dot(sensitivities.T, d))
				# for index, value in enumerate(m):
				# 	if value <= 0:
				# 		m[index] = 0
				m_all.append(m)
			m_all = np.array(m_all)
			m_std = np.std(m_all, axis=(0))

			core_voxel_grid.plot(voxel_values=m_std, colorbar=['rainbow', 'Emissivity std [W/m3]'])
			plt.title('std of the inverted emissivity\nwith noise on foil power of ' + str(noise_on_power) + '\n mean of the voxel power std of ' + str(np.mean(m_std)))
			plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
			plt.savefig(path_for_plots + '/noise.eps')
			plt.close()



		# I need to print stats to know what was going on

		header = ['# Record of what was found in this simulation of inversion']
		to_write = [['grid resolution in cm', str(grid_resolution)],
					['foil horizontal resolution in pixels', str(pixel_h)],
					['foil vertical resolution in pixels', str(pixel_v)],
					['shape of sensitivity matrix ', str(np.shape(sensitivities))],
					['with noise, std of noise on power, time averaging ',
					 str([with_noise, noise_on_power, time_averaging])],
					['extra informations in the sensitivity matrix ', str(is_this_extra)],
					['Alpha that made possible SVG decomposition ', str(alpha_record.tolist())],
					# ['record of the values of the treshold on eigenvalues ', str(treshold_record)],
					['record of the residuals on solution ', str(score_x_record.tolist())],
					['record of the residuals on smoothing ', str(score_y_record.tolist())],
					# ['relative to the best alpha: value, eigenvalue threshold, residuals on solution, residuals on smoothing',
					# 	str([alpha, treshold, score_x_record[index_best_fit], score_y_record[index_best_fit]])],
					['relative to the best alpha: value, residuals on solution, residuals on smoothing',
						str([alpha, score_x_record[index_best_fit], score_y_record[index_best_fit]])],

					['total input radiated power', str(total_input_radiated_power)],
					['total estimated radiated power', str(total_estimated_radiated_power)],
					['fraction of the total power detected',str(total_estimated_radiated_power / total_input_radiated_power)]
					]
		# to_write='1, 1, '+str(foil_fake_corner1)[8:-1]+', '+str(foil_fake_corner2)[8:-1]+', '+str(foil_fake_corner3)[8:-1]+', '+str(foil_fake_corner4)[8:-1]

		with open(path_for_plots + '/stats.csv', mode='w') as f:
			writer = csv.writer(f)
			writer.writerow(header)
			for row in to_write:
				writer.writerow(row)
		f.close()




print('J O B   D O N E !')


#
# foil_resolution_all = [187, 93, 62, 47, 37, 31, 26, 19]
#
# import concurrent.futures as cf
# with cf.ProcessPoolExecutor() as executor:
# 	executor.map(_run_, foil_resolution_all)

# exit()
#
# # I skip the rest because less relevant
#
#
#
#
# plt.plot(alpha_record,check,'o')
# plt.plot(alpha_record,check)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('check')
# plt.savefig(path_sensitivity+'/gnappo5.eps')
# plt.close()
#
#
#
#
#
# if False:
# 	# This bit is to check different tresholds
#
# 	# print('alpha')
# 	# print(alpha_record)
# 	# print('check')
# 	# print(check)
# 	alpha=0
# 	test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
# 	# print(np.dot(sensitivities.T, sensitivities))
#
# 	shape=np.shape(test_matrix)
# 	U, s, Vh = np.linalg.svd(test_matrix)
#
# 	for exp in np.linspace(-19,-11,num=20):
# 		treshold=1*10**(exp)
# 		print('treshold ' + str(treshold))
# 		zero=[]
# 		for i in range(shape[0]):
# 			if s[i]<treshold	:
# 				zero.append(i)
# 		if zero==[]:
# 			zero.append(shape[0])
# 		print('zero' + str(zero[0]))
# 		sigma = np.zeros((zero[0],zero[0]))
# 		U=U[:,:zero[0]]
# 		Vh=Vh[:zero[0]]
# 		for i in range(zero[0]):
# 			if s[i]<treshold:
# 				sigma[i, i]=0
# 			else:
# 				sigma[i, i] = s[i]
# 		a1 = np.dot(U, np.dot(sigma, Vh))
#
# 		inv_sigma = np.zeros((zero[0],zero[0]))
# 		for i in range(zero[0]):
# 			inv_sigma[i, i] = 1/s[i]
# 		a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
#
# 		# print(np.trace(np.dot(a1,a1_inv)))
# 		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).max())
# 		# print((np.dot(a1,np.linspace(0,155-1,155)) - np.dot(test_matrix,np.linspace(0,155-1,155))).min())
#
# 		d=np.dot(sensitivities,power_on_voxels)
# 		m=np.dot(a1_inv,np.dot(sensitivities.T,d))
# 		score = np.sum((power_on_voxels - m) ** 2)
# 		print('check='+str(score))
# 		# print('m')
# 		# print(m)
# 		# print('power_on_voxels')
# 		# print(power_on_voxels)
#
#
# 		core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]','log'])
# 		plt.title('residuals = '+str(score))
# 		plt.savefig(path_sensitivity+'/treshold_'+str(treshold)+'.eps')
# 		plt.close()
#
# elif True:
#
# 	# I do this to verify which voxel inversion is better than the other
#
# 	back=0
# 	spike=1
# 	path=path_sensitivity + '/voxel_by_voxel_back'+str(back)+'-spike'+str(spike)
# 	if not os.path.exists(path):
# 		os.makedirs(path)
#
# 	check=[]
# 	a1_inv = np.array((scipy.sparse.load_npz(path_sensitivity + '/inverse_sensitivity.npz')).todense())
# 	for test_voxel in range(core_voxel_grid.count):
# 		power_on_voxels = np.ones((core_voxel_grid.count))*back
# 		power_on_voxels[test_voxel] = spike
#
# 		if False:
# 			weight = np.ones(len(power_on_voxels))
# 			weight_string='uniform'
# 		elif True:
# 			weight = power_on_voxels / (power_on_voxels).max()
# 			weight_string = 'linear on input emissivity'
#
# 		if (test_voxel % 20) == 0:
# 			core_voxel_grid.plot(voxel_values=power_on_voxels, colorbar=['rainbow', 'Emissivity [W/m3]'])
# 			plt.title('Input emissivity')
# 			plt.savefig(path + '/voxel_'+str(test_voxel)+'input.eps')
# 			plt.close()
# 		d = np.dot(sensitivities, power_on_voxels)
# 		m = np.dot(a1_inv, np.dot(sensitivities.T, d))
# 		score = np.sum(((power_on_voxels - m) ** 2) * weight)
# 		if (test_voxel % 20) == 0:
# 			core_voxel_grid.plot(voxel_values=m,colorbar=['rainbow','Emissivity [W/m3]'])
# 			plt.title('Estimated emissivity \n Score '+str(score))
# 			plt.savefig(path + '/voxel_'+str(test_voxel)+'estimation.eps')
# 			plt.close()
# 		check.append(score)
# 	check=np.array(check)
# 	print(repr(check))
# 	print(np.shape(check))
# 	core_voxel_grid.plot(voxel_values=check, colorbar=['rainbow', 'score [au]'])
# 	plt.title(weight_string+' weight score to find one single pixel')
# 	plt.savefig(path + '/all_scores.eps')
# 	plt.close()
