# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8


from scipy.ndimage import geometric_transform

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

# added to reat the .ptw
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
import pyradi.ryptw as ryptw

# degree of polynomial of choice
n=3
# folder of the parameters path
# pathparams='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
pathparams='/home/ffederic/work/irvb/2021-12-07_window_multiple_search_for_parameters'
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
pathparams_BB='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
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

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']

path = '/home/ffederic/work/irvb/MAST-U/'

seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s


continue_after_FAST = False
override_FAST_analysis = True
do_inversions = False

every_pixel_independent = False
overwrite_oscillation_filter = True
overwrite_binning = False
skip_second_pass = True

if overwrite_oscillation_filter:
	overwrite_binning = True

name='IRVB-MASTU_shot-45401.ptw'
# i_day,day = 0,'2021-10-21'
# name='IRVB-MASTU_shot-45371.ptw'

shot_list = get_data(path+'shot_list2.ods')
temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
for i in range(1,len(shot_list['Sheet1'])):
	if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
		date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
i_day,day = 0,str(date.date())
laser_to_analyse=path+day+'/'+name


print('starting '+laser_to_analyse)

shot_number = int(laser_to_analyse[-9:-4])
path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
path_for_plots = path_power_output + '/truncated_SVD'
if not os.path.exists(path_for_plots):
	os.makedirs(path_for_plots)

laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
laser_int_time = laser_dict['IntegrationTime']

full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
full_saved_file_dict_FAST.allow_pickle=True
full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)

full_saved_file_dict_FAST['first_pass'] = full_saved_file_dict_FAST['first_pass'].all()
inverted_dict = full_saved_file_dict_FAST['first_pass']['inverted_dict']

grid_resolution = 2
time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
# inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
# inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
binning_type = inverted_dict[str(grid_resolution)]['binning_type']
original_counts_shape = inverted_dict[str(grid_resolution)]['original_counts_shape']
foil_power =  inverted_dict[str(grid_resolution)]['foil_power']
foil_power_std =  inverted_dict[str(grid_resolution)]['foil_power_std']
shrink_factor_t = int(binning_type[binning_type.find('bin')+len('bin'):binning_type.find('x')])
shrink_factor_x = int(binning_type[binning_type.find('x')+len('x'):binning_type[binning_type.find('x')+len('x'):].find('x')+binning_type.find('x')+len('x')])


EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)



# for grid_resolution in [8, 4, 2]:
# for grid_resolution in [2,4]:
for grid_resolution in [2]:
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
		temp=0#1e-4
		temp2=0#1e-4
	elif grid_resolution==4:
		temp=0
		temp2=0

	sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = coleval.reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False,restrict_polygon=_MASTU_CORE_GRID_POLYGON)


	plt.figure()
	# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities_reshaped,axis=(0,1)),marker='s')
	# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.std(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
	plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.sum(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.colorbar()
	plt.pause(0.01)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')
	plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_reshaped_masked,axis=(0,1)),marker='s',cmap='rainbow',norm=LogNorm())#,vmin=2e-8)
	try:
		temp = np.abs(efit_reconstruction.time-0.65).argmin()
		for i in range(len(all_time_sep_r[temp])):
			plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
		plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
		plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
	except:
		pass
	# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.std(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
	# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
	plt.ylim(top=0,bottom=-2.2)
	plt.xlim(left=0.2,right=1.65)
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.colorbar().set_label('LOS density [au]')
	plt.xlabel('R [m]')
	plt.ylabel('Z [m]')
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
	# binning_type = 'bin' + str(all_shrink_factor_t[0]) + 'x' + str(all_shrink_factor_x[0]) + 'x' + str(all_shrink_factor_x[0])
	shape = list(original_counts_shape[1:])
	if shape!=list(sensitivities_reshaped_masked.shape[:-1]):
		shape.extend([len(grid_laplacian_masked)])
		def mapping(output_coords):
			return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
		sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)
	else:
		sensitivities_reshaped_masked2 = cp.deepcopy(sensitivities_reshaped_masked)
	plt.figure()
	plt.imshow(np.sum(sensitivities_reshaped_masked2,axis=-1),'rainbow',origin='lower')
	plt.colorbar()
	plt.pause(0.01)

	plt.figure()
	plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked2,axis=(0,1)),marker='s')
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
	# for shrink_factor_x in [5,3]:
	sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
	sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
	sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

	plt.figure()
	plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned,axis=(0,1)),marker='s')
	plt.colorbar()
	plt.pause(0.01)

	ROI1 = np.array([[0.03,0.82],[0.03,0.90]])	# horizontal, vertical
	ROI2 = np.array([[0.03,0.68],[0.03,0.91]])
	ROI_beams = np.array([[0.,0.32],[0.42,1]])
	# suggestion from Jack: keep the view as poloidal as I can, remove the central colon bit
	ROI1 = np.array([[0.03,0.82],[0.03,0.90]])	# horizontal, vertical
	ROI2 = np.array([[0.03,0.68],[0.03,0.91]])
	ROI_beams = np.array([[0.,0.32],[0.42,1]])
	sensitivities_binned_crop,selected_ROI,ROI1,ROI2,ROI_beams = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse,additional_output=True)

	plt.figure()
	plt.imshow(np.flip(np.transpose(selected_ROI,(1,0)),axis=1),'rainbow',origin='lower')
	plt.plot([ROI1[0,0]-0.5,ROI1[0,1]-0.5,ROI1[0,1]-0.5,ROI1[0,0]-0.5,ROI1[0,0]-0.5],[ROI1[1,0]-0.5,ROI1[1,0]-0.5,ROI1[1,1]-0.5,ROI1[1,1]-0.5,ROI1[1,0]-0.5],'k')
	plt.plot([ROI2[0,0]-0.5,ROI2[0,1]-0.5,ROI2[0,1]-0.5,ROI2[0,0]-0.5,ROI2[0,0]-0.5],[ROI2[1,0]-0.5,ROI2[1,0]-0.5,ROI2[1,1]-0.5,ROI2[1,1]-0.5,ROI2[1,0]-0.5],'--k')
	plt.plot([ROI_beams[0,0]-0.5,ROI_beams[0,1]-0.5,ROI_beams[0,1]-0.5,ROI_beams[0,0]-0.5,ROI_beams[0,0]-0.5],[ROI_beams[1,0]-0.5,ROI_beams[1,0]-0.5,ROI_beams[1,1]-0.5,ROI_beams[1,1]-0.5,ROI_beams[1,0]-0.5],'-.k')
	# plt.colorbar()
	plt.pause(0.01)

	plt.figure()
	plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
	plt.plot([ROI1[0,0]-0.5,ROI1[0,1]-0.5,ROI1[0,1]-0.5,ROI1[0,0]-0.5,ROI1[0,0]-0.5],[ROI1[1,0]-0.5,ROI1[1,0]-0.5,ROI1[1,1]-0.5,ROI1[1,1]-0.5,ROI1[1,0]-0.5],'k')
	plt.plot([ROI2[0,0]-0.5,ROI2[0,1]-0.5,ROI2[0,1]-0.5,ROI2[0,0]-0.5,ROI2[0,0]-0.5],[ROI2[1,0]-0.5,ROI2[1,0]-0.5,ROI2[1,1]-0.5,ROI2[1,1]-0.5,ROI2[1,0]-0.5],'--k')
	plt.plot([ROI_beams[0,0]-0.5,ROI_beams[0,1]-0.5,ROI_beams[0,1]-0.5,ROI_beams[0,0]-0.5,ROI_beams[0,0]-0.5],[ROI_beams[1,0]-0.5,ROI_beams[1,0]-0.5,ROI_beams[1,1]-0.5,ROI_beams[1,1]-0.5,ROI_beams[1,0]-0.5],'-.k')
	plt.colorbar()
	plt.pause(0.01)

	plt.figure()
	plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned_crop,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
	# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
	plt.colorbar()
	plt.pause(0.01)

	select_foil_region_with_plasma = coleval.select_region_with_plasma(sensitivities_binned_crop,selected_ROI)

	plt.figure()
	plt.imshow(np.flip(np.transpose(select_foil_region_with_plasma,(1,0)),axis=1),'rainbow',origin='lower')
	# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
	# plt.colorbar()
	plt.pause(0.01)


	# select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-4)
	selected_ROI_no_plasma = np.logical_and(selected_ROI,np.logical_not(select_foil_region_with_plasma))
	plt.figure(figsize=(10,6))
	# plt.imshow(np.flip(np.transpose(1*selected_ROI + 1*select_foil_region_with_plasma,(1,0)),axis=1),'rainbow',origin='lower')
	plt.imshow(np.flip(1*selected_ROI + 1*select_foil_region_with_plasma,axis=0),'rainbow',origin='lower')
	# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
	# plt.colorbar()
	plt.title('blue: excluded area\nred: plasma area')
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_plasma_region.eps')
	plt.pause(0.01)
	select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()

	plt.figure()
	plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s')
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
		temp=0#1e-4
		temp2=0#5e-5
		# temp2=np.sum(sensitivities_binned_crop,axis=(0,1)).max()*1e-4
	elif grid_resolution==4:
		temp=0
		temp2=0

	sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp,restrict_polygon=_MASTU_CORE_GRID_POLYGON,chop_corner_close_to_baffle=False)
	selected_edge_cells_for_laplacian = np.sum(sensitivities_binned_crop,axis=(0,1))>np.sum(sensitivities_binned_crop,axis=(0,1)).max()*0.2
	grid_laplacian_masked_crop = build_laplacian(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)


	sensitivities_binned_crop_shape = sensitivities_binned_crop.shape
	sensitivities_binned_crop = sensitivities_binned_crop.reshape((sensitivities_binned_crop.shape[0]*sensitivities_binned_crop.shape[1],sensitivities_binned_crop.shape[2]))

	if shrink_factor_x > 1:
		foil_resolution = str(shrink_factor_x) + 'x' + str(shrink_factor_x)
	else:
		foil_resolution = str(shape[0])

	foil_res = '_foil_pixel_h_' + str(foil_resolution)
	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
	path_sensitivity_original = cp.deepcopy(path_sensitivity)

	# binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
	# print('starting '+binning_type)



	U, s, Vh = np.linalg.svd(sensitivities_binned_crop)

	how_many_eigenvalues_to_keep = 180	# 180	# for 2B
	range_of_how_many_eigenvalues_to_keep = np.linspace(100,500,num=21).astype(int)
	a1_inv_all = []
	for how_many_eigenvalues_to_keep in range_of_how_many_eigenvalues_to_keep:
		# sigma = np.zeros_like(sensitivities_binned_crop)
		# np.fill_diagonal(sigma,s)
		inv_sigma = np.zeros_like(sensitivities_binned_crop.T)
		np.fill_diagonal(inv_sigma,1/s)
		inv_sigma[how_many_eigenvalues_to_keep:]=0	# 180	# for 2B
		# a1 = np.dot(U, np.dot(sigma, Vh))
		a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
		# m = np.dot(a1_inv, b_).T
		a1_inv_all.append(a1_inv)
	a1_inv_all = np.array(a1_inv_all)

	powernoback_full_orig = foil_power.reshape((foil_power.shape[0],foil_power.shape[1]*foil_power.shape[2]))
	inverted_data_all = np.dot(a1_inv_all,powernoback_full_orig.T).T
	temp = np.zeros((inverted_data.shape[0],inverted_data.shape[1]+2))
	temp[:,:-2] = inverted_data
	recompose_voxel_sigma = []
	for i_t in range(len(time_full_binned_crop)):
		recompose_voxel_sigma.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),temp[i_t],np.mean(grid_data_masked_crop,axis=1))[1])
	recompose_voxel_sigma = np.array(recompose_voxel_sigma)

	extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
	image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
	ani,trash = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_sigma,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot ' + laser_to_analyse[-9:-4] ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)

	ani.save(path_for_plots + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_eigen'+str(how_many_eigenvalues_to_keep)+ '_reconstruct_emissivity_truncated_svd.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
	plt.close()


	#
	#
	#
	#
