# Created 06/08/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

# just to import _MASTU_CORE_GRID_POLYGON
calculate_tangency_angle_for_poloidal_section=coleval.calculate_tangency_angle_for_poloidal_section
exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())


# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

import pickle
from scipy.ndimage import geometric_transform
# import cherab.mastu.bolometry.grid_construction

from multiprocessing import Pool,cpu_count,set_start_method
# set_start_method('spawn',force=True)
try:
	number_cpu_available = open('/proc/cpuinfo').read().count('processor\t:')
except:
	number_cpu_available = cpu_count()
number_cpu_available = 8	# the previous cheks never work
print('Number of cores available: '+str(number_cpu_available))

from scipy.signal import find_peaks, peak_prominences as get_proms
import time as tm
import pyuda as uda
client = uda.Client()

path = '/home/ffederic/work/irvb/MAST-U/'
to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25','2021-06-29','2021-06-30','2021-07-01','2021-07-06','2021-07-08','2021-07-09','2021-07-15','2021-07-27','2021-07-28','2021-07-29','2021-08-04']
# to_do = ['2021-06-29','2021-07-01']
# to_do = ['2021-08-05']
to_do = np.flip(to_do,axis=0)
# to_do = ['2021-06-03']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']


f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.append(dirnames)
days_available = f[0]
shot_available = []
for i_day,day in enumerate(to_do):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path+day+'/'):
		f.append(filenames)
	shot_available.append([])
	for name in f[0]:
		if name[-3:]=='ats' or name[-3:]=='ptw':
			shot_available[i_day].append(name)



i_day,day = 0,'2021-10-12'
name='IRVB-MASTU_shot-45239.ptw'
# name='IRVB-MASTU_shot-45235.ptw'
laser_to_analyse=path+day+'/'+name


print('starting '+laser_to_analyse)

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

inverted_dict = np.load(laser_to_analyse[:-4]+'_FAST.npz')
# inverted_dict = np.load(laser_to_analyse[:-4]+'_FAST.npz')
inverted_dict.allow_pickle=True
inverted_dict = dict(inverted_dict)
inverted_dict = inverted_dict['first_pass'].all()['inverted_dict']

EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
efit_reconstruction_CD = coleval.mclass(EFIT_path_default+'/epm0'+'45235'+'.nc',pulse_ID='45235')
all_time_sep_r_CD,all_time_sep_z_CD,r_fine_CD,z_fine_CD = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction_CD)
all_time_separatrix_CD = coleval.return_all_time_separatrix(efit_reconstruction_CD,all_time_sep_r_CD,all_time_sep_z_CD,r_fine_CD,z_fine_CD)


# for grid_resolution in [8, 4, 2]:
# for grid_resolution in [2,4]:
for grid_resolution in [2]:
	# grid_resolution = 8  # in cm
	foil_resolution = '187'

	foil_res = '_foil_pixel_h_' + str(foil_resolution)

	grid_type = 'core_res_' + str(grid_resolution) + 'cm'
	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'
	# path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_res_2cm_foil_pixel_h_187_power_stand_off_0.045_pinhole_4'
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
	# sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = coleval.reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)
	sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = coleval.reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,sum_treshold=temp2,std_treshold = temp,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False,restrict_polygon=_MASTU_CORE_GRID_POLYGON)

	if False:
		plt.figure()
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
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

	shrink_factor_t=7
	shrink_factor_x=3
	# this step is to adapt the matrix to the size of the foil I measure, that can be slightly different
	binning_type = 'bin' + str(1) + 'x' + str(1) + 'x' + str(1)
	shape = list(np.array(saved_file_dict_short[binning_type].all()['powernoback_full'].shape[1:])+2)	# +2 spatially because I remove +/- 1 pixel when I calculate the laplacian of the temperature
	if shape!=list(sensitivities_reshaped_masked.shape[:-1]):
		shape.extend([len(grid_laplacian_masked)])
		def mapping(output_coords):
			return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
		sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)
	else:
		sensitivities_reshaped_masked2 = cp.deepcopy(sensitivities_reshaped_masked)

	sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
	sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
	sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

	if False:
		plt.figure()
		plt.imshow(np.sum(sensitivities_reshaped_masked2,axis=-1),'rainbow',origin='lower')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked2,axis=(0,1)),marker='s')
		plt.colorbar()
		plt.pause(0.01)

	# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
	# ROI = np.array([[0.2,0.85],[0.1,0.9]])
	# ROI = np.array([[0.05,0.95],[0.05,0.95]])
	# ROI = np.array([[0.2,0.95],[0.1,1]])
	ROI1 = np.array([[0.03,0.82],[0.03,0.90]])	# horizontal, vertical
	ROI2 = np.array([[0.03,0.68],[0.03,0.91]])
	ROI_beams = np.array([[0.,0.32],[0.42,1]])
	# # suggestion from Jack: keep the view as poloidal as I can, remove the central colon bit
	# ROI1 = np.array([[0.03,0.65],[0.03,0.90]])	# horizontal, vertical
	# ROI2 = np.array([[0.03,0.65],[0.03,0.91]])
	ROI_beams = np.array([[0.,0.32],[0.42,1]])
	sensitivities_binned_crop,selected_ROI,ROI1,ROI2,ROI_beams = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse,additional_output=True)

	additional_polygons_dict = dict([])
	additional_polygons_dict['time'] = np.array([0])	# in this case I plot the same polygon for the whole movie
	additional_polygons_dict['0'] = np.array([[[ROI1[0,0],ROI1[0,1],ROI1[0,1],ROI1[0,0],ROI1[0,0]],[ROI1[1,0],ROI1[1,0],ROI1[1,1],ROI1[1,1],ROI1[1,0]]]])
	additional_polygons_dict['1'] = np.array([[[ROI2[0,0],ROI2[0,1],ROI2[0,1],ROI2[0,0],ROI2[0,0]],[ROI2[1,0],ROI2[1,0],ROI2[1,1],ROI2[1,1],ROI2[1,0]]]])
	additional_polygons_dict['2'] = np.array([[[ROI_beams[0,0],ROI_beams[0,1],ROI_beams[0,1],ROI_beams[0,0],ROI_beams[0,0]],[ROI_beams[1,0],ROI_beams[1,0],ROI_beams[1,1],ROI_beams[1,1],ROI_beams[1,0]]]])
	additional_polygons_dict['number_of_polygons'] = 3
	additional_polygons_dict['marker'] = ['--k','--k','--k']

	if False:
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
	# sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp,chop_corner_close_to_baffle=False)

	# # I add another step of filtering purely fo eliminate cells too isolated from the bulk and can let the laplacian grow
	# select = np.max(grid_laplacian_masked_crop,axis=0)<2.9
	# grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
	# select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
	# grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
	# select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
	# grid_laplacian_masked_crop_temp = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=select)
	# select = np.max(grid_laplacian_masked_crop_temp,axis=0)<2.9
	# sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,sum_treshold=0,std_treshold = 0,restrict_polygon=[],chop_corner_close_to_baffle=False,cells_to_exclude=select)

	selected_super_x_cells = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.85,np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.65)
	# select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-3)	# this does not work if you change the ROI
	select_foil_region_with_plasma = coleval.select_region_with_plasma(sensitivities_binned_crop,selected_ROI)
	selected_ROI_no_plasma = np.logical_and(selected_ROI,np.logical_not(select_foil_region_with_plasma))
	select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()

	if False:
		plt.figure(figsize=(10,6))
		# plt.imshow(np.flip(np.transpose(1*selected_ROI + 1*select_foil_region_with_plasma,(1,0)),axis=1),'rainbow',origin='lower')
		plt.imshow(np.flip(1*selected_ROI + 1*select_foil_region_with_plasma,axis=0),'rainbow',origin='lower')
		# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		# plt.colorbar()
		plt.title('blue: excluded area\nred: plasma area')
		# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_plasma_region.eps')
		plt.pause(0.01)
		select_foil_region_with_plasma = select_foil_region_with_plasma.flatten()

		plt.close('all')

		plt.figure()
		# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s',norm=LogNorm())
		plt.tricontourf(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],np.log10(np.sum(sensitivities_binned_crop,axis=(0,1))),levels=10,vmin=-4,vmax=0,cmap='rainbow')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.std(sensitivities_binned_crop,axis=(0,1)),marker='s',norm=LogNorm())
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.max(grid_laplacian_masked_crop,axis=(0)),marker='s')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=np.max(grid_Z_derivate_masked_crop,axis=(1)),marker='s')
		plt.colorbar()
		plt.pause(0.01)

	x1 = [1.55,0.25]	# r,z
	x2 = [1.1,-0.15]
	interp = interp1d([x1[0],x2[0]],[x1[1],x2[1]],fill_value="extrapolate",bounds_error=False)
	select = np.mean(grid_data_masked_crop,axis=1)[:,1]>interp(np.mean(grid_data_masked_crop,axis=1)[:,0])
	selected_central_border_cells = np.logical_and(select,np.logical_and(np.max(grid_Z_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,1]>-0.5))
	selected_central_border_cells = np.dot(grid_laplacian_masked_crop,selected_central_border_cells*np.random.random(selected_central_border_cells.shape))!=0

	selected_central_column_border_cells = np.logical_and(np.logical_and(np.max(grid_R_derivate_masked_crop,axis=(1))==1,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)
	selected_central_column_border_cells = np.logical_and(np.logical_and(np.dot(grid_laplacian_masked_crop,selected_central_column_border_cells*np.random.random(selected_central_column_border_cells.shape))!=0,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.7)

	# selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))
	# selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6))
	selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<2))
	selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<2))

	if False:
		selected_edge_cells_for_laplacian = np.logical_and(np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.6),np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
		if grid_resolution<8:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		if grid_resolution<4:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
			# selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

		# def temp_func():	# I package it only so it doesn't mess up other variables
		# 	from shapely.geometry import Point
		# 	from shapely.geometry.polygon import Polygon
		# 	select = np.zeros_like(selected_edge_cells_for_laplacian).astype(bool)
		# 	polygon = Polygon(FULL_MASTU_CORE_GRID_POLYGON)
		# 	for i_e in range(len(grid_data_masked_crop)):
		# 		if np.sum([polygon.contains(Point((grid_data_masked_crop[i_e][i__e,0],grid_data_masked_crop[i_e][i__e,1]))) for i__e in range(4)])==0:
		# 			select[i_e] = True
		# 	selected_cells_to_exclude = np.logical_or(selected_edge_cells_for_laplacian,select)
		# 	return selected_cells_to_exclude
		# selected_cells_to_exclude = temp_func()
		selected_cells_to_exclude = np.zeros_like(selected_edge_cells_for_laplacian)

	else:	# strange thing. I want less regularisation where the sensitivity is much higher than the rest of the image
		selected_edge_cells_for_laplacian = np.sum(sensitivities_binned_crop,axis=(0,1))>np.sum(sensitivities_binned_crop,axis=(0,1)).max()*0.2
		grid_laplacian_masked_crop = coleval.build_laplacian(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)
		grid_Z_derivate_masked_crop = coleval.build_Z_derivate(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)
		grid_R_derivate_masked_crop = coleval.build_R_derivate(grid_data_masked_crop,cells_to_exclude=selected_edge_cells_for_laplacian)

		def temp_func():	# I package it only so it doesn't mess up other variables
			from shapely.geometry import Point
			from shapely.geometry.polygon import Polygon
			select = np.zeros_like(selected_edge_cells_for_laplacian).astype(bool)
			polygon = Polygon(FULL_MASTU_CORE_GRID_POLYGON)
			for i_e in range(len(grid_data_masked_crop)):
				if np.sum([polygon.contains(Point((grid_data_masked_crop[i_e][i__e,0],grid_data_masked_crop[i_e][i__e,1]))) for i__e in range(4)])==0:
					select[i_e] = True
			selected_cells_to_exclude = np.logical_or(selected_edge_cells_for_laplacian,select)
			return selected_cells_to_exclude
		selected_cells_to_exclude = temp_func()

	if False:
		plt.figure(figsize=(6,10))
		# select = (np.max(grid_laplacian_masked_crop,axis=0)<=4) * (np.mean(grid_data_masked_crop,axis=1)[:,0]<=1.4) * (np.mean(grid_data_masked_crop,axis=1)[:,0]>=0.68) * (np.mean(grid_data_masked_crop,axis=1)[:,1]<np.interp(np.mean(grid_data_masked_crop,axis=1)[:,0],[0.44,1.1,1.6],[-1.3,-1.9,-1.7]))
		# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=select)
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells,marker='s')
		plt.title('edge region with emissivity\nrequired to be negligible')
		plt.colorbar()
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region1.eps')
		plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells_for_laplacian,marker='s')
		plt.title('edge region with\nlaplacian of emissivity\nrequired to be low')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.colorbar()
		# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region2.eps')
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_central_border_cells,marker='s')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.title('central column region with\nlimited radial derivate')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_central_column_border_cells,marker='s')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.colorbar()
		# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_centrel_column_low_R_der.eps')
		plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.title('super-x region with\nlaplacian of emissivity\nless restricted')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_super_x_cells,marker='s')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.colorbar()
		# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_super-x.eps')
		plt.pause(0.01)

	sensitivities_binned_crop_shape = sensitivities_binned_crop.shape
	sensitivities_binned_crop = sensitivities_binned_crop.reshape((sensitivities_binned_crop.shape[0]*sensitivities_binned_crop.shape[1],sensitivities_binned_crop.shape[2]))

	# plot for the paper
	# plt.figure(figsize=(8,14))
	# # plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities_reshaped,axis=(0,1)),marker='s')
	# # plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.std(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
	# temp = np.sum(sensitivities,axis=(0))/(4*np.pi) * (0.09*0.07/(240*187))
	# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=temp,s=5,marker='s',cmap='rainbow',norm=LogNorm(),vmin=temp.max()*1e-4)
	# # plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities,axis=(0)),marker='s',cmap='rainbow',norm=LogNorm(),vmin=np.mean(sensitivities,axis=(0)).max()*1e-2)
	# plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	# temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
	# try:
	# 	for i in range(len(all_time_sep_r[temp])):
	# 		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
	# except:
	# 	pass
	# plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
	# plt.plot(efit_reconstruction.upper_xpoint_r[temp],efit_reconstruction.upper_xpoint_z[temp],'xr')
	# plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
	# plt.plot(efit_reconstruction.strikepointR[temp],-efit_reconstruction.strikepointZ[temp],'xr')
	# plt.gca().set_aspect('equal')
	# plt.xlabel('R [m]')
	# plt.ylabel('Z [m]')
	# plt.colorbar().set_label('power on foil per voxel [W/(W/m^3)]')
	# plt.pause(0.01)



	if shrink_factor_x > 1:
		foil_resolution = str(shrink_factor_x) + 'x' + str(shrink_factor_x)
	else:
		foil_resolution = str(shape[0])

	foil_res = '_foil_pixel_h_' + str(foil_resolution)
	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_'+grid_type[5:]+foil_res+'_power'
	path_sensitivity_original = cp.deepcopy(path_sensitivity)

	binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
	print('starting '+binning_type)

	time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
	i_t = np.abs(time_full_binned_crop-0.5).argmin()

	temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
	phantom_variant = '1'
	scan_type = 'stand_off_0.045_pinhole_4' + '_'+phantom_variant

	try:
		temp_save = np.load(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan.npz')
		temp_save.allow_pickle=True
		phantom = temp_save[scan_type].all()['phantom']
	except:
		if True:	# this phantom scans the whole FOV
			phantom = []
			for i in np.arange(sensitivities_binned_crop_shape[-1])[::7]:
				if np.sum(grid_laplacian_masked_crop[i]!=0)>0:
					r = np.mean(grid_data_masked_crop[grid_laplacian_masked_crop[i]!=0],axis=(0,1))[0]
					z = np.mean(grid_data_masked_crop[grid_laplacian_masked_crop[i]!=0],axis=(0,1))[1]
					if r>=0.7 and r<=1.0 and z<=-1.3:
						phantom.append(grid_laplacian_masked_crop[i]!=0)
				continue
		elif False:	# scan of the target surface
			phantom = (np.max(grid_laplacian_masked_crop,axis=0)<=4) * (np.mean(grid_data_masked_crop,axis=1)[:,0]<=1.4) * (np.mean(grid_data_masked_crop,axis=1)[:,0]>=0.60) * (np.mean(grid_data_masked_crop,axis=1)[:,1]<np.interp(np.mean(grid_data_masked_crop,axis=1)[:,0],[0.44,1.1,1.6],[-1.3,-1.9,-1.7]))
			phantom = (grid_laplacian_masked_crop!=0)[phantom]
			phantom = phantom[::2]
		else:	# this phantom scans only the outer leg
			phantom = []
			for i in np.arange(sensitivities_binned_crop_shape[-1])[::1]:
				r,z = np.mean(grid_data_masked_crop[i],axis=0)
				if z<=efit_reconstruction.lower_xpoint_z[temp]:
					# if r>=efit_reconstruction.lower_xpoint_r[temp]:
					if r>=0.68:	# I already did the scan with a SXD, the the radious before is already mapped
						for i_ in range(len(all_time_sep_r[temp])):
							if np.nanmin((r_fine[all_time_sep_r[temp][i_]]-r)**2 + (z_fine[all_time_sep_z[temp][i_]]-z)**2) <= 0.005**2:
								phantom.append(grid_laplacian_masked_crop[i]!=0)
								continue
			phantom = phantom[::1]
		phantom = np.array(phantom).astype(float)
		phantom[phantom!=0] = 1e5	# W/m^3/st
		phantom = np.flip(phantom,axis=0)

		# temporary!
		phantom = phantom[24:]
		# phantom = np.flip(phantom,axis=0)

		temp_save = dict([])
		temp_save[scan_type] = dict([])

	recompose_voxel_emissivity_input_all = []
	x_optimal_input_full_res_all = []
	for i_phantom_int,phantom_int in enumerate(phantom):
		original_x_optimal = np.concatenate([phantom_int,[0],[0]])	# W/m^3/st
		x_optimal_input_full_res,recompose_voxel_emissivity_input = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),original_x_optimal,np.mean(grid_data,axis=1))
		x_optimal_input_full_res_all.append(x_optimal_input_full_res)
		recompose_voxel_emissivity_input_all.append(recompose_voxel_emissivity_input)

	extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
	image_extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
	ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_emissivity_input_all,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(recompose_voxel_emissivity_input_all))*0.5,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=False,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=True)#,extvmin=0,extvmax=4e4)
	ani.save(laser_to_analyse[:-4] + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan'+phantom_variant+'.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
	# plt.pause(0.01)
	plt.close()

	recompose_voxel_emissivity_out = []
	recompose_voxel_emissivity_excluded_out = []
	recompose_voxel_emissivity_sigma_out = []
	recompose_voxel_emissivity_sigma_excluded_out = []
	covariance_out = []
	x_optimal_out = []

	i_t_new_out = []
	fitted_foil_power_in = []
	fitted_foil_power_out = []
	regolarisation_coeff_all = []
	foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix
	# trash,selected_ROI_full_res = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_reshaped,ROI1,ROI2,ROI_beams,laser_to_analyse)

	diffusivity =1.347180363576353e-05
	thickness =2.6854735451543735e-06
	emissivity =0.9999999999
	Ptthermalconductivity=71.6 #[W/(m·K)]
	zeroC=273.15 #K / C
	sigmaSB=5.6704e-08 #[W/(m2 K4)]
	# 2022/10/14 Laser_data_analysis3_3.py, I separate well the digitizers now
	sigma_diffusivity = 0.13893985595434794
	sigma_emissivity_foil =0.08610352405403708
	sigma_thickness =0.1357454922343387
	sigma_rec_diffusivity = sigma_diffusivity

	grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
	grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
	grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
	number_cells_ROI = np.sum(selected_ROI)
	number_cells_plasma = np.sum(select_foil_region_with_plasma)

	homogeneous_scaling=1e-4
	# target_chi_square = sensitivities_binned_crop.shape[1]	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
	# target_chi_square_sigma = 200	# this should be tight, because for such a high number of degrees of freedom things should average very well
	# regolarisation_coeff_edge = 10
	# regolarisation_coeff_edge_multiplier = 100
	regolarisation_coeff_central_border_Z_derivate_multiplier = 0
	regolarisation_coeff_central_column_border_R_derivate_multiplier = 0
	regolarisation_coeff_edge_laplacian_multiplier = 0
	regolarisation_coeff_divertor_multiplier = 1
	regolarisation_coeff_non_negativity_multiplier = 200
	regolarisation_coeff_offsets_multiplier = 0#1e-10
	# regolarisation_coeff_edge_laplacian = 0.01#0.001#0.02
	regolarisation_coeff_edge = 0# 500 while writing the thesis ~13/12/2022 I realised this is not necessary

	sigma_emissivity = 1e6	# 2e3	# this is completely arbitrary
	# sigma_emissivity_2 = sigma_emissivity**2
	r_int = np.mean(grid_data_masked_crop,axis=1)[:,0]
	r_int_2 = r_int**2

	selected_ROI_internal = selected_ROI.flatten()
	not_selected_super_x_cells = np.logical_not(selected_super_x_cells)

	dt = 1/383*2	# the *2 from the 2 digitizers that were averaged
	dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']
	# proportional = 3.4e-14
	proportional = 1.203672266998437e-13
	additive = 1689.849903379957
	params_BB = np.ones((*(sensitivities_reshaped.shape[:-1]),2))
	params_BB[:,:,0] = proportional
	params_BB[:,:,1] = additive
	errparams_BB = np.zeros((*np.shape(params_BB),2))
	# errparams_BB[:,:,:,0] = 6.435877532718499e-35
	errparams_BB[:,:,:,0] = 6.435877532718499e-37
	photon_dict = coleval.calc_interpolators_BB(wavewlength_top=5.1,wavelength_bottom=1.5,inttime=2)
	time_per_iteration = []
	all_time_binned = []
	all_powernoback_full = []
	all_powernoback_full_std = []
	all_temperature_evolution_with_noise = []
	all_temperature_std_crop = []
	score_x_all = []
	score_y_all = []
	regolarisation_coeff_range_all = []
	regolarisation_coeff_upper_limit = 0.1
	# regolarisation_coeff_lower_limit = 2e-4
	regolarisation_coeff_lower_limit = 5e-5
	curvature_fit_regularisation_interval = 0.05
	fraction_of_L_curve_for_fit = 0.08
	plt.figure(10,figsize=(20, 10))
	plt.title('L-curve evolution\nlight=early, dark=late')
	plt.figure(11,figsize=(20, 10))
	plt.title('L-curve curvature evolution\nlight=early, dark=late')
	for i_phantom_int,phantom_int in enumerate(phantom):
		start = tm.time()

		try:
			temp_save = np.load(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan.npz')
			temp_save.allow_pickle=True
			temp_save = dict(temp_save)
			temp_save[scan_type] = temp_save[scan_type].all()
			temperature_evolution_with_noise = temp_save[scan_type]['all_temperature_evolution_with_noise'][i_phantom_int]
			temperature_std_crop = temp_save[scan_type]['all_temperature_std_crop'][i_phantom_int]
			peak_oscillation_freq = temp_save[scan_type]['peak_oscillation_freq']
			ref_temperature = temp_save[scan_type]['ref_temperature']
			counts_full_resolution_std = temp_save[scan_type]['counts_full_resolution_std']
			time_full_res_int = temp_save[scan_type]['time_full_res_int']
		except:

			temperature_evolution_with_noise,temperature_std_crop,peak_oscillation_freq,ref_temperature,counts_full_resolution_std,time_full_res_int = coleval.calculate_temperature_from_phantom2(phantom_int,grid_data_masked_crop,grid_data,sensitivities_reshaped,params_BB,errparams_BB,dt,dx,foil_position_dict,diffusivity,thickness,emissivity,Ptthermalconductivity,photon_dict,shrink_factor_t)

			all_temperature_evolution_with_noise.append(temperature_evolution_with_noise)
			all_temperature_std_crop.append(temperature_std_crop)
			temp_save[scan_type]['all_temperature_evolution_with_noise'] = all_temperature_evolution_with_noise
			temp_save[scan_type]['all_temperature_std_crop'] = all_temperature_std_crop
			temp_save[scan_type]['counts_full_resolution_std'] = counts_full_resolution_std
			temp_save[scan_type]['time_full_res_int'] = time_full_res_int
			temp_save[scan_type]['peak_oscillation_freq'] = peak_oscillation_freq
			temp_save[scan_type]['ref_temperature'] = ref_temperature
			temp_save[scan_type]['grid_data_masked_crop'] = grid_data_masked_crop
			np.savez_compressed(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan',**temp_save)

			if i_phantom_int>0:
				all_powernoback_full = cp.deepcopy(temp_save[scan_type]['all_powernoback_full'])
				all_powernoback_full_std = cp.deepcopy(temp_save[scan_type]['all_powernoback_full_std'])


		try:
			temp_save = np.load(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan.npz')
			temp_save.allow_pickle=True
			temp_save = dict(temp_save)
			temp_save[scan_type] = temp_save[scan_type].all()
			powernoback_full_extra = temp_save[scan_type]['all_powernoback_full'][i_phantom_int]
			powernoback_full_std_extra = temp_save[scan_type]['all_powernoback_full_std'][i_phantom_int]
			i_t_new = temp_save[scan_type]['i_t_new_out'][i_phantom_int]
		except:
			time_binned,powernoback_full_extra,powernoback_full_std_extra = coleval.from_temperature_to_power_on_foil(temperature_evolution_with_noise,counts_full_resolution_std,peak_oscillation_freq,dt,shrink_factor_t,shrink_factor_x,params_BB,errparams_BB,emissivity,thickness,diffusivity,Ptthermalconductivity,sigma_thickness,sigma_rec_diffusivity,sigma_emissivity_foil,ref_temperature,temperature_std_crop,time_full_res_int,foil_position_dict,photon_dict)

			all_powernoback_full.append(powernoback_full_extra)
			all_powernoback_full_std.append(powernoback_full_std_extra)
			temp_save[scan_type]['all_powernoback_full'] = all_powernoback_full
			temp_save[scan_type]['all_powernoback_full_std'] = all_powernoback_full_std


			# i_t = temp_save['i_t']
			i_t_new = len(time_binned)-1
			print(time_binned)
			i_t_new_out.append(i_t_new)

		powernoback_full_orig = cp.deepcopy(powernoback_full_extra[i_t_new])
		sigma_powernoback_full = cp.deepcopy(powernoback_full_std_extra[i_t_new])
		del powernoback_full_extra
		del powernoback_full_std_extra

		if False:
			plt.figure(figsize=(15,12))
			plt.imshow(powernoback_full_orig)
			plt.colorbar().set_label('input fitted_foil_power [W/m2]')
			plt.title(csv_file.name[-60:-28])
			plt.pause(0.01)

			plt.figure(figsize=(15,12))
			plt.imshow(foil_power_residuals)
			plt.colorbar().set_label('foil_power_residuals [W/m2]')
			plt.title(csv_file.name[-60:-28])
			plt.pause(0.01)

			plt.figure(figsize=(15,12))
			plt.imshow(fitted_foil_power)
			plt.colorbar().set_label('fitted_foil_power [W/m2]')
			plt.title(csv_file.name[-60:-28])
			plt.pause(0.01)

			plt.figure(figsize=(15,12))
			plt.imshow(sigma_powernoback_full)
			plt.colorbar().set_label('input sigma_powernoback_full [W/m2]')
			plt.title(csv_file.name[-60:-28])
			plt.pause(0.01)

			plt.close('all')
		else:
			pass

		powernoback_full_orig[np.logical_not(selected_ROI)] = 0
		sigma_powernoback_full[np.logical_not(selected_ROI)] = np.nan

		reference_sigma_powernoback = np.nanmedian(sigma_powernoback_full)
		regolarisation_coeff = 1e-3	# ok for np.median(sigma_powernoback_full)=78.18681

		sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1e10
		# inverted_dict[str(grid_resolution)]['foil_power'] = powernoback_full_orig

		powernoback = powernoback_full_orig.flatten()
		sigma_powernoback = sigma_powernoback_full.flatten()
		# sigma_powernoback = np.ones_like(powernoback)*10
		sigma_powernoback_2 = sigma_powernoback**2

		guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2


		prob_and_gradient,calc_hessian = coleval.define_fitting_functions(homogeneous_scaling,regolarisation_coeff_divertor_multiplier,regolarisation_coeff_central_column_border_R_derivate_multiplier,regolarisation_coeff_central_border_Z_derivate_multiplier,regolarisation_coeff_edge_laplacian_multiplier,sensitivities_binned_crop,selected_ROI_internal,select_foil_region_with_plasma,grid_laplacian_masked_crop_scaled,not_selected_super_x_cells,selected_edge_cells_for_laplacian,selected_super_x_cells,selected_central_column_border_cells,selected_central_border_cells,regolarisation_coeff_non_negativity_multiplier,selected_edge_cells,r_int,regolarisation_coeff_edge,regolarisation_coeff_offsets_multiplier,number_cells_ROI,reference_sigma_powernoback,number_cells_plasma,r_int_2)

		# regolarisation_coeff_range = np.array([1e-2])
		regolarisation_coeff_range = np.array([5e-3])
		# regolarisation_coeff_range = 10**np.arange(-0.5,-7,-0.06)

		guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
		x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,voxels_centre,recompose_voxel_emissivity_excluded_all = coleval.loop_fit_over_regularisation(prob_and_gradient,regolarisation_coeff_range,guess,grid_data_masked_crop,powernoback,sigma_powernoback,sigma_emissivity,factr=1e10,excluded_cells = selected_edge_cells_for_laplacian)
		# if first_guess == []:

		if True:
			regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
			x_optimal_all = np.flip(x_optimal_all,axis=0)
			recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)

			recompose_voxel_emissivity = recompose_voxel_emissivity_all[0]
			recompose_voxel_emissivity_excluded = recompose_voxel_emissivity_excluded_all[0]
			x_optimal = x_optimal_all[0]
			regolarisation_coeff = regolarisation_coeff_range[0]
		else:
			regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
			x_optimal_all = np.flip(x_optimal_all,axis=0)
			recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)
			recompose_voxel_emissivity_excluded_all = np.flip(recompose_voxel_emissivity_excluded_all,axis=0)
			y_opt_all = np.flip(y_opt_all,axis=0)
			opt_info_all = np.flip(opt_info_all,axis=0)

			score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
			score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,(np.logical_not(selected_edge_cells_for_laplacian)*np.array(x_optimal_all)[:,:-2]).T).T) ** 2) / (sigma_emissivity**2),axis=1)
			score_x_all.append(score_x)
			score_y_all.append(score_y)
			regolarisation_coeff_range_all.append(regolarisation_coeff_range)

			score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks,best_index,recompose_voxel_emissivity_excluded = coleval.find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,recompose_voxel_emissivity_excluded_all,regolarisation_coeff_upper_limit=regolarisation_coeff_upper_limit,regolarisation_coeff_lower_limit=regolarisation_coeff_lower_limit,curvature_fit_regularisation_interval=curvature_fit_regularisation_interval,fraction_of_L_curve_for_fit=fraction_of_L_curve_for_fit)

			plt.figure(10)
			# plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
			plt.plot(np.log(score_x_all[-1]),np.log(score_y_all[-1]),color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
			plt.plot(score_x,score_y,'+',color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
			plt.plot(score_x[best_index],score_y[best_index],'o',color='b')
			plt.plot(score_x[peaks],score_y[peaks],'o',color=str(0.9-i_phantom_int/(len(phantom)/0.9)),fillstyle='none',markersize=10)
			plt.plot(score_x[np.abs(regolarisation_coeff_range-regolarisation_coeff_upper_limit).argmin()],score_y[np.abs(regolarisation_coeff_range-regolarisation_coeff_upper_limit).argmin()],'s',color='r')
			plt.plot(score_x[np.abs(regolarisation_coeff_range-regolarisation_coeff_lower_limit).argmin()],score_y[np.abs(regolarisation_coeff_range-regolarisation_coeff_lower_limit).argmin()],'s',color='r')
			plt.xlabel('log ||Gm-d||2')
			plt.ylabel('log ||Laplacian(m)||2')
			plt.title('shot ' + laser_to_analyse[-9:-4]+' '+'L-curve evolution\nlight=early, dark=late\ncurvature_range = '+str(curvature_range)+'\ntime_per_iteration [s] ')
			plt.savefig(laser_to_analyse[:-4] + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan'+phantom_variant+'_L_curve_evolution.eps')
			plt.figure(11)
			plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
			plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+',color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
			plt.plot(regolarisation_coeff_range[best_index],Lcurve_curvature[best_index-curvature_range],'o',color='b')
			plt.plot(regolarisation_coeff_range[peaks],Lcurve_curvature[peaks-curvature_range],'o',color=str(0.9-i_phantom_int/(len(phantom)/0.9)),fillstyle='none',markersize=10)
			plt.axvline(x=regolarisation_coeff_upper_limit,color='r')
			plt.axvline(x=regolarisation_coeff_lower_limit,color='r')
			plt.semilogx()
			plt.xlabel('regularisation coeff')
			plt.ylabel('L-curve turvature')
			plt.title('shot ' + laser_to_analyse[-9:-4]+' '+'L-curve curvature evolution\nlight=early, dark=late\ncurvature_range = '+str(curvature_range)+'\ncurvature_range_left_all = '+str(curvature_range_left_all)+'\ncurvature_range_right_all = '+str(curvature_range_right_all))
			plt.savefig(laser_to_analyse[:-4] + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan'+phantom_variant+'_L_curve_curvature_evolution.eps')

		fitted_foil_power = (np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig.shape)

		args = [powernoback,sigma_powernoback,sigma_emissivity,regolarisation_coeff,sigma_powernoback**2,sigma_emissivity**2]
		hessian=calc_hessian(x_optimal,*args)
		covariance = np.linalg.inv(hessian)
		trash,recompose_voxel_sigma,recompose_voxel_excluded_sigma = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),np.diag(covariance)**0.5,np.mean(grid_data_masked_crop,axis=1),cells_to_exclude=selected_edge_cells_for_laplacian)

		recompose_voxel_emissivity_out.append(recompose_voxel_emissivity)
		recompose_voxel_emissivity_excluded_out.append(recompose_voxel_emissivity_excluded)
		recompose_voxel_emissivity_sigma_out.append(recompose_voxel_sigma)
		recompose_voxel_emissivity_sigma_excluded_out.append(recompose_voxel_excluded_sigma)
		regolarisation_coeff_all.append(regolarisation_coeff)
		covariance_out.append(covariance)
		x_optimal_out.append(x_optimal)
		fitted_foil_power_out.append(fitted_foil_power)
		time_per_iteration.append(tm.time()-start)

		extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
		image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
		additional_each_frame_label_description = ['reg coeff=']*len(recompose_voxel_emissivity_out)
		additional_each_frame_label_number = np.array(regolarisation_coeff_all)
		ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_emissivity_out,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(recompose_voxel_emissivity_out))*0.5,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=False,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=True,additional_each_frame_label_description=additional_each_frame_label_description,additional_each_frame_label_number=additional_each_frame_label_number)#,extvmin=0,extvmax=4e4)
		ani.save(laser_to_analyse[:-4]+'_stand_off_0.045_pinhole_4' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan_output'+phantom_variant+'.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
		# plt.pause(0.01)
		plt.close()


		temp_save = np.load(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan.npz',allow_pickle = True)
		temp_save.allow_pickle=True
		temp_save = dict(temp_save)
		temp_save[scan_type] = temp_save[scan_type].all()
		temp_save[scan_type]['x_optimal_input_full_res_all'] = x_optimal_input_full_res_all
		temp_save[scan_type]['phantom'] = phantom
		temp_save[scan_type]['recompose_voxel_emissivity_out'] = recompose_voxel_emissivity_out
		temp_save[scan_type]['recompose_voxel_emissivity_excluded_out'] = recompose_voxel_emissivity_excluded_out
		temp_save[scan_type]['recompose_voxel_emissivity_sigma_out'] = recompose_voxel_emissivity_sigma_out
		temp_save[scan_type]['recompose_voxel_emissivity_sigma_excluded_out'] = recompose_voxel_emissivity_sigma_excluded_out
		temp_save[scan_type]['covariance_out'] = covariance_out
		temp_save[scan_type]['x_optimal_out'] = x_optimal_out
		temp_save[scan_type]['recompose_voxel_emissivity_input_all'] = recompose_voxel_emissivity_input_all
		temp_save[scan_type]['i_t_new_out'] = i_t_new_out
		temp_save[scan_type]['fitted_foil_power_in'] = fitted_foil_power_in
		temp_save[scan_type]['fitted_foil_power_out'] = fitted_foil_power_out
		temp_save[scan_type]['time_per_iteration'] = time_per_iteration
		temp_save[scan_type]['all_time_binned'] = all_time_binned
		temp_save[scan_type]['grid_data_masked_crop'] = grid_data_masked_crop
		if False:
			temp_save[scan_type]['score_x_all'] = score_x_all
			temp_save[scan_type]['score_y_all'] = score_y_all
			temp_save[scan_type]['regolarisation_coeff_range_all'] = regolarisation_coeff_range_all
		np.savez_compressed(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan',**temp_save)


exit()

# manual bit
phantom_variant = '4'
scan_type = 'stand_off_0.045_pinhole_4' + '_'+phantom_variant
temp_save = np.load(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan.npz')
temp_save.allow_pickle = True
temp_save = dict(temp_save)
x_optimal_out = temp_save[scan_type].all()['x_optimal_out']
x_optimal_input_full_res_all = temp_save[scan_type].all()['x_optimal_input_full_res_all']
recompose_voxel_emissivity_out = temp_save[scan_type].all()['recompose_voxel_emissivity_out']
phantom = temp_save[scan_type].all()['phantom']
regolarisation_coeff = 1e-2
regolarisation_coeff_all = np.array([regolarisation_coeff]*len(phantom))

recompose_voxel_emissivity_input_all = []
recompose_voxel_emissivity_output_all = []
recompose_voxel_emissivity_difference_all = []
for i_phantom_int,phantom_int in enumerate(phantom):
	x_optimal_input = np.concatenate([phantom_int,[0],[0]])	# W/m^3/st
	x_optimal_output = x_optimal_out[i_phantom_int]
	x_optimal_difference = x_optimal_out[i_phantom_int] - np.concatenate([phantom_int,[0],[0]])
	recompose_voxel_emissivity_input_all.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),x_optimal_input,np.mean(grid_data_masked_crop,axis=1))[1])
	recompose_voxel_emissivity_output_all.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),x_optimal_output,np.mean(grid_data_masked_crop,axis=1))[1])
	recompose_voxel_emissivity_difference_all.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),x_optimal_difference,np.mean(grid_data_masked_crop,axis=1))[1])

extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
additional_each_frame_label_description = ['reg coeff=']*len(recompose_voxel_emissivity_out)
additional_each_frame_label_number = np.array(regolarisation_coeff_all)
ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(np.abs(recompose_voxel_emissivity_difference_all),(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(recompose_voxel_emissivity_out))*0.5,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=False,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=True,additional_each_frame_label_description=additional_each_frame_label_description,additional_each_frame_label_number=additional_each_frame_label_number)#,extvmin=0,extvmax=4e4)
ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan_difference'+phantom_variant+'.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
# plt.pause(0.01)
plt.close()

case = 15
plt.figure(figsize=(9,5))
im = plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_input_all[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_input_all[case])*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_input_all[case])*1e-3)
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal*4*np.pi*np.logical_not(selected_cells_to_exclude),s=50,marker='s',cmap='rainbow',vmin=x_optimal[np.logical_not(selected_cells_to_exclude)].min()*4*np.pi,vmax=x_optimal[np.logical_not(selected_cells_to_exclude)].max()*4*np.pi)#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=power_on_voxels,s=50,marker='s',norm=LogNorm(vmin=1000, vmax=total_radiation_density.max()),cmap='rainbow')
# plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--k')
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=eps.T[0],s=50,marker='s',cmap='rainbow',vmin=eps.T[0][np.logical_not(selected_cells_to_exclude)].min(),vmax=eps.T[0][np.logical_not(selected_cells_to_exclude)].max())#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(efit_reconstruction.time-0.65).argmin()
for i in range(len(all_time_sep_r[temp])):
	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
cb = plt.colorbar(im,fraction=0.0399, pad=0.04).set_label('Emissivity kW/m^3')
plt.title('Separatrix from '+str(efit_reconstruction.shotnumber)+' %.3gms\n radiated power %.4gkW' %(efit_reconstruction.time[temp]*1e3,4*np.pi*np.sum(phantom[case]*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*((grid_resolution/100)**2))/1000))
plt.ylim(bottom=-2.1,top=efit_reconstruction.lower_xpoint_z[temp]+0.2)
plt.xlim(left=0.2,right=1.6)
plt.pause(0.01)

plt.figure(figsize=(9,5))
im = plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_output_all[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_output_all[case])*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_output_all[case])*1e-3)
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal*4*np.pi*np.logical_not(selected_cells_to_exclude),s=50,marker='s',cmap='rainbow',vmin=x_optimal[np.logical_not(selected_cells_to_exclude)].min()*4*np.pi,vmax=x_optimal[np.logical_not(selected_cells_to_exclude)].max()*4*np.pi)#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=power_on_voxels,s=50,marker='s',norm=LogNorm(vmin=1000, vmax=total_radiation_density.max()),cmap='rainbow')
# plt.plot(eq.lcfs_polygon[:,0],eq.lcfs_polygon[:,1],'--k')
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=eps.T[0],s=50,marker='s',cmap='rainbow',vmin=eps.T[0][np.logical_not(selected_cells_to_exclude)].min(),vmax=eps.T[0][np.logical_not(selected_cells_to_exclude)].max())#,vmin=min(SART_x_optimal1.min(),SART_x_optimal.min()),vmax=max(SART_x_optimal1.max(),SART_x_optimal.max()))
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(efit_reconstruction.time-0.65).argmin()
for i in range(len(all_time_sep_r[temp])):
	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
cb = plt.colorbar(im,fraction=0.0399, pad=0.04).set_label('Emissivity kW/m^3')
# plt.colorbar().set_label('Emissivity kW/m^3')
plt.title('Separatrix from '+str(efit_reconstruction.shotnumber)+' %.3gms\n radiated power %.4gkW' %(efit_reconstruction.time[temp]*1e3,4*np.pi*np.sum(x_optimal_out[case][:-2]*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*((grid_resolution/100)**2))/1000))
plt.ylim(bottom=-2.1,top=efit_reconstruction.lower_xpoint_z[temp]+0.2)
plt.xlim(left=0.2,right=1.6)
plt.pause(0.01)


difference_sum = np.nansum(np.abs(recompose_voxel_emissivity_difference_all),axis=(1,2))
radiator_position_r_out = np.nansum(np.array(x_optimal_out)[:,:-2]*(np.mean(grid_data_masked_crop,axis=1)[:,0]),axis=(1))/np.nansum(np.array(x_optimal_out)[:,:-2],axis=(1))
radiator_position_z_out = np.nansum(np.array(x_optimal_out)[:,:-2]*(np.mean(grid_data_masked_crop,axis=1)[:,1]),axis=(1))/np.nansum(np.array(x_optimal_out)[:,:-2],axis=(1))
peak_radiator_position_r_out = np.mean(grid_data_masked_crop,axis=1)[:,0][np.array(x_optimal_out).argmax(axis=1)]
peak_radiator_position_z_out = np.mean(grid_data_masked_crop,axis=1)[:,1][np.array(x_optimal_out).argmax(axis=1)]
radiator_position_r_in = (np.nansum(phantom*(np.mean(grid_data_masked_crop,axis=1)[:,0]),axis=(1))/np.nansum(phantom,axis=(1)))[:len(phantom)-(len(phantom)-len(difference_sum))]
radiator_position_z_in = (np.nansum(phantom*(np.mean(grid_data_masked_crop,axis=1)[:,1]),axis=(1))/np.nansum(phantom,axis=(1)))[:len(phantom)-(len(phantom)-len(difference_sum))]
plt.figure()
plt.plot(radiator_position_r_in,((radiator_position_r_in-peak_radiator_position_r_out)**2+(radiator_position_r_in-peak_radiator_position_r_out)**2)**0.5,'b')
plt.plot(radiator_position_r_in,((radiator_position_r_in-peak_radiator_position_r_out)**2+(radiator_position_r_in-peak_radiator_position_r_out)**2)**0.5,'+b')
plt.axvline(x=0.76,color='k',linestyle='--',label='acceptable level')
plt.axhline(y=0.02,color='r',linestyle='--',label='grid resolution')
plt.legend(loc='best', fontsize='small')
plt.xlabel('phanrom radii [m]')
plt.ylabel('movement of the peak radiation [m]')
plt.grid()
plt.pause(0.01)

plt.figure()
# plt.plot(radiator_position_r_out,radiator_position_z_out,'r')
# plt.plot(radiator_position_r_out,radiator_position_z_out,'+r')
plt.plot(radiator_position_r_in,radiator_position_z_in,'b',label='radiator input position')
plt.plot(radiator_position_r_in,radiator_position_z_in,'+b')
# plt.plot(peak_radiator_position_r_out,peak_radiator_position_z_out,'y')
# plt.plot(peak_radiator_position_r_out,peak_radiator_position_z_out,'+y')
# for i in np.arange(len(radiator_position_r_out)):
# 	plt.plot([peak_radiator_position_r_out[i],radiator_position_r_in[i]],[peak_radiator_position_z_out[i],radiator_position_z_in[i]],'--k')
for i in np.arange(len(radiator_position_r_out))[1::2]:
	if ((peak_radiator_position_r_out[i]-radiator_position_r_in[i])**2+(peak_radiator_position_z_out[i]-radiator_position_z_in[i])**2)**0.5>0.02:
		plt.plot(radiator_position_r_in[i],radiator_position_z_in[i],'ok',alpha=0.5,markersize=3)
		plt.arrow(radiator_position_r_in[i],radiator_position_z_in[i],(peak_radiator_position_r_out[i]-radiator_position_r_in[i]),(peak_radiator_position_z_out[i]-radiator_position_z_in[i]), head_width=0.03, head_length=0.05, fc='k', ec='k',ls='--',length_includes_head=True,alpha=0.5)
	# plt.plot([radiator_position_r_out[i],radiator_position_r_in[i]],[radiator_position_z_out[i],radiator_position_z_in[i]],'--r')
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k',label='peak radiation movement')
temp = np.abs(radiator_position_r_in-0.70).argmin()
plt.plot([radiator_position_r_in[temp],radiator_position_r_in[temp],radiator_position_r_in[temp]+0.5],[radiator_position_z_in[temp]-0.5,radiator_position_z_in[temp],radiator_position_z_in[temp]],'--r',label='limit for power')
temp = np.abs(radiator_position_r_in-0.76).argmin()
plt.plot([radiator_position_r_in[temp],radiator_position_r_in[temp],radiator_position_r_in[temp]+0.5],[radiator_position_z_in[temp]-0.5,radiator_position_z_in[temp],radiator_position_z_in[temp]],'--k',label='limit for shape')
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.ylim(bottom=-2.1,top=efit_reconstruction.lower_xpoint_z[temp]+0.2)
plt.xlim(left=0.2,right=1.6)
plt.legend(loc='best', fontsize='small')
plt.pause(0.01)

plt.figure()
temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
for i in range(len(all_time_sep_r[temp])):
	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
temp = np.abs(efit_reconstruction_CD.time-time_full_binned_crop[i_t]).argmin()
for i in range(len(all_time_sep_r_CD[temp])):
	plt.plot(r_fine[all_time_sep_r_CD[temp][i]],z_fine[all_time_sep_z_CD[temp][i]],'--b')
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(radiator_position_r_in-0.70).argmin()
plt.plot([radiator_position_r_in[temp],radiator_position_r_in[temp],radiator_position_r_in[temp]+0.5],[radiator_position_z_in[temp]-0.5,radiator_position_z_in[temp],radiator_position_z_in[temp]],'--r',label='limit for power')
temp = np.abs(radiator_position_r_in-0.76).argmin()
plt.plot([radiator_position_r_in[temp],radiator_position_r_in[temp],radiator_position_r_in[temp]+0.5],[radiator_position_z_in[temp]-0.5,radiator_position_z_in[temp],radiator_position_z_in[temp]],'--k',label='limit for shape')
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.ylim(bottom=-2.1,top=efit_reconstruction.lower_xpoint_z[temp]+0.2)
plt.xlim(left=0.2,right=1.6)
plt.legend(loc='best', fontsize='small')
plt.pause(0.01)


temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
select_below_outside_x_point = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]<=efit_reconstruction.lower_xpoint_z[temp],np.mean(grid_data_masked_crop,axis=1)[:,0]>=efit_reconstruction.lower_xpoint_r[temp])
select_peak_dist = ((np.array([np.mean(grid_data_masked_crop,axis=1)[:,0]]*len(peak_radiator_position_r_out)).T-peak_radiator_position_r_out)**2 + (np.array([np.mean(grid_data_masked_crop,axis=1)[:,1]]*len(peak_radiator_position_r_out)).T-peak_radiator_position_z_out)**2).T**0.5 < 0.1

power_out_peak_dist = np.nansum(np.array(x_optimal_out)[:,:-2]*select_peak_dist*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)*4*np.pi*1e-3
power_out_below_outside_x_point = np.nansum(np.array(x_optimal_out)[:,:-2]*select_below_outside_x_point*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)*4*np.pi*1e-3
power_out_all = np.nansum(np.array(x_optimal_out)[:,:-2]*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)*4*np.pi*1e-3
power_in = np.nansum(phantom*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)[:len(phantom)-(len(phantom)-len(difference_sum))]*4*np.pi*1e-3
plt.figure()
p1 = plt.plot(radiator_position_r_in,power_in,label='input')
plt.plot(radiator_position_r_in,power_in,'+',color=p1[0].get_color())
plt.plot(radiator_position_r_in,power_out_all,label='total output')
plt.plot(radiator_position_r_in,power_out_below_outside_x_point,label='output below/outsode x-point')
plt.plot(radiator_position_r_in,power_out_peak_dist,label='output within 0.1m of peak')
plt.axvline(x=0.70,color='k',linestyle='--',label='acceptable level')
plt.legend(loc='best', fontsize='small')
plt.xlabel('phanrom radii [m]')
plt.ylabel('power [kW]')
plt.grid()
plt.pause(0.01)






grid_resolution_m = grid_resolution*0.01
spatial_coord=np.meshgrid(np.arange(np.shape(recompose_voxel_emissivity_output_all)[2]),np.arange(np.shape(recompose_voxel_emissivity_output_all)[1]))
def gaussian_2D_fitting_plus_gradient(full_output,data):
	def internal(args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		diameter_2 = ((x-args[2])*grid_resolution_m)**2+((y-args[1])*grid_resolution_m)**2
		out = args[0]*np.exp(- diameter_2/(2*(args[3]**2)) )
		full_out = out-data
		grad_a0 = np.nansum( 2*full_out*np.exp(- diameter_2/(2*(args[3]**2)) ) )
		grad_a1 = -np.nansum( 2*full_out*out*(-1/(2*(args[3]**2)))*2*(grid_resolution_m**2)*(y-args[1]))
		grad_a2 = -np.nansum( 2*full_out*out*(-1/(2*(args[3]**2)))*2*(grid_resolution_m**2)*(x-args[2]))
		grad_a3 = np.nansum( 2*full_out*out*( diameter_2/(args[3]**3)) )
		out = full_out**2
		if full_output==True:
			return np.nansum(out),np.array([grad_a0,grad_a1,grad_a2,grad_a3])
		else:
			return np.nansum(out)
	return internal

if False:	# only for testinf the prob_and_gradient function
	temp = recompose_voxel_emissivity_output_all[-2]
	temp[np.isnan(temp)] = 0
	guess = [temp.max(),*np.unravel_index(temp.argmax(),np.shape(temp)),0.01]
	# guess = [temp.max(),20,20,0.03]
	target = 3
	scale = 1e-4
	# guess[target] = 1e5
	temp1 = gaussian_2D_fitting_plus_gradient(True,temp)(guess)
	guess[target] +=scale
	temp2 = gaussian_2D_fitting_plus_gradient(True,temp)(guess)
	guess[target] += -2*scale
	temp3 = gaussian_2D_fitting_plus_gradient(True,temp)(guess)
	guess[target] += scale
	print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))

bds = [[0,np.inf],[-np.inf,np.inf],[-np.inf,np.inf],[0,np.inf]]
x_optimal_output_all = []
fit_output = []
x_optimal_input_all = []
fit_input = []
x = spatial_coord[0]	# horizontal
y = spatial_coord[1]	# vertical
for i_phantom_int,phantom_int in enumerate(phantom):
	temp = recompose_voxel_emissivity_output_all[i_phantom_int]
	temp[np.isnan(temp)] = 0
	guess = [temp.max(),*np.unravel_index(temp.argmax(),np.shape(temp)),0.01]
	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,recompose_voxel_emissivity_output_all[i_phantom_int]), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
	x_optimal_output_all.append(x_optimal)
	diameter_2 = ((x-x_optimal[2])*grid_resolution_m)**2+((y-x_optimal[1])*grid_resolution_m)**2
	out = x_optimal[0]*np.exp(- diameter_2/(2*(x_optimal[3]**2)) )
	fit_output.append(out)

	temp = recompose_voxel_emissivity_input_all[i_phantom_int]
	temp[np.isnan(temp)] = 0
	guess = [temp.max(),*np.unravel_index(temp.argmax(),np.shape(temp)),0.01]
	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,recompose_voxel_emissivity_input_all[i_phantom_int]), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
	x_optimal_input_all.append(x_optimal)
	diameter_2 = ((x-x_optimal[2])*grid_resolution_m)**2+((y-x_optimal[1])*grid_resolution_m)**2
	out = x_optimal[0]*np.exp(- diameter_2/(2*(x_optimal[3]**2)) )
	fit_input.append(out)
x_optimal_output_all = np.array(x_optimal_output_all)	# A,y,x,sigma
fit_output = np.array(fit_output)
x_optimal_input_all = np.array(x_optimal_input_all)	# A,y,x,sigma
fit_input = np.array(fit_input)

extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
image_extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(np.abs(fit_output--np.array(recompose_voxel_emissivity_output_all)),(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(fit))*0.5,integration=laser_int_time/1000,barlabel='Emissivity |fit - output| [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=False,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=False)#,extvmin=0,extvmax=4e4)
ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan3.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
# plt.pause(0.01)
plt.close()

plt.figure()
plt.scatter(np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_output_all[:,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_output_all[:,2].astype(int)],c=x_optimal_output_all[:,3]/x_optimal_input_all[:,3],s=50,marker='s',cmap='rainbow')
plt.colorbar()
for i_phantom_int,phantom_int in enumerate(phantom):
	plt.plot([np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_input_all[i_phantom_int,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_output_all[i_phantom_int,1].astype(int)]],[np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_input_all[i_phantom_int,2].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_output_all[i_phantom_int,2].astype(int)]],'k--')
plt.plot(np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_input_all[:,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_input_all[:,2].astype(int)],'bo')
# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.gca().set_aspect('equal')
plt.ylim(top=-0.5)
plt.pause(0.01)

plt.figure()
plt.scatter(np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_output_all[:,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_output_all[:,2].astype(int)],c=np.sum(recompose_voxel_emissivity_output_all,axis=(1,2))/np.max(recompose_voxel_emissivity_output_all,axis=(1,2)),s=50,marker='s',cmap='rainbow')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.gca().set_aspect('equal')
plt.ylim(top=-0.5)
plt.pause(0.01)


regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
x_optimal_all = np.flip(x_optimal_all,axis=0)
recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)

score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,np.array(x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)

plt.figure(10)
plt.plot(np.log(score_x),np.log(score_y),'--',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))

regolarisation_coeff_upper_limit = 10**-0.2
score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks,best_index = coleval.find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,regolarisation_coeff_upper_limit=regolarisation_coeff_upper_limit,forward_model_residuals=True)

plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(score_x,score_y,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(score_x[best_index],score_y[best_index],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(score_x[peaks],score_y[peaks],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
plt.xlabel('log ||Gm-d||2')
plt.ylabel('log ||Laplacian(m)||2')
plt.title(csv_file.name[-60:-28])
# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_evolution.eps')
plt.figure(11)
plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(regolarisation_coeff_range[best_index],Lcurve_curvature[best_index-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(regolarisation_coeff_range[peaks],Lcurve_curvature[peaks-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
plt.axvline(x=regolarisation_coeff_upper_limit,color='r')
plt.semilogx()
plt.xlabel('regularisation coeff')
plt.ylabel('L-curve turvature')
# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution.eps')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

fitted_foil_power = (np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig.shape)
fitted_foil_power_input = (np.dot(sensitivities_binned_crop,x_optimal_input[:-2])+x_optimal_input[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal_input[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig.shape)
foil_power = powernoback_full_orig
foil_power_residuals = powernoback_full_orig-fitted_foil_power
foil_power_std = cp.deepcopy(sigma_powernoback_full)
foil_power_std[foil_power_std==1e10]=np.nan

plt.figure(figsize=(12,13))
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
try:
	for i in range(len(all_time_sep_r[temp])):
		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
except:
	pass
plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
plt.colorbar().set_label('emissivity [W/m3]')
plt.ylim(top=0.5)
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(12,13))
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity-recompose_voxel_emissivity_input,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
try:
	for i in range(len(all_time_sep_r[temp])):
		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
except:
	pass
plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
plt.colorbar().set_label('emissivity inversion error [W/m3]')
plt.ylim(top=0.5)
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)


plt.figure(figsize=(15,12))
plt.imshow(foil_power)
plt.colorbar().set_label('foil_power [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(fitted_foil_power_input)
plt.colorbar().set_label('fitted_foil_power_input [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(fitted_foil_power)
plt.colorbar().set_label('fitted_foil_power [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(foil_power_residuals)
plt.colorbar().set_label('foil_power_residuals [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(foil_power_std)
plt.colorbar().set_label('foil_power_std [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(foil_power_residuals_simulated)
plt.colorbar().set_label('foil_power_residuals_simulated [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.plot(fitted_foil_power.flatten(),foil_power_residuals_simulated.flatten(),'+')
plt.grid()
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)


 # IMPORTANT - power / std correlation
# plt.figure(figsize=(15,12))
# plt.plot(np.abs(foil_power).flatten(),foil_power_std.flatten(),'+')
# # plt.plot(np.sort(np.abs(foil_power).flatten()),np.polyval(np.polyfit(np.abs(foil_power[np.isfinite(foil_power_std)]).flatten(),foil_power_std[np.isfinite(foil_power_std)].flatten(),1),np.sort(np.abs(foil_power).flatten())),'--')
# plt.plot(np.sort(np.abs(foil_power).flatten()),np.polyval(np.polyfit([0,10,15,20,25],[11.2,11.21,11.24,11.28,11.33],2),np.sort(np.abs(foil_power).flatten())),'--')
# plt.pause(0.01)

temp_save = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test_export.npz')
temp_save.allow_pickle = True
temp_save = dict(temp_save)
temp_save['stand_off_0.045_pinhole_4']['fitted_foil_power'] = fitted_foil_power
temp_save['stand_off_0.045_pinhole_4']['foil_power'] = foil_power
temp_save['stand_off_0.045_pinhole_4']['foil_power_residuals'] = foil_power_residuals
temp_save['stand_off_0.045_pinhole_4']['regolarisation_coeff'] = regolarisation_coeff
temp_save['stand_off_0.045_pinhole_4']['x_optimal'] = x_optimal
temp_save['stand_off_0.045_pinhole_4']['sigma_powernoback'] = sigma_powernoback
np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian_test_export',**temp_save)







if False:	# only visualisation
	plt.figure()
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature)
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+')
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o')
	plt.xlabel('regularisation coeff')
	plt.ylabel('L-curve turvature')
	plt.semilogx()
	plt.pause(0.01)

	plt.figure()
	plt.plot(score_x_record_rel,score_y_record_rel)
	plt.plot(score_x_record_rel,score_y_record_rel,'+')
	plt.xlabel('log ||Gm-d||2')
	plt.ylabel('log ||Laplacian(m)||2')
	plt.grid()
	plt.plot(score_x_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],score_y_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],'o')
	plt.pause(0.01)


	plt.figure(figsize=(12,13))
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
	plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
	try:
		for i in range(len(all_time_sep_r[temp])):
			plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
	except:
		pass
	plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
	plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
	plt.colorbar().set_label('emissivity [W/m3]')
	plt.ylim(top=0.5)
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
	plt.pause(0.01)


	# piece of code to plot the traces on the foil of the MASTU geometry and separatrix with changing pinhole location and standoff
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')
	i_time = np.abs(efit_reconstruction.time-0.3).argmin()
	pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z
	# pinhole_offset_extra = np.array([+0.012/(2**0.5),-0.012/(2**0.5)])
	pinhole_offset_extra = np.array([0,0])
	stand_off_length = 0.075	# m
	# Rf=1.54967	# m	radius of the centre of the foil
	Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
	plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
	centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
	pinhole_offset += pinhole_offset_extra
	pinhole_location = coleval.locate_pinhole(pinhole_offset=pinhole_offset)
	all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	structure_point_location_on_foil = coleval.return_structure_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	cv0 = np.zeros((61,78)).T
	foil_size = [0.07,0.09]
	structure_alpha=0.5
	for i in range(len(fueling_point_location_on_foil)):
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
	for i in range(len(structure_point_location_on_foil)):
		plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	plt.plot(all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	plt.plot(all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--r')
	for __i in range(len(all_time_strike_points_location_rot[i_time])):
		plt.plot(all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	for __i in range(len(all_time_separatrix[i_time])):
		plt.plot(all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--b')
	plt.axhline(y=0,color='k'),plt.axhline(y=np.shape(cv0)[0],color='k'),plt.axvline(x=0,color='k'),plt.axvline(x=np.shape(cv0)[1],color='k')
	plt.title('pinhole additional position [%.3g,%.3g]' %(pinhole_offset_extra[0]*1e3,pinhole_offset_extra[1]*1e3)+'mm\nstand off '+str(stand_off_length*1e3)+'mm')
	plt.pause(0.01)







if False:	# only for testinf the prob_and_gradient function
	target = len(guess)-2
	scale = 1e-3
	# guess[target] = 1e5
	temp1 = prob_and_gradient(guess,*args)
	guess[target] +=scale
	temp2 = prob_and_gradient(guess,*args)
	guess[target] += -2*scale
	temp3 = prob_and_gradient(guess,*args)
	guess[target] += scale
	print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))

	target = 1
	scale = 1e-3
	# guess[target] = 1e5
	temp1 = distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]])(guess)
	guess[target] +=scale
	temp2 = distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]])(guess)
	guess[target] += -2*scale
	temp3 = distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]])(guess)
	guess[target] += scale
	print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))



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
				# recompose_voxel_emissivity[i_r,i_z] = (x_optimal2-x_optimal3)[index]
				recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
				# recompose_voxel_emissivity[i_r,i_z] = likelihood_emissivity_laplacian[index]
	recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi

else:
	pass

temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
xpoint_r = efit_reconstruction.lower_xpoint_r[temp]
xpoint_z = efit_reconstruction.lower_xpoint_z[temp]
z_,r_ = np.meshgrid(np.unique(voxels_centre[:,1]),np.unique(voxels_centre[:,0]))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[z_>xpoint_z] = 0
temp[r_<xpoint_r] = 0
outer_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[z_>xpoint_z] = 0
temp[r_>xpoint_r] = 0
inner_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[z_<xpoint_z] = 0
temp[z_>0] = 0
core_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.10] = 0
x_point_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))

if False:	# just for visualisation

	plt.figure(figsize=(12,13))
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
	plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
	for i in range(len(all_time_sep_r[temp])):
		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
	plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
	plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
	plt.colorbar().set_label('emissivity [W/m3]')
	plt.ylim(top=0.5)
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
	plt.pause(0.01)



	temp = x_optimal
	temp[1226]=0
	plt.figure(figsize=(6,12))
	plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
	plt.pause(0.01)


	plt.figure(figsize=(15,12))
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nplasma region offset %.3g, whole foil offset %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,x_optimal[-2],x_optimal[-1]))
	plt.imshow(foil_power_guess.reshape(powernoback_full_orig[i_t].shape))
	plt.colorbar().set_label('power [W/m2]')
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example20.eps')
	plt.pause(0.01)

	plt.figure(figsize=(15,12))
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
	plt.imshow(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal_no_plasma).reshape(powernoback_full_orig[i_t].shape))
	plt.colorbar().set_label('power error [W/m2]')
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example21.eps')
	plt.pause(0.01)

	plt.figure()
	plt.imshow(powernoback_full_orig[i_t])
	plt.colorbar().set_label('power [W/m2]')
	plt.title('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example0.eps')
	plt.pause(0.01)

	plt.figure()
	plt.imshow(sigma_powernoback_full[i_t])
	plt.colorbar().set_label('power [W/m2]')
	plt.title('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example0.eps')
	plt.pause(0.01)

	plt.figure()
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=likelihood_emissivity_central_border_Z_derivate,marker='s')
	plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,marker='s')
	plt.colorbar()
	plt.pause(0.01)

	from cherab.core.math import Interpolate2DLinear
	temp = cp.deepcopy(recompose_voxel_emissivity)
	temp[np.isnan(temp)] = 0
	rad_interp = Interpolate2DLinear(np.unique(voxels_centre[:,0]),np.unique(voxels_centre[:,1]),temp)
	gna_dict = dict([])
	gna_dict['recompose_voxel_emissivity'] = temp
	gna_dict['R'] = np.unique(voxels_centre[:,0])
	gna_dict['Z'] = np.unique(voxels_centre[:,1])
	np.savez_compressed(laser_to_analyse[:-4]+'_test',**gna_dict)

	import numpy as np
	from cherab.core.math import Interpolate2DLinear
	gna_dict = np.load('/home/ffederic/work/irvb/MAST-U/2021-09-28/IRVB-MASTU_shot-45071_test.npz')
	rad_interp = Interpolate2DLinear(gna_dict['R'],gna_dict['Z'],gna_dict['recompose_voxel_emissivity'],extrapolate=True)
	from mubol.phantoms.solps import SOLPSPhantom
	import xarray as xr

	class Foo(SOLPSPhantom):

		def __init__(self):
			# self.prad_interpolator = rad_interp
			data = np.zeros((len(gna_dict['Z']), len(gna_dict['R'])))
			self.sampled_emiss = xr.DataArray(data, coords=[('z', gna_dict['Z']), ('r', gna_dict['R'])])
			super().__init__(prad_interpolator=rad_interp,system='sxdl')

	boh = Foo()
	boh.brightness


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
	regolarisation_coeff_all.append(regolarisation_coeff)
	outer_leg_tot_rad_power_all.append(outer_leg_tot_rad_power)
	inner_leg_tot_rad_power_all.append(inner_leg_tot_rad_power)
	core_tot_rad_power_all.append(core_tot_rad_power)
	x_point_tot_rad_power_all.append(x_point_tot_rad_power)
	time_per_iteration.append(tm.time()-time_start)
	for value in points_removed:
		Lcurve_curvature = np.concatenate([Lcurve_curvature[:value],[np.nan],Lcurve_curvature[value:]])
	Lcurve_curvature_all.append(Lcurve_curvature)

	plt.figure(10)
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_evolution.eps')
	plt.close()
	plt.figure(11)
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution.eps')
	plt.close()

	inverted_data = np.array(inverted_data)
	inverted_data_likelihood = -np.array(inverted_data_likelihood)
	inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
	inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
	fit_error = np.array(fit_error)
	chi_square_all = np.array(chi_square_all)
	regolarisation_coeff_all = np.array(regolarisation_coeff_all)
	outer_leg_tot_rad_power_all = np.array(outer_leg_tot_rad_power_all)
	inner_leg_tot_rad_power_all = np.array(inner_leg_tot_rad_power_all)
	core_tot_rad_power_all = np.array(core_tot_rad_power_all)
	x_point_tot_rad_power_all = np.array(x_point_tot_rad_power_all)
	time_per_iteration = np.array(time_per_iteration)
	fitted_foil_power = np.array(fitted_foil_power)
	foil_power = np.array(foil_power)
	foil_power_residuals = np.array(foil_power_residuals)
	score_x_all = np.array(score_x_all)
	score_y_all = np.array(score_y_all)
	regolarisation_coeff_range_all = np.array(regolarisation_coeff_range_all)
	Lcurve_curvature_all = np.array(Lcurve_curvature_all)

	path_for_plots = path_power_output + '/invertions_log/'+binning_type
	if not os.path.exists(path_for_plots):
		os.makedirs(path_for_plots)



	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(time_per_iteration)],time_per_iteration)
	# plt.semilogy()
	plt.title('time spent per iteration')
	plt.xlabel('time [s]')
	plt.ylabel('time [s]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_time_trace_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(inverted_data_likelihood)],inverted_data_likelihood)
	# plt.semilogy()
	plt.title('Fit log likelihood')
	plt.xlabel('time [s]')
	plt.ylabel('log likelihoog [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_likelihood_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(chi_square_all)],chi_square_all)
	# plt.plot(time_full_binned_crop,np.ones_like(time_full_binned_crop)*target_chi_square,'--k')
	# plt.semilogy()
	if False:
		plt.title('chi square obtained vs requested\nfixed regularisation of '+str(regolarisation_coeff))
	else:
		plt.title('chi square obtained vs requested\nflexible regolarisation coefficient')
	plt.xlabel('time [s]')
	plt.ylabel('chi square [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_chi_square_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(regolarisation_coeff_all)],regolarisation_coeff_all)
	# plt.semilogy()
	plt.title('regolarisation coefficient obtained')
	plt.semilogy()
	plt.xlabel('time [s]')
	plt.ylabel('regolarisation coefficient [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_regolarisation_coeff_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(fit_error)],fit_error)
	# plt.semilogy()
	plt.title('Fit error ( sum((image-fit)^2)^0.5/num pixels )')
	plt.xlabel('time [s]')
	plt.ylabel('average fit error [W/m2]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_fit_error_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(inverted_data_plasma_region_offset)],inverted_data_plasma_region_offset,label='plasma region')
	plt.plot(time_full_binned_crop[:len(inverted_data_homogeneous_offset)],inverted_data_homogeneous_offset,label='whole foil')
	plt.title('Offsets to match foil power')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('time [s]')
	plt.ylabel('power density [W/m2]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_offsets_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(outer_leg_tot_rad_power_all)],outer_leg_tot_rad_power_all/1e3,label='outer_leg')
	plt.plot(time_full_binned_crop[:len(inner_leg_tot_rad_power_all)],inner_leg_tot_rad_power_all/1e3,label='inner_leg')
	plt.plot(time_full_binned_crop[:len(core_tot_rad_power_all)],core_tot_rad_power_all/1e3,label='core')
	plt.plot(time_full_binned_crop[:len(x_point_tot_rad_power_all)],x_point_tot_rad_power_all/1e3,label='x_point')
	plt.plot(time_full_binned_crop[:len(outer_leg_tot_rad_power_all)],outer_leg_tot_rad_power_all/1e3+inner_leg_tot_rad_power_all/1e3+core_tot_rad_power_all/1e3,label='tot')
	plt.title('radiated power in the lower half of the machine')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('time [s]')
	plt.ylabel('power [kW]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_tot_rad_power_bayesian.eps')
	plt.close()


if False:	# I want to take results from the archive
	inverted_dict = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test.npz')
	inverted_dict.allow_pickle=True
	inverted_dict = dict(inverted_dict)
	score_x_all = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['score_x_all']
	score_y_all = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['score_y_all']
	Lcurve_curvature_all = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['Lcurve_curvature_all']
	time_full_binned_crop = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop']

	plt.figure(10)
	plt.title('L-curve evolution\nlight=early, dark=late')
	plt.figure(11)
	plt.title('L-curve curvature evolution\nlight=early, dark=late')
	i_t = 18
	for i_t in range(len(time_full_binned_crop)):
		print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
		regolarisation_coeff_range = 10**np.linspace(2,-4,num=40)
		regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
		score_x = score_x_all[i_t]
		score_y = score_y_all[i_t]

		points_removed = []
		counter_score_x = collections.Counter(score_x)
		counter_score_y = collections.Counter(score_y)
		test = np.logical_and( [value in np.array(list(counter_score_x.items()))[:,0][np.array(list(counter_score_x.items()))[:,1]>1] for value in score_x] , [value in np.array(list(counter_score_y.items()))[:,0][np.array(list(counter_score_y.items()))[:,1]>1] for value in score_y] )
		while np.sum(test)>0:
			i__ = test.argmax()
			print(i__)
			regolarisation_coeff_range = np.concatenate([regolarisation_coeff_range[:i__],regolarisation_coeff_range[i__+1:]])
			# x_optimal_all = np.concatenate([x_optimal_all[:i__],x_optimal_all[i__+1:]])
			recompose_voxel_emissivity_all = np.concatenate([recompose_voxel_emissivity_all[:i__],recompose_voxel_emissivity_all[i__+1:]])
			score_x = np.concatenate([score_x[:i__],score_x[i__+1:]])
			score_y = np.concatenate([score_y[:i__],score_y[i__+1:]])
			counter_score_x = collections.Counter(score_x)
			counter_score_y = collections.Counter(score_y)
			test = np.logical_and( [value in np.array(list(counter_score_x.items()))[:,0][np.array(list(counter_score_x.items()))[:,1]>1] for value in score_x] , [value in np.array(list(counter_score_y.items()))[:,0][np.array(list(counter_score_y.items()))[:,1]>1] for value in score_y] )
			points_removed.append(i__)
		points_removed = np.flip(points_removed,axis=0)

		# plt.figure()
		# plt.plot(score_x,score_y)
		# plt.plot(score_x,score_y,'+')
		# plt.xlabel('||Gm-d||2')
		# plt.ylabel('||Laplacian(m)||2')
		# plt.grid()
		# plt.pause(0.01)


		score_y = np.log(score_y)
		score_x = np.log(score_x)

		score_y_record_rel = (score_y-score_y.min())/(score_y.max()-score_y.min())
		score_x_record_rel = (score_x-score_x.min())/(score_x.max()-score_x.min())


		# plt.figure()
		# plt.plot(regolarisation_coeff_range,score_x_record_rel,label='fit error')
		# plt.plot(regolarisation_coeff_range,score_y_record_rel,label='laplacian error')
		# plt.legend()
		# plt.semilogx()
		# plt.pause(0.01)
		#
		# plt.figure()
		# plt.plot(score_x,score_y)
		# plt.plot(score_x,score_y,'+')
		# plt.xlabel('log ||Gm-d||2')
		# plt.ylabel('log ||Laplacian(m)||2')
		# plt.grid()
		# plt.pause(0.01)

		# if I use a fine regularisation range I need to fit the curvature over more points. this takes care of that.
		# curvature_range was originally = 2
		curvature_range = max(1,int(np.ceil(np.abs(-1.8/np.median(np.diff(np.log10(regolarisation_coeff_range)))-1)/2)))
		print('curvature_range = '+str(curvature_range))


		def distance_spread(coord):
			def int(trash,px,py):
				x = coord[0]
				y = coord[1]
				dist = ((x-px)**2 + (y-py)**2)**0.5
				spread = np.sum((dist-np.mean(dist))**2)
				# print(spread)
				return [spread]*5
			return int

		def distance_spread_and_gradient(coord):
			def int(arg):
				x = coord[0]
				y = coord[1]
				px = arg[0]
				py = arg[1]
				dist = ((x-px)**2 + (y-py)**2)**0.5
				spread = np.sum((dist-np.mean(dist))**2)
				temp = (((x-px)**2 + (y-py)**2)**-0.5)
				derivate = np.array([np.sum(2*(dist-np.mean(dist))*( -0.5*temp*2*(x-px) - np.mean(-0.5*temp*2*(x-px)) )) , np.sum(2*(dist-np.mean(dist))*( -0.5*temp*2*(y-py) - np.mean(-0.5*temp*2*(y-py)) ))])
				return spread,derivate
			return int

		# plt.figure()
		# plt.plot(score_x_record_rel,score_y_record_rel)
		curvature_radious = []
		for ii in range(curvature_range,len(score_y_record_rel)-curvature_range):
			# try:
			# 	guess = centre[0]
			# except:
			print(ii)
			try:
				guess = np.max([score_y_record_rel[ii-curvature_range:ii+curvature_range+1]*10,score_x_record_rel[ii-curvature_range:ii+curvature_range+1]*10],axis=1)

				# bds = [[np.min(score_y_record_rel[ii-2:ii+2+1]),np.min(score_x_record_rel[ii-2:ii+2+1])],[np.inf,np.inf]]
				# centre = curve_fit(distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]]),[0]*5,[0]*5,p0=guess,bounds = bds,maxfev=1e5,gtol=1e-12,verbose=1)

				# bds = [[np.min(score_y_record_rel[ii-curvature_range:ii+curvature_range+1]),np.inf],[np.min(score_x_record_rel[ii-curvature_range:ii+curvature_range+1]),np.inf]]
				bds = [[score_y_record_rel[ii],np.inf],[score_x_record_rel[ii],np.inf]]
				centre, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(distance_spread_and_gradient([score_x_record_rel[ii-curvature_range:ii+curvature_range+1],score_y_record_rel[ii-curvature_range:ii+curvature_range+1]]), x0=guess, bounds = bds, iprint=0, factr=1e8, pgtol=1e-8)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				centre = [centre]

				dist = ((score_x_record_rel[ii-curvature_range:ii+curvature_range+1]-centre[0][0])**2 + (score_y_record_rel[ii-curvature_range:ii+curvature_range+1]-centre[0][1])**2)**0.5
				radious = np.mean(dist)
				# plt.plot(score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1],'+')
				# # plt.plot(centre[0][0],centre[0][1],'o')
				# # plt.plot(np.linspace(centre[0][0]-radious,centre[0][0]+radious),centre[0][1]+(radious**2-np.linspace(-radious,+radious)**2)**0.5)
				# # plt.plot(np.linspace(centre[0][0]-radious,centre[0][0]+radious),centre[0][1]-(radious**2-np.linspace(-radious,+radious)**2)**0.5)
				# plt.axhline(y=np.min(score_y_record_rel[ii-2:ii+2+1]),linestyle='--')
				# plt.axvline(x=np.min(score_x_record_rel[ii-2:ii+2+1]),linestyle='--')
				# plt.pause(0.01)
			except:
				radious = np.inf
			curvature_radious.append(radious)
		# curvature_radious = [np.max(curvature_radious)]+curvature_radious+[np.max(curvature_radious)]
		Lcurve_curvature = 1/np.array(curvature_radious)

		# plt.figure()
		# plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature)
		# plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+')
		# plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o')
		# plt.xlabel('regularisation coeff')
		# plt.ylabel('L-curve turvature')
		# plt.semilogx()
		# plt.pause(0.01)
		#
		# plt.figure()
		# plt.plot(score_x_record_rel,score_y_record_rel)
		# plt.plot(score_x_record_rel,score_y_record_rel,'+')
		# plt.xlabel('log ||Gm-d||2')
		# plt.ylabel('log ||Laplacian(m)||2')
		# plt.grid()
		# plt.plot(score_x_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],score_y_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],'o')
		# plt.pause(0.01)

		plt.figure(10)
		plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.plot(score_x,score_y,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.plot(score_x[curvature_range:-curvature_range][Lcurve_curvature.argmax()],score_y[curvature_range:-curvature_range][Lcurve_curvature.argmax()],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.xlabel('log ||Gm-d||2')
		plt.ylabel('log ||Laplacian(m)||2')
		plt.figure(11)
		plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.semilogx()
		plt.xlabel('regularisation coeff')
		plt.ylabel('L-curve turvature')
	plt.figure(10)
	plt.grid()
	plt.figure(11)
	plt.grid()



	recompose_voxel_emissivity = recompose_voxel_emissivity_all[Lcurve_curvature.argmax()+curvature_range]
	regolarisation_coeff = regolarisation_coeff_range[Lcurve_curvature.argmax()+curvature_range]
	x_optimal = x_optimal_all[Lcurve_curvature.argmax()+curvature_range]
