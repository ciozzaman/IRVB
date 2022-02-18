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



if False:
	for i_day,day in enumerate(to_do):
	# for i_day,day in enumerate(np.flip(to_do,axis=0)):
		# for name in shot_available[i_day]:
		for name in np.flip(shot_available[i_day],axis=0):
			laser_to_analyse=path+day+'/'+name

			# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process.py").read())
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2.py").read())
			# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3.py").read())
else:
	# i_day,day = 0,'2021-07-29'
	# name='IRVB-MASTU_shot-44578.ptw'
	# i_day,day = 0,'2021-09-01'
	# name='IRVB-MASTU_shot-44863.ptw'
	# i_day,day = 0,'2021-08-13'
	# name='IRVB-MASTU_shot-44677.ptw'
	# i_day,day = 0,'2021-08-13'
	# name='IRVB-MASTU_shot-44683.ptw'
	i_day,day = 0,'2021-10-12'
	name='IRVB-MASTU_shot-45239.ptw'
	# i_day,day = 0,'2021-10-22'
	# name='IRVB-MASTU_shot-45401.ptw'
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

inverted_dict = np.load(laser_to_analyse[:-4]+'_FAST_temp_to_erase.npz')
# inverted_dict = np.load(laser_to_analyse[:-4]+'_FAST.npz')
inverted_dict.allow_pickle=True
inverted_dict = dict(inverted_dict)
inverted_dict = inverted_dict['inverted_dict'].all()

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
	# path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_' + grid_type[5:] + foil_res + '_power'
	path_sensitivity = '/home/ffederic/work/analysis_scripts/sensitivity_matrix_res_2cm_foil_pixel_h_187_power_stand_off_0.06_pinhole_4'
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
	binning_type = 'bin' + str(1) + 'x' + str(1) + 'x' + str(1)
	shape = list(np.array(saved_file_dict_short[binning_type].all()['powernoback_full'].shape[1:])+2)	# +2 spatially because I remove +/- 1 pixel when I calculate the laplacian of the temperature
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
	for shrink_factor_x in [3]:
		sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
		sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
		sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

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
		ROI_beams = np.array([[0.,0.32],[0.4,1]])
		sensitivities_binned_crop,selected_ROI = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_binned,ROI1,ROI2,ROI_beams,laser_to_analyse)

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

		if True:	# setting zero to the sensitivities I want to exclude
			sensitivities_binned_crop = cp.deepcopy(sensitivities_binned)
			sensitivities_binned_crop[np.logical_not(selected_ROI),:] = 0
		else:	# cutting sensitivity out of ROI
			sensitivities_binned_crop = sensitivities_binned[sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]

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
			temp=1e-4
			temp2=np.sum(sensitivities_binned_crop,axis=(0,1)).max()*1e-3
		elif grid_resolution==4:
			temp=0
			temp2=0

		sensitivities_binned_crop,grid_laplacian_masked_crop,grid_data_masked_crop,grid_Z_derivate_masked_crop,grid_R_derivate_masked_crop = coleval.reduce_voxels(sensitivities_binned_crop,grid_laplacian_masked,grid_data_masked,sum_treshold=temp2,std_treshold = temp)

		select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-3)
		selected_ROI_no_plasma = np.logical_and(selected_ROI,np.logical_not(select_foil_region_with_plasma))
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

		# # I want to disconnect the 2 voxels with highest sensitivity from the others, because they have an almost homogeneous effect on the foil
		# # if there is unaccounted and uniform light all over the foil this is reflected in this voxels, so it doesn't have much sense to have them connected via the laplacian to all the rest
		# voxels_centre = np.mean(grid_data_masked_crop,axis=1)
		# dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
		# dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
		# voxel_to_disconnect = np.mean(sensitivities_binned_crop,axis=(0,1)).argmax()
		# voxels_to_disconnect = np.logical_and( np.abs(np.mean(grid_data_masked_crop,axis=1)[:,0]-np.mean(grid_data_masked_crop,axis=1)[:,0][voxel_to_disconnect])<2*dr , np.abs(np.mean(grid_data_masked_crop,axis=1)[:,1]-np.mean(grid_data_masked_crop,axis=1)[:,1][voxel_to_disconnect])<2*dz)
		# grid_laplacian_masked_crop[voxels_to_disconnect] = 0
		# grid_laplacian_masked_crop[:,voxels_to_disconnect] = 0

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

		selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.65))
		selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=5.5,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.65))

		selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells*np.random.random(selected_edge_cells.shape))!=0)
		if grid_resolution<8:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)
		if grid_resolution<4:
			selected_edge_cells_for_laplacian = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05,np.dot(grid_laplacian_masked_crop,selected_edge_cells_for_laplacian*np.random.random(selected_edge_cells_for_laplacian.shape))!=0)

		plt.figure(figsize=(6,10))
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
		# for shrink_factor_t in [3,5,2]:
		for shrink_factor_t in [7]:
			binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
			print('starting '+binning_type)

			time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
			# inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
			# x_optimal_ext = inverted_dict[str(grid_resolution)]['x_optimal_ext']
			# fitted_foil_power = inverted_dict[str(grid_resolution)]['fitted_foil_power']

			# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(fitted_foil_power,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_full_binned_crop)),integration=laser_int_time/1000,time_offset=time_full_binned_crop[0],xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)

			# The sensitivity AND voxels indexing wil be different than before, so I need to find the correlation between the old and new geometry

			temp_save = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test_export.npz')
			temp_save.allow_pickle = True
			temp_save = dict(temp_save)
			original_voxels_centre = temp_save['voxels_centre']
			# original_x_optimal = temp_save['x_optimal']
			# foil_power_residuals = temp_save['foil_power_residuals']
			sigma_powernoback = temp_save['sigma_powernoback']
			# fitted_foil_power = temp_save['fitted_foil_power']
			# foil_power_residuals_mean_interp = temp_save['foil_power_residuals_mean_interp'].tolist()
			# foil_power_residuals_std_interp = temp_save['foil_power_residuals_std_interp'].tolist()
			# foil_power_std_interp = temp_save['foil_power_std_interp'].tolist()
			# original_sensitivities_binned_crop = temp_save['sensitivities_binned_crop']
			original_grid_laplacian_masked_crop = temp_save['grid_laplacian_masked_crop']
			original_grid_data_masked_crop = temp_save['grid_data_masked_crop']
			i_t = temp_save['i_t']


			# phantom = ((original_grid_laplacian_masked_crop!=0)[::10] * 1e5).T
			# phantom = (np.dot(original_grid_laplacian_masked_crop,phantom)!=0) * 1e5
			# phantom = phantom.T

			temp = np.abs(efit_reconstruction.time-0.5).argmin()
			sep_r = all_time_sep_r[temp]
			sep_z = all_time_sep_z[temp]
			phantom = []
			for i in range(len(original_grid_laplacian_masked_crop)):
				r,z = np.mean(original_grid_data_masked_crop[i],axis=0)
				if z<-0.9:
					for i_ in range(len(sep_r)):
						if np.nanmin((r_fine[sep_r[i_]]-r)**2 + (z_fine[sep_z[i_]]-z)**2) <= 0.03**2:
							phantom.append((original_grid_laplacian_masked_crop!=0)[i])
							continue
			phantom = phantom[::9]
			phantom.append((original_grid_laplacian_masked_crop!=0)[0])
			phantom.extend(original_grid_laplacian_masked_crop[2:][(np.diff(np.mean(original_grid_data_masked_crop,axis=1)[:,1])<0)[:-1]][::4])
			phantom = np.array(phantom)
			phantom[phantom!=0] = 1e5
			phantom = np.flip(phantom,axis=0)

			recompose_voxel_emissivity_input_all = []
			x_optimal_input_full_res_all = []
			for i_phantom_int,phantom_int in enumerate(phantom):
				original_x_optimal = np.concatenate([phantom_int,[0],[0]])
				x_optimal_input_full_res,recompose_voxel_emissivity_input = coleval.translate_emissivity_profile_with_homo_temp(original_voxels_centre,original_x_optimal,np.mean(grid_data,axis=1))
				x_optimal_input_full_res_all.append(x_optimal_input_full_res)
				recompose_voxel_emissivity_input_all.append(recompose_voxel_emissivity_input)

			extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
			image_extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
			ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_emissivity_input_all,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(recompose_voxel_emissivity_input_all))*0.5,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=True)#,extvmin=0,extvmax=4e4)
			ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
			# plt.pause(0.01)
			plt.close()

			plt.figure(10,figsize=(20, 10))
			plt.title('L-curve evolution\nlight=early, dark=late')
			plt.figure(11,figsize=(20, 10))
			plt.title('L-curve curvature evolution\nlight=early, dark=late')
			recompose_voxel_emissivity_out = []
			x_optimal_out = []
			regolarisation_coeff_out = []
			i_t_new_out = []
			fitted_foil_power_in = []
			fitted_foil_power_out = []
			regolarisation_coeff_all = []
			foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizw',0.09),('foilvertw',0.07),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix
			time_per_iteration = []
			for i_phantom_int,phantom_int in enumerate(phantom):
				start = tm.time()
				original_x_optimal = np.concatenate([phantom_int,[0],[0]])
				x_optimal_input_full_res = x_optimal_input_full_res_all[i_phantom_int]
				x_optimal_input_low_res,trash = coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data,axis=1),x_optimal_input_full_res,grid_data_masked_crop)


				if False:	# visualisation only
					plt.figure(figsize=(12,13))
					# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
					plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_input,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data[:,:,0].min(),grid_data[:,:,0].max(),grid_data[:,:,1].min(),grid_data[:,:,1].max()])
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
					plt.colorbar().set_label('input emissivity [W/m3]')
					plt.ylim(top=0.5)
					plt.title(csv_file.name[-60:-28])
					plt.pause(0.01)
				else:
					pass

				homogeneous_scaling=1e-4
				# time = time_full_binned_crop[i_t]
				trash,selected_ROI_full_res = coleval.cut_sensitivity_matrix_based_on_foil_anysotropy(sensitivities_reshaped,ROI1,ROI2,ROI_beams,laser_to_analyse)
				fitted_foil_power_full_res_no_homo = (np.dot(sensitivities_reshaped,x_optimal_input_full_res[:-2])).reshape(selected_ROI_full_res.shape)#+x_optimal_input_full_res[-2]*select_foil_region_with_plasma_full_res*homogeneous_scaling+x_optimal_input_full_res[-1]*(selected_ROI_full_res.flatten())*homogeneous_scaling).reshape(selected_ROI_full_res.shape)
				fitted_foil_power_full_res_no_homo = np.flip(fitted_foil_power_full_res_no_homo,axis=1)
				fitted_foil_power_in.append(fitted_foil_power_full_res_no_homo)
				temperature_full_resolution = np.zeros(np.array(np.shape(fitted_foil_power_full_res_no_homo))+2)
				dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']
				if i_phantom_int==0:
					horizontal_coord = np.arange(np.shape(temperature_full_resolution)[1])
					vertical_coord = np.arange(np.shape(temperature_full_resolution)[0])
					horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
					grid = np.array([[horizontal_coord.flatten()]*4,[vertical_coord.flatten()]*4]).T
					grid_laplacian = -coleval.build_laplacian(grid,diagonal_factor=0.5) / (dx**2) / 2	# the /2 comes from the fact that including the diagonals amounts to double counting, so i do a mean by summing half of it
				ref_temperature = 25
				thickness=2.0531473351462095e-06
				emissivity=0.9999999999999
				diffusivity=1.0283685197530968e-05
				Ptthermalconductivity=71.6
				time_full_res = saved_file_dict_short['bin1x1x1'].all()['time_full_binned']
				dt = np.mean(np.diff(time_full_res))
				temperature_evolution = []
				temperature_evolution.append(temperature_full_resolution)
				time_full_res_int = []
				time_full_res_int.append(0)
				for i in range(shrink_factor_t*4):
					x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(coleval.find_temperature_from_power_residuals(dt,grid_laplacian,fitted_foil_power_full_res_no_homo,temperature_evolution[-1],ref_temperature=25,thickness=thickness,emissivity=emissivity,diffusivity=diffusivity,Ptthermalconductivity=Ptthermalconductivity), x0=temperature_evolution[-1][1:-1,1:-1], iprint=2, factr=1e7,pgtol=5e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
					temperature_full_resolution = np.zeros_like(temperature_evolution[0])
					temperature_full_resolution[1:-1,1:-1] = x_optimal.reshape(temperature_full_resolution[1:-1,1:-1].shape)
					temperature_evolution.append(temperature_full_resolution)
					time_full_res_int.append((i+1)*dt)


				if False:	# only to check derivatives
					target = 2
					scale = 1e-5
					# guess[target] = 1e5
					temp1 = find_temperature_from_power_residuals(dt,grid_laplacian,fitted_foil_power_full_res_no_homo,temperature_full_resolution)(guess)
					guess[target,10] +=scale
					temp2 = find_temperature_from_power_residuals(dt,grid_laplacian,fitted_foil_power_full_res_no_homo,temperature_full_resolution)(guess)
					guess[target,10] += -2*scale
					temp3 = find_temperature_from_power_residuals(dt,grid_laplacian,fitted_foil_power_full_res_no_homo,temperature_full_resolution)(guess)
					guess[target,10] += scale
					print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][np.ravel_multi_index((target,10),np.shape(guess))],((temp2[0]-temp3[0])/(2*scale))))


				temperature_evolution = np.array(temperature_evolution)
				temperature_evolution += ref_temperature
				temperature_evolution = temperature_evolution[:,1:-1,1:-1]	# because I added a pixel with fixed temperature before, and now I remove it
				# temperature_evolution = np.flip(temperature_evolution,axis=2)


				# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temperature_evolution,(0,2,1)),axis=2)]), 1/np.median(np.diff(time_full_res)),integration=laser_int_time/1000,time_offset=time_full_res[0],xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4]+'\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)

				averaged_params = np.array([-1.40845607e-07,  6.08795383e-03, -2.53336005e+01])
				counts_full_resolution = (-averaged_params[1]+((averaged_params[1]**2 - 4*averaged_params[0]*(averaged_params[2]-temperature_evolution))**0.5))/(2*averaged_params[0])
				counts_full_resolution_std = coleval.estimate_counts_std(counts_full_resolution,int_time=laser_int_time/1000)
				counts_full_resolution_with_noise = np.random.normal(loc=counts_full_resolution,scale=counts_full_resolution_std)
				temperature_evolution_with_noise = averaged_params[-1] + averaged_params[-2] * counts_full_resolution_with_noise + averaged_params[-3] * (counts_full_resolution_with_noise**2)

				temperature_crop_binned,nan_ROI_mask = coleval.proper_homo_binning_t_2D(temperature_evolution_with_noise,shrink_factor_t,shrink_factor_x)
				temperature_minus_background_crop_binned = temperature_crop_binned-ref_temperature
				counts_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(coleval.proper_homo_binning_t_2D(counts_full_resolution_std**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5)
				averaged_BB_proportional_crop_binned = 1.17347598197361e-13	* np.ones_like(temperature_minus_background_crop_binned[0])
				averaged_BB_proportional_std_crop_binned = 1.1647691499440713e-17/((shrink_factor_t**0.5)*shrink_factor_x)*np.ones_like(temperature_minus_background_crop_binned[0])
				temp_ref_counts_std_crop_binned = 5.077188644392399/((shrink_factor_t**0.5)*shrink_factor_x)*np.ones_like(temperature_minus_background_crop_binned)
				photon_flux_over_temperature_interpolator = interp1d([20,30,40,50],[2.22772831e+14, 3.10040403e+14, 4.22648744e+14, 5.65430883e+14],fill_value="extrapolate",bounds_error=False)
				photon_flux_over_temperature = photon_flux_over_temperature_interpolator(temperature_evolution_with_noise)
				temperature_std_crop = counts_full_resolution_std/(photon_flux_over_temperature*np.mean(averaged_BB_proportional_crop_binned))
				temperature_std_crop_binned = 1/(shrink_factor_t*shrink_factor_x**2)*(coleval.proper_homo_binning_t_2D(temperature_std_crop**2,shrink_factor_t,shrink_factor_x,type='np.nansum')[0]**0.5)
				time_binned = coleval.proper_homo_binning_t(time_full_res_int,shrink_factor_t)

				dx=foil_position_dict['foilhorizw']/foil_position_dict['foilhorizwpixel']*shrink_factor_x
				if i_phantom_int==0:
					horizontal_coord = np.arange(np.shape(temperature_minus_background_crop_binned)[2])
					vertical_coord = np.arange(np.shape(temperature_minus_background_crop_binned)[1])
					horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
					grid_binned = np.array([[horizontal_coord.flatten()]*4,[vertical_coord.flatten()]*4]).T
					grid_laplacian_binned = -coleval.build_laplacian(grid_binned,diagonal_factor=0.5) / (dx**2) / 2	# the /2 comes from the fact that including the diagonals amounts to double counting, so i do a mean by summing half of it

				dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std = coleval.calc_temp_to_power_BB_1(photon_flux_over_temperature_interpolator,temperature_minus_background_crop_binned,ref_temperature,time_binned,dx,counts_std_crop_binned,averaged_BB_proportional_crop_binned,averaged_BB_proportional_std_crop_binned,temp_ref_counts_std_crop_binned,temperature_std_crop_binned,nan_ROI_mask,grid_laplacian=grid_laplacian_binned)
				BBrad,diffusion,timevariation,powernoback_full,BBrad_std,diffusion_std,timevariation_std,powernoback_full_std = coleval.calc_temp_to_power_BB_2(dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4_T04,T4_T04_std,nan_ROI_mask,emissivity,thickness,1/diffusivity,Ptthermalconductivity)

				# i_t = temp_save['i_t']
				i_t_new = len(time_binned)-1-1-1
				i_t_new_out.append(i_t_new)

				powernoback_full_orig = powernoback_full[i_t_new]
				sigma_powernoback_full = powernoback_full_std[i_t_new]
				sigma_powernoback_full[sigma_powernoback.reshape(sigma_powernoback_full.shape)==1e10]=1e10
				powernoback_full_orig[sigma_powernoback.reshape(sigma_powernoback_full.shape)==1e10]=0

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

				grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
				grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
				grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
				reference_sigma_powernoback = np.nanmedian(sigma_powernoback_full)
				not_selected_super_x_cells = np.logical_not(selected_super_x_cells)

				sigma_emissivity = 1e6	# this is completely arbitrary
				sigma_emissivity_2 = sigma_emissivity**2
				r_int = np.mean(grid_data_masked_crop,axis=1)[:,0]
				r_int_2 = r_int**2

				sigma_powernoback_full[np.isnan(sigma_powernoback_full)] = 1e10
				selected_ROI_internal = selected_ROI.flatten()

				# plt.figure(10,figsize=(20, 10))
				# plt.title('L-curve evolution\nlight=early, dark=late')
				# plt.figure(11,figsize=(20, 10))
				# plt.title('L-curve curvature evolution\nlight=early, dark=late')


				powernoback = powernoback_full_orig.flatten()
				# powernoback += x_optimal_input[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal_input[-1]*(selected_ROI.flatten())*homogeneous_scaling
				sigma_powernoback = sigma_powernoback_full.flatten()
				# sigma_powernoback = np.ones_like(powernoback)*10
				sigma_powernoback_2 = sigma_powernoback**2
				# homogeneous_scaling=1e-4
				# guess = np.ones(sensitivities_binned_crop.shape[1]+2)*1e2
				# if i_phantom_int>0:
				# 	guess = first_guess
				guess = cp.deepcopy(x_optimal_input_low_res)

				# regolarisation_coeff_edge = 10
				regolarisation_coeff_central_border_Z_derivate_multiplier = 0
				regolarisation_coeff_central_column_border_R_derivate_multiplier = 0
				regolarisation_coeff_edge_laplacian_multiplier = 1e1
				regolarisation_coeff_divertor_multiplier = 1
				regolarisation_coeff_edge_multiplier = 2e2
				regolarisation_coeff_non_negativity_multiplier = 10

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
					regolarisation_coeff_edge_laplacian = regolarisation_coeff*regolarisation_coeff_edge_laplacian_multiplier
					regolarisation_coeff_edge = regolarisation_coeff*regolarisation_coeff_edge_multiplier
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
					likelihood_homogeneous_offset = 0#(homogeneous_offset/reference_sigma_powernoback)**2
					likelihood_homogeneous_offset_plasma = 0#(homogeneous_offset_plasma/reference_sigma_powernoback)**2
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
					likelihood_homogeneous_offset_derivate = 0#2*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_homogeneous_offset_plasma_derivate = 0#2*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate
					# print(tm.time()-time_start)
					# time_start = tm.time()
					return likelihood,likelihood_derivate

				regolarisation_coeff_range = 10**np.linspace(-1,-8,num=93)
				x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,voxels_centre = coleval.loop_fit_over_regularisation(prob_and_gradient,regolarisation_coeff_range,guess,grid_data_masked_crop,powernoback,sigma_powernoback,sigma_emissivity,factr=1e8,pgtol=5e-8,iprint=1)
				first_guess = cp.deepcopy(x_optimal_all[0])

				regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
				x_optimal_all = np.flip(x_optimal_all,axis=0)
				recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)

				score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
				score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,np.array(x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)

				plt.figure(10)
				plt.plot(np.log(score_x),np.log(score_y),'--',color=str(0.9-i_phantom_int/(len(phantom)/0.9)))

				regolarisation_coeff_upper_limit = 10**-0.2
				score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks,best_index = coleval.find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,regolarisation_coeff_upper_limit=regolarisation_coeff_upper_limit,forward_model_residuals=True)

				fitted_foil_power = (np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig.shape)

				recompose_voxel_emissivity_out.append(recompose_voxel_emissivity)
				regolarisation_coeff_all.append(regolarisation_coeff)
				x_optimal_out.append(x_optimal)
				regolarisation_coeff_out.append(regolarisation_coeff)
				fitted_foil_power_out.append(fitted_foil_power)
				time_per_iteration.append(tm.time()-start)

				plt.plot(score_x,score_y,color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
				plt.plot(score_x,score_y,'+',color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
				plt.plot(score_x[best_index],score_y[best_index],'o',color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
				plt.plot(score_x[peaks],score_y[peaks],'o',color=str(0.9-i_phantom_int/(len(phantom)/0.9)),fillstyle='none',markersize=10)
				plt.xlabel('log ||Gm-d||2')
				plt.ylabel('log ||Laplacian(m)||2')
				plt.title(csv_file.name[-60:-28])
				plt.title('L-curve evolution\nlight=early, dark=late\n'+csv_file.name[-60:-28]+'\ntime_per_iteration [s] '+str(np.round(time_per_iteration).astype(int)))
				plt.savefig('/home/ffederic/work/irvb/MAST-U' + '/'+'stand_off_0.060_pinhole_4' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_L_curve_evolution'+'_radiator_scan.eps')
				plt.figure(11)
				plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
				plt.plot(regolarisation_coeff_range[best_index],Lcurve_curvature[best_index-curvature_range],'o',color=str(0.9-i_phantom_int/(len(phantom)/0.9)))
				plt.plot(regolarisation_coeff_range[peaks],Lcurve_curvature[peaks-curvature_range],'o',color=str(0.9-i_phantom_int/(len(phantom)/0.9)),fillstyle='none',markersize=10)
				plt.axvline(x=regolarisation_coeff_upper_limit,color='r')
				plt.semilogx()
				plt.xlabel('regularisation coeff')
				plt.ylabel('L-curve turvature')
				plt.title('L-curve curvature evolution\nlight=early, dark=late'+csv_file.name[-60:-28])
				plt.savefig('/home/ffederic/work/irvb/MAST-U' + '/'+'stand_off_0.060_pinhole_4' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution'+'_radiator_scan.eps')
				plt.title(csv_file.name[-60:-28])
				# plt.pause(0.01)

				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				additional_each_frame_label_description = ['reg coeff=']*len(recompose_voxel_emissivity_out)
				additional_each_frame_label_number = np.array(regolarisation_coeff_all)
				ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_emissivity_out,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(recompose_voxel_emissivity_out))*0.5,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=True,additional_each_frame_label_description=additional_each_frame_label_description,additional_each_frame_label_number=additional_each_frame_label_number)#,extvmin=0,extvmax=4e4)
				ani.save('/home/ffederic/work/irvb/MAST-U' + '/'+'stand_off_0.060_pinhole_4' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan_output.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
				# plt.pause(0.01)
				plt.close()


			# temp_save = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test_export_radiator_scan.npz')
			# temp_save.allow_pickle = True
			temp_save = dict([])
			temp_save['stand_off_0.060_pinhole_4'] = dict([])
			temp_save['stand_off_0.060_pinhole_4']['x_optimal_input_full_res_all'] = x_optimal_input_full_res_all
			temp_save['stand_off_0.060_pinhole_4']['phantom'] = phantom
			temp_save['stand_off_0.060_pinhole_4']['recompose_voxel_emissivity_out'] = recompose_voxel_emissivity_out
			temp_save['stand_off_0.060_pinhole_4']['x_optimal_out'] = x_optimal_out
			temp_save['stand_off_0.060_pinhole_4']['recompose_voxel_emissivity_input_all'] = recompose_voxel_emissivity_input_all
			temp_save['stand_off_0.060_pinhole_4']['i_t_new_out'] = i_t_new_out
			temp_save['stand_off_0.060_pinhole_4']['fitted_foil_power_in'] = fitted_foil_power_in
			temp_save['stand_off_0.060_pinhole_4']['fitted_foil_power_out'] = fitted_foil_power_out
			temp_save['stand_off_0.060_pinhole_4']['time_per_iteration'] = time_per_iteration
			np.savez_compressed(laser_to_analyse[:-4]+'_stand_off_0.060_pinhole_4_inverted_baiesian_test_export_radiator_scan',**temp_save)


exit()

# manual bit

temp_save = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test_export.npz')
temp_save.allow_pickle = True
temp_save = dict(temp_save)
x_optimal_all = temp_save['stand_off_0.060_pinhole_4'].all()['x_optimal_all']
recompose_voxel_emissivity_all = temp_save['stand_off_0.060_pinhole_4'].all()['recompose_voxel_emissivity_all']
voxels_centre = temp_save['stand_off_0.060_pinhole_4'].all()['voxels_centre']
y_opt_all = temp_save['stand_off_0.060_pinhole_4'].all()['y_opt_all']
opt_info_all = temp_save['stand_off_0.060_pinhole_4'].all()['opt_info_all']
foil_power_residuals_simulated = temp_save['stand_off_0.060_pinhole_4'].all()['foil_power_residuals_simulated']


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
temp_save['stand_off_0.060_pinhole_4']['fitted_foil_power'] = fitted_foil_power
temp_save['stand_off_0.060_pinhole_4']['foil_power'] = foil_power
temp_save['stand_off_0.060_pinhole_4']['foil_power_residuals'] = foil_power_residuals
temp_save['stand_off_0.060_pinhole_4']['regolarisation_coeff'] = regolarisation_coeff
temp_save['stand_off_0.060_pinhole_4']['x_optimal'] = x_optimal
temp_save['stand_off_0.060_pinhole_4']['sigma_powernoback'] = sigma_powernoback
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
