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
	i_day,day = 0,'2021-09-28'
	name='IRVB-MASTU_shot-45068.ptw'
	# i_day,day = 0,'2021-10-12'
	# name='IRVB-MASTU_shot-45238.ptw'
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

all_binning_type = list(saved_file_dict_short.keys())
all_shrink_factor_x = []
all_shrink_factor_t = []
for binning_type in all_binning_type:
	all_shrink_factor_t.append(int(binning_type[binning_type.find('bin')+len('bin'):binning_type.find('x')]))
	all_shrink_factor_x.append(int(binning_type[binning_type.find('x')+len('x'):binning_type[binning_type.find('x')+len('x'):].find('x')+binning_type.find('x')+len('x')]))
all_shrink_factor_t = np.unique(all_shrink_factor_t)
all_shrink_factor_x = np.unique(all_shrink_factor_x)


EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)



# inverted_dict = dict([])
# inverted_dict['efit_reconstruction'] = efit_reconstruction

# for grid_resolution in [8, 4, 2]:
for grid_resolution in [4, 2]:
# for grid_resolution in [4]:
	# inverted_dict[str(grid_resolution)] = dict([])
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

	if False:
		# masking the voxels whose emission does not reach the foil
		# select = np.sum(sensitivities_reshaped,axis=(0,1))>0.05
		select = np.logical_and(np.std(sensitivities_reshaped,axis=(0,1))>1e-6,np.mean(sensitivities_reshaped,axis=(0,1))>0.005)
		plt.figure()
		plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=select,marker='s')
		plt.colorbar()
		plt.pause(0.01)
		grid_laplacian_masked = grid_laplacian[select]
		grid_laplacian_masked = grid_laplacian_masked[:,select]
		grid_data_masked = grid_data[select]
		sensitivities_reshaped_masked = sensitivities_reshaped[:,:,select]

		plt.figure()
		plt.imshow(np.sum(sensitivities_reshaped_masked,axis=-1),'rainbow',origin='lower')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked,axis=(0,1)),marker='s')
		plt.colorbar()
		plt.pause(0.01)

		# this is not enough because the voxels close to the pinhole have a too large influence on it and the inversion is weird
		# select = np.median(sensitivities_reshaped_masked,axis=(0,1))<5*np.mean(np.median(sensitivities_reshaped_masked,axis=(0,1)))
		select = np.logical_or(np.mean(grid_data_masked,axis=1)[:,0]<1.15,np.mean(grid_data_masked,axis=1)[:,1]<-1.3)
		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=select,marker='s')
		plt.colorbar()
		plt.pause(0.01)
		grid_laplacian_masked = grid_laplacian_masked[select]
		grid_laplacian_masked = grid_laplacian_masked[:,select]
		grid_data_masked = grid_data_masked[select]
		sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
	else:

		if grid_resolution==8:
			# temp=1e-3
			temp=1e-7
		elif grid_resolution==2:
			temp=1e-4
		elif grid_resolution==4:
			temp=0

		def reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,std_treshold = temp, sum_treshold = 0.000, core_radious_treshold = 1.9, divertor_radious_treshold = 1.9,chop_top_corner = False, extra_chop_top_corner = False , chop_corner_close_to_baffle = False, restrict_polygon = FULL_MASTU_CORE_GRID_POLYGON):
			from shapely.geometry import Point
			from shapely.geometry.polygon import Polygon
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
			if len(restrict_polygon)>0:
				polygon = Polygon(restrict_polygon)
				for i_e in range(len(grid_data_masked)):
					if np.sum([polygon.contains(Point((grid_data_masked[i_e][i__e,0],grid_data_masked[i_e][i__e,1]))) for i__e in range(4)])==0:
						select[i_e] = False
			grid_data_masked = grid_data_masked[select]
			sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
			select = np.logical_or(np.mean(grid_data_masked,axis=1)[:,0]<divertor_radious_treshold,np.mean(grid_data_masked,axis=1)[:,1]>-1.3)
			grid_data_masked = grid_data_masked[select]
			sensitivities_reshaped_masked = sensitivities_reshaped_masked[:,:,select]
			grid_laplacian_masked = coleval.build_laplacian(grid_data_masked)
			grid_Z_derivate_masked = coleval.build_Z_derivate(grid_data_masked)
			grid_R_derivate_masked = coleval.build_R_derivate(grid_data_masked)


			return sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked

	sensitivities_reshaped_masked,grid_laplacian_masked,grid_data_masked,grid_Z_derivate_masked,grid_R_derivate_masked = reduce_voxels(sensitivities_reshaped,grid_laplacian,grid_data,chop_top_corner = False,chop_corner_close_to_baffle = False, core_radious_treshold = 1.9,extra_chop_top_corner=False)


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
		# inverted_dict[str(grid_resolution)][str(shrink_factor_x)] = dict([])
		sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
		sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
		sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

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
		ROI_beams = np.array([[0.,0.3],[0.5,1]])

		# ROI = np.array([[0.2,0.95],[0.1,1]])
		ROI1 = np.round((ROI1.T*foil_shape).T).astype(int)
		ROI2 = np.round((ROI2.T*foil_shape).T).astype(int)
		ROI_beams = np.round((ROI_beams.T*foil_shape).T).astype(int)
		a,b = np.meshgrid(np.arange(foil_shape[1]),np.arange(foil_shape[0]))
		selected_ROI = np.logical_and(np.logical_and(a>=ROI1[1,0],a<ROI1[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI1[0,1],b<sensitivities_binned.shape[0]-ROI1[0,0]))
		selected_ROI = np.logical_or(selected_ROI,np.logical_and(np.logical_and(a>=ROI2[1,0],a<ROI2[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI2[0,1],b<sensitivities_binned.shape[0]-ROI2[0,0])))
		if False:
			selected_ROI = np.logical_and(selected_ROI,np.logical_not(np.logical_and(np.logical_and(a>=ROI_beams[1,0],a<ROI_beams[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI_beams[0,1],b<sensitivities_binned.shape[0]-ROI_beams[0,0]))))

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

		plt.figure()
		plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned_crop,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
		# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-4)
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

		selected_edge_cells = np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=6,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.35),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1,np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.65))
		selected_edge_cells = np.logical_or(selected_edge_cells,np.logical_and(np.logical_and(np.logical_and(np.max(grid_laplacian_masked_crop,axis=(0))<=6,np.mean(grid_data_masked_crop,axis=1)[:,0]>1.05),np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.5),np.mean(grid_data_masked_crop,axis=1)[:,1]<-0.65))

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
		# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region1.eps')
		plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells_for_laplacian,marker='s')
		plt.title('ede region with\nlaplacian of emissivity\nrequired to be low')
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
		for shrink_factor_t in [3]:
			# inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)] = dict([])
			binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
			print('starting '+binning_type)
			# powernoback_full = saved_file_dict_short[binning_type].all()['powernoback_full']
			# powernoback_std_full = saved_file_dict_short[binning_type].all()['powernoback_std_full']

			if False:
				phantom = np.array([1e5*np.logical_and(np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.6,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.8),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.,np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.2))]).T
			elif False:
				phantom = 1e5*np.logical_and(np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.6,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.7),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.,np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1))
				phantom[np.logical_and(np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,0]>0.8,np.mean(grid_data_masked_crop,axis=1)[:,0]<0.9),np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.,np.mean(grid_data_masked_crop,axis=1)[:,1]>-1.1))] = 1e5
				phantom = np.array([phantom]).T
			elif True:
				if grid_resolution == 4 :
					phantom = ((grid_laplacian_masked_crop!=0)[::2] * 1e5).T
				elif grid_resolution == 2:
					phantom = ((grid_laplacian_masked_crop!=0)[::4] * 1e5).T
					phantom = (np.dot(grid_laplacian_masked_crop,phantom)!=0) * 1e5
			powernoback_full_orig = np.dot(sensitivities_binned_crop,phantom).T.reshape((phantom.shape[-1],*sensitivities_binned_crop_shape[:-1]))
			sigma_powernoback_full = powernoback_full_orig*0.2
			sigma_powernoback_full[sigma_powernoback_full<np.median(np.unique(powernoback_full_orig))*0.2] = np.median(np.unique(powernoback_full_orig))*0.2

			plt.close('all')

			grid_laplacian_masked_crop_scaled = grid_laplacian_masked_crop/((1e-2*grid_resolution)**2)
			grid_Z_derivate_masked_crop_scaled = grid_Z_derivate_masked_crop/((1e-2*grid_resolution)**1)
			grid_R_derivate_masked_crop_scaled = grid_R_derivate_masked_crop/((1e-2*grid_resolution)**1)
			reference_sigma_powernoback = np.nanmedian(sigma_powernoback_full)
			regolarisation_coeff = 1e-3	# ok for np.median(sigma_powernoback_full)=78.18681
			if grid_resolution==4:
				regolarisation_coeff = 1e-3
			elif grid_resolution==2:
				regolarisation_coeff = 5e-4
			# regolarisation_coeff = 1e-5 / ((reference_sigma_powernoback/78.18681)**0.5)
			if True:	# these are the settings that seem to work with regolarisation_coeff = 1e-5
				sigma_emissivity = 1e4	# this is completely arbitrary
				# sigma_emissivity = 1e4 * ((np.median(sigma_powernoback_full)/78)**0.5)	# I think it must go hand in hand with the uncertanty in the pixels
				regolarisation_coeff_edge = 10	# I raiset it artificially from 1e-3 to engage regolarisation_coeff_central_border_Z_derivate and regolarisation_coeff_central_column_border_R_derivate
				regolarisation_coeff_central_border_Z_derivate = 1e-20
				regolarisation_coeff_central_column_border_R_derivate = 1e-20
				regolarisation_coeff_divertor = regolarisation_coeff/1.5
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
				regolarisation_coeff_divertor = regolarisation_coeff*1e0

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
			regolarisation_coeff_all = []
			time_per_iteration = []
			phantom_radial_all = []
			for i_t in range(len(powernoback_full_orig)):
				time_start = tm.time()

				powernoback = powernoback_full_orig[i_t].flatten()
				sigma_powernoback = sigma_powernoback_full[i_t].flatten()
				# sigma_powernoback = np.ones_like(powernoback)*10
				sigma_powernoback_2 = sigma_powernoback**2
				homogeneous_scaling=1e-4
				# if time_full_binned_crop[i_t]<0.2:
				# 	if False:	# I can start from a random sample no problem
				# 		guess = final_emissivity+(np.random.random(final_emissivity.shape)-0.5)*1e3	# the 1e3 is to make things harder for the solver, but also to go away a bit from the best fit of other functions
				# 		guess[selected_edge_cells] = guess[selected_edge_cells]*1e-4
				# 		guess = np.concatenate((guess,[0.1/homogeneous_scaling,0.1/homogeneous_scaling]))
				# 	else:
				# 		guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
				# else:
				# 	guess = cp.deepcopy(x_optimal)
				# 	guess[:-2] += (np.random.random(x_optimal.shape[0]-2)-0.5)*1e3	# I still add a bit of scramble to give it the freedom to find the best configuration
				# guess = np.ones_like(final_emissivity[i_t])*1e4

				if i_t==0:
					guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
				else:
					try:
						guess = cp.deepcopy(x_optimal)
						guess[:-2] += (np.random.random(x_optimal.shape[0]-2)-0.5)*1e2	# I still add a bit of scramble to give it the freedom to find the best configuration
					except:
						guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
				# target_chi_square = 1020	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
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
					likelihood_chi_square =0# ((likelihood_power_fit-target_chi_square)/target_chi_square_sigma)**2
					likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
					likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian*np.logical_not(selected_super_x_cells) /sigma_emissivity)**2))
					likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian*selected_super_x_cells /sigma_emissivity)**2))
					likelihood_emissivity_edge_laplacian = 0#(regolarisation_coeff_edge**2)* np.sum(((emissivity_laplacian*selected_edge_cells_for_laplacian /sigma_emissivity)**2))
					likelihood_emissivity_edge = (regolarisation_coeff_edge**2)*np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
					likelihood_emissivity_central_border_Z_derivate = (regolarisation_coeff_central_border_Z_derivate**2)* np.sum((Z_derivate*selected_central_border_cells/sigma_emissivity)**2)
					likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate*selected_central_column_border_cells/sigma_emissivity)**2)
					likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge_laplacian + likelihood_emissivity_edge + likelihood_emissivity_central_border_Z_derivate + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_laplacian_superx
					likelihood_homogeneous_offset = 0#(homogeneous_offset/reference_sigma_powernoback)**2
					likelihood_homogeneous_offset_plasma = (homogeneous_offset_plasma/reference_sigma_powernoback)**2
					likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma + likelihood_chi_square

					likelihood_power_fit_derivate = np.concatenate((-2*np.dot((foil_power_error/sigma_powernoback_2),sensitivities_binned_crop),[-2*np.sum(foil_power_error*select_foil_region_with_plasma/sigma_powernoback_2)*homogeneous_scaling,-2*np.sum(foil_power_error*selected_ROI_internal/sigma_powernoback_2)*homogeneous_scaling]))
					likelihood_chi_square_derivate =0# likelihood_power_fit_derivate * 2 *(likelihood_power_fit-target_chi_square)/(target_chi_square_sigma**2)
					likelihood_emissivity_pos_derivate = 2*(np.minimum(0.,emissivity)**2)/emissivity/sigma_emissivity_2
					likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*np.logical_not(selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian*selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_edge_laplacian_derivate = 0#2*(regolarisation_coeff_edge**2) * np.dot(emissivity_laplacian*selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
					likelihood_emissivity_edge_derivate = 2*(regolarisation_coeff_edge**2)*emissivity*selected_edge_cells/sigma_emissivity_2
					likelihood_emissivity_central_border_Z_derivate_derivate = 2*(regolarisation_coeff_central_border_Z_derivate**2)*np.dot(Z_derivate*selected_central_border_cells,grid_Z_derivate_masked_crop_scaled)/sigma_emissivity_2
					likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate*selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
					likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_edge_laplacian_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_border_Z_derivate_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate + likelihood_emissivity_laplacian_derivate_superx
					likelihood_homogeneous_offset_derivate = 0#2*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_homogeneous_offset_plasma_derivate = 2*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
					likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate + likelihood_chi_square_derivate
					# likelihood_derivate = likelihood_emissivity_central_border_derivate
					# print([likelihood,likelihood_derivate.max(),likelihood_derivate.min()])
					return likelihood,likelihood_derivate

				x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), iprint=1, factr=1e0, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				if opt_info['warnflag']>0:
					print('incomplete fit so restarted')
					x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=x_optimal, args = (powernoback), disp=1, factr=1e0, pgtol=1e-8)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				# x_optimal[-2:] *= homogeneous_scaling
				x_optimal[-2:] *= np.array([homogeneous_scaling,homogeneous_scaling])

				foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal
				foil_power_error = powernoback - foil_power_guess
				chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
				print('chi_square '+str(chi_square))


				voxels_centre = np.mean(grid_data_masked_crop,axis=1)
				dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
				dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
				dist_mean = (dz**2 + dr**2)/2
				recompose_voxel_emissivity = np.zeros((len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
				phantom_radial = np.zeros((len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
				for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
					for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
						dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
						if dist.min()<dist_mean/2:
							index = np.abs(dist).argmin()
							# recompose_voxel_emissivity[i_r,i_z] = guess[index]
							# recompose_voxel_emissivity[i_r,i_z] = (x_optimal-guess)[index]
							# recompose_voxel_emissivity[i_r,i_z] = (x_optimal2-x_optimal3)[index]
							recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
							phantom_radial[i_r,i_z] = phantom[index,i_t]
							# recompose_voxel_emissivity[i_r,i_z] = likelihood_emissivity_laplacian[index]

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
				regolarisation_coeff_all.append(x_optimal[-3])
				time_per_iteration.append(tm.time()-time_start)
				phantom_radial_all.append(phantom_radial)


			inverted_data = np.array(inverted_data)
			inverted_data_likelihood = -np.array(inverted_data_likelihood)
			inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
			inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
			fit_error = np.array(fit_error)
			chi_square_all = np.array(chi_square_all)
			regolarisation_coeff_all = np.array(regolarisation_coeff_all)
			time_per_iteration = np.array(time_per_iteration)
			fitted_foil_power = np.array(fitted_foil_power)
			foil_power = np.array(foil_power)
			foil_power_residuals = np.array(foil_power_residuals)
			phantom_radial_all = np.array(phantom_radial_all)

			if False:
				plt.figure(figsize=(12,13))
				# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
				plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
				plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
				plt.colorbar().set_label('emissivity [W/m3]')
				plt.ylim(top=0.5)
				# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
				plt.pause(0.01)

				temp = x_optimal
				temp[1226]=0
				plt.figure(figsize=(6,12))
				plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
				plt.pause(0.01)


				plt.figure(figsize=(15,12))
				plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nplasma region offset %.3g, whole foil offset %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,x_optimal[-2],x_optimal[-1]))
				plt.imshow((np.dot(sensitivities_binned_crop,x_optimal[:-3])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal_no_plasma).reshape(powernoback_full_orig[i_t].shape))
				plt.colorbar().set_label('power [W/m2]')
				# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example20.eps')
				plt.pause(0.01)

				plt.figure(figsize=(15,12))
				plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
				plt.imshow(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-3])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal_no_plasma).reshape(powernoback_full_orig[i_t].shape))
				plt.colorbar().set_label('power error [W/m2]')
				# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example21.eps')
				plt.pause(0.01)

				plt.figure()
				plt.imshow(powernoback_full_orig[i_t])
				plt.colorbar().set_label('power [W/m2]')
				plt.title('phantom on foil')
				# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example0.eps')
				plt.pause(0.01)

				plt.figure()
				# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=likelihood_emissivity_central_border_Z_derivate,marker='s')
				plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=phantom,marker='s')
				plt.colorbar()
				plt.pause(0.01)

				print('Power in the phantom = %.3gW, in the inverted data = %.3gW' %(np.nansum(phantom*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*dr*dr),np.nansum(x_optimal[:-2]*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*dr*dr)))

			else:
				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.arange(len(inverted_data))*2,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nregolarisation_coeff_divertor %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,regolarisation_coeff_divertor,grid_resolution) ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
				ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
				# plt.pause(0.01)
				plt.close()

				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data-phantom_radial_all,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.arange(len(inverted_data))*2,integration=laser_int_time/1000,barlabel='Emissivity-Phantom [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nregolarisation_coeff_divertor %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,regolarisation_coeff_divertor,grid_resolution) ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
				ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_difference_bayesian_radiator_scan.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
				# plt.pause(0.01)
				plt.close()

				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(foil_power_residuals,(0,2,1)),axis=2)]), 1 ,timesteps=np.arange(len(inverted_data))*2,integration=laser_int_time/1000,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Fitted power on foil [W/m2]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,grid_resolution),overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
				ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_fitted_foil_residuals_bayesian_radiator_scan.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
				plt.close('all')

				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(fitted_foil_power,(0,2,1)),axis=2)]), 1 ,timesteps=np.arange(len(inverted_data))*2,integration=laser_int_time/1000,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Fitted power on foil [W/m2]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,grid_resolution),overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
				ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_fitted_foil_power_bayesian_radiator_scan.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
				plt.close('all')

				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(foil_power,(0,2,1)),axis=2)]), 1 ,timesteps=np.arange(len(inverted_data))*2,integration=laser_int_time/1000,xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Fitted power on foil [W/m2]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,grid_resolution),overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)
				ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_input_foil_power_bayesian_radiator_scan.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
				plt.close('all')
