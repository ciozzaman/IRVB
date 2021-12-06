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



inverted_dict = dict([])
inverted_dict['efit_reconstruction'] = efit_reconstruction

# for grid_resolution in [8, 4, 2]:
# for grid_resolution in [2,4]:
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

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')
	plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked,axis=(0,1)),marker='s',cmap='rainbow',norm=LogNorm(),vmin=2e-8)
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
		inverted_dict[str(grid_resolution)][str(shrink_factor_x)] = dict([])
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
		if coleval.check_beams_on(laser_to_analyse[-9:-4]):
			selected_ROI = np.logical_and(selected_ROI,np.logical_not(np.logical_and(np.logical_and(a>=ROI_beams[1,0],a<ROI_beams[1,1]),np.logical_and(b>=sensitivities_binned.shape[0]-ROI_beams[0,1],b<sensitivities_binned.shape[0]-ROI_beams[0,0]))))

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

		select_foil_region_with_plasma = (np.sum(sensitivities_binned_crop,axis=-1)>1e-4)
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
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region1.eps')
		plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_edge_cells_for_laplacian,marker='s')
		plt.title('edge region with\nlaplacian of emissivity\nrequired to be low')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.colorbar()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_edge_region2.eps')
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
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_centrel_column_low_R_der.eps')
		plt.pause(0.01)

		plt.figure(figsize=(6,10))
		plt.title('super-x region with\nlaplacian of emissivity\nless restricted')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=selected_super_x_cells,marker='s')
		ax = plt.gca() #you first need to get the axis handle
		ax.set_aspect(1)
		plt.colorbar()
		plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+ 'bin' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) +'_gridres'+str(grid_resolution)+'cm_super-x.eps')
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

			emissivity_steps = 5
			thickness_steps = 9
			rec_diffusivity_steps = 9
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

			tend = coleval.get_tend(laser_to_analyse[-9:-4])+0.01	 # I add 10ms just for safety and to catch disruptions

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

			if False:	# this can be skipped now
				powernoback_full = (np.array([[[BBrad_full_crop.tolist()]*rec_diffusivity_steps]*thickness_steps]*emissivity_steps).T*emissivity_array).T	# emissivity, thickness, rec_diffusivity
				powernoback_full += (np.array([(np.array([[diffusion_full_crop.tolist()]*rec_diffusivity_steps]*thickness_steps).T*thickness_array).T.tolist()]*emissivity_steps))
				powernoback_full += (np.array([(np.array([(np.array([timevariation_full_crop.tolist()]*rec_diffusivity_steps).T*rec_diffusivity_array).T.tolist()]*thickness_steps).T*thickness_array).T.tolist()]*emissivity_steps))

				neg_powernoback_full_penalty = np.zeros((emissivity_steps,thickness_steps,rec_diffusivity_steps))	# emissivity, thickness, rec_diffusivity
				neg_powernoback_full_penalty[np.min(powernoback_full,axis=(-1,-2,-3))<0] = np.min(powernoback_full,axis=(-1,-2,-3))[np.min(powernoback_full,axis=(-1,-2,-3))<0] - np.max(np.min(powernoback_full,axis=(-1,-2,-3)))
				likelihood = neg_powernoback_full_penalty + rec_diffusivity_log_prob
				likelihood = (likelihood.T + emissivity_log_prob).T
				likelihood = np.transpose(np.transpose(likelihood, (0,2,1)) + thickness_log_prob, (0,2,1))
				likelihood = likelihood -np.log(np.trapz(np.trapz(np.trapz(np.exp(likelihood),x=rec_diffusivity_array),x=thickness_array),x=emissivity_array))	# normalisation for logarithmic probabilities

				plt.figure()
				plt.imshow(np.trapz(np.exp(likelihood),x=rec_diffusivity_array))
				plt.pause(0.01)

				plt.figure()
				plt.imshow(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0))
				plt.pause(0.01)


				# this was the blobal properties, looking at the most likely average value of property

				# neg_powernoback_full_penalty = np.zeros((emissivity_steps,thickness_steps,rec_diffusivity_steps,powernoback_full.shape[-2],powernoback_full.shape[-1]))	# emissivity, thickness, rec_diffusivity
				neg_powernoback_full_penalty = np.zeros_like(powernoback_full)	# emissivity, thickness, rec_diffusivity
				# neg_powernoback_full_penalty[np.min(powernoback_full,axis=(-3))<0] = np.min(powernoback_full,axis=(-3))[np.min(powernoback_full,axis=(-3))<0] - np.max(np.min(powernoback_full,axis=(-3)))
				# neg_powernoback_full_penalty = np.min(powernoback_full,axis=(-3)) - np.max(np.min(powernoback_full,axis=(-3)))
				likelihood = np.transpose(np.transpose(neg_powernoback_full_penalty, (1,2,3,4,5,0)) + emissivity_log_prob, (5,0,1,2,3,4))
				likelihood = np.transpose(np.transpose(likelihood, (0,2,3,4,5,1)) + thickness_log_prob, (0,5,1,2,3,4))
				likelihood = np.transpose(np.transpose(likelihood, (0,1,3,4,5,2)) + rec_diffusivity_log_prob, (0,1,5,2,3,4))
				likelihood = likelihood -np.log(np.trapz(np.trapz(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0))	# normalisation for logarithmic probabilities
				total_volume = np.trapz(np.trapz(np.trapz(np.ones((emissivity_steps,thickness_steps,rec_diffusivity_steps)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)

				# plt.figure()
				# plt.imshow(np.trapz(np.exp(likelihood[:,:,:,20,20]),x=rec_diffusivity_array))
				# plt.pause(0.01)
				#
				# plt.figure()
				# plt.imshow(np.trapz(np.exp(likelihood[:,:,:,20,20]),x=emissivity_array,axis=0))
				# plt.pause(0.01)

				plt.figure()
				plt.imshow(np.argmax(np.trapz(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0),x=thickness_array,axis=0),axis=0))
				plt.colorbar()
				plt.pause(0.01)

				plt.figure()
				plt.imshow(np.argmax(np.trapz(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0),x=rec_diffusivity_array,axis=1),axis=0))
				plt.colorbar()
				plt.pause(0.01)

				plt.figure()
				plt.imshow(np.argmax(np.trapz(np.trapz(np.exp(likelihood),x=thickness_array,axis=2),x=rec_diffusivity_array,axis=1),axis=0))
				plt.colorbar()
				plt.pause(0.01)
			else:
				pass


			# this only looks at one constrain, though, the power on the foil.
			# now i include the relationship with the volume.
			if False:
				import numpy as np
				import scipy.sparse as sps

				def solve(A,b,non_neg=True,max_iter=2000,tol=1e-2,lam=None,verbose=True,x0=None):
					"""
					Scott Silburn's Simultaneous Algebraic Reconstruction Technique (SART) solver.
					Ported from MATLAB PhD code by James Harrison & Scott Silburn
					Iteratively solves a linear equations A*x = b
					Written for tomographic reconstruction of camera data
					Parameters:
						A (scipy sparse matrix)  : Matrix
						b (scipy sparse matrix)  : Data vector
						non_neg (bool)		   : Whether to enforce non-negativity on the solution
						max_iter (int)		   : Maximum allowed number of iterations
						tol (float)			  : Difference in fractional error between consecutive iterations where the \
												   result is considered converged and the solver stops.
						lam (float)			  : Starting value for lambda parameter ("strength" of each iteration). \
												   This will be sanity checked by the solver anyway.
						verbose (bool)		   : Whether to print status messages
					Returns:
						Numpy matrix containing solution vector x
						Array with as many elements as the number of iterations containing the relative error
						at each iteration. Can be used to see how the convergence goes.
					"""
					A = sps.csc_matrix(A)
					b = sps.csc_matrix(b)
					equations,unknowns = A.shape
					if verbose:
						print('SART Solver: solving system of {:d} equations with {:d} unknowns.'.format(equations,unknowns))

					# Initialise output matrix
					if np.sum(x0==None)==1:
						x = sps.csc_matrix(np.ones((unknowns,1))*np.exp(-1))
					else:
						x = sps.csc_matrix((np.ones((1,unknowns))*x0).T)
					x0 = cp.deepcopy(x)
					# Check if we're given lambda or want to optimise it automatically
					if lam is None:
						optimise_lam = True
						lam = 1.
					else:
						optimise_lam = False
						if verbose:
							print('Using given lambda = {:.1e}'.format(lam))
					# Lambda is weighted by the structure of the geometry matrix
					colsum = np.abs(np.array(A.T * np.ones((equations, 1))))
					lamda = np.ones(colsum.shape) * lam / colsum
					lamda[colsum == 0] = 0
					lamda = sps.csc_matrix(lamda)
					if optimise_lam:
						# Run 2 iterations and see if the norm of the difference between successive
						# iterations is going up or down. Decrease lambda until it goes down.
						if verbose:
							print('Finding maximum starting lambda value...')
						while True:
							deltas = [0,0]
							for i in range(3):
								x1 = x + lamda.multiply( A.T*(b.T - A*x) )
								if i > 0:
									deltas[i-1] = sps.linalg.norm(x1 - x) / sps.linalg.norm(x)
								x = x1
							if deltas[1] > deltas[0]:
								lamda = np.divide(lamda , 1.1)
								lam = lam / 1.1
								# x[:] = np.exp(-1)
							else:
								break
						if verbose:
							print('   Using lambda = {:.1e}'.format(lam))
					# List to store the running errror
					err = []
					x = cp.deepcopy(x0)
					iteration_number = 1
					converged = False
					print('Starting iterations...')
					# from Joe Allcock
					c1 = lam/np.sum(A,axis=0).transpose()
					c2 = np.transpose(A)
					c3 = np.transpose(1/np.sum(c2,axis=0))
					c1 = csr_matrix(c1)
					c3 = csr_matrix(c3)

					while iteration_number <= max_iter:
						# Calculate updated solution
						if False:
							x1 = x + lamda.multiply( A.T * (b.T - A * x))
						else:
							# from Joe Allcock
							x1 = x + c1.multiply(c2*(c3.multiply(b.T - A * x)))
						#Non-negativity constraint, if required
						if non_neg:
							x1[x1 < 0] = 0
							x1.eliminate_zeros()
						# Current relative error
						err.append( sps.linalg.norm(b.T - A*x1) / sps.linalg.norm(x) )
						# Update the current solution
						x = x1
						# Check if we are converged enough
						if iteration_number >= 2:
							print('iteration '+str(iteration_number)+' relative error %.3g over %.3g' %(np.abs(err[-1] - err[-2])/err[-2],tol))
							if np.abs(err[-1] - err[-2])/err[-2] < tol:
								if verbose:
									print('   Reached convergence criterion after {:d} iterations.'.format(iteration_number))
								converged = True
								break
						iteration_number += 1
					if not converged and verbose:
						print('   Stopped at {:d} iteration limit without reaching convergence criterion.'.format(max_iter))
					return x.todense(), np.array(err)

				print('starting t=%.4gms' %(time_full_binned[30]*1e3))
				import scipy.sparse as sps
				alpha = 1e-5
				A_ = np.dot(sensitivities_binned_crop.T, sensitivities_binned_crop) + (alpha**2) * np.dot(grid_laplacian_masked_crop.T, grid_laplacian_masked_crop)
				d=powernoback_full[4,4,4,30].flatten()
				temp = np.zeros((sensitivities_binned_crop.shape[1]))
				temp[270]=1
				d = np.dot(sensitivities_binned_crop,temp)
				b_ = np.dot(sensitivities_binned_crop.T,d)

				plt.figure()
				plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=temp,s=100,marker='s',cmap='rainbow')
				plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				# temp = np.abs(efit_reconstruction.time-time_full_binned[i_t]).argmin()
				# for i in range(len(all_time_sep_r[temp])):
				# 	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
				# plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
				# plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
				plt.colorbar().set_label('emissivity [W/m3]')
				plt.pause(0.01)

				m,err = solve(sensitivities_binned_crop,d,non_neg=True,max_iter=1000,tol=1e-6,verbose=True,x0=m)
				m = np.array(m)[:,0]


				m,err = solve(A_,b_,non_neg=True,lam=1,max_iter=300,tol=1e-6,verbose=True)
				m = np.array(m)[:,0]
				m,err = solve(A_,b_,non_neg=True,max_iter=30000,tol=1e-10,verbose=True,x0=m)
				m = np.array(m)[:,0]

				U, s, Vh = np.linalg.svd(A_)
				sigma = np.diag(s)
				inv_sigma = np.diag(1 / s)
				a1 = np.dot(U, np.dot(sigma, Vh))
				a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))
				m = np.dot(a1_inv, b_).T

				plt.figure()
				plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=m,s=100,marker='s',cmap='rainbow')
				plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				# temp = np.abs(efit_reconstruction.time-time_full_binned[i_t]).argmin()
				# for i in range(len(all_time_sep_r[temp])):
				# 	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
				# plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
				# plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
				plt.colorbar().set_label('emissivity [W/m3]')
				plt.pause(0.01)

				plt.figure()
				plt.imshow(rotate(np.sum(sensitivities_binned_crop*m,axis=1).reshape(np.shape(powernoback_full[0,0,0,0])),-90),'rainbow',origin='lower')
				# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
				plt.colorbar()
				plt.pause(0.01)

				plt.figure()
				plt.imshow(rotate(d.reshape(powernoback_full[4,4,4,30].shape),-90),'rainbow',origin='lower')
				# plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
				plt.colorbar()
				plt.pause(0.01)

			elif False:
				# bayesian process with predefined prior from a range of possible values of material properties
				alpha = 1e-4
				A_ = np.dot(sensitivities_binned_crop.T, sensitivities_binned_crop) + (alpha**2) * np.dot(grid_laplacian_masked_crop.T, grid_laplacian_masked_crop)
				d=powernoback_full.reshape(emissivity_steps*thickness_steps*rec_diffusivity_steps*powernoback_full.shape[-3],powernoback_full.shape[-2]*powernoback_full.shape[-1])
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
					neg_powernoback_full_penalty = neg_powernoback_full_penalty.reshape((emissivity_steps*thickness_steps*rec_diffusivity_steps*powernoback_full.shape[-3],neg_powernoback_full_penalty.shape[-2]*neg_powernoback_full_penalty.shape[-1]))
					neg_powernoback_full_penalty = np.dot(a1_inv, np.dot(sensitivities_binned_crop.T,neg_powernoback_full_penalty.T)).T
					neg_powernoback_full_penalty -= np.max(neg_powernoback_full_penalty)
					# neg_powernoback_full_penalty[neg_powernoback_full_penalty<0] = -10*neg_powernoback_full_penalty[neg_powernoback_full_penalty<0]/np.min(neg_powernoback_full_penalty[neg_powernoback_full_penalty<0])
					neg_powernoback_full_penalty = -20*neg_powernoback_full_penalty/np.median(np.flip(np.sort(neg_powernoback_full_penalty[neg_powernoback_full_penalty<0]),axis=0)[-np.sum(neg_powernoback_full_penalty<0)//10:])
					neg_powernoback_full_penalty = neg_powernoback_full_penalty.reshape(neg_m_penalty.shape)
				else:
					neg_powernoback_full_penalty = np.zeros_like(neg_m_penalty)
				# neg_powernoback_full_penalty = np.zeros_like(powernoback_full[:,:,:,0])	# emissivity, thickness, rec_diffusivity,coord_1,coord_2
				likelihood = np.transpose(np.transpose(neg_powernoback_full_penalty, (1,2,3,4,0)) + emissivity_log_prob, (4,0,1,2,3))
				likelihood = np.transpose(np.transpose(likelihood, (0,2,3,4,1)) + thickness_log_prob, (0,4,1,2,3))
				likelihood = np.transpose(np.transpose(likelihood, (0,1,3,4,2)) + rec_diffusivity_log_prob, (0,1,4,2,3))
				likelihood += neg_m_penalty
				likelihood += edge_penalty
				# likelihood = np.sum(likelihood, axis=-3)
				likelihood = likelihood -np.log(np.trapz(np.trapz(np.trapz(np.exp(likelihood),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0))	# normalisation for logarithmic probabilities
				total_volume = np.trapz(np.trapz(np.trapz(np.ones((emissivity_steps,thickness_steps,rec_diffusivity_steps)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)
				final_emissivity = np.trapz(np.trapz(np.trapz(np.exp(likelihood)*(m.reshape(neg_m_penalty.shape)),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0)


				voxels_centre = np.mean(grid_data_masked_crop,axis=1)
				dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
				dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
				dist_mean = dz**2 + dr**2
				recompose_voxel_emissivity = np.zeros((len(final_emissivity),len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
				for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
					for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
						dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
						if dist.min()<dist_mean/2:
							index = np.abs(dist).argmin()
							recompose_voxel_emissivity[:,i_r,i_z] = final_emissivity[:,index]

				# plt.figure()
				# # plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=final_emissivity[20],s=100,marker='s',cmap='rainbow')
				# plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2),axis=2),axis=1)[20],extent=[np.unique(voxels_centre[:,0]).min(),np.unique(voxels_centre[:,0]).max(),np.unique(voxels_centre[:,1]).min(),np.unique(voxels_centre[:,1]).max()])
				# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
				# # temp = np.abs(efit_reconstruction.time-time_full_binned[i_t]).argmin()
				# # for i in range(len(all_time_sep_r[temp])):
				# # 	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
				# # plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
				# # plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
				# plt.colorbar().set_label('emissivity [W/m3]')
				# plt.pause(0.01)



				# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				ani,trash = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned))), extent = extent, image_extent=image_extent,timesteps=time_full_binned,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n',overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
				ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_reconstruct_emissivity2.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned))))/383, writer='ffmpeg',codec='mpeg4')
				plt.close()
				# plt.pause(0.01)


				temp = np.sum(likelihood,axis=(-1,-2))/(likelihood.shape[-1]*likelihood.shape[-2])
				temp = temp -np.log(np.trapz(np.trapz(np.trapz(np.exp(temp),x=emissivity_array,axis=0),x=thickness_array,axis=0),x=rec_diffusivity_array,axis=0))	# normalisation for logarithmic probabilities
				plt.figure()
				plt.imshow(np.trapz(np.exp(temp),x=rec_diffusivity_array,axis=-1))
				plt.pause(0.01)

				plt.figure()
				plt.imshow(np.trapz(np.exp(temp),x=emissivity_array,axis=0))
				plt.pause(0.01)


				d_simple=powernoback_full[-1,4,4].reshape(powernoback_full.shape[-3],powernoback_full.shape[-2]*powernoback_full.shape[-1])
				b_simple = np.dot(sensitivities_binned_crop.T,d_simple.T)
				m_simple = np.dot(a1_inv, b_simple).T

				recompose_voxel_emissivity_simple = np.zeros((len(m_simple),len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))
				for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
					for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
						dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
						if dist.min()<dist_mean/2:
							index = np.abs(dist).argmin()
							recompose_voxel_emissivity_simple[:,i_r,i_z] = m_simple[:,index]

				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity_simple,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
				plt.pause(0.01)

				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity_simple-recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
				plt.pause(0.01)

				recompose_voxel_emissivity_comparison = np.zeros((recompose_voxel_emissivity.shape[0],recompose_voxel_emissivity.shape[1]*2,recompose_voxel_emissivity.shape[2]))
				recompose_voxel_emissivity_comparison[:,:recompose_voxel_emissivity.shape[1],:] = recompose_voxel_emissivity
				recompose_voxel_emissivity_comparison[:,-recompose_voxel_emissivity.shape[1]:,:] = recompose_voxel_emissivity_simple

				ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity_comparison,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned))),timesteps=time_full_binned,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',extvmin=0,extvmax=4e4)
				# plt.pause(0.01)
				ani.save(path_power_output + '/' + str(shot_number)+'_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '_reconstruct_emissivity.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned))))/383, writer='ffmpeg',codec='mpeg4')
				plt.close()

				m2 = jacobi(A,b,N=25,x=m,tol=1e-6)
				def iverative(A,b,N=25,x=None,tol=1e-2,method='GS',relax=None):
					"""
					Solves the equation Ax=b via the Jacobi iterative method.
					"""
					# Create an initial guess if needed
					if x is None:
						x = np.zeros(len(A[0]))

					# Create a vector of the diagonal elements of A
					# and subtract them from A
					if method == 'J':
						D = np.diag(A)
						R = A - np.diagflat(D)
						rec_D = np.diagflat(1/D)
						# J = np.dot(rec_D,R)
						iter_matrix = np.dot(rec_D,R)
						fixed_coeff = np.dot(np.diagflat(1/D),b)
					if method == 'GS':
						D = np.diagflat(np.diag(A))
						E = np.tril(A) - D
						F = np.triu(A) - D
						D_E_1 = np.linalg.inv(D-E)
						# L1 = np.dot(D_E_1,F)
						iter_matrix = np.dot(D_E_1,F)
						fixed_coeff = np.dot(D_E_1,b)
					if method == 'SOR':
						D = np.diagflat(np.diag(A))
						E = np.tril(A) - D
						F = np.triu(A) - D
						D_E_1 = np.linalg.inv(D-relax*E)
						# L1 = np.dot(D_E_1,F)
						iter_matrix = np.dot(D_E_1,(1-relax)*D + relax*F)
						fixed_coeff = np.dot(D_E_1,relax*b)

					err = []
					iteration_number = 1
					converged = False
					# Iterate for N times
					for i in range(N):
						# x1 = (b - np.dot(R,x)) / D
						x1 = np.dot(iter_matrix,x) + fixed_coeff

						err.append( np.linalg.norm(x-x1) / np.linalg.norm(x) )
						if iteration_number >= 2:
							x1[x1<0]=0
							print('iteration '+str(iteration_number)+' relative error %.3g over %.3g' %(np.abs(err[-1] - err[-2])/err[-2],tol))
							if np.abs(err[-1] - err[-2])/err[-2] < tol:
								if verbose:
									print('   Reached convergence criterion after {:d} iterations.'.format(iteration_number))
								converged = True
								x=x1
								break
						iteration_number += 1
						x=x1

					return x


				plt.figure()
				plt.imshow(rotate(np.sum(sensitivities_binned_crop*m2,axis=1).reshape(np.shape(powernoback_full[4,4,4,10])),-90),'rainbow',origin='lower')
				plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
				plt.colorbar()
				plt.pause(0.01)

			else:
				# alternative method using a function that takes the likelihood based on the emissivity and finds the peak of that
				# I think that using the log prob or prob is the same, from the point of view of finding the peak, so I use the log prob because is softer than the prob itself

				# I start with the previous method just to get a decent guess
				# no, this method is strong enough that I can start from scratch
				if True:
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
					sigma_emissivity = 1e5	# this is completely arbitrary
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

				if False:	# This seems actually unnecessary
					powernoback = np.ones((powernoback_full_orig.shape[1]*powernoback_full_orig.shape[2]))
					sigma_powernoback = sigma_powernoback_full[0].flatten()	# sigma when I have the least signal
					sigma_powernoback_2 = sigma_powernoback**2
					def prob_and_gradient_preliminary(emissivity,*powernoback):
						emissivity[emissivity==0] = 1e-10
						foil_power_guess = np.dot(sensitivities_binned_crop,emissivity)
						foil_power_error = powernoback - foil_power_guess
						emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
						Z_derivate = np.dot(grid_Z_derivate_masked_crop_scaled,emissivity)
						R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)

						likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
						likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
						likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian*np.logical_not(selected_super_x_cells) /sigma_emissivity)**2))
						likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian*selected_super_x_cells /sigma_emissivity)**2))
						likelihood_emissivity_edge_laplacian = (regolarisation_coeff_edge**2)* np.sum(((emissivity_laplacian*selected_edge_cells_for_laplacian /sigma_emissivity)**2))
						likelihood_emissivity_edge = np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
						likelihood_emissivity_central_border_Z_derivate = (regolarisation_coeff_central_border_Z_derivate**2)* np.sum((Z_derivate*selected_central_border_cells/sigma_emissivity)**2)
						likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate*selected_central_column_border_cells/sigma_emissivity)**2)
						likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge_laplacian + likelihood_emissivity_edge + likelihood_emissivity_central_border_Z_derivate + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_laplacian_superx
						likelihood = likelihood

						likelihood_power_fit_derivate = -2*np.dot((foil_power_error/sigma_powernoback_2),sensitivities_binned_crop)
						likelihood_emissivity_pos_derivate = 2*(np.minimum(0.,emissivity)**2)/emissivity/sigma_emissivity_2
						likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*np.logical_not(selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
						likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
						likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge**2) * np.dot(emissivity_laplacian*selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
						likelihood_emissivity_edge_derivate = 2*emissivity*selected_edge_cells/sigma_emissivity_2
						likelihood_emissivity_central_border_Z_derivate_derivate = 2*(regolarisation_coeff_central_border_Z_derivate**2)*np.dot(Z_derivate*selected_central_border_cells,grid_Z_derivate_masked_crop_scaled)/sigma_emissivity_2
						likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate*selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
						likelihood_derivate = likelihood_power_fit_derivate + likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_edge_laplacian_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_border_Z_derivate_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate + likelihood_emissivity_laplacian_derivate_superx
						# likelihood_derivate = likelihood_emissivity_central_border_derivate
						# print([likelihood,likelihood_derivate.max(),likelihood_derivate.min()])
						return likelihood,likelihood_derivate

					m_homogeneous_offset, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient_preliminary, x0=np.ones((len(grid_laplacian_masked_crop))), args = (powernoback), disp=1, factr=1e5, pgtol=5e-8)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)

				else:
					pass

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
				outer_leg_tot_rad_power_all = []
				inner_leg_tot_rad_power_all = []
				core_tot_rad_power_all = []
				x_point_tot_rad_power_all = []
				time_per_iteration = []
				score_x_all = []
				score_y_all = []
				Lcurve_curvature_all = []
				for i_t in range(len(time_full_binned_crop)):
					time_start = tm.time()

					print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
					# plt.figure()
					# plt.imshow(powernoback_full_orig[i_t])
					# plt.colorbar()
					# plt.pause(0.01)
					#
					# plt.figure()
					# plt.imshow(sigma_powernoback_full[i_t])
					# plt.colorbar()
					# plt.pause(0.01)

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

					if False:
						def prob_and_gradient(emissivity,sensitivities_binned_crop,powernoback,sigma_powernoback,selected_edge_cells,regolarisation_coeff,regolarisation_coeff_edge):
							# print('called')
							# the algorithm will minimize what I give to it, so this likelihood is the negative of the real one.
							sensitivities_binned_crop_times_emissivity = sensitivities_binned_crop*emissivity
							foil_power_guess = np.sum(sensitivities_binned_crop_times_emissivity,axis=1)

							likelihood_power_fit = np.sum(((powernoback - foil_power_guess)/sigma_powernoback)**2)
							# plt.figure()
							# plt.imshow((((powernoback - foil_power_guess)/sigma_powernoback)**2).reshape(powernoback_full_orig[i_t].shape))
							# plt.colorbar()
							# plt.pause(0.01)

							# plt.figure()
							# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=sensitivities_binned_crop[62],marker='s')
							# plt.colorbar()
							# plt.pause(0.01)

							# likelihood_power_pos = np.sum((np.minimum(0.,foil_power_guess)/sigma_powernoback)**2)
							# likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
							# likelihood_emissivity_edge = np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
							# # likelihood_emissivity_laplacian = np.sum((regolarisation_coeff**2)*grid_laplacian_masked_crop_2 * ((emissivity/sigma_emissivity)**2))
							# likelihood_emissivity_laplacian = np.sum((regolarisation_coeff*grid_laplacian_masked_crop*emissivity/sigma_emissivity)**2)
							# likelihood_emissivity_edge_laplacian = np.sum((regolarisation_coeff_edge*grid_laplacian_masked_crop*emissivity*selected_edge_cells/sigma_emissivity)**2)
							likelihood_power_pos = 0
							likelihood_emissivity_pos = 0
							likelihood_emissivity_edge = 0
							likelihood_emissivity_laplacian = 0
							likelihood_emissivity_edge_laplacian = 0
							likelihood = likelihood_power_fit + likelihood_power_pos + likelihood_emissivity_pos + likelihood_emissivity_edge + likelihood_emissivity_laplacian + likelihood_emissivity_edge_laplacian
							print([likelihood,likelihood_power_fit,likelihood_power_pos,likelihood_emissivity_pos,likelihood_emissivity_edge,likelihood_emissivity_laplacian,likelihood_emissivity_edge_laplacian])

							likelihood_power_fit_derivate = -np.sum(((powernoback-sensitivities_binned_crop_times_emissivity.T)/sigma_powernoback_2).T*sensitivities_binned_crop,axis=0)
							# likelihood_power_pos_derivate = np.sum((np.minimum(0.,sensitivities_binned_crop_times_emissivity)**2/emissivity).T/sigma_powernoback_2,axis=1)
							# likelihood_emissivity_pos_derivate = np.minimum(0.,emissivity)**2/emissivity/sigma_emissivity_2
							# likelihood_emissivity_edge_derivate = emissivity*selected_edge_cells/sigma_emissivity_2
							# # likelihood_emissivity_laplacian_derivate = (regolarisation_coeff**2)*np.dot(grid_laplacian_masked_crop_2,emissivity)/sigma_emissivity_2
							# likelihood_emissivity_laplacian_derivate = (regolarisation_coeff**2)*np.dot(grid_laplacian_masked_crop**2,emissivity)/sigma_emissivity_2
							# likelihood_emissivity_edge_laplacian_derivate = (regolarisation_coeff_edge**2)*np.dot(grid_laplacian_masked_crop**2,emissivity*selected_edge_cells)/sigma_emissivity_2
							likelihood_power_pos_derivate = 0
							likelihood_emissivity_pos_derivate = 0
							likelihood_emissivity_edge_derivate = 0
							likelihood_emissivity_laplacian_derivate = 0
							likelihood_emissivity_edge_laplacian_derivate = 0
							likelihood_derivate = 1e5*(likelihood_power_fit_derivate + likelihood_power_pos_derivate + likelihood_emissivity_pos_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_edge_laplacian_derivate)
							print([likelihood_derivate.max(),likelihood_derivate.min()])#,likelihood_power_fit_derivate.max(),likelihood_power_pos_derivate.max(),likelihood_emissivity_pos_derivate.max(),likelihood_emissivity_edge_derivate.max(),likelihood_emissivity_laplacian_derivate.max(),likelihood_emissivity_edge_laplacian_derivate.max()])
							# print([likelihood,likelihood_derivate])
							return likelihood,likelihood_derivate

					elif False:
						if time_full_binned_crop[i_t]<0.2:
							if False:	# I can start from a random sample no problem
								guess = final_emissivity+(np.random.random(final_emissivity.shape)-0.5)*1e3	# the 1e3 is to make things harder for the solver, but also to go away a bit from the best fit of other functions
								guess[selected_edge_cells] = guess[selected_edge_cells]*1e-4
								guess = np.concatenate((guess,[0.1/homogeneous_scaling,0.1/homogeneous_scaling]))
							else:
								guess = np.random.random(sensitivities_binned_crop.shape[1]+2)*1e2
						else:
							guess = cp.deepcopy(x_optimal)
							guess[:-2] += (np.random.random(x_optimal.shape[0]-2)-0.5)*1e3	# I still add a bit of scramble to give it the freedom to find the best configuration
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

						if time_full_binned_crop[i_t]<0.1:
							x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-6)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						else:
							x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), iprint=0, factr=1e0, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						# if opt_info['warnflag']>0:
						# 	print('incomplete fit so restarted')
						# 	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=x_optimal, args = (powernoback), disp=1, factr=1e0, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						# x_optimal[-2:] *= homogeneous_scaling
						x_optimal[-2:] *= np.array([homogeneous_scaling,homogeneous_scaling])

						foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal
						foil_power_error = powernoback - foil_power_guess
						chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
						print('chi_square '+str(chi_square))
					elif False:	# here I make the regularisation coefficient adaptive such that chi square = number of degrees of freedom = number of voxels
						target_chi_square = 820	# obtained doing a scan of the regularisation coefficient. this was the result for regolarisation_coeff~1e-3
						target_chi_square_sigma = 2	# this should be tight, because for such a high number of degrees of freedom things should average very well
						regolarisation_coeff_TO_regolarisation_coeff_divertor = 1/1.5

						if time_full_binned_crop[i_t]<0.2:
							if False:	# I can start from a random sample no problem
								guess = final_emissivity+(np.random.random(final_emissivity.shape)-0.5)*1e3	# the 1e3 is to make things harder for the solver, but also to go away a bit from the best fit of other functions
								guess[selected_edge_cells] = guess[selected_edge_cells]*1e-4
								guess = np.concatenate((guess,[1e-3/(homogeneous_scaling**1.5),0.1/homogeneous_scaling,0.1/homogeneous_scaling]))
							else:
								guess = np.random.random(sensitivities_binned_crop.shape[1]+3)*1e2
								guess[-3] = 1e-3/(homogeneous_scaling**1.5)
						else:
							guess = cp.deepcopy(x_optimal)
							guess[:-3] += (np.random.random(x_optimal.shape[0]-3)-0.5)*1e2	# I still add a bit of scramble to give it the freedom to find the best configuration
							guess[-3:] /= np.array([1*(homogeneous_scaling**1.5),homogeneous_scaling,homogeneous_scaling])

						def prob_and_gradient(emissivity_plus,*powernoback):
							homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
							homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
							regolarisation_coeff = emissivity_plus[-3]*(homogeneous_scaling**1.5)	# scaling added such that all variables have the same order of magnitude
							print(regolarisation_coeff)
							regolarisation_coeff_divertor = regolarisation_coeff*regolarisation_coeff_TO_regolarisation_coeff_divertor
							# print(homogeneous_offset,homogeneous_offset_plasma)
							emissivity = emissivity_plus[:-3]
							emissivity[emissivity==0] = 1e-10
							foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal_no_plasma*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
							foil_power_error = powernoback - foil_power_guess
							emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
							Z_derivate = np.dot(grid_Z_derivate_masked_crop_scaled,emissivity)
							R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)

							likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
							likelihood_chi_square = ((likelihood_power_fit-target_chi_square)/target_chi_square_sigma)**2
							likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
							likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian*np.logical_not(selected_super_x_cells) /sigma_emissivity)**2))
							likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian*selected_super_x_cells /sigma_emissivity)**2))
							likelihood_emissivity_edge_laplacian = (regolarisation_coeff_edge**2)* np.sum(((emissivity_laplacian*selected_edge_cells_for_laplacian /sigma_emissivity)**2))
							likelihood_emissivity_edge = np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
							likelihood_emissivity_central_border_Z_derivate = (regolarisation_coeff_central_border_Z_derivate**2)* np.sum((Z_derivate*selected_central_border_cells/sigma_emissivity)**2)
							likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate*selected_central_column_border_cells/sigma_emissivity)**2)
							likelihood = likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge_laplacian + likelihood_emissivity_edge + likelihood_emissivity_central_border_Z_derivate + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_laplacian_superx + likelihood_chi_square + likelihood_power_fit
							likelihood_homogeneous_offset = (homogeneous_offset/reference_sigma_powernoback)**2
							likelihood_homogeneous_offset_plasma = (homogeneous_offset_plasma/reference_sigma_powernoback)**2
							# likelihood_regolarisation_coeff =0 -np.log(regolarisation_coeff**2)
							likelihood_regolarisation_coeff =0 +0.1/regolarisation_coeff	# the 0.1 acts as a weight of this penalty
							likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma + likelihood_regolarisation_coeff

							likelihood_power_fit_derivate = np.concatenate((-2*np.dot((foil_power_error/sigma_powernoback_2),sensitivities_binned_crop),[0],[-2*np.sum(foil_power_error*select_foil_region_with_plasma/sigma_powernoback_2)*homogeneous_scaling,-2*np.sum(foil_power_error*selected_ROI_internal_no_plasma/sigma_powernoback_2)*homogeneous_scaling]))
							likelihood_chi_square_derivate = likelihood_power_fit_derivate * 2 *(likelihood_power_fit-target_chi_square)/(target_chi_square_sigma**2)
							likelihood_emissivity_pos_derivate = 2*(np.minimum(0.,emissivity)**2)/emissivity/sigma_emissivity_2
							likelihood_emissivity_laplacian_derivate = np.concatenate((2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*np.logical_not(selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2),[2*likelihood_emissivity_laplacian/regolarisation_coeff * (homogeneous_scaling**1.5)],[0],[0]))
							likelihood_emissivity_laplacian_derivate_superx = np.concatenate((2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian*selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2),[2*likelihood_emissivity_laplacian_superx/regolarisation_coeff_divertor * (homogeneous_scaling**1.5) * regolarisation_coeff_TO_regolarisation_coeff_divertor],[0],[0]))
							likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge**2) * np.dot(emissivity_laplacian*selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
							likelihood_emissivity_edge_derivate = 2*emissivity*selected_edge_cells/sigma_emissivity_2
							likelihood_emissivity_central_border_Z_derivate_derivate = 2*(regolarisation_coeff_central_border_Z_derivate**2)*np.dot(Z_derivate*selected_central_border_cells,grid_Z_derivate_masked_crop_scaled)/sigma_emissivity_2
							likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate*selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
							likelihood_derivate = likelihood_emissivity_pos_derivate + likelihood_emissivity_edge_laplacian_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_central_border_Z_derivate_derivate + likelihood_emissivity_central_column_border_R_derivate_derivate
							likelihood_homogeneous_offset_derivate = 2*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
							likelihood_homogeneous_offset_plasma_derivate = 2*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
							# likelihood_regolarisation_coeff_derivate =0-2*(homogeneous_scaling**1.5)/(regolarisation_coeff)
							likelihood_regolarisation_coeff_derivate =0-0.1*(homogeneous_scaling**1.5)/(regolarisation_coeff**2)
							likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_regolarisation_coeff_derivate,likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_laplacian_derivate_superx + likelihood_chi_square_derivate + likelihood_power_fit_derivate
							# likelihood_derivate = likelihood_emissivity_central_border_derivate
							# print([likelihood,likelihood_derivate.max(),likelihood_derivate.min()])
							return likelihood,likelihood_derivate

						bds = np.array([([-np.inf]*(len(guess)-3))+[1e-10]+[-np.inf]*2,[np.inf]*len(guess)]).T
						# x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, disp=1, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						if time_full_binned_crop[i_t]<0.1:
							x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), disp=1, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						else:
							x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (powernoback), disp=1, factr=1e0, pgtol=1e-7,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						# if opt_info['warnflag']>0:
						# 	print('incomplete fit so restarted')
						# 	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=x_optimal, args = (powernoback), disp=1, factr=1e0, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						# x_optimal[-2:] *= homogeneous_scaling
						x_optimal[-3:] *= np.array([1*(homogeneous_scaling**1.5),homogeneous_scaling,homogeneous_scaling])

						foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-3])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal_no_plasma
						foil_power_error = powernoback - foil_power_guess
						chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
						print('chi_square '+str(chi_square))
					elif True:	# here I want to do the L-curve search
						guess = np.ones(sensitivities_binned_crop.shape[1]+2)*1e3

						regolarisation_coeff_range = 10**np.linspace(-2,-6,num=15)
						regolarisation_coeff_edge = 10
						def prob_and_gradient(emissivity_plus,*args):
							powernoback = args[0]
							sigma_powernoback = args[1]
							sigma_powernoback_2 = sigma_powernoback**2
							sigma_emissivity = args[2]
							sigma_emissivity_2 = sigma_emissivity**2
							regolarisation_coeff = args[3]
							homogeneous_offset = emissivity_plus[-1]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
							homogeneous_offset_plasma = emissivity_plus[-2]*homogeneous_scaling	# scaling added such that all variables have the same order of magnitude
							regolarisation_coeff_divertor = regolarisation_coeff/1
							regolarisation_coeff_central_column_border_R_derivate = regolarisation_coeff*4e2
							regolarisation_coeff_edge_laplacian = regolarisation_coeff*1e1
							# print(homogeneous_offset,homogeneous_offset_plasma)
							emissivity = emissivity_plus[:-2]
							emissivity[emissivity==0] = 1e-10
							foil_power_guess = np.dot(sensitivities_binned_crop,emissivity) + selected_ROI_internal*homogeneous_offset + homogeneous_offset_plasma*select_foil_region_with_plasma
							foil_power_error = powernoback - foil_power_guess
							emissivity_laplacian = np.dot(grid_laplacian_masked_crop_scaled,emissivity)
							R_derivate = np.dot(grid_R_derivate_masked_crop_scaled,emissivity)

							likelihood_power_fit = np.sum((foil_power_error/sigma_powernoback)**2)
							likelihood_emissivity_pos = np.sum((np.minimum(0.,emissivity)/sigma_emissivity)**2)
							likelihood_emissivity_laplacian = (regolarisation_coeff**2)* np.sum(((emissivity_laplacian*np.logical_not(selected_super_x_cells) /sigma_emissivity)**2))
							likelihood_emissivity_laplacian_superx = (regolarisation_coeff_divertor**2)* np.sum(((emissivity_laplacian*selected_super_x_cells /sigma_emissivity)**2))
							likelihood_emissivity_edge_laplacian = (regolarisation_coeff_edge_laplacian**2)* np.sum(((emissivity_laplacian*selected_edge_cells_for_laplacian /sigma_emissivity)**2))
							likelihood_emissivity_edge = (regolarisation_coeff_edge**2)*np.sum((emissivity*selected_edge_cells/sigma_emissivity)**2)
							likelihood_emissivity_central_column_border_R_derivate = (regolarisation_coeff_central_column_border_R_derivate**2)* np.sum((R_derivate*selected_central_column_border_cells/sigma_emissivity)**2)
							likelihood = likelihood_power_fit + likelihood_emissivity_pos + likelihood_emissivity_laplacian + likelihood_emissivity_edge + likelihood_emissivity_laplacian_superx + likelihood_emissivity_central_column_border_R_derivate + likelihood_emissivity_edge_laplacian
							likelihood_homogeneous_offset = 0#(homogeneous_offset/reference_sigma_powernoback)**2
							likelihood_homogeneous_offset_plasma = (homogeneous_offset_plasma/reference_sigma_powernoback)**2
							likelihood = likelihood + likelihood_homogeneous_offset + likelihood_homogeneous_offset_plasma

							likelihood_power_fit_derivate = np.concatenate((-2*np.dot((foil_power_error/sigma_powernoback_2),sensitivities_binned_crop),[-2*np.sum(foil_power_error*select_foil_region_with_plasma/sigma_powernoback_2)*homogeneous_scaling,-2*np.sum(foil_power_error*selected_ROI_internal/sigma_powernoback_2)*homogeneous_scaling]))
							likelihood_emissivity_pos_derivate = 2*(np.minimum(0.,emissivity)**2)/emissivity/sigma_emissivity_2
							likelihood_emissivity_laplacian_derivate = 2*(regolarisation_coeff**2) * np.dot(emissivity_laplacian*np.logical_not(selected_super_x_cells) , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
							likelihood_emissivity_laplacian_derivate_superx = 2*(regolarisation_coeff_divertor**2) * np.dot(emissivity_laplacian*selected_super_x_cells , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
							likelihood_emissivity_edge_laplacian_derivate = 2*(regolarisation_coeff_edge_laplacian**2) * np.dot(emissivity_laplacian*selected_edge_cells_for_laplacian , grid_laplacian_masked_crop_scaled) / (sigma_emissivity**2)
							likelihood_emissivity_edge_derivate = 2*(regolarisation_coeff_edge**2)*emissivity*selected_edge_cells/sigma_emissivity_2
							likelihood_emissivity_central_column_border_R_derivate_derivate = 2*(regolarisation_coeff_central_column_border_R_derivate**2)*np.dot(R_derivate*selected_central_column_border_cells,grid_R_derivate_masked_crop_scaled)/sigma_emissivity_2
							likelihood_derivate = np.zeros_like(emissivity) + likelihood_emissivity_pos_derivate + likelihood_emissivity_laplacian_derivate + likelihood_emissivity_edge_derivate + likelihood_emissivity_laplacian_derivate_superx + likelihood_emissivity_central_column_border_R_derivate_derivate + likelihood_emissivity_edge_laplacian_derivate
							likelihood_homogeneous_offset_derivate = 0#2*homogeneous_offset*homogeneous_scaling/(reference_sigma_powernoback**2)
							likelihood_homogeneous_offset_plasma_derivate = 2*homogeneous_offset_plasma*homogeneous_scaling/(reference_sigma_powernoback**2)
							likelihood_derivate = np.concatenate((likelihood_derivate,[likelihood_homogeneous_offset_plasma_derivate,likelihood_homogeneous_offset_derivate])) + likelihood_power_fit_derivate
							return likelihood,likelihood_derivate

						x_optimal_all = []
						recompose_voxel_emissivity_all = []
						for regolarisation_coeff in regolarisation_coeff_range:
							print(regolarisation_coeff)
							args = [powernoback.astype(float),sigma_powernoback,sigma_emissivity,regolarisation_coeff]
							if time_full_binned_crop[i_t]<0.1:
								x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (args), iprint=2, factr=1e3, pgtol=1e-5)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
							else:
								x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=guess, args = (args), iprint=2, factr=1e3, pgtol=1e-5)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
							x_optimal_all.append(x_optimal)
							guess = x_optimal

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
										recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
							recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi
							recompose_voxel_emissivity_all.append(recompose_voxel_emissivity)

temp_save = dict([])
temp_save['grid_data_masked_crop'] = grid_data_masked_crop
temp_save['powernoback_full_orig'] = powernoback_full_orig
temp_save['sigma_powernoback_full'] = sigma_powernoback_full
temp_save['time_full_binned_crop'] = time_full_binned_crop
temp_save['sensitivities_binned_crop'] = sensitivities_binned_crop
temp_save['selected_ROI_internal'] = selected_ROI_internal
temp_save['select_foil_region_with_plasma'] = select_foil_region_with_plasma
temp_save['grid_laplacian_masked_crop_scaled'] = grid_laplacian_masked_crop_scaled
temp_save['grid_R_derivate_masked_crop_scaled'] = grid_R_derivate_masked_crop_scaled
temp_save['regolarisation_coeff_edge'] = regolarisation_coeff_edge
temp_save['selected_edge_cells'] = selected_edge_cells
temp_save['regolarisation_coeff_central_column_border_R_derivate'] = regolarisation_coeff_central_column_border_R_derivate
temp_save['selected_central_column_border_cells'] = selected_central_column_border_cells
temp_save['selected_central_column_border_cells'] = selected_central_column_border_cells
temp_save['selected_central_column_border_cells'] = selected_central_column_border_cells
temp_save['selected_central_column_border_cells'] = selected_central_column_border_cells
temp_save['selected_central_column_border_cells'] = selected_central_column_border_cells
np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian_test_export',**temp_save)


						regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
						x_optimal_all = np.flip(x_optimal_all,axis=0)
						recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)

						score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
						score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,np.array(x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)

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

						def distance_spread(coord):
							def int(trash,px,py):
								x = coord[0]
								y = coord[1]
								dist = ((x-px)**2 + (y-py)**2)**0.5
								spread = np.sum((dist-np.mean(dist))**2)
								# print(spread)
								return [spread]*5
							return int

						# plt.figure()
						# plt.plot(score_x_record_rel,score_y_record_rel)
						curvature_radious = []
						for ii in range(2,len(score_y_record_rel)-2):
							# try:
							# 	guess = centre[0]
							# except:
							try:
								guess = np.max([score_y_record_rel[ii-2:ii+2+1],score_x_record_rel[ii-2:ii+2+1]],axis=1)
								bds = [[np.min(score_y_record_rel[ii-2:ii+2+1]),np.min(score_x_record_rel[ii-2:ii+2+1])],[np.inf,np.inf]]
								centre = curve_fit(distance_spread([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]]),[0]*5,[0]*5,p0=guess,bounds = bds,maxfev=1e5,gtol=1e-12,verbose=1)
								dist = ((score_x_record_rel[ii-2:ii+2+1]-centre[0][0])**2 + (score_y_record_rel[ii-2:ii+2+1]-centre[0][1])**2)**0.5
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
						curvature_radious = [np.max(curvature_radious)]+curvature_radious+[np.max(curvature_radious)]
						Lcurve_curvature = 1/np.array(curvature_radious)

						recompose_voxel_emissivity = recompose_voxel_emissivity_all[Lcurve_curvature.argmax()+1]
						regolarisation_coeff = regolarisation_coeff_range[Lcurve_curvature.argmax()+1]
						x_optimal = x_optimal_all[Lcurve_curvature.argmax()+1]

						if False:	# only visualisation
							plt.figure()
							plt.plot(regolarisation_coeff_range[1:-1],Lcurve_curvature)
							plt.plot(regolarisation_coeff_range[1:-1],Lcurve_curvature,'+')
							plt.plot(regolarisation_coeff_range[1:-1][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o')
							plt.semilogx()
							plt.pause(0.01)

							plt.figure()
							plt.plot(score_x_record_rel,score_y_record_rel)
							plt.plot(score_x_record_rel,score_y_record_rel,'+')
							plt.xlabel('log ||Gm-d||2')
							plt.ylabel('log ||Laplacian(m)||2')
							plt.grid()
							plt.plot(score_x_record_rel[1:-1][Lcurve_curvature.argmax()],score_y_record_rel[1:-1][Lcurve_curvature.argmax()],'o')
							plt.pause(0.01)



							extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
							image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
							ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(recompose_voxel_emissivity_all,(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=regolarisation_coeff_range,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nregolarisation_coeff_divertor %.3g\ngrid resolution %.3g\n' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,regolarisation_coeff_divertor,grid_resolution) ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
							plt.pause(0.01)
							plt.close()


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
							plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
							plt.colorbar().set_label('emissivity [W/m3]')
							plt.ylim(top=0.5)
							# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
							plt.pause(0.01)

						# if opt_info['warnflag']>0:
						# 	print('incomplete fit so restarted')
						# 	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(prob_and_gradient, x0=x_optimal, args = (powernoback), disp=1, factr=1e0, pgtol=1e-7)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
						# x_optimal[-2:] *= homogeneous_scaling
						x_optimal[-2:] *= np.array([homogeneous_scaling,homogeneous_scaling])

						foil_power_guess = np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal
						foil_power_error = powernoback - foil_power_guess
						chi_square = np.sum((foil_power_error/sigma_powernoback)**2)
						print('chi_square '+str(chi_square))



					if False:	# only for testinf the prob_and_gradient function
						target = 832
						scale = 1e-4
						# guess[target] = 1e5
						temp1 = prob_and_gradient(guess,*args)
						guess[target] +=scale
						temp2 = prob_and_gradient(guess,*args)
						guess[target] += -2*scale
						temp3 = prob_and_gradient(guess,*args)
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
					score_x_all.append(score_x)
					score_y_all.append(score_y)
					Lcurve_curvature_all.append(Lcurve_curvature)

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
				Lcurve_curvature_all = np.array(Lcurve_curvature_all)

				path_for_plots = path_power_output + '/invertions_log/'+binning_type
				if not os.path.exists(path_for_plots):
					os.makedirs(path_for_plots)



				plt.figure(figsize=(20, 10))
				plt.plot(time_full_binned_crop,time_per_iteration)
				# plt.semilogy()
				plt.title('time spent per iteration')
				plt.xlabel('time [s]')
				plt.ylabel('time [s]')
				plt.grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_time_trace_bayesian.eps')
				plt.close()

				plt.figure(figsize=(20, 10))
				plt.plot(time_full_binned_crop,inverted_data_likelihood)
				# plt.semilogy()
				plt.title('Fit log likelihood')
				plt.xlabel('time [s]')
				plt.ylabel('log likelihoog [au]')
				plt.grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_likelihood_bayesian.eps')
				plt.close()

				plt.figure(figsize=(20, 10))
				plt.plot(time_full_binned_crop,chi_square_all)
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
				plt.plot(time_full_binned_crop,regolarisation_coeff_all)
				# plt.semilogy()
				plt.title('regolarisation coefficient obtained')
				plt.semilogy()
				plt.xlabel('time [s]')
				plt.ylabel('regolarisation coefficient [au]')
				plt.grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_regolarisation_coeff_bayesian.eps')
				plt.close()

				plt.figure(figsize=(20, 10))
				plt.plot(time_full_binned_crop,fit_error)
				# plt.semilogy()
				plt.title('Fit error ( sum((image-fit)^2)^0.5/num pixels )')
				plt.xlabel('time [s]')
				plt.ylabel('average fit error [W/m2]')
				plt.grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_fit_error_bayesian.eps')
				plt.close()

				plt.figure(figsize=(20, 10))
				plt.plot(time_full_binned_crop,inverted_data_plasma_region_offset,label='plasma region')
				plt.plot(time_full_binned_crop,inverted_data_homogeneous_offset,label='whole foil')
				plt.title('Offsets to match foil power')
				plt.legend(loc='best', fontsize='x-small')
				plt.xlabel('time [s]')
				plt.ylabel('power density [W/m2]')
				plt.grid()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_offsets_bayesian.eps')
				plt.close()

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
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_tot_rad_power_bayesian.eps')
				plt.close()

				# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(recompose_voxel_emissivity,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))),integration=laser_int_time/1000,barlabel='Emissivity [W/m3]')#,extvmin=0,extvmax=4e4)
				extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				image_extent = [grid_data_masked_crop[:,:,0].min(), grid_data_masked_crop[:,:,0].max(), grid_data_masked_crop[:,:,1].min(), grid_data_masked_crop[:,:,1].max()]
				ani,trash = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(inverted_data,(0,2,1)),axis=2)]), 1/(np.mean(np.diff(time_full_binned_crop))), extent = extent, image_extent=image_extent,timesteps=time_full_binned_crop,integration=laser_int_time/1000,barlabel='Emissivity [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n'+'sigma_emissivity %.3g' %(sigma_emissivity) ,overlay_structure=True,include_EFIT=True,EFIT_output_requested=True,efit_reconstruction=efit_reconstruction,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=True,overlay_strike_points=True,overlay_separatrix=True)#,extvmin=0,extvmax=4e4)
				ani.save(path_power_output + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian.mp4', fps=5*(1/(np.mean(np.diff(time_full_binned_crop))))/383, writer='ffmpeg',codec='mpeg4')
				plt.close()

				plt.figure()
				plt.imshow(Lcurve_curvature_all,'rainbow',origin='lower')
				plt.colorbar()
				plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_bayesian.eps')
				plt.close()

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
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['outer_leg_tot_rad_power_all'] = outer_leg_tot_rad_power_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['inner_leg_tot_rad_power_all'] = inner_leg_tot_rad_power_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['core_tot_rad_power_all'] = core_tot_rad_power_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['x_point_tot_rad_power_all'] = x_point_tot_rad_power_all
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry'] = dict([])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['R'] = np.unique(voxels_centre[:,0])
			inverted_dict[str(grid_resolution)][str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['Z'] = np.unique(voxels_centre[:,1])

np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian_test',**inverted_dict)
# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_power_to_emissivity2.py").read())
