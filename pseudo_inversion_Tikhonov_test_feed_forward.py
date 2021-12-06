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
	# i_day,day = 0,'2021-06-24'
	# name='IRVB-MASTU_shot-44308.ptw'
	# i_day,day = 0,'2021-08-13'
	# name='IRVB-MASTU_shot-44677.ptw'
	i_day,day = 0,'2021-10-12'
	name='IRVB-MASTU_shot-45239.ptw'
	laser_to_analyse=path+day+'/'+name

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


# feed forward example in which I start from a known profile and return to it via the inversion

inverted_dict = dict([])

# for grid_resolution in [8, 2]:
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

	if False:
		plt.figure()
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.mean(sensitivities_reshaped,axis=(0,1)),marker='s')
		plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.std(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
		# plt.scatter(np.mean(grid_data,axis=1)[:,0],np.mean(grid_data,axis=1)[:,1],c=np.sum(sensitivities_reshaped,axis=(0,1)),marker='s',norm=LogNorm())
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.mean(sensitivities_reshaped_masked,axis=(0,1)),marker='s')
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.std(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
		# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_reshaped_masked,axis=(0,1)),marker='s',norm=LogNorm())
		plt.colorbar()
		plt.pause(0.01)

	# this step is to adapt the matrix to the size of the foil I measure, that can be slightly different
	binning_type = 'bin' + str(all_shrink_factor_t[0]) + 'x' + str(all_shrink_factor_x[0]) + 'x' + str(all_shrink_factor_x[0])
	shape = list(np.array(saved_file_dict_short[binning_type].all()['powernoback_full'].shape[1:])+2)	# +2 spatially because I remove +/- 1 pixel when I calculate the laplacian of the temperature
	shape.extend([len(grid_laplacian_masked)])
	def mapping(output_coords):
		return(output_coords[0]/shape[0]*pixel_h,output_coords[1]/shape[1]*pixel_v,output_coords[2])
	sensitivities_reshaped_masked2 = geometric_transform(sensitivities_reshaped_masked,mapping,output_shape=shape)

	if False:
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
	for shrink_factor_x in all_shrink_factor_x:
		inverted_dict[str(grid_resolution)][str(shrink_factor_x)] = dict([])
		sensitivities_binned = coleval.proper_homo_binning_1D_1D_1D(sensitivities_reshaped_masked2,shrink_factor_x,shrink_factor_x,1,type='np.nanmean')
		sensitivities_binned = sensitivities_binned[1:-1,1:-1]	# i need to remove 2 pixels per coordinate because this is done to calculate the lalacian
		sensitivities_binned = np.flip(sensitivities_binned,axis=1)	# it turns ou that I need to flip it

		if False:

			plt.figure()
			plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned,axis=(0,1)),marker='s')
			plt.colorbar()
			plt.pause(0.01)

		# additional cropping of the foil to exlude regions without plasma LOS, the frame of the foil and gas puff
		foil_shape = np.shape(sensitivities_binned)[:-1]
		# ROI = np.array([[0.2,0.85],[0.1,0.9]])
		# ROI = np.round((ROI.T*foil_shape).T).astype(int)
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

		# radiator fixed at r=0.6m and z=-1.2m
		# size of 20cm
		radiation_profile = 1e6 * np.logical_and(np.abs(np.mean(grid_data_masked,axis=1)[:,0] - 0.6)<0.1 , np.abs(np.mean(grid_data_masked,axis=1)[:,1] - (-1.2))<0.1)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=radiation_profile,marker='s')
		plt.colorbar()
		plt.pause(0.01)

		radiation_profile_on_foil = np.sum(sensitivities_binned*radiation_profile,axis=-1)
		plt.figure()
		plt.imshow(np.flip(np.transpose(radiation_profile_on_foil,(1,0)),axis=1),'rainbow',origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		if True:	# setting zero to the sensitivities I want to exclude
			sensitivities_binned_crop = cp.deepcopy(sensitivities_binned)
			sensitivities_binned_crop[np.logical_not(selected_ROI),:] = 0
		else:	# cutting sensitivity out of ROI
			sensitivities_binned_crop = sensitivities_binned[sensitivities_binned.shape[0]-ROI[0,1]:sensitivities_binned.shape[0]-ROI[0,0],ROI[1,0]:ROI[1,1]]

		plt.figure()
		plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned_crop,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=np.sum(sensitivities_binned_crop,axis=(0,1)),marker='s')
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






		weight1 = np.ones((len(radiation_profile_on_foil.flatten())))	# I'm not sure if I can even use weights, so I set them at 1
		weight2 = np.ones((len(grid_laplacian_masked_crop)))	# I'm not sure if I can even use weights, so I set them at 1
		# alpha_exponents_to_test = np.unique(np.concatenate([np.arange(-10,0,0.25).astype(float),np.arange(-1.5,0,0.1).astype(float)]))
		if grid_resolution==8:
			alpha_exponents_to_test = np.arange(-6,0.5,0.25).astype(float)
		elif grid_resolution==2:
			alpha_exponents_to_test = np.arange(-9,5,0.25).astype(float)

		regen_inverse_anyway = True

		def calculate_SVG_with_Tikonov_reg(arg, sensitivities=sensitivities_binned_crop, laplacian=grid_laplacian_masked_crop, d=radiation_profile_on_foil.flatten(),weight1=weight1,weight2=weight2,path_sensitivity_original=path_sensitivity_original,regen_inverse_anyway=regen_inverse_anyway):
			index_ext = arg[0]
			exp = arg[1]
			alpha=1.*10**(exp)
			if exp==-50:
				alpha=0
			# test_matrix = np.dot(sensitivities.T, sensitivities) + (alpha**2) * laplacian
			test_matrix = np.dot(sensitivities.T, sensitivities) + (alpha**2) * np.dot(laplacian.T, laplacian)

			if (not os.path.exists(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp) + '.npz')) or regen_inverse_anyway:
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

				if not os.path.exists(path_sensitivity_original):
					os.makedirs(path_sensitivity_original)
				np.savez_compressed(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp), a1_inv=a1_inv,s=s)
				m = np.dot(a1_inv, np.dot(sensitivities.T, d))
			else:
				try:
					a1_inv = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp) + '.npz')['a1_inv']
					s = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp) + '.npz')['s']
					print('the SVG decomposition of alpha = ' + str(alpha) + ' is imported')
					m = np.dot(a1_inv, np.dot(sensitivities.T, d))
				except:
					try:
						print('Trying to recover SVG decomposition ' + path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp))
						U, s, Vh = np.linalg.svd(test_matrix)
						sigma = np.diag(s)
						inv_sigma = np.diag(1 / s)
						a1 = np.dot(U, np.dot(sigma, Vh))
						a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))

						if not os.path.exists(path_sensitivity_original):
							os.makedirs(path_sensitivity_original)
						np.savez_compressed(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp), a1_inv=a1_inv,s=s)
						print('stored SVG decomposition ' + path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp) + ' recovered')
					except:
						print('the SVG decomposition of alpha = ' + str(alpha) + ' did not converge')
						# continue
						output = calc_stuff_output(exp, alpha, np.nan, np.nan)
						return output
					m = np.dot(a1_inv, np.dot(sensitivities.T, d))


			# m = np.dot(a1_inv, np.dot(sensitivities.T, d))
			# plt.figure()
			# plt.scatter(np.mean(grid_data_masked,axis=1)[:,0],np.mean(grid_data_masked,axis=1)[:,1],c=m,marker='s')
			# plt.pause(0.01)
			#
			#
			# plt.figure()
			# plt.imshow(np.flip(np.transpose(powernoback_full[i],(1,0)),axis=1),'rainbow',origin='lower')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop_binned[:,:,:180],axis=(-1,-2))[temp])
			# plt.pause(0.01)

			print('alpha='+str(alpha))
			score_x = np.sum(((np.dot(sensitivities, m) - d) ** 2) * weight1)
			print('||Gm-d||2=' + str(score_x))
			score_y = np.sum(((np.dot(laplacian,m)) ** 2) * weight2)
			print('||Laplacian(m)||2=' + str(score_y))

			output = calc_stuff_output(exp, alpha, score_x, score_y)

			return output

		pool = Pool(number_cpu_available)
		# pool = Pool(1)

		composed_array = [*pool.map(calculate_SVG_with_Tikonov_reg, enumerate(alpha_exponents_to_test))]
		# composed_array = set(composed_array)
		print('np.shape(composed_array)' + str(np.shape(composed_array)))

		pool.close()
		pool.join()

		composed_array = list(composed_array)


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

		if True:	# old check
			print('Alpha that made possible SVG decomposition ')
			print(str(alpha_record.tolist()))
			print('record of the residuals on solution ')
			print(str(score_x_record.tolist()))
			print('record of the residuals on smoothing ')
			print(str(score_y_record.tolist()))

		if False:
			score_y_record_derivative = np.gradient(np.log(score_y_record),alpha_record)[1:-1]
			# score_y_record_derivative = (score_y_record[2:] - score_y_record[:-2]) / (alpha_record[2:] - alpha_record[:-2])
			# Lcurve_curvature = -(((alpha_record[1:-1] ** 2) * score_y_record[1:-1] * (score_x_record[1:-1] ** 2)) + 2 * (alpha_record[1:-1] * (score_y_record[1:-1] ** 2) * (score_x_record[1:-1] ** 2) / score_y_record_derivative) + ((alpha_record[1:-1] ** 4) * (score_y_record[1:-1] ** 2) * score_x_record[1:-1])) / ((((alpha_record[1:-1] ** 4) * (score_y_record[1:-1] ** 2)) + (score_x_record[1:-1] ** 2)) ** (3 / 2))
			Lcurve_curvature = np.abs(((alpha_record[1:-1] ** 2) * score_y_record[1:-1] * (score_x_record[1:-1] ** 2)) + 2 * (alpha_record[1:-1] * (score_y_record[1:-1] ** 2) * (score_x_record[1:-1] ** 2) / score_y_record_derivative) + ((alpha_record[1:-1] ** 4) * (score_y_record[1:-1] ** 2) * score_x_record[1:-1])) / ((((alpha_record[1:-1] ** 4) * (score_y_record[1:-1] ** 2)) + (score_x_record[1:-1] ** 2)) ** (3 / 2))
		elif False:
			score_y_record = np.log(score_y_record)
			score_x_record = np.log(score_x_record)
			# alpha_record2 = np.array(alpha_record)
			alpha_record2 = np.log(alpha_record)
			score_y_record_derivative = np.gradient(score_y_record,alpha_record2)
			score_y_record_second_derivative = np.gradient(score_y_record_derivative,alpha_record2)[1:-1]
			score_y_record_derivative = score_y_record_derivative[1:-1]
			# score_y_record_second_derivative = (score_y_record[2:] -2*score_y_record[1:-1] + score_y_record[:-2]) / (((alpha_record2[2:] - alpha_record2[:-2])/2)**2)
			# score_y_record_second_derivative = (((score_y_record[2:] -score_y_record[1:-1])/(alpha_record2[2:] - alpha_record2[1:-1])) - ((score_y_record[1:-1] - score_y_record[:-2]) / (alpha_record2[1:-1] - alpha_record2[:-2]))) / (np.mean([alpha_record2[2:],alpha_record2[1:-1]],axis=0) - np.mean([alpha_record2[1:-1],alpha_record2[:-2]],axis=0))
			score_x_record_derivative = np.gradient(score_x_record,alpha_record2)
			score_x_record_second_derivative = np.gradient(score_x_record_derivative,alpha_record2)[1:-1]
			score_x_record_derivative = score_x_record_derivative[1:-1]
			# score_x_record_second_derivative = (score_x_record[2:] -2*score_x_record[1:-1] + score_x_record[:-2]) / (((alpha_record2[2:] - alpha_record2[:-2])/2)**2)
			# score_x_record_second_derivative = (((score_x_record[2:] -score_x_record[1:-1])/(alpha_record2[2:] - alpha_record2[1:-1])) - ((score_x_record[1:-1] - score_x_record[:-2]) / (alpha_record2[1:-1] - alpha_record2[:-2]))) / (np.mean([alpha_record2[2:],alpha_record2[1:-1]],axis=0) - np.mean([alpha_record2[1:-1],alpha_record2[:-2]],axis=0))

			Lcurve_curvature = np.abs(score_y_record_derivative*score_x_record_second_derivative - score_y_record_second_derivative*score_x_record_derivative) / ((score_y_record_derivative**2 + score_x_record_derivative**2)**(3/2))
			# Lcurve_curvature = Lcurve_curvature[1:-1]
		elif True:
			# basically fitting a circle on 5 points, rather than looking only at 3 at a time
			score_y_record = np.log(score_y_record)
			score_x_record = np.log(score_x_record)

			score_y_record_rel = (score_y_record-score_y_record.min())/(score_y_record.max()-score_y_record.min())
			score_x_record_rel = (score_x_record-score_x_record.min())/(score_x_record.max()-score_x_record.min())

			def distance_spread(coord):
				def int(trash,px,py):
					x = coord[0]
					y = coord[1]
					dist = ((x-px)**2 + (y-py)**2)**0.5
					spread = np.sum((dist-np.mean(dist))**2)
					# print(spread)
					return [spread]*5
				return int

			curvature_radious = []
			for ii in range(2,len(score_y_record_rel)-2):
				guess = np.max([score_y_record_rel[ii-2:ii+2+1],score_x_record_rel[ii-2:ii+2+1]],axis=1)
				bds = [[np.min(score_y_record_rel[ii-2:ii+2+1]),np.min(score_x_record_rel[ii-2:ii+2+1])],[np.inf,np.inf]]
				centre = curve_fit(distance_spread([score_y_record_rel[ii-2:ii+2+1],score_x_record_rel[ii-2:ii+2+1]]),[0]*5,[0]*5,p0=guess,bounds = bds,maxfev=1e5)
				curvature_radious.append(np.mean(((score_y_record_rel[ii-2:ii+2+1]-centre[0][0])**2 + (score_x_record_rel[ii-2:ii+2+1]-centre[0][1])**2)**0.5))
			curvature_radious = [np.max(curvature_radious)]+curvature_radious+[np.max(curvature_radious)]
			Lcurve_curvature = 1/np.array(curvature_radious)

		# plt.figure()
		# plt.plot(exp_indexes[1:-1], generic_filter(Lcurve_curvature,np.mean,size=5))
		# plt.plot(exp_indexes[1:-1], Lcurve_curvature, 'x')
		# # plt.semilogx()
		# plt.pause(0.01)
		# plt.figure()
		# plt.plot(score_x_record,score_y_record)
		# plt.plot(score_x_record,score_y_record, 'x')
		# plt.semilogx()
		# plt.semilogy()
		# plt.pause(0.01)


		#
		# peaks = find_peaks(Lcurve_curvature)[0]
		# proms = get_proms(Lcurve_curvature,peaks)[0]
		# plt.plot(exp_indexes[1:-1][peaks],proms,'x')

		plt.figure(figsize=(20, 10))
		if False:	# made True 16/12/3019 because it is better to analyse solps max seed case
			fine_alpha = np.logspace(np.log10((alpha_record[:-2][Lcurve_curvature > 0]).min()),np.log10((alpha_record[:-2][Lcurve_curvature > 0]).max()), 1000,endpoint=True)
			# Lcurve_curvature_interp = interp1d(alpha_record, Lcurve_curvature,fill_value='extrapolate', kind = 'cubic')
			# Lcurve_curvature_interp = splrep(np.log10(alpha_record[Lcurve_curvature>0]), np.log10(Lcurve_curvature[Lcurve_curvature>0]), k=3)
			Lcurve_curvature_interp = splrep(np.log10(alpha_record[:-2]),Lcurve_curvature, k=3)
			Lcurve_curvature_interp = splev(np.log10(fine_alpha), Lcurve_curvature_interp)
			plt.plot(fine_alpha, Lcurve_curvature_interp, '--')
			plt.plot(alpha_record[:-2], Lcurve_curvature, 'x')
			# plt.plot(alpha_record[:-2], -Lcurve_curvature, 'y.')
			plt.yscale('log')
			plt.xscale('log')

			index_best_fit = np.array(Lcurve_curvature).argmax()

			# 21/10/2019 lines added to deal with an anbiguous detection of the maximum curvature
			temp = cp.deepcopy(Lcurve_curvature)
			temp[temp<0]=0
			peaks = find_peaks(temp)[0]
			proms = get_proms(temp, peaks)[0]
			peaks = np.array([peaks for _, peaks in sorted(zip(proms, peaks))])
			proms = np.log10(np.sort(proms))
			if ((proms[-2] - proms[-1]) / proms[-1] < 0.2 and peaks[-2] < peaks[-1]):
				plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit ], 'r+', markersize=9)
				index_best_fit = peaks[-2]
				plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit ], 'gx')
				select = np.logical_and(fine_alpha<alpha_record[index_best_fit+1],fine_alpha>alpha_record[index_best_fit-1])
				most_lickely_alpha_index = Lcurve_curvature_interp[select].argmax()
				plt.plot(fine_alpha[select][most_lickely_alpha_index],Lcurve_curvature_interp[select][most_lickely_alpha_index],'go', mfc='none')
				most_lickely_alpha = fine_alpha[select][most_lickely_alpha_index]
			else:
				plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit ], 'gx')
				most_lickely_alpha_index = Lcurve_curvature_interp.argmax()
				plt.plot(fine_alpha[most_lickely_alpha_index],Lcurve_curvature_interp[most_lickely_alpha_index],'go', mfc='none')
				most_lickely_alpha = fine_alpha[most_lickely_alpha_index]
			# plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit-1], 'gx')

		elif False:
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
			peaks = find_peaks(temp)[0]
			proms = get_proms(temp, peaks)[0]
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
		elif False:	# created 2021-08-13
			fine_alpha = np.logspace(np.log10((alpha_record[1:-1][Lcurve_curvature > 0]).min()),np.log10((alpha_record[1:-1][Lcurve_curvature > 0]).max()), 1000,endpoint=True)
			# Lcurve_curvature_interp = interp1d(alpha_record, Lcurve_curvature,fill_value='extrapolate', kind = 'cubic')
			# Lcurve_curvature_interp = splrep(np.log10(alpha_record[Lcurve_curvature>0]), np.log10(Lcurve_curvature[Lcurve_curvature>0]), k=3)
			Lcurve_curvature_interp = splrep(np.log10(alpha_record[1:-1]),np.log10(Lcurve_curvature), k=3)
			Lcurve_curvature_interp = 10**splev(np.log10(fine_alpha), Lcurve_curvature_interp)
			plt.plot(fine_alpha, Lcurve_curvature_interp, '--')

			temp = cp.deepcopy(generic_filter(Lcurve_curvature,np.mean,size=1))
			plt.plot(alpha_record[1:-1], temp, 'y-')
			plt.plot(alpha_record[1:-1], Lcurve_curvature, 'x')
			plt.xscale('log')

			temp[temp<0]=0
			peaks = find_peaks(temp, distance=4)[0]
			proms = get_proms(np.log10(temp), peaks)[0]
			try:
				peaks = np.array([peaks for _, peaks in sorted(zip(proms, peaks))])
				proms = np.sort(proms)
				index_best_fit = min(peaks[-2:])+1
				# peaks = peaks[proms>0.25]
				# proms = proms[proms>0.25]
				# if len(peaks)>2:
				# 	index_best_fit = peaks[0+1]+1
				# elif len(peaks)<=2:
				# 	index_best_fit = peaks[0]+1
				# index_best_fit = peaks[np.array(proms).argmax()]+1
			except:
				print("prominences are too low so I can't filter")
				temp[temp<0]=0
				peaks = find_peaks(temp)[0]
				proms = get_proms(temp, peaks)[0]
				index_best_fit = peaks[0]+1

			# index_best_fit = peaks[0]+1
			# index_best_fit = Lcurve_curvature.argmax()+1
			plt.plot(alpha_record[Lcurve_curvature.argmax()+1], Lcurve_curvature[Lcurve_curvature.argmax()], 'r+', markersize=9)
			plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit-1], 'gx')
			select = np.logical_and(fine_alpha<alpha_record[index_best_fit+1],fine_alpha>alpha_record[index_best_fit-1])
			most_lickely_alpha_index = Lcurve_curvature_interp[select].argmax()+1
			plt.plot(fine_alpha[select][most_lickely_alpha_index],Lcurve_curvature_interp[select][most_lickely_alpha_index],'go', mfc='none')
			most_lickely_alpha = fine_alpha[select][most_lickely_alpha_index]
		else:	# created 2021-09-01 to use the curvature from fitting the L-curve directly
			fine_alpha = np.logspace(np.log10((alpha_record[1:-1][Lcurve_curvature > 0]).min()),np.log10((alpha_record[1:-1][Lcurve_curvature > 0]).max()), 1000,endpoint=True)
			# Lcurve_curvature_interp = interp1d(alpha_record, Lcurve_curvature,fill_value='extrapolate', kind = 'cubic')
			# Lcurve_curvature_interp = splrep(np.log10(alpha_record[Lcurve_curvature>0]), np.log10(Lcurve_curvature[Lcurve_curvature>0]), k=3)
			Lcurve_curvature_interp = splrep(np.log10(alpha_record[1:-1]),np.log10(Lcurve_curvature), k=3)
			Lcurve_curvature_interp = 10**splev(np.log10(fine_alpha), Lcurve_curvature_interp)
			plt.plot(fine_alpha, Lcurve_curvature_interp, '--')

			temp = cp.deepcopy(generic_filter(Lcurve_curvature,np.mean,size=1))
			plt.plot(alpha_record[1:-1], temp, 'y-')
			plt.plot(alpha_record[1:-1], Lcurve_curvature, 'x')
			plt.xscale('log')

			temp[temp<0]=0
			peaks = find_peaks(temp, distance=4)[0]
			proms = get_proms(np.log10(temp), peaks)[0]
			peaks = np.array([peaks for _, peaks in sorted(zip(proms, peaks))])
			proms = np.sort(proms)
			# this is pretty much arbitrary. here I provilege a higher resolution even if the result is a bit more noisy
			if False:
				if len(peaks)>=2:
					index_best_fit = peaks[-2]+1
				else:
					index_best_fit = peaks[-1]+1
			elif True:
				peaks = peaks[peaks>1]
				peaks = peaks[peaks<len(temp)-1]
				index_best_fit = np.min(peaks[-2:]) + 1
			elif False:
				index_best_fit = np.max(peaks[-2:]) + 1
			elif False:
				index_best_fit = int(np.mean(peaks[-2:])) + 1

			# index_best_fit = peaks[0]+1
			# index_best_fit = Lcurve_curvature.argmax()+1
			plt.plot(alpha_record[Lcurve_curvature.argmax()+1], Lcurve_curvature[Lcurve_curvature.argmax()], 'r+', markersize=9)
			plt.plot(alpha_record[index_best_fit], Lcurve_curvature[index_best_fit-1], 'gx')
			select = np.logical_and(fine_alpha<alpha_record[index_best_fit+1],fine_alpha>alpha_record[index_best_fit-1])
			most_lickely_alpha_index = Lcurve_curvature_interp[select].argmax()+1
			plt.plot(fine_alpha[select][most_lickely_alpha_index],Lcurve_curvature_interp[select][most_lickely_alpha_index],'go', mfc='none')
			most_lickely_alpha = fine_alpha[select][most_lickely_alpha_index]



		best_alpha = alpha_record[index_best_fit]
		plt.title('Best tested regularization coefficient is alpha=' + '%.3g' % best_alpha +'\neven if the best is likely ~'+ '%.3g' % most_lickely_alpha )
		plt.xlabel('Regularisation parameter')
		plt.ylabel('L-curve curvature')
		plt.grid()
		# plt.semilogy()
		plt.savefig(path_for_plots + '/L_curve_curvature_grid'+str(grid_resolution)+'cm_time%.4g.eps' %(time_full_binned[i_t]*1e3))
		plt.close()

		plt.figure(figsize=(20, 10))
		plt.plot(score_x_record, score_y_record)
		plt.plot(score_x_record,score_y_record, 'x')
		plt.plot(score_x_record[0],score_y_record[0], 'kx', label = 'reg param '+ '%.3g' % alpha_record[0])
		plt.plot(score_x_record[-1],score_y_record[-1], 'rx', label = 'reg param '+ '%.3g' % alpha_record[-1])
		plt.plot(score_x_record[index_best_fit], score_y_record[index_best_fit], 'gx', label='reg param ' + '%.3g' % alpha_record[index_best_fit])
		plt.legend(loc = 'best')
		# plt.yscale('log')
		# plt.xscale('log')
		plt.title('Best regularization coefficient is alpha=' + str(best_alpha) + '\nwith residuals ||Gm-d||2=' + str(score_x_record[index_best_fit]))
		plt.xlabel('log ||Gm-d||2')
		plt.ylabel('log ||Laplacian(m)||2')
		plt.grid()
		plt.savefig(path_for_plots + '/L_curve_grid'+str(grid_resolution)+'cm_time%.4g.eps' %(time_full_binned[i_t]*1e3))
		plt.close()


		# print('alpha')
		# print(alpha_record)
		# print('check')
		# print(check)
		alpha = alpha_record[index_best_fit]
		exp = exp_indexes[index_best_fit]
			# treshold = treshold_record[index_best_fit]


		print('Best alpha='+str(alpha))
		# test_matrix = np.dot(sensitivities.T, sensitivities) + alpha * laplacian
		# test_matrix = np.dot(sensitivities_binned_crop.T, sensitivities_binned_crop) + (alpha ** 2) * np.dot(grid_laplacian_masked_crop.T, grid_laplacian_masked_crop)
		# print(np.dot(sensitivities.T, sensitivities))
		try:
			a1_inv = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp) + '.npz')['a1_inv']
			s = np.load(path_sensitivity_original + '/svg_decomp_spatial_ave_' + str(shrink_factor_x) + '_alpha_1e' + str(exp) + '.npz')['s']
			print('the SVG decomposition of alpha = ' + str(alpha) + ' is imported')
			sigma = np.diag(s)
			inv_sigma = np.diag(1 / s)
		except:
			print('attempt to import SVG decomposition failed\nthis should not have happened')
			test_matrix = np.dot(sensitivities_binned_crop.T, sensitivities_binned_crop) + (alpha ** 2) * np.dot(grid_laplacian_masked_crop.T, grid_laplacian_masked_crop)
			U, s, Vh = np.linalg.svd(test_matrix)
			sigma = np.diag(s)
			inv_sigma = np.diag(1 / s)
			a1_inv = np.dot(Vh.T, np.dot(inv_sigma, U.T))


		m = np.dot(a1_inv, np.dot(sensitivities_binned_crop.T, radiation_profile_on_foil.flatten()))

		plt.figure()
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=m,s=100,marker='s',cmap='rainbow')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar().set_label('emissivity [W/m3]')
		plt.pause(0.01)

		select = m<0
		plt.figure()
		plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0][select],np.mean(grid_data_masked_crop,axis=1)[:,1][select],c=m[select],s=100,marker='s',cmap='rainbow')
		plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
		plt.colorbar().set_label('emissivity [W/m3]')
		plt.pause(0.01)

		sensitivities_binned_crop_2 = cp.deepcopy(sensitivities_binned_crop.reshape(sensitivities_binned_crop_shape))
		sensitivities_binned_crop_2[:,:,m>m.min()/2] = 0

		plt.figure()
		plt.imshow(np.flip(np.transpose(np.sum(sensitivities_binned_crop_2,axis=-1),(1,0)),axis=1),'rainbow',origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.imshow(np.flip(np.transpose(np.dot(sensitivities_binned_crop_2,m),(1,0)),axis=1),'rainbow',origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		plt.imshow(rotate(np.sum(sensitivities_binned_crop*m,axis=1).reshape(np.shape(powernoback_full[i_t])),-90),'rainbow',origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		m2 = cp.deepcopy(m)
		m2[m2<0]=0
		plt.figure()
		plt.imshow(rotate(np.sum(sensitivities_binned_crop*m2,axis=1).reshape(np.shape(powernoback_full[i_t])),-90),'rainbow',origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

		plt.figure()
		# plt.imshow(rotate(powernoback_full[i_t],-90),'rainbow',vmax=np.sum(sensitivities_binned_crop*m,axis=1).max(),vmin=0,origin='lower')
		plt.imshow(rotate(powernoback_full[i_t],-90),'rainbow',vmax=np.sum(sensitivities_binned_crop*m,axis=1).max(),origin='lower')
		plt.plot([ROI[0,0]-0.5,ROI[0,1]-0.5,ROI[0,1]-0.5,ROI[0,0]-0.5,ROI[0,0]-0.5],[ROI[1,0]-0.5,ROI[1,0]-0.5,ROI[1,1]-0.5,ROI[1,1]-0.5,ROI[1,0]-0.5],'k')
		plt.colorbar()
		plt.pause(0.01)

	select = np.mean(grid_data_masked_crop,axis=1)[:,1]<20	# 0 to select only the lower half, such that the tot rad power is twice that
	total_radiated_power = np.sum(m[select] * np.pi * (((np.mean(grid_data_masked_crop,axis=1)[:,0]+grid_resolution*1e-2/2)**2 - (np.mean(grid_data_masked_crop,axis=1)[:,0]-grid_resolution*1e-2/2)**2)[select]) * grid_resolution*1e-2)



	all_m = np.array(all_m)
	all_alpha = np.array(all_alpha)
	all_total_radiated_power = np.array(all_total_radiated_power)

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned,all_alpha)
	plt.semilogy()
	plt.title('Regularization coefficient')
	plt.xlabel('time [s]')
	plt.ylabel('alpha [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_alpha_grid'+str(grid_resolution)+'cm.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned,all_total_radiated_power)
	plt.title('total radiated power detected')
	plt.xlabel('time [s]')
	plt.ylabel('total radiated power [W]')
	plt.grid()
	# plt.semilogy()
	plt.ylim(bottom=0,top=all_total_radiated_power[np.logical_and(time_full_binned>0.2,time_full_binned<0.7)].max()*2)
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_rad_power_grid'+str(grid_resolution)+'cm.eps')
	plt.close()

	voxels_centre = np.mean(grid_data_masked_crop,axis=1)
	dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
	dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
	dist_mean = dz**2 + dr**2
	recompose_voxel_emissivity = np.zeros((len(all_alpha),len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))
	for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
		for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
			dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
			if dist.min()<dist_mean/2:
				index = np.abs(dist).argmin()
				recompose_voxel_emissivity[:,i_r,i_z] = all_m[:,index]
