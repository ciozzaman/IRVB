# Created 06/08/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

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

# added to check if a point is inside a polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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
	i_day,day = 0,'2021-08-05'
	name='IRVB-MASTU_shot-44611.ptw'
	# i_day,day = 0,'2021-08-13'
	# name='IRVB-MASTU_shot-44683.ptw'
	# i_day,day = 0,'2021-09-28'
	# name='IRVB-MASTU_shot-45070.ptw'
	# i_day,day = 0,'2021-10-05'
	# name='IRVB-MASTU_shot-45156.ptw'
	laser_to_analyse=path+day+'/'+name

print('starting '+laser_to_analyse)

shot_number = int(laser_to_analyse[-9:-4])
path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
path_for_plots = path_power_output + '/invertions_log'
if not os.path.exists(path_for_plots):
	os.makedirs(path_for_plots)


inverted_dict = np.load(laser_to_analyse[:-4]+'_inverted_baiesian.npz')
inverted_dict.allow_pickle=True
inverted_dict = dict(inverted_dict)

if 'efit_reconstruction' in list(inverted_dict.keys()):
	efit_reconstruction = inverted_dict['efit_reconstruction'].all()
else:
	EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
	efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
all_time_strike_points_location = coleval.return_all_time_strike_points_location_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)


all_grid_type = np.unique([int(value) if value!='efit_reconstruction' else 4 for value in list(inverted_dict.keys()) ])
for grid_resolution in all_grid_type:
	all_shrink_factor_x = np.unique( list(inverted_dict[str(grid_resolution)].all().keys()) ).astype(int)
	for shrink_factor_x in all_shrink_factor_x:
		all_shrink_factor_t = np.unique( list(inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)].keys()) ).astype(int)
		for shrink_factor_t in all_shrink_factor_t:
			binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
			print('starting '+binning_type)

			time_full_binned_crop = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop']
			inversion_R = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['R']
			inversion_Z = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['Z']
			inverted_data = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data']
			a,b = np.meshgrid(inversion_Z,inversion_R)
			a_flat = a.flatten()
			b_flat = b.flatten()
			leg_resolution = 0.1

			data_length = 0
			local_mean_emis_all = []
			local_power_all = []
			leg_length_all = []
			for i_t in range(len(time_full_binned_crop)):
				print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
				try:
					i_efit_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
					if np.abs((r_fine[all_time_sep_r[i_efit_time][1]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][1]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).min() < np.abs((r_fine[all_time_sep_r[i_efit_time][3]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][3]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).min():
						i_closer_separatrix_to_x_point = 1
					else:
						i_closer_separatrix_to_x_point = 3
					i_where_x_point_is = np.abs((r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]]- efit_reconstruction.lower_xpoint_r[i_efit_time])**2 + (z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]]- efit_reconstruction.lower_xpoint_z[i_efit_time])**2 ).argmin()

					temp = np.array([((all_time_strike_points_location[i_efit_time][0][i] - r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2 + (all_time_strike_points_location[i_efit_time][1][i] - z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2).min() for i in range(len(all_time_strike_points_location[i_efit_time][1]))])
					temp[np.isnan(temp)] = np.inf
					i_which_strike_point_is = temp.argmin()
					i_where_strike_point_is = ((all_time_strike_points_location[i_efit_time][0][i_which_strike_point_is] - r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2 + (all_time_strike_points_location[i_efit_time][1][i_which_strike_point_is] - z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][:i_where_x_point_is])**2).argmin()
					# plt.figure()
					# plt.plot(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is])
					# plt.pause(0.001)
					r_coord_smooth = generic_filter(r_fine[all_time_sep_r[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],np.mean,size=11)
					z_coord_smooth = generic_filter(z_fine[all_time_sep_z[i_efit_time][i_closer_separatrix_to_x_point]][i_where_strike_point_is:i_where_x_point_is],np.mean,size=11)
					leg_length = np.sum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5)
					# I arbitrarily decide to cut the leg in 10cm pieces
					target_length = 0 + leg_resolution
					i_ref_points = [0]
					ref_points = [[r_coord_smooth[0],z_coord_smooth[0]]]
					while target_length < leg_length + leg_resolution:
						temp = np.abs(np.cumsum((np.diff(r_coord_smooth)**2 + np.diff(z_coord_smooth)**2)**0.5) - target_length).argmin() + 1
						i_ref_points.append(temp)
						ref_points.append([r_coord_smooth[temp],z_coord_smooth[temp]])
						target_length += leg_resolution
					ref_points = np.array(ref_points)
					ref_points_1 = []
					ref_points_2 = []
					m = -1/((z_coord_smooth[i_ref_points[0]]-z_coord_smooth[i_ref_points[0]+1])/(r_coord_smooth[i_ref_points[0]]-r_coord_smooth[i_ref_points[0]+1]))
					ref_points_1.append([r_coord_smooth[i_ref_points[0]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[0]] - m*leg_resolution/((1+m**2)**0.5)])
					ref_points_2.append([r_coord_smooth[i_ref_points[0]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[0]] + m*leg_resolution/((1+m**2)**0.5)])
					for i_ref_point in range(1,len(i_ref_points)-1):
						m = -1/((z_coord_smooth[i_ref_points[i_ref_point]-1]-z_coord_smooth[i_ref_points[i_ref_point]+1])/(r_coord_smooth[i_ref_points[i_ref_point]-1]-r_coord_smooth[i_ref_points[i_ref_point]+1]))
						ref_points_1.append([r_coord_smooth[i_ref_points[i_ref_point]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[i_ref_point]] - m*leg_resolution/((1+m**2)**0.5)])
						ref_points_2.append([r_coord_smooth[i_ref_points[i_ref_point]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[i_ref_point]] + m*leg_resolution/((1+m**2)**0.5)])
					m = -1/((z_coord_smooth[i_ref_points[-1]-1]-z_coord_smooth[i_ref_points[-1]])/(r_coord_smooth[i_ref_points[-1]-1]-r_coord_smooth[i_ref_points[-1]]))
					ref_points_1.append([r_coord_smooth[i_ref_points[-1]] - leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[-1]] - m*leg_resolution/((1+m**2)**0.5)])
					ref_points_2.append([r_coord_smooth[i_ref_points[-1]] + leg_resolution/((1+m**2)**0.5) , z_coord_smooth[i_ref_points[-1]] + m*leg_resolution/((1+m**2)**0.5)])
					ref_points_1 = np.array(ref_points_1)
					ref_points_2 = np.array(ref_points_2)

					# plt.figure()
					# plt.plot(r_coord_smooth,z_coord_smooth)
					# plt.plot(r_coord_smooth,z_coord_smooth,'+')
					# plt.plot(ref_points[:,0],ref_points[:,1],'+')
					# plt.plot(ref_points_1[:,0],ref_points_1[:,1],'o')
					# plt.plot(ref_points_2[:,0],ref_points_2[:,1],'+')
					# plt.pause(0.001)

					local_mean_emis = []
					local_power = []
					emissivity_flat = inverted_data[i_t].flatten()
					for i_ref_point in range(1,len(i_ref_points)-1):
						select = []
						polygon = Polygon([ref_points_1[i_ref_point-1], ref_points_1[i_ref_point], ref_points_2[i_ref_point], ref_points_2[i_ref_point-1]])
						for i_e in range(len(emissivity_flat)):
							point = Point((b_flat[i_e],a_flat[i_e]))
							select.append(polygon.contains(point))
						local_mean_emis.append(np.nanmean(emissivity_flat[select]))
						local_power.append(2*np.pi*np.nansum(emissivity_flat[select]*b_flat[select]*((grid_resolution*1e-2)**2)))
					local_mean_emis = np.array(local_mean_emis)
					local_power = np.array(local_power)
					local_mean_emis = local_mean_emis[np.logical_not(np.isnan(local_mean_emis))].tolist()
					local_power = local_power[np.logical_not(np.isnan(local_power))].tolist()
					local_mean_emis_all.append(local_mean_emis)
					local_power_all.append(local_power)
					data_length = max(data_length,len(local_power))
					leg_length_all.append(leg_length)
				except:
					local_mean_emis_all.append([])
					local_power_all.append([])
					leg_length_all.append(0)

			for i_t in range(len(time_full_binned_crop)):
				if len(local_mean_emis_all[i_t])<data_length:
					local_mean_emis_all[i_t].extend([0]*(data_length-len(local_mean_emis_all[i_t])))
					local_power_all[i_t].extend([0]*(data_length-len(local_power_all[i_t])))


			plt.figure(figsize=(10, 20))
			plt.imshow(local_power_all,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(local_power_all[:-4]),vmax=np.max(local_power_all[:-4]))
			plt.plot(leg_length_all,time_full_binned_crop,'--k')
			plt.colorbar().set_label('power [W]')
			plt.title('Power radiated on the outer leg')
			plt.ylabel('time [s]')
			plt.xlabel('distance from the strike point [m]')
			# plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_rad_power.eps')
			plt.close()

			plt.figure(figsize=(10, 20))
			plt.imshow(local_mean_emis_all,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(local_mean_emis_all[:-4]),vmax=np.max(local_mean_emis_all[:-4]))
			plt.plot(leg_length_all,time_full_binned_crop,'--k')
			plt.colorbar().set_label('emissivity [W/m3]')
			plt.title('Emissivity on the outer leg')
			plt.ylabel('time [s]')
			plt.xlabel('distance from the strike point [m]')
			# plt.grid()
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_emissivity.eps')
			plt.close()


			inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['local_outer_leg_power'] = local_power_all
			inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['local_outer_leg_mean_emissivity'] = local_mean_emis_all
			inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['leg_length_all'] = leg_length_all

			additional_points_dict,radiator_xpoint_distance_all,radiator_above_xpoint_all,radiator_magnetic_radious_all = coleval.find_radiator_location(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)




np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian',**inverted_dict)
