from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
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

			local_mean_emis_all,local_power_all,leg_length_all,data_length,leg_resolution = coleval.track_outer_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)

			fig, ax = plt.subplots( 1,2,figsize=(10, 20), squeeze=False,sharey=True)
			im1 = ax[0,0].imshow(local_power_all,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(local_power_all[:-4]),vmax=np.max(local_power_all[:-4]))
			ax[0,0].plot(leg_length_all,time_full_binned_crop,'--k')
			im2 = ax[0,1].imshow(local_mean_emis_all,'rainbow',origin='lower',extent=[(0-0.5)*leg_resolution,(data_length+0.5)*leg_resolution,time_full_binned_crop[0]-np.diff(time_full_binned_crop)[0]/2,time_full_binned_crop[-1]+np.diff(time_full_binned_crop)[-1]/2],aspect=10,vmin=np.min(local_mean_emis_all[:-4]),vmax=np.max(local_mean_emis_all[:-4]))
			ax[0,1].plot(leg_length_all,time_full_binned_crop,'--k')
			fig.suptitle('tracking radiation on the outer leg')
			ax[0,0].set_xlabel('distance from the strike point [m]')
			ax[0,0].grid()
			ax[0,0].set_ylabel('time [s]')
			plt.colorbar(im1,ax=ax[0,0]).set_label('Integrated power [W]')
			# ax[0,0].colorbar().set_label('Integrated power [W]')
			ax[0,1].set_xlabel('distance from the strike point [m]')
			ax[0,1].grid()
			# ax[0,1].colorbar().set_label('Emissivity [W/m3]')
			plt.colorbar(im2,ax=ax[0,1]).set_label('Emissivity [W/m3]')
			plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')
			plt.close()

			inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['local_outer_leg_power'] = local_power_all
			inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['local_outer_leg_mean_emissivity'] = local_mean_emis_all
			inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['leg_length_all'] = leg_length_all


np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian',**inverted_dict)
