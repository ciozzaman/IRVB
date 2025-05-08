from mastu_exhaust_analysis.calc_pohm import calc_pohm

def temp_function(full_saved_file_dict_FAST):
	try:
		try:
			# trash = full_saved_file_dict_FAST['multi_instrument']
			full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
		except:
			full_saved_file_dict_FAST['multi_instrument'] = dict([])
		grid_resolution = 2	# cm

		try:	# I do this first so I have all the memory I can
			try:
				inverted_data_covariance_log = inverted_dict[str(grid_resolution)]['inverted_data_covariance_log']
				covariance_out = np.exp(inverted_data_covariance_log.astype(np.float32)).astype(np.float32)
				del inverted_data_covariance_log
				inverted_data_covariance_is_positive_packed = inverted_dict[str(grid_resolution)]['inverted_data_covariance_is_positive_packed']
				covariance_out_sing = np.unpackbits(inverted_data_covariance_is_positive_packed)[:np.prod(np.shape(covariance_out))].reshape(np.shape(covariance_out)).astype(int)
				del inverted_data_covariance_is_positive_packed
				covariance_out_sing *= 2
				covariance_out_sing -= 1
				covariance_out_sing = np.round(covariance_out_sing,2)	# just in case I generated some 0.9999999
				# covariance_out = (covariance_out*covariance_out_sing).astype(np.float32)
				covariance_out *= covariance_out_sing	# this should be more memory efficient
				del covariance_out_sing
			except:	# this is the new method (24/09/2024) sugested by Kevin, where the covariance matrix is in a reparate file
				inverted_data_covariance_log = covariance_dict['inverted_data_covariance_log']
				covariance_out = np.exp(inverted_data_covariance_log.astype(np.float32)).astype(np.float32)
				del inverted_data_covariance_log
				inverted_data_covariance_is_positive_packed = covariance_dict['inverted_data_covariance_is_positive_packed']
				covariance_out_sing = np.unpackbits(inverted_data_covariance_is_positive_packed)[:np.prod(np.shape(covariance_out))].reshape(np.shape(covariance_out)).astype(int)
				del inverted_data_covariance_is_positive_packed
				covariance_out_sing *= 2
				covariance_out_sing -= 1
				covariance_out_sing = np.round(covariance_out_sing,2)	# just in case I generated some 0.9999999
				# covariance_out = (covariance_out*covariance_out_sing).astype(np.float32)
				covariance_out *= covariance_out_sing	# this should be more memory efficient
				del covariance_out_sing
		except Exception as e:
			print('inverted_data_covariance_log reading failed because of '+e)
			logging.exception('with error: ' + str(e))
			try:
				inverted_data_covariance_scaling_factor = inverted_dict[str(grid_resolution)]['inverted_data_covariance_scaling_factor']
				inverted_data_covariance_scaled = inverted_dict[str(grid_resolution)]['inverted_data_covariance_scaled']
				covariance_out = (inverted_data_covariance_scaled.astype(np.float32).T * inverted_data_covariance_scaled.astype(np.float32).T).T
			except:
				covariance_out = inverted_dict[str(grid_resolution)]['inverted_data_covariance']
		time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
		inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
		inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
		binning_type = inverted_dict[str(grid_resolution)]['binning_type']
		shotnumber = int(laser_to_analyse[-9:-4])

		outer_leg_tot_rad_power_all = inverted_dict[str(grid_resolution)]['outer_leg_tot_rad_power_all']
		inner_leg_tot_rad_power_all = inverted_dict[str(grid_resolution)]['inner_leg_tot_rad_power_all']
		core_tot_rad_power_all = inverted_dict[str(grid_resolution)]['core_tot_rad_power_all']
		sxd_tot_rad_power_all = inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_all']
		x_point_tot_rad_power_all = inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_all']
		outer_leg_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['outer_leg_tot_rad_power_sigma_all']
		inner_leg_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['inner_leg_tot_rad_power_sigma_all']
		core_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['core_tot_rad_power_sigma_all']
		sxd_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_sigma_all']
		x_point_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_sigma_all']
		grid_data_masked_crop = inverted_dict[str(grid_resolution)]['grid_data_masked_crop']

		local_inner_leg_mean_emissivity = inverted_dict[str(grid_resolution)]['local_inner_leg_mean_emissivity']
		local_outer_leg_mean_emissivity = inverted_dict[str(grid_resolution)]['local_outer_leg_mean_emissivity']
		try:
			x_point_region_radious = inverted_dict[str(grid_resolution)]['x_point_region_radious']
		except:
			if pass_number==0:
				x_point_region_radious = 0.2
			else:
				x_point_region_radious = 0.1
			inverted_dict[str(grid_resolution)]['x_point_region_radious'] = x_point_region_radious

		EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
		efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
		inversion_R = inverted_dict[str(grid_resolution)]['geometry']['R']
		inversion_Z = inverted_dict[str(grid_resolution)]['geometry']['Z']

		client=pyuda.Client()
		try:
			ne_bar_dict=calc_ne_bar(shotnumber, efit_data = None)
			greenwald_density = np.interp(time_full_binned_crop,ne_bar_dict['t'],ne_bar_dict['greenwald_density'])
			ne_bar = np.interp(time_full_binned_crop,ne_bar_dict['t'],ne_bar_dict['data'])	# this should be the line averaged density
			density_data_missing = False
			full_saved_file_dict_FAST['multi_instrument']['greenwald_density'] = greenwald_density
			full_saved_file_dict_FAST['multi_instrument']['ne_bar'] = ne_bar
			data = client.get('/ANE/DENSITY',shotnumber)
			# core_density = np.array([data.data[np.abs(data.time.data-time).argmin()] for time in time_full_binned_crop])
			core_density = np.interp(time_full_binned_crop,data.time.data,data.data)	# this is core line integrated density
			full_saved_file_dict_FAST['multi_instrument']['line_integrated_density'] = core_density
		except:
			density_data_missing = True
		try:	# last ditch effort to see if the data was saved already
			greenwald_density = full_saved_file_dict_FAST['multi_instrument']['greenwald_density']
			ne_bar = full_saved_file_dict_FAST['multi_instrument']['ne_bar']
			core_density = full_saved_file_dict_FAST['multi_instrument']['line_integrated_density']
			if len(ne_bar)==len(time_full_binned_crop):
				density_data_missing = False
		except:
			pass

		filename_root = inverted_dict[str(grid_resolution)]['filename_root']
		filename_root_add = inverted_dict[str(grid_resolution)]['filename_root_add']
		# fitted_foil_power = inverted_dict[str(grid_resolution)]['fitted_foil_power']
		# fitted_foil_power_excluded = inverted_dict[str(grid_resolution)]['fitted_foil_power_excluded']
		# foil_power = inverted_dict[str(grid_resolution)]['foil_power']
		# full_foil_power = inverted_dict[str(grid_resolution)]['full_foil_power']

		if True:	# added to deal with the fact that the peak radiation is extremely volatile, and hard to track consistently
			local_inner_leg_mean_emissivity = np.array(local_inner_leg_mean_emissivity)
			local_inner_leg_mean_emissivity[local_inner_leg_mean_emissivity==0] = np.nan
			temp = []
			for i in range(len(local_inner_leg_mean_emissivity)):
				if np.sum(np.isfinite([local_inner_leg_mean_emissivity[i]]))==0:
					temp.append(np.nan)
				elif not (local_inner_leg_mean_emissivity[i]<np.nanmax(local_inner_leg_mean_emissivity[i])/2)[0]:
					temp.append(0)
				else:
					temp.append(((np.arange(len(local_inner_leg_mean_emissivity[i]))[(local_inner_leg_mean_emissivity[i]<np.nanmax(local_inner_leg_mean_emissivity[i])/2)])[-1]+0.5) / ((np.arange(len(local_inner_leg_mean_emissivity[i]))[np.isfinite(local_inner_leg_mean_emissivity[i])])[-1]))
			movement_local_inner_leg_mean_emissivity = np.array(temp)

			local_outer_leg_mean_emissivity = np.array(local_outer_leg_mean_emissivity)
			local_outer_leg_mean_emissivity[local_outer_leg_mean_emissivity==0] = np.nan
			temp = []
			for i in range(len(local_outer_leg_mean_emissivity)):
				if np.sum(np.isfinite([local_outer_leg_mean_emissivity[i]]))==0:
					temp.append(np.nan)
				elif not (local_outer_leg_mean_emissivity[i]<np.nanmax(local_outer_leg_mean_emissivity[i])/2)[0]:
					temp.append(0)
				else:
					temp.append(((np.arange(len(local_outer_leg_mean_emissivity[i]))[(local_outer_leg_mean_emissivity[i]<np.nanmax(local_outer_leg_mean_emissivity[i])/2)])[-1]+0.5) / ((np.arange(len(local_outer_leg_mean_emissivity[i]))[np.isfinite(local_outer_leg_mean_emissivity[i])])[-1]))
			movement_local_outer_leg_mean_emissivity = np.array(temp)
			inverted_dict[str(grid_resolution)]['movement_local_inner_leg_mean_emissivity'] = movement_local_inner_leg_mean_emissivity
			inverted_dict[str(grid_resolution)]['movement_local_outer_leg_mean_emissivity'] = movement_local_outer_leg_mean_emissivity

		# # temporary
		# plt.figure()
		# plt.imshow(inverted_data[15])
		# plt.title('shot ' + laser_to_analyse[-9:-4]+' '+'\ntemp')
		# plt.savefig(filename_root+filename_root_add+'_temp.eps')
		# plt.close()

		outer_local_mean_emis_all,outer_local_power_all,outer_local_L_poloidal_all,outer_leg_length_interval_all,outer_leg_length_all,outer_data_length,outer_leg_resolution,outer_emissivity_baricentre_all,outer_emissivity_peak_all,outer_L_poloidal_baricentre_all,outer_L_poloidal_peak_all,outer_L_poloidal_peak_only_leg_all,outer_L_poloidal_baricentre_only_leg_all,trash,dr_sep_in,dr_sep_out,outer_L_poloidal_x_point_all,outer_L_poloidal_midplane_all,outer_leg_reliable_power_all,outer_leg_reliable_power_sigma_all,DMS_equivalent_all,DMS_equivalent_sigma_all,MWI_equivalent_all,MWI_equivalent_sigma_all,x_point_tot_rad_power_all,x_point_tot_rad_power_sigma_all,sxd_tot_rad_power_all,sxd_tot_rad_power_sigma_all,outer_half_peak_L_pol_all,outer_half_peak_divertor_L_pol_all,outer_sideways_leg_resolution = coleval.baricentre_outer_separatrix_radiation(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,covariance_out,grid_data_masked_crop,x_point_region_radious=x_point_region_radious)
		inverted_dict[str(grid_resolution)]['outer_local_mean_emis_all'] = outer_local_mean_emis_all
		inverted_dict[str(grid_resolution)]['outer_local_power_all'] = outer_local_power_all
		inverted_dict[str(grid_resolution)]['outer_local_L_poloidal_all'] = outer_local_L_poloidal_all
		inverted_dict[str(grid_resolution)]['outer_leg_length_interval_all'] = outer_leg_length_interval_all
		inverted_dict[str(grid_resolution)]['outer_leg_length_all'] = outer_leg_length_all
		inverted_dict[str(grid_resolution)]['outer_data_length'] = outer_data_length
		inverted_dict[str(grid_resolution)]['outer_leg_resolution'] = outer_leg_resolution
		inverted_dict[str(grid_resolution)]['outer_emissivity_baricentre_all'] = outer_emissivity_baricentre_all
		inverted_dict[str(grid_resolution)]['outer_emissivity_peak_all'] = outer_emissivity_peak_all
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_baricentre_all'] = outer_L_poloidal_baricentre_all
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_all'] = outer_L_poloidal_peak_all
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_only_leg_all'] = outer_L_poloidal_peak_only_leg_all
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_baricentre_only_leg_all'] = outer_L_poloidal_baricentre_only_leg_all
		inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_all'] = x_point_tot_rad_power_all
		inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_sigma_all'] = x_point_tot_rad_power_sigma_all
		full_saved_file_dict_FAST['multi_instrument']['time_full_binned_crop'] = time_full_binned_crop
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_in'] = dr_sep_in
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_out'] = dr_sep_out
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_x_point_all'] = outer_L_poloidal_x_point_all
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_midplane_all'] = outer_L_poloidal_midplane_all
		inverted_dict[str(grid_resolution)]['outer_leg_reliable_power_all'] = outer_leg_reliable_power_all
		inverted_dict[str(grid_resolution)]['outer_leg_reliable_power_sigma_all'] = outer_leg_reliable_power_sigma_all
		inverted_dict[str(grid_resolution)]['DMS_equivalent_all'] = DMS_equivalent_all
		inverted_dict[str(grid_resolution)]['DMS_equivalent_sigma_all'] = DMS_equivalent_sigma_all
		inverted_dict[str(grid_resolution)]['MWI_equivalent_all'] = MWI_equivalent_all
		inverted_dict[str(grid_resolution)]['MWI_equivalent_sigma_all'] = MWI_equivalent_sigma_all
		inverted_dict[str(grid_resolution)]['outer_half_peak_L_pol_all'] = outer_half_peak_L_pol_all
		inverted_dict[str(grid_resolution)]['outer_half_peak_divertor_L_pol_all'] = outer_half_peak_divertor_L_pol_all
		inverted_dict[str(grid_resolution)]['outer_sideways_leg_resolution'] = outer_sideways_leg_resolution
		inner_local_mean_emis_all,inner_local_power_all,inner_local_L_poloidal_all,inner_leg_length_interval_all,inner_leg_length_all,inner_data_length,inner_leg_resolution,inner_emissivity_baricentre_all,inner_emissivity_peak_all,inner_L_poloidal_baricentre_all,inner_L_poloidal_peak_all,inner_L_poloidal_peak_only_leg_all,inner_L_poloidal_baricentre_only_leg_all,inner_L_poloidal_midplane_all,inner_leg_reliable_power_all,inner_leg_reliable_power_sigma_all,trash,dr_sep_in,dr_sep_out,inner_L_poloidal_x_point_all,inner_half_peak_L_pol_all,inner_half_peak_divertor_L_pol_all,inner_sideways_leg_resolution = coleval.baricentre_inner_separatrix_radiation(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,covariance_out,grid_data_masked_crop,x_point_region_radious=x_point_region_radious)
		inverted_dict[str(grid_resolution)]['inner_local_mean_emis_all'] = inner_local_mean_emis_all
		inverted_dict[str(grid_resolution)]['inner_local_power_all'] = inner_local_power_all
		inverted_dict[str(grid_resolution)]['inner_local_L_poloidal_all'] = inner_local_L_poloidal_all
		inverted_dict[str(grid_resolution)]['inner_leg_length_interval_all'] = inner_leg_length_interval_all
		inverted_dict[str(grid_resolution)]['inner_leg_length_all'] = inner_leg_length_all
		inverted_dict[str(grid_resolution)]['inner_data_length'] = inner_data_length
		inverted_dict[str(grid_resolution)]['inner_leg_resolution'] = inner_leg_resolution
		inverted_dict[str(grid_resolution)]['inner_emissivity_baricentre_all'] = inner_emissivity_baricentre_all
		inverted_dict[str(grid_resolution)]['inner_emissivity_peak_all'] = inner_emissivity_peak_all
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_baricentre_all'] = inner_L_poloidal_baricentre_all
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_all'] = inner_L_poloidal_peak_all
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_only_leg_all'] = inner_L_poloidal_peak_only_leg_all
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_baricentre_only_leg_all'] = inner_L_poloidal_baricentre_only_leg_all
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_midplane_all'] = inner_L_poloidal_midplane_all
		inverted_dict[str(grid_resolution)]['inner_leg_reliable_power_all'] = inner_leg_reliable_power_all
		inverted_dict[str(grid_resolution)]['inner_leg_reliable_power_sigma_all'] = inner_leg_reliable_power_sigma_all
		inverted_dict[str(grid_resolution)]['inner_half_peak_L_pol_all'] = inner_half_peak_L_pol_all
		inverted_dict[str(grid_resolution)]['inner_half_peak_divertor_L_pol_all'] = inner_half_peak_divertor_L_pol_all
		inverted_dict[str(grid_resolution)]['inner_sideways_leg_resolution'] = inner_sideways_leg_resolution
		full_saved_file_dict_FAST['multi_instrument']['time_full_binned_crop'] = time_full_binned_crop
		# full_saved_file_dict_FAST['multi_instrument']['greenwald_density'] = greenwald_density
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_in'] = dr_sep_in
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_out'] = dr_sep_out
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_x_point_all'] = inner_L_poloidal_x_point_all

		real_core_radiation_all,real_core_radiation_sigma_all,real_non_core_radiation_all,real_non_core_radiation_sigma_all,out_VV_radiation_all,out_VV_radiation_sigma_all,real_inner_core_radiation_all,real_inner_core_radiation_sigma_all,real_outer_core_radiation_all,real_outer_core_radiation_sigma_all = coleval.inside_vs_outside_separatrix_radiation(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,covariance_out,grid_data_masked_crop,x_point_region_radious=x_point_region_radious)
		inverted_dict[str(grid_resolution)]['real_core_radiation_all'] = real_core_radiation_all
		inverted_dict[str(grid_resolution)]['real_core_radiation_sigma_all'] = real_core_radiation_sigma_all
		inverted_dict[str(grid_resolution)]['real_non_core_radiation_all'] = real_non_core_radiation_all
		inverted_dict[str(grid_resolution)]['real_non_core_radiation_sigma_all'] = real_non_core_radiation_sigma_all
		inverted_dict[str(grid_resolution)]['out_VV_radiation_all'] = out_VV_radiation_all
		inverted_dict[str(grid_resolution)]['out_VV_radiation_sigma_all'] = out_VV_radiation_sigma_all
		inverted_dict[str(grid_resolution)]['real_inner_core_radiation_all'] = real_inner_core_radiation_all
		inverted_dict[str(grid_resolution)]['real_inner_core_radiation_sigma_all'] = real_inner_core_radiation_sigma_all
		inverted_dict[str(grid_resolution)]['real_outer_core_radiation_all'] = real_outer_core_radiation_all
		inverted_dict[str(grid_resolution)]['real_outer_core_radiation_sigma_all'] = real_outer_core_radiation_sigma_all
		all_lower_volume_radiation_all = inverted_data[:,:,inversion_Z<0]
		all_lower_volume_radiation_all = np.nansum(np.nansum(all_lower_volume_radiation_all,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
		all_lower_volume_radiation_sigma_all = ((np.mean(grid_data_masked_crop,axis=1)[:,1]<=0)*2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*(np.mean(np.diff(inversion_R))**2)).astype(np.float32)
		all_lower_volume_radiation_sigma_all = np.nansum((np.transpose(covariance_out[:,:len(grid_data_masked_crop),:len(grid_data_masked_crop)]*all_lower_volume_radiation_sigma_all,(0,2,1))*all_lower_volume_radiation_sigma_all),axis=(1,2))**0.5
		# all_lower_volume_radiation_all = real_core_radiation_all + real_non_core_radiation_all
		# all_lower_volume_radiation_sigma_all = (real_core_radiation_sigma_all**2 + real_non_core_radiation_sigma_all**2)**0.5
		inverted_dict[str(grid_resolution)]['all_lower_volume_radiation_all'] = all_lower_volume_radiation_all
		inverted_dict[str(grid_resolution)]['all_lower_volume_radiation_sigma_all'] = all_lower_volume_radiation_sigma_all
		all_separatrix_radiation_all = outer_leg_reliable_power_all+sxd_tot_rad_power_all+inner_leg_reliable_power_all+real_core_radiation_all+x_point_tot_rad_power_all
		all_separatrix_radiation_sigma_all = (outer_leg_reliable_power_sigma_all**2 + sxd_tot_rad_power_sigma_all**2 + inner_leg_reliable_power_sigma_all**2 + real_core_radiation_sigma_all**2 + x_point_tot_rad_power_sigma_all**2)**0.5
		inverted_dict[str(grid_resolution)]['all_separatrix_radiation_all'] = all_separatrix_radiation_all
		inverted_dict[str(grid_resolution)]['all_separatrix_radiation_sigma_all'] = all_separatrix_radiation_sigma_all
		divertor_tot_rad_power_all = outer_leg_reliable_power_all+inner_leg_reliable_power_all+sxd_tot_rad_power_all+x_point_tot_rad_power_all
		divertor_tot_rad_power_sigma_all = (outer_leg_reliable_power_sigma_all**2+inner_leg_reliable_power_sigma_all**2+sxd_tot_rad_power_sigma_all**2 + x_point_tot_rad_power_sigma_all**2)**0.5
		inverted_dict[str(grid_resolution)]['divertor_tot_rad_power_all'] = divertor_tot_rad_power_all
		inverted_dict[str(grid_resolution)]['divertor_tot_rad_power_sigma_all'] = divertor_tot_rad_power_sigma_all

		inner_SOL_leg_all,inner_SOL_leg_sigma_all,outer_SOL_leg_all,outer_SOL_leg_sigma_all,outer_SOL_all,outer_SOL_sigma_all,inner_SOL_all,inner_SOL_sigma_all,psiN_min_lower_baffle_all,psiN_min_lower_target_all,psiN_min_central_column_all,psiN_min_upper_target_all,psiN_min_upper_baffle_all = coleval.symplified_out_core_regions(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,covariance_out,grid_data_masked_crop,x_point_region_radious=x_point_region_radious)
		inverted_dict[str(grid_resolution)]['inner_SOL_leg_all'] = inner_SOL_leg_all
		inverted_dict[str(grid_resolution)]['inner_SOL_leg_sigma_all'] = inner_SOL_leg_sigma_all
		inverted_dict[str(grid_resolution)]['inner_SOL_all'] = inner_SOL_all	#  (including part of X-point region)
		inverted_dict[str(grid_resolution)]['inner_SOL_sigma_all'] = inner_SOL_sigma_all
		inverted_dict[str(grid_resolution)]['outer_SOL_leg_all'] = outer_SOL_leg_all	#  (including part of X-point region)
		inverted_dict[str(grid_resolution)]['outer_SOL_leg_sigma_all'] = outer_SOL_leg_sigma_all
		inverted_dict[str(grid_resolution)]['outer_SOL_all'] = outer_SOL_all
		inverted_dict[str(grid_resolution)]['outer_SOL_sigma_all'] = outer_SOL_sigma_all
		# 2024/11/25 I want the minimum psiN to see hot much the plasma goes close to the surface
		inverted_dict[str(grid_resolution)]['psiN_min_lower_baffle_all'] = psiN_min_lower_baffle_all
		inverted_dict[str(grid_resolution)]['psiN_min_lower_target_all'] = psiN_min_lower_target_all
		inverted_dict[str(grid_resolution)]['psiN_min_central_column_all'] = psiN_min_central_column_all
		inverted_dict[str(grid_resolution)]['psiN_min_upper_target_all'] = psiN_min_upper_target_all
		inverted_dict[str(grid_resolution)]['psiN_min_upper_baffle_all'] = psiN_min_upper_baffle_all

		equivalent_res_bolo_view_all,equivalent_res_bolo_view_sigma_all,all_out_of_sxd_all,all_out_of_sxd_sigma_all = coleval.equivalent_res_bolo_view(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,covariance_out,grid_data_masked_crop)
		inverted_dict[str(grid_resolution)]['equivalent_res_bolo_view_all'] = equivalent_res_bolo_view_all
		inverted_dict[str(grid_resolution)]['equivalent_res_bolo_view_sigma_all'] = equivalent_res_bolo_view_sigma_all
		inverted_dict[str(grid_resolution)]['all_out_of_sxd_all'] = all_out_of_sxd_all
		inverted_dict[str(grid_resolution)]['all_out_of_sxd_sigma_all'] = all_out_of_sxd_sigma_all

		shot_list = get_data(path+'shot_list2.ods')
		temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
		scenario = ''
		experiment = ''
		useful = ''
		Preshot = ''
		Postshot = ''
		SC = ''
		SL = ''
		try:
			for i in range(1,len(shot_list['Sheet1'])):
				if shot_list['Sheet1'][i][temp1] == shotnumber:
					scenario = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Scenario').argmax()]
					experiment = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax()]
					useful = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Useful').argmax()]
					Preshot = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Preshot').argmax()]
					Postshot = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Postshot').argmax()]
					SC = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='SC').argmax()]
					SL = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='SL').argmax()]
					break
		except:
			pass
		full_saved_file_dict_FAST['multi_instrument']['scenario'] = scenario
		full_saved_file_dict_FAST['multi_instrument']['experiment'] = experiment
		full_saved_file_dict_FAST['multi_instrument']['useful'] = useful
		full_saved_file_dict_FAST['multi_instrument']['Preshot'] = Preshot
		full_saved_file_dict_FAST['multi_instrument']['Postshot'] = Postshot
		full_saved_file_dict_FAST['multi_instrument']['SC'] = SC
		full_saved_file_dict_FAST['multi_instrument']['SL'] = SL

		plt.figure(figsize=(30, 10))
		# plt.errorbar(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,yerr=outer_leg_tot_rad_power_sigma_all/1e3,label='outer_leg\nwith x-point',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,outer_leg_reliable_power_all/1e3,yerr=outer_leg_reliable_power_sigma_all/1e3,label='outer_leg\nno x-point\n+/-'+str(outer_sideways_leg_resolution)+'m from sep\nno sxd',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,sxd_tot_rad_power_all/1e3,yerr=sxd_tot_rad_power_sigma_all/1e3,label='sxd',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,inner_leg_reliable_power_all/1e3,yerr=inner_leg_reliable_power_sigma_all/1e3,label='inner_leg\nno x-point\n+/-'+str(inner_sideways_leg_resolution)+'m from sep',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,real_core_radiation_all/1e3,yerr=real_core_radiation_sigma_all/1e3,label='core\naccurate\nno x-point',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,x_point_tot_rad_power_all/1e3,yerr=x_point_tot_rad_power_sigma_all/1e3,label='x_point (dist<%.3gm)' %(x_point_region_radious),capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,all_separatrix_radiation_all/1e3,yerr=all_separatrix_radiation_sigma_all/1e3,label='tot\nwithin separatrix',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,all_lower_volume_radiation_all/1e3,yerr=all_lower_volume_radiation_sigma_all/1e3,label='tot',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,inner_SOL_leg_all/1e3,yerr=inner_SOL_leg_sigma_all/1e3,label='inner SOL\n+inner div',capsize=5,elinewidth=1)
		plt.errorbar(time_full_binned_crop,DMS_equivalent_all/1e3,yerr=DMS_equivalent_sigma_all/1e3,label='DMS equivalent\nLOS19 V2',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,MWI_equivalent_all/1e3,yerr=MWI_equivalent_sigma_all/1e3,label='MWI equivalent',capsize=5,linestyle=':')
		plt.errorbar(time_full_binned_crop,inner_SOL_all/1e3,yerr=inner_SOL_sigma_all/1e3,label='inner SOL',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,outer_SOL_leg_all/1e3,yerr=outer_SOL_leg_sigma_all/1e3,label='outer SOL\n+outer div\n+sxd',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,outer_SOL_all/1e3,yerr=outer_SOL_sigma_all/1e3,label='outer SOL',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,out_VV_radiation_all/1e3,yerr=out_VV_radiation_sigma_all/1e3,label='tot\nout VV',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,equivalent_res_bolo_view_all/1e3,yerr=equivalent_res_bolo_view_sigma_all/1e3,label='equivalent\nto res bolo',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,all_out_of_sxd_all/1e3,yerr=all_out_of_sxd_sigma_all/1e3,label='out sxd',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,divertor_tot_rad_power_all/1e3,yerr=divertor_tot_rad_power_sigma_all/1e3,label='whole divertor\nwith X-point',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,real_inner_core_radiation_all/1e3,yerr=real_inner_core_radiation_sigma_all/1e3,label='core\naccurate\nno x-point\ninner',capsize=5,elinewidth=1,linestyle='--')
		plt.errorbar(time_full_binned_crop,real_outer_core_radiation_all/1e3,yerr=real_outer_core_radiation_sigma_all/1e3,label='core\naccurate\nno x-point\nouter',capsize=5,elinewidth=1,linestyle='--')
		plt.title('shot ' + laser_to_analyse[-9:-4]+' '+scenario+'\nradiated power in the lower half of the machine')
		plt.ylim(bottom=0,top=median_filter(all_lower_volume_radiation_all,size=[max(1,len(all_lower_volume_radiation_all)//8)*2+1],mode='constant',cval=0).max()/1e3)	# arbitrary limit to see better if there is a disruption at the end of the shot
		plt.legend(loc='best', fontsize='xx-small')
		plt.xlabel('time [s]')
		plt.ylabel('power [kW]')
		plt.grid()
		plt.savefig(filename_root+filename_root_add+'_FAST_tot_rad_power.eps')
		plt.close()

		local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = coleval.track_outer_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,leg_resolution=0.05)
		try:
			peak_location,midpoint_location = coleval.plot_leg_radiation_tracking(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution,filename_root,filename_root_add,laser_to_analyse,scenario,which_leg='outer',x_point_L_pol=outer_L_poloidal_x_point_all)
		except Exception as e:
			logging.exception('with error: ' + str(e))
			print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')
		inverted_dict[str(grid_resolution)]['outer_leg_only_local_power'] = local_power_all
		inverted_dict[str(grid_resolution)]['outer_leg_only_local_emissivity'] = local_mean_emis_all
		inverted_dict[str(grid_resolution)]['outer_leg_only_length_all'] = leg_length_all
		inverted_dict[str(grid_resolution)]['outer_leg_only_length_interval_all'] = leg_length_interval_all
		inverted_dict[str(grid_resolution)]['outer_leg_only_peak_location'] = peak_location
		inverted_dict[str(grid_resolution)]['outer_leg_only_midpoint_location'] = midpoint_location

		local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = coleval.track_inner_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)
		try:
			peak_location,midpoint_location = coleval.plot_leg_radiation_tracking(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution,filename_root,filename_root_add,laser_to_analyse,scenario,which_leg='inner',x_point_L_pol=inner_L_poloidal_x_point_all)
		except Exception as e:
			logging.exception('with error: ' + str(e))
			print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_inner_leg_radiation_tracking.eps')
		inverted_dict[str(grid_resolution)]['inner_leg_only_local_power'] = local_power_all
		inverted_dict[str(grid_resolution)]['inner_leg_only_local_emissivity'] = local_mean_emis_all
		inverted_dict[str(grid_resolution)]['inner_leg_only_length_all'] = leg_length_all
		inverted_dict[str(grid_resolution)]['inner_leg_only_length_interval_all'] = leg_length_interval_all
		inverted_dict[str(grid_resolution)]['inner_leg_only_peak_location'] = peak_location
		inverted_dict[str(grid_resolution)]['inner_leg_only_midpoint_location'] = midpoint_location


		local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = coleval.track_outer_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,type='separatrix',leg_resolution=0.05)
		try:
			peak_location,midpoint_location = coleval.plot_leg_radiation_tracking(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution,filename_root,filename_root_add,laser_to_analyse,scenario,which_leg='outer',x_point_L_pol=outer_L_poloidal_x_point_all,which_part_of_separatrix='separatrix')
		except Exception as e:
			logging.exception('with error: ' + str(e))
			print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_outer_leg_radiation_tracking.eps')
		inverted_dict[str(grid_resolution)]['outer_separatrix_local_power'] = local_power_all
		inverted_dict[str(grid_resolution)]['outer_separatrix_local_emissivity'] = local_mean_emis_all
		inverted_dict[str(grid_resolution)]['outer_separatrix_length_all'] = leg_length_all
		inverted_dict[str(grid_resolution)]['outer_separatrix_length_interval_all'] = leg_length_interval_all
		inverted_dict[str(grid_resolution)]['outer_separatrix_peak_location'] = peak_location
		inverted_dict[str(grid_resolution)]['outer_separatrix_midpoint_location'] = midpoint_location

		local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution = coleval.track_inner_leg_radiation(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction,type='separatrix')
		try:
			peak_location,midpoint_location = coleval.plot_leg_radiation_tracking(inverted_data,inversion_R,inversion_Z,time_full_binned_crop,local_mean_emis_all,local_power_all,leg_length_interval_all,leg_length_all,data_length,leg_resolution,filename_root,filename_root_add,laser_to_analyse,scenario,which_leg='inner',x_point_L_pol=inner_L_poloidal_x_point_all,which_part_of_separatrix='separatrix')
		except Exception as e:
			logging.exception('with error: ' + str(e))
			print('failed to print\n'+'/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_inner_leg_radiation_tracking.eps')
		inverted_dict[str(grid_resolution)]['inner_separatrix_local_power'] = local_power_all
		inverted_dict[str(grid_resolution)]['inner_separatrix_local_emissivity'] = local_mean_emis_all
		inverted_dict[str(grid_resolution)]['inner_separatrix_length_all'] = leg_length_all
		inverted_dict[str(grid_resolution)]['inner_separatrix_length_interval_all'] = leg_length_interval_all
		inverted_dict[str(grid_resolution)]['inner_separatrix_peak_location'] = peak_location
		inverted_dict[str(grid_resolution)]['inner_separatrix_midpoint_location'] = midpoint_location

		gas = get_gas_info(shotnumber)
		gas_core = 0.
		gas_core_valves = []
		gas_div = 0.
		gas_div_valves = []
		gas_all = 0.
		gas_all_valves = []
		gas_inner = 0.
		gas_inner_valves = []
		gas_outer = 0.
		gas_outer_valves = []
		for valve in gas:
			gas_time = valve['time']
			select = np.logical_and(gas_time>time_full_binned_crop.min(),gas_time<time_full_binned_crop.max())
			gas_time = gas_time[select]
			gas_all += valve['flux'][select]
			gas_all_valves.append(valve['valve'])
			if valve['valve'][:3] == 'hfs':
				gas_core += valve['flux'][select]
				gas_core_valves.append(valve['valve'])
				gas_inner += valve['flux'][select]
				gas_inner_valves.append(valve['valve'])
			elif valve['valve'][:4] in ['pfr_','lfsd','lfss']:
				gas_div += valve['flux'][select]
				gas_div_valves.append(valve['valve'])
			elif valve['valve'][:4] in ['lfs_','lfsv']:
				gas_core += valve['flux'][select]
				gas_core_valves.append(valve['valve'])
				gas_outer += valve['flux'][select]
				gas_outer_valves.append(valve['valve'])

		full_saved_file_dict_FAST['multi_instrument']['gas_time'] = gas_time
		full_saved_file_dict_FAST['multi_instrument']['gas_all'] = gas_all
		full_saved_file_dict_FAST['multi_instrument']['gas_all_valves'] = gas_all_valves
		full_saved_file_dict_FAST['multi_instrument']['gas_core'] = gas_core
		full_saved_file_dict_FAST['multi_instrument']['gas_core_valves'] = gas_core_valves
		full_saved_file_dict_FAST['multi_instrument']['gas_inner'] = gas_inner
		full_saved_file_dict_FAST['multi_instrument']['gas_inner_valves'] = gas_inner_valves
		full_saved_file_dict_FAST['multi_instrument']['gas_outer'] = gas_outer
		full_saved_file_dict_FAST['multi_instrument']['gas_outer_valves'] = gas_outer_valves
		full_saved_file_dict_FAST['multi_instrument']['gas_div'] = gas_div
		full_saved_file_dict_FAST['multi_instrument']['gas_div_valves'] = gas_div_valves

		print('marker pre LP')
		# reading LP data
		import pandas as pd
		badLPs_V0 = np.array(pd.read_csv('/home/ffederic/work/analysis_scripts/scripts/from_Peter/30473_badLPs_V0.csv',index_col=0).index.tolist())
		FF_LP_path = '/home/ffederic/work/irvb/from_pryan_LP'
		path_alternate='/common/uda-scratch/pryan'

		import subprocess
		import tempfile
		from scipy.interpolate import interp1d
		# 2024/07/02 I have to modify this code, as it works only with python 3.7, and not 3.9.
		# I tried to find a reasonable way to do it, but I couldn't really find it, so I simply skip this entirely if I'm in Python3.9
		# Note: the code seems to run well also in Python 3.7
		from mastu_exhaust_analysis.eich_gui import prepare_data_for_Eich_fit_fn_time,Eich_fit_fn_time
		from mastu_exhaust_analysis.flux_expansion import calc_fpol
		fpol=calc_fpol(shotnumber,max_s=20e-2,number_steps=30)
		lambda_q_compression = (fpol['fpol_lower']/np.sin(fpol['poloidal_angle_lower']))[:,0]
		lambda_q_compression_lower_interp = interp1d(fpol['time'],lambda_q_compression,fill_value="extrapolate",bounds_error=False)
		lambda_q_compression = (fpol['fpol_upper']/np.sin(fpol['poloidal_angle_upper']))[:,0]
		lambda_q_compression_upper_interp = interp1d(fpol['time'],lambda_q_compression,fill_value="extrapolate",bounds_error=False)

		# to go from wall coord to s
		# client=pyuda.Client()
		limiter=client.geometry("/limiter/efit",50000, no_cal=False)
		limiter_r=limiter.data.R
		limiter_z=limiter.data.Z
		s_lookup=get_s_coords_tables_mastu(limiter_r, limiter_z, ds=1e-4, debug_plot=False)
		# coleval.reset_connection(client)
		# del client

		found = False
		for LP_file_type in ['alp0','elp0']:
			for LP_file_path in [path_alternate,FF_LP_path]:
				# print(str([LP_file_path,LP_file_type]))
				if os.path.exists(LP_file_path + '/' + LP_file_type + str(shotnumber) + '.nc'):
					found = True
					break
			if found:
				print(LP_file_type + str(shotnumber) + '.nc' + ' found in '+LP_file_path)
				break
		# if os.path.exists(path_alternate + '/' + 'elp0' + str(shotnumber) + '.nc') or os.path.exists(path_alternate + '/' + 'alp0' + str(shotnumber) + '.nc'):
		# 	LP_file_path = cp.deepcopy(path_alternate)
		# 	print('LP file found in '+LP_file_path)
		# elif os.path.exists(FF_LP_path + '/' + 'elp0' + str(shotnumber) + '.nc') or os.path.exists(FF_LP_path + '/' + 'alp0' + str(shotnumber) + '.nc'):
		# 	LP_file_path = cp.deepcopy(FF_LP_path)
		# 	print('LP file found in '+LP_file_path)
		if not found:
			newest_file = None
			newest_time = None
			for root, dirs, files in os.walk(path_alternate):
				# Skip the root folder
				if root == path_alternate:
					continue
				for LP_file_type_int in ['alp0','elp0']:
					if LP_file_type_int + str(shotnumber) + '.nc' in files:
						full_path = os.path.join(root, LP_file_type_int + str(shotnumber) + '.nc')
						file_time = os.path.getmtime(full_path)
						if newest_time is None or file_time > newest_time:
							newest_time = file_time
							newest_file = full_path
							LP_file_type = cp.deepcopy(LP_file_type_int)
			if newest_file:
				LP_file_path = os.path.dirname(newest_file)
				print('file not found in \n'+path_alternate+  '\n' + FF_LP_path + '\nbut the newest file was found in\n'+LP_file_path)
			else:
				print('file not found, LP analysis will fail')
				LP_file_path = cp.deepcopy(FF_LP_path)

		try:
			# I don't think is necessary, in reality.
			# lp_data.contour_plot does not work in Python 3.9, but compare_shots works fine
			# Eich_fit_fn_time works only in python 3.7 because of the function iminuit.Struct
			if sys.version_info.major==3 and sys.version_info.minor==7:
				lambda_q_determination = True
			else:
				print('LPs can be real properly only with Pithon 3.7, and it was not used here, so this step is skipped')
				lambda_q_determination = False
				# sblu=sgne	# I want an error to occour, as LPs can be real properly only with Pithon 3.7
			# try:
			# 	fdir = coleval.uda_transfer(shotnumber,'elp',extra_path='/0'+str(shotnumber)[:2])
			# 	lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path = os.path.split(fdir)[0])
			# 	os.remove(fdir)
			# except:
			# 	lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path=LP_file_path)
			try:	# considering that Peter just (12/02/2025) renwed the analisys code and is saving the files locally, I look at that first now
				lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path=LP_file_path,LP_file_type=LP_file_type)
			except:
				fdir = coleval.uda_transfer(shotnumber,'elp',extra_path='/0'+str(shotnumber)[:2])
				lp_data,output_contour1 = coleval.read_LP_data(shotnumber,path = os.path.split(fdir)[0])
				os.remove(fdir)
			if True:	# use ot Peter Ryan function to combine data within time slice and to filter when strike point is on a dead channel
				temp = np.logical_and(time_full_binned_crop>output_contour1['time'][0][0].min(),time_full_binned_crop<output_contour1['time'][0][0].max())
				trange = tuple(map(list, np.array([time_full_binned_crop[temp] - np.mean(np.diff(time_full_binned_crop))/2,time_full_binned_crop[temp] + np.mean(np.diff(time_full_binned_crop))/2]).T))
				# LP_lambdaq_time = np.mean(trange,axis=1)
				try:
					output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T1','T2','T3','T4','T5'],show=False)
					temp = output_contour1['y'][0][0]
					temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
					temp[np.isnan(temp)] = 0
					if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
						for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
							if probe_name in badLPs_V0:
								temp[:,i_] = 0
					# s10_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[5,1]),axis=0)	# threshold for broken probes
					s10_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
					s10_lower_s = output_contour1['s'][0][0]
					s10_lower_r = output_contour1['R'][0][0]
					s10_lower_z = output_contour1['Z'][0][0]
					output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=10, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False)
					s10_lower_jsat = []
					s10_lower_jsat_sigma = []
					s10_lower_jsat_r = []
					s10_lower_jsat_s = []
					for i in range(len(output)):
						s10_lower_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
						s10_lower_jsat_sigma.append(output[i]['y_error'][0][0])
						s10_lower_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
						s10_lower_jsat_s.append(output[i]['s'][0][0])
					if lambda_q_determination:	# section to calculate OMP lambda_q
						output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=10, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
						x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
						fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

						lambda_q_compression = lambda_q_compression_lower_interp(time_all_finite)
						lambda_q_10_lower = fit_store['lambda_q']/lambda_q_compression
						if not len(lambda_q_10_lower)==0:	# this means that there was no data to fit
							lambda_q_10_lower = lambda_q_10_lower[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_10_lower_sigma = fit_store['lambda_q_err']/lambda_q_compression
							lambda_q_10_lower_sigma = lambda_q_10_lower_sigma[fit_store['acceptable_fit'].astype(bool)]
							time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_10_lower = [lambda_q_10_lower[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
							lambda_q_10_lower_sigma = [lambda_q_10_lower_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
						else:
							lambda_q_10_lower = np.ones((len(time_full_binned_crop)))*np.nan
							lambda_q_10_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
						if False:
							# plt.figure()
							# for i in range(len(Eich)):
							# 	try:
							# 		i = i*3
							# 		gna, = plt.plot(Eich[i]['R'][0][0][0],Eich[i]['y'][0][0][0],'--',label=r'$\lambda_q =$ %.3gmm, time=%.3gms' %(Eich[i]['lambda_q'][0][0][0]*1000,np.nanmean(Eich[i]['time'][0][0][0])*1000))
							# 		# plt.plot(output[i]['R'][0][0],output[i]['y'][0][0],'--',color=gna.get_color())
							# 		plt.errorbar(output[i]['R'][0][0],output[i]['y'][0][0],yerr=output[i]['y_error'][0][0],color=gna.get_color())
							# 	except:
							# 		pass
							# plt.legend()
							from mastu_exhaust_analysis.eich_gui import prepare_data_for_Eich_fit_fn_time,Eich_fit_fn_time,Eich_fit_GUI
							fit_store_edit, start_limits_store_edit=Eich_fit_GUI(time_all_finite,x_all_finite,y_all_finite,y_err_all_finite,fit_store,start_limits_store)
				except:
					s10_lower_good_probes = np.zeros((10)).astype(bool)
					s10_lower_s = np.zeros((10))
					s10_lower_r = np.zeros((10))
					s10_lower_jsat = np.zeros((len(trange),10))
					s10_lower_jsat_sigma = np.zeros((len(trange),10))
					s10_lower_jsat_r = np.zeros((len(trange),10))
					s10_lower_jsat_s = np.zeros((len(trange),10))
					lambda_q_10_lower = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_10_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				plt.close('all')

				try:
					output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=4, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
					temp = output_contour1['y'][0][0]
					temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
					temp[np.isnan(temp)] = 0
					if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
						for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
							if probe_name in badLPs_V0:
								temp[:,i_] = 0
					# s4_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
					s4_lower_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
					s4_lower_s = output_contour1['s'][0][0]
					s4_lower_r = output_contour1['R'][0][0]
					s4_lower_z = output_contour1['Z'][0][0]
					output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=4, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False)
					s4_lower_jsat = []
					s4_lower_jsat_sigma = []
					s4_lower_jsat_r = []
					s4_lower_jsat_s = []
					for i in range(len(output)):
						s4_lower_jsat.append(output[i]['y'][0][0])	# here there are only small probes
						s4_lower_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only small probes
						s4_lower_jsat_r.append(output[i]['R'][0][0])	# here there are only small probes
						s4_lower_jsat_s.append(output[i]['s'][0][0])
					if lambda_q_determination:	# section to calculate OMP lambda_q
						output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=4, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
						x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
						fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

						lambda_q_compression = lambda_q_compression_lower_interp(time_all_finite)
						lambda_q_4_lower = fit_store['lambda_q']/lambda_q_compression
						if not len(lambda_q_4_lower)==0:	# this means that there was no data to fit
							lambda_q_4_lower = lambda_q_4_lower[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_4_lower_sigma = fit_store['lambda_q_err']/lambda_q_compression
							lambda_q_4_lower_sigma = lambda_q_4_lower_sigma[fit_store['acceptable_fit'].astype(bool)]
							time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_4_lower = [lambda_q_4_lower[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
							lambda_q_4_lower_sigma = [lambda_q_4_lower_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
						else:
							lambda_q_4_lower = np.ones((len(time_full_binned_crop)))*np.nan
							lambda_q_4_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				except:
					s4_lower_good_probes = np.zeros((10)).astype(bool)
					s4_lower_s = np.zeros((10))
					s4_lower_r = np.zeros((10))
					s4_lower_z = np.zeros((10))
					s4_lower_jsat = np.zeros((len(trange),10))
					s4_lower_jsat_sigma = np.zeros((len(trange),10))
					s4_lower_jsat_r = np.zeros((len(trange),10))
					s4_lower_jsat_s = np.zeros((len(trange),10))
					lambda_q_4_lower = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_4_lower_sigma = np.ones((len(time_full_binned_crop)))*np.nan
					plt.close('all')

				try:
					output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=4, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
					temp = output_contour1['y'][0][0]
					temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
					temp[np.isnan(temp)] = 0
					if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
						for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
							if probe_name in badLPs_V0:
								temp[:,i_] = 0
					# s4_upper_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
					s4_upper_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
					s4_upper_s = output_contour1['s'][0][0]
					s4_upper_r = output_contour1['R'][0][0]
					s4_upper_z = output_contour1['Z'][0][0]
					output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=4, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False)
					s4_upper_jsat = []
					s4_upper_jsat_sigma = []
					s4_upper_jsat_r = []
					s4_upper_jsat_s = []
					for i in range(len(output)):
						s4_upper_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
						s4_upper_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only standard probes
						s4_upper_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
						s4_upper_jsat_s.append(output[i]['s'][0][0])
					if lambda_q_determination:	# section to calculate OMP lambda_q
						output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=4, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4','T5'],time_combine=True,show=False,Eich_fit=False)
						x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
						fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

						lambda_q_compression = lambda_q_compression_upper_interp(time_all_finite)
						lambda_q_4_upper = fit_store['lambda_q']/lambda_q_compression
						if not len(lambda_q_4_upper)==0:	# this means that there was no data to fit
							lambda_q_4_upper = lambda_q_4_upper[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_4_upper_sigma = fit_store['lambda_q_err']/lambda_q_compression
							lambda_q_4_upper_sigma = lambda_q_4_upper_sigma[fit_store['acceptable_fit'].astype(bool)]
							time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_4_upper = [lambda_q_4_upper[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
							lambda_q_4_upper_sigma = [lambda_q_4_upper_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
						else:
							lambda_q_4_upper = np.ones((len(time_full_binned_crop)))*np.nan
							lambda_q_4_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				except:
					s4_upper_good_probes = np.zeros((10)).astype(bool)
					s4_upper_s = np.zeros((10))
					s4_upper_r = np.zeros((10))
					s4_upper_jsat = np.zeros((len(trange),10))
					s4_upper_jsat_sigma = np.zeros((len(trange),10))
					s4_upper_jsat_r = np.zeros((len(trange),10))
					s4_upper_jsat_s = np.zeros((len(trange),10))
					lambda_q_4_upper = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_4_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				plt.close('all')

				try:
					output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['C5','C6','T2','T3','T4'],show=False)
					temp = output_contour1['y'][0][0]
					temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
					temp[np.isnan(temp)] = 0
					if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
						for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
							if probe_name in badLPs_V0:
								temp[:,i_] = 0
					# s10_upper_std_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
					s10_upper_std_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
					s10_upper_std_s = output_contour1['s'][0][0]
					s10_upper_std_r = output_contour1['R'][0][0]
					s10_upper_std_z = output_contour1['Z'][0][0]
					output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-4,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4'],time_combine=True,show=False)
					s10_upper_std_jsat = []
					s10_upper_std_jsat_sigma = []
					s10_upper_std_jsat_r = []
					s10_upper_std_jsat_s = []
					for i in range(len(output)):
						s10_upper_std_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
						s10_upper_std_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only standard probes
						s10_upper_std_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
						s10_upper_std_jsat_s.append(output[i]['s'][0][0])
					if lambda_q_determination:	# section to calculate OMP lambda_q
						output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'q_tile', coordinate='s',tiles=['C5','C6','T2','T3','T4'],time_combine=True,show=False,Eich_fit=False)
						x_all_finite,y_all_finite,y_err_all_finite,time_all_finite = prepare_data_for_Eich_fit_fn_time(output)
						fit_store,start_limits_store=Eich_fit_fn_time(x_all_finite,y_all_finite,y_err_all_finite,time_all_finite)

						lambda_q_compression = lambda_q_compression_upper_interp(time_all_finite)
						lambda_q_10_upper = fit_store['lambda_q']/lambda_q_compression
						if not len(lambda_q_10_upper)==0:	# this means that there was no data to fit
							lambda_q_10_upper = lambda_q_10_upper[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_10_upper_sigma = fit_store['lambda_q_err']/lambda_q_compression
							lambda_q_10_upper_sigma = lambda_q_10_upper_sigma[fit_store['acceptable_fit'].astype(bool)]
							time_all_finite = time_all_finite[fit_store['acceptable_fit'].astype(bool)]
							lambda_q_10_upper = [lambda_q_10_upper[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
							lambda_q_10_upper_sigma = [lambda_q_10_upper_sigma[np.abs(time_all_finite-time).argmin()] if np.nanmin(np.abs(time_all_finite-time))<np.diff(time_full_binned_crop).mean()/3 else np.nan for time in time_full_binned_crop]
						else:
							lambda_q_10_upper = np.ones((len(time_full_binned_crop)))*np.nan
							lambda_q_10_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				except:
					s10_upper_std_good_probes = np.zeros((10)).astype(bool)
					s10_upper_std_s = np.zeros((10))
					s10_upper_std_r =np.zeros((10))
					s10_upper_std_jsat = np.zeros((len(trange),10))
					s10_upper_std_jsat_sigma = np.zeros((len(trange),10))
					s10_upper_std_jsat_r = np.zeros((len(trange),10))
					s10_upper_std_jsat_s = np.zeros((len(trange),10))
					lambda_q_10_upper = np.ones((len(time_full_binned_crop)))*np.nan
					lambda_q_10_upper_sigma = np.ones((len(time_full_binned_crop)))*np.nan
				plt.close('all')

				try:
					output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='R',tiles=['T5'],show=False)
					temp = output_contour1['y'][0][0]
					temp = generic_filter(temp,np.nanmean,size=(int(0.002/np.diff(output_contour1['time'][0][0]).mean()),1))	# this is a way to fix that there are sometimes 6 nans for each good data point. i smooth ver 2ms, just to make sure
					temp[np.isnan(temp)] = 0
					if shotnumber < 45514:	# 21/02/2025 Peter told me the the bad probes issue arises only in MU01 as most of it was dome in swept mode. from late MU01 onward he only did Isat mode, that does not have this issue
						for i_,probe_name in enumerate(output_contour1['probe_name'][0][0]):
							if probe_name in badLPs_V0:
								temp[:,i_] = 0
					# s10_upper_large_good_probes = np.nanmax(median_filter((temp>0.005),size=[1,5]),axis=0)	# threshold for broken probes
					s10_upper_large_good_probes = np.nanmax(median_filter((temp>0.005),size=[int((0.015/(np.diff(output_contour1['time'][0][0]).mean()))//2*2+1),1]),axis=0)	# threshold for broken probes	# I change the median interval from 5 data point to the equivalent to 10ms, just to be sure that there is enough data
					s10_upper_large_s = output_contour1['s'][0][0]
					s10_upper_large_r = output_contour1['R'][0][0]
					s10_upper_large_z = output_contour1['Z'][0][0]
					output,Eich=compare_shots(filepath=LP_file_path+'/',shot=[shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='s',tiles=['T5'],time_combine=True,show=False)
					s10_upper_large_jsat = []
					s10_upper_large_jsat_sigma = []
					s10_upper_large_jsat_r = []
					s10_upper_large_jsat_s = []
					for i in range(len(output)):
						s10_upper_large_jsat.append(output[i]['y'][0][0])	# here there are only standard probes
						s10_upper_large_jsat_sigma.append(output[i]['y_error'][0][0])	# here there are only standard probes
						s10_upper_large_jsat_r.append(output[i]['R'][0][0])	# here there are only standard probes
						s10_upper_large_jsat_s.append(output[i]['s'][0][0])
				except:
					s10_upper_large_good_probes = np.zeros((10)).astype(bool)
					s10_upper_large_s = np.zeros((10))
					s10_upper_large_r = np.zeros((10))
					s10_upper_large_jsat = np.zeros((len(trange),10))
					s10_upper_large_jsat_sigma = np.zeros((len(trange),10))
					s10_upper_large_jsat_r = np.zeros((len(trange),10))
					s10_upper_large_jsat_s = np.zeros((len(trange),10))
				plt.close('all')

				closeness_limit_to_dead_channels = 0.01	# m
				closeness_limit_for_good_channels = np.median(np.abs(np.diff(s10_lower_s).tolist()+np.diff(s4_lower_s).tolist()+np.diff(s4_upper_s).tolist()+np.diff(s10_upper_std_s).tolist()+np.diff(s10_upper_large_s).tolist()))*2	# *5
				jsat_upper_inner_small_max = []
				jsat_upper_outer_small_max = []
				jsat_upper_inner_mid_max = []
				jsat_upper_outer_mid_max = []
				jsat_upper_inner_large_max = []	# T5
				jsat_upper_outer_large_max = []	# T5
				jsat_lower_inner_small_max = []	#central column
				jsat_lower_outer_small_max = []	#central column
				jsat_lower_inner_mid_max = []
				jsat_lower_outer_mid_max = []
				jsat_lower_inner_large_max = []
				jsat_lower_outer_large_max = []

				jsat_upper_inner_small_max_sigma = []
				jsat_upper_outer_small_max_sigma = []
				jsat_upper_inner_mid_max_sigma = []
				jsat_upper_outer_mid_max_sigma = []
				jsat_upper_inner_large_max_sigma = []	# T5
				jsat_upper_outer_large_max_sigma = []	# T5
				jsat_lower_inner_small_max_sigma = []	#central column
				jsat_lower_outer_small_max_sigma = []	#central column
				jsat_lower_inner_mid_max_sigma = []
				jsat_lower_outer_mid_max_sigma = []
				jsat_lower_inner_large_max_sigma = []
				jsat_lower_outer_large_max_sigma = []

				jsat_upper_inner_small_integrated = []
				jsat_upper_outer_small_integrated = []
				jsat_upper_inner_mid_integrated = []
				jsat_upper_outer_mid_integrated = []
				jsat_upper_inner_large_integrated = []	# T5
				jsat_upper_outer_large_integrated = []	# T5
				jsat_lower_inner_small_integrated = []	#central column
				jsat_lower_outer_small_integrated = []	#central column
				jsat_lower_inner_mid_integrated = []
				jsat_lower_outer_mid_integrated = []
				jsat_lower_inner_large_integrated = []
				jsat_lower_outer_large_integrated = []

				jsat_upper_inner_small_integrated_sigma = []
				jsat_upper_outer_small_integrated_sigma = []
				jsat_upper_inner_mid_integrated_sigma = []
				jsat_upper_outer_mid_integrated_sigma = []
				jsat_upper_inner_large_integrated_sigma = []	# T5
				jsat_upper_outer_large_integrated_sigma = []	# T5
				jsat_lower_inner_small_integrated_sigma = []	#central column
				jsat_lower_outer_small_integrated_sigma = []	#central column
				jsat_lower_inner_mid_integrated_sigma = []
				jsat_lower_outer_mid_integrated_sigma = []
				jsat_lower_inner_large_integrated_sigma = []
				jsat_lower_outer_large_integrated_sigma = []

				# presenecting only good probes
				s10_lower_jsat = np.array(s10_lower_jsat)[:,s10_lower_good_probes]
				s10_lower_jsat_sigma = np.array(s10_lower_jsat_sigma)[:,s10_lower_good_probes]
				s10_lower_jsat_r = np.array(s10_lower_jsat_r)[:,s10_lower_good_probes]
				s10_lower_jsat_s = np.array(s10_lower_jsat_s)[:,s10_lower_good_probes]
				s4_lower_jsat = np.array(s4_lower_jsat)[:,s4_lower_good_probes]
				s4_lower_jsat_sigma = np.array(s4_lower_jsat_sigma)[:,s4_lower_good_probes]
				s4_lower_jsat_r = np.array(s4_lower_jsat_r)[:,s4_lower_good_probes]
				s4_lower_jsat_s = np.array(s4_lower_jsat_s)[:,s4_lower_good_probes]
				s4_upper_jsat = np.array(s4_upper_jsat)[:,s4_upper_good_probes]
				s4_upper_jsat_sigma = np.array(s4_upper_jsat_sigma)[:,s4_upper_good_probes]
				s4_upper_jsat_r = np.array(s4_upper_jsat_r)[:,s4_upper_good_probes]
				s4_upper_jsat_s = np.array(s4_upper_jsat_s)[:,s4_upper_good_probes]
				s10_upper_std_jsat = np.array(s10_upper_std_jsat)[:,s10_upper_std_good_probes]
				s10_upper_std_jsat_sigma = np.array(s10_upper_std_jsat_sigma)[:,s10_upper_std_good_probes]
				s10_upper_std_jsat_r = np.array(s10_upper_std_jsat_r)[:,s10_upper_std_good_probes]
				s10_upper_std_jsat_s = np.array(s10_upper_std_jsat_s)[:,s10_upper_std_good_probes]
				s10_upper_large_jsat = np.array(s10_upper_large_jsat)[:,s10_upper_large_good_probes]
				s10_upper_large_jsat_sigma = np.array(s10_upper_large_jsat_sigma)[:,s10_upper_large_good_probes]
				s10_upper_large_jsat_r = np.array(s10_upper_large_jsat_r)[:,s10_upper_large_good_probes]
				s10_upper_large_jsat_s = np.array(s10_upper_large_jsat_s)[:,s10_upper_large_good_probes]

				# skipped = 0
				for time in time_full_binned_crop:
					if time<np.min(output[0]['time']) or time>np.max(output[-1]['time']):
						jsat_upper_inner_small_max.append(np.nan)
						jsat_upper_outer_small_max.append(np.nan)
						jsat_upper_inner_small_integrated.append(np.nan)
						jsat_upper_outer_small_integrated.append(np.nan)
						jsat_upper_inner_mid_max.append(np.nan)
						jsat_upper_inner_mid_integrated.append(np.nan)
						jsat_upper_outer_mid_max.append(np.nan)
						jsat_upper_outer_mid_integrated.append(np.nan)
						jsat_upper_inner_large_max.append(np.nan)
						jsat_upper_inner_large_integrated.append(np.nan)
						jsat_upper_outer_large_max.append(np.nan)
						jsat_upper_outer_large_integrated.append(np.nan)
						jsat_lower_inner_small_max.append(np.nan)
						jsat_lower_inner_small_integrated.append(np.nan)
						jsat_lower_outer_small_max.append(np.nan)
						jsat_lower_outer_small_integrated.append(np.nan)
						jsat_lower_inner_mid_max.append(np.nan)
						jsat_lower_inner_mid_integrated.append(np.nan)
						jsat_lower_outer_mid_max.append(np.nan)
						jsat_lower_outer_mid_integrated.append(np.nan)
						jsat_lower_inner_large_max.append(np.nan)
						jsat_lower_outer_large_max.append(np.nan)
						jsat_lower_inner_large_integrated.append(np.nan)
						jsat_lower_outer_large_integrated.append(np.nan)

						jsat_upper_inner_small_max_sigma.append(np.nan)
						jsat_upper_outer_small_max_sigma.append(np.nan)
						jsat_upper_inner_small_integrated_sigma.append(np.nan)
						jsat_upper_outer_small_integrated_sigma.append(np.nan)
						jsat_upper_inner_mid_max_sigma.append(np.nan)
						jsat_upper_inner_mid_integrated_sigma.append(np.nan)
						jsat_upper_outer_mid_max_sigma.append(np.nan)
						jsat_upper_outer_mid_integrated_sigma.append(np.nan)
						jsat_upper_inner_large_max_sigma.append(np.nan)
						jsat_upper_inner_large_integrated_sigma.append(np.nan)
						jsat_upper_outer_large_max_sigma.append(np.nan)
						jsat_upper_outer_large_integrated_sigma.append(np.nan)
						jsat_lower_inner_small_max_sigma.append(np.nan)
						jsat_lower_inner_small_integrated_sigma.append(np.nan)
						jsat_lower_outer_small_max_sigma.append(np.nan)
						jsat_lower_outer_small_integrated_sigma.append(np.nan)
						jsat_lower_inner_mid_max_sigma.append(np.nan)
						jsat_lower_inner_mid_integrated_sigma.append(np.nan)
						jsat_lower_outer_mid_max_sigma.append(np.nan)
						jsat_lower_outer_mid_integrated_sigma.append(np.nan)
						jsat_lower_inner_large_max_sigma.append(np.nan)
						jsat_lower_outer_large_max_sigma.append(np.nan)
						jsat_lower_inner_large_integrated_sigma.append(np.nan)
						jsat_lower_outer_large_integrated_sigma.append(np.nan)
						# skipped += 1
						continue

					i_time = np.abs(time-np.mean(trange,axis=1)).argmin()

					i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
					inner_strike_point = [np.min(efit_reconstruction.strikepointR[i_efit_time][:2]),0,0]
					inner_strike_point[1] = -efit_reconstruction.strikepointZ[i_efit_time][np.abs(efit_reconstruction.strikepointR[i_efit_time]-inner_strike_point[0]).argmin()]
					temp = get_nearest_s_coordinates_mastu([inner_strike_point[0]],[inner_strike_point[1]],s_lookup, tol=5e-1)
					inner_strike_point[2] = np.abs(temp[0][0])	# this is all info for the upper divertor [R,Z,s]
					outer_strike_point = [np.max(efit_reconstruction.strikepointR[i_efit_time][:2]),0,0]
					outer_strike_point[1] = -efit_reconstruction.strikepointZ[i_efit_time][np.abs(efit_reconstruction.strikepointR[i_efit_time]-outer_strike_point[0]).argmin()]
					temp = get_nearest_s_coordinates_mastu([outer_strike_point[0]],[outer_strike_point[1]],s_lookup, tol=5e-1)
					outer_strike_point[2] = np.abs(temp[0][0])	# this is all info for the upper divertor [R,Z,s]
					mid_strike_point = [np.mean(efit_reconstruction.strikepointR[i_efit_time][:2]),-np.mean(efit_reconstruction.strikepointZ[i_efit_time][:2]),0]
					temp = get_nearest_s_coordinates_mastu([mid_strike_point[0]],[mid_strike_point[1]],s_lookup, tol=5e-1)
					mid_strike_point[2] = np.abs(temp[0][0])	# this is all info for the upper divertor [R,Z,s]

					# upper divertor
					# 19/02/2025 I split the second condition so that I need close good plobes on both sides of the strike point to be able to say anything
					# inner SP
					temp = s4_upper_s-inner_strike_point[2]
					s4_upper_inner_test = np.sum(np.logical_not(s4_upper_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_upper_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_upper_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s10_upper_std_s-inner_strike_point[2]
					s10_upper_std_inner_test = np.sum(np.logical_not(s10_upper_std_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s10_upper_large_s-inner_strike_point[2]
					s10_upper_large_inner_test = np.sum(np.logical_not(s10_upper_large_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					# outer SP
					temp = s4_upper_s-outer_strike_point[2]
					s4_upper_outer_test = np.sum(np.logical_not(s4_upper_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_upper_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_upper_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s10_upper_std_s-outer_strike_point[2]
					s10_upper_std_outer_test = np.sum(np.logical_not(s10_upper_std_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_std_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s10_upper_large_s-outer_strike_point[2]
					s10_upper_large_outer_test = np.sum(np.logical_not(s10_upper_large_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_upper_large_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0

					jsat_upper_inner_small_max.append(0)
					jsat_upper_inner_small_max_sigma.append(0)
					jsat_upper_outer_small_max.append(0)
					jsat_upper_outer_small_max_sigma.append(0)
					jsat_upper_inner_small_integrated.append(0)
					jsat_upper_inner_small_integrated_sigma.append(0)
					jsat_upper_outer_small_integrated.append(0)
					jsat_upper_outer_small_integrated_sigma.append(0)

					temp = [np.nan]
					if s4_upper_inner_test:
						temp = np.concatenate([[0],temp,s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]])
					if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp = np.concatenate([[0],temp,s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s4_upper_inner_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]])
					if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_upper_inner_mid_max.append(np.nanmax(temp))
					jsat_upper_inner_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = [np.nan]
					temp_sigma_extra = 0
					if s4_upper_inner_test:
						temp.append( np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]) )
					if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp.append( np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]) )
					if np.nanmax(temp) - np.nanmin(temp)<np.nanmean(temp)/2:	# step added to see if averaging among the 2 sectors rather than taking the max matters
						temp_sigma_extra = np.nanmax(temp) - np.nanmin(temp)
						temp =np.nanmean(temp)
					else:
						temp = np.nan
					if not(s4_upper_inner_test) and not(s10_upper_std_inner_test):
						temp = np.nan
					temp_sigma = 0
					if s4_upper_inner_test:
						temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]])**2,x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]])])
					if s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]])**2,x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]])])
					if not(s4_upper_inner_test) and not(s10_upper_std_inner_test):
						temp_sigma = np.nan
					jsat_upper_inner_mid_integrated.append(temp)
					jsat_upper_inner_mid_integrated_sigma.append(max(temp_sigma**0.5,temp_sigma_extra))

					temp = [np.nan]
					if s4_upper_outer_test:
						temp = np.concatenate([[0],temp,s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]])
					if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp = np.concatenate([[0],temp,s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s4_upper_outer_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]])
					if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_upper_outer_mid_max.append(np.nanmax(temp))
					jsat_upper_outer_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = [np.nan]
					temp_sigma_extra = 0
					if s4_upper_outer_test:
						temp.append( np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]) )
					if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp.append( np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]) )
					if np.nanmax(temp) - np.nanmin(temp)<np.nanmean(temp)/2:	# step added to see if averaging among the 2 sectors rather than taking the max matters
						temp_sigma_extra = np.nanmax(temp) - np.nanmin(temp)
						temp =np.nanmean(temp)
					else:
						temp = np.nan
					if not(s4_upper_outer_test) and not(s10_upper_std_outer_test):
						temp = np.nan
					temp_sigma = 0
					if s4_upper_outer_test:
						temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s4_upper_jsat_sigma[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]])**2,x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]])])
					if s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp_sigma = np.nanmax([temp_sigma,coleval.np_trapz_error((s10_upper_std_jsat_sigma[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]])**2,x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]])])
					if not(s4_upper_outer_test) and not(s10_upper_std_outer_test):
						temp_sigma = np.nan
					jsat_upper_outer_mid_integrated.append(temp)
					jsat_upper_outer_mid_integrated_sigma.append(max(temp_sigma**0.5,temp_sigma_extra))

					temp = [np.nan]
					if s10_upper_large_inner_test:
						temp = np.concatenate([[0],temp,s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s10_upper_large_inner_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_upper_inner_large_max.append(np.nanmax(temp))
					jsat_upper_inner_large_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = 0
					if s10_upper_large_inner_test:
						temp += np.trapz(s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					temp_sigma = 0
					if s10_upper_large_inner_test:
						temp_sigma += coleval.np_trapz_error((s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])**2,x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp_sigma = np.nan
					jsat_upper_inner_large_integrated.append(temp)
					jsat_upper_inner_large_integrated_sigma.append(temp_sigma**0.5)

					temp = [np.nan]
					if s10_upper_large_outer_test:
						temp = np.concatenate([[0],temp,s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s10_upper_large_outer_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_upper_outer_large_max.append(np.nanmax(temp))
					jsat_upper_outer_large_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = 0
					if s10_upper_large_outer_test:
						temp += np.trapz(s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					temp_sigma = 0
					if s10_upper_large_outer_test:
						temp_sigma += coleval.np_trapz_error((s10_upper_large_jsat_sigma[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])**2,x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp_sigma = np.nan
					jsat_upper_outer_large_integrated.append(temp)
					jsat_upper_outer_large_integrated_sigma.append(temp_sigma**0.5)


					# lower divertor
					# I add the second part to exluce the case there are NO good channel close to the strike point, diving a bit of slack allowing twice the normal distance between probes
					# 19/02/2025 I split the second condition so that I need close good plobes on both sides of the strike point to be able to say anything
					temp = s4_lower_s-(-inner_strike_point[2])
					s4_lower_inner_test = np.sum(np.logical_not(s4_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s10_lower_s-(-inner_strike_point[2])
					s10_lower_inner_test = np.sum(np.logical_not(s10_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s4_lower_s-(-outer_strike_point[2])
					s4_lower_outer_test = np.sum(np.logical_not(s4_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s4_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s4_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0
					temp = s10_lower_s-(-outer_strike_point[2])
					s10_lower_outer_test = np.sum(np.logical_not(s10_lower_good_probes)[np.abs(temp)<closeness_limit_to_dead_channels])==0 and np.sum(s10_lower_good_probes[np.logical_and(temp>0 , np.abs(temp)<closeness_limit_for_good_channels)])>0 and np.sum(s10_lower_good_probes[np.logical_and(temp<0 , np.abs(temp)<closeness_limit_for_good_channels)])>0

					temp = [np.nan]
					if s4_lower_inner_test:
						temp = np.concatenate([[0],temp,s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s4_lower_inner_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_lower_inner_small_max.append(np.nanmax(temp))
					jsat_lower_inner_small_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = 0
					if s4_lower_inner_test:
						temp += np.trapz(s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]],x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					temp_sigma = 0
					if s4_lower_inner_test:
						temp_sigma += coleval.np_trapz_error((s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])**2,x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp_sigma = np.nan
					jsat_lower_inner_small_integrated.append(-temp)
					jsat_lower_inner_small_integrated_sigma.append(temp_sigma**0.5)

					temp = [np.nan]
					if s4_lower_outer_test:
						temp = np.concatenate([[0],temp,s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s4_lower_outer_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_lower_outer_small_max.append(np.nanmax(temp))
					jsat_lower_outer_small_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = 0
					if s4_lower_outer_test:
						temp += np.trapz(s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]],x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					temp_sigma = 0
					if s4_lower_outer_test:
						temp_sigma += coleval.np_trapz_error((s4_lower_jsat_sigma[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])**2,x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp_sigma = np.nan
					jsat_lower_outer_small_integrated.append(-temp)
					jsat_lower_outer_small_integrated_sigma.append(temp_sigma**0.5)

					temp = [np.nan]
					if s10_lower_inner_test:
						temp = np.concatenate([[0],temp,s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s10_lower_inner_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_lower_inner_mid_max.append(np.nanmax(temp))
					jsat_lower_inner_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = 0
					if s10_lower_inner_test:
						temp += np.trapz(s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]],x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					temp_sigma = 0
					if s10_lower_inner_test:
						temp_sigma += coleval.np_trapz_error((s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])**2,x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp_sigma = np.nan
					jsat_lower_inner_mid_integrated.append(-temp)
					jsat_lower_inner_mid_integrated_sigma.append(temp_sigma**0.5)

					temp = [np.nan]
					if s10_lower_outer_test:
						temp = np.concatenate([[0],temp,s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]])
					temp_sigma = [np.nan]
					if s10_lower_outer_test:
						temp_sigma = np.concatenate([[0],temp_sigma,s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_lower_outer_mid_max.append(np.nanmax(temp))
					jsat_lower_outer_mid_max_sigma.append(temp_sigma[np.nanargmax(np.concatenate([temp,[-np.inf]]))])

					temp = 0
					if s10_lower_outer_test:
						temp += np.trapz(s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]],x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					temp_sigma = 0
					if s10_lower_outer_test:
						temp_sigma += coleval.np_trapz_error((s10_lower_jsat_sigma[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])**2,x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp_sigma = np.nan
					jsat_lower_outer_mid_integrated.append(-temp)
					jsat_lower_outer_mid_integrated_sigma.append(temp_sigma**0.5)

					jsat_lower_inner_large_max.append(0)
					jsat_lower_inner_large_max_sigma.append(0)
					jsat_lower_outer_large_max.append(0)
					jsat_lower_outer_large_max_sigma.append(0)
					jsat_lower_inner_large_integrated.append(0)
					jsat_lower_inner_large_integrated_sigma.append(0)
					jsat_lower_outer_large_integrated.append(0)
					jsat_lower_outer_large_integrated_sigma.append(0)

				jsat_time = cp.deepcopy(time_full_binned_crop)
				jsat_read = True

				jsat_lower_inner_small_integrated = np.array(jsat_lower_inner_small_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_small_integrated'] = jsat_lower_inner_small_integrated
				jsat_lower_outer_small_integrated = np.array(jsat_lower_outer_small_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_small_integrated'] = jsat_lower_outer_small_integrated
				jsat_lower_inner_mid_integrated = np.array(jsat_lower_inner_mid_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_mid_integrated'] = jsat_lower_inner_mid_integrated
				jsat_lower_outer_mid_integrated = np.array(jsat_lower_outer_mid_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_integrated'] = jsat_lower_outer_mid_integrated
				jsat_lower_inner_large_integrated = np.array(jsat_lower_inner_large_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_large_integrated'] = jsat_lower_inner_large_integrated
				jsat_lower_outer_large_integrated = np.array(jsat_lower_outer_large_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_large_integrated'] = jsat_lower_outer_large_integrated
				jsat_upper_inner_small_integrated = np.array(jsat_upper_inner_small_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_small_integrated'] = jsat_upper_inner_small_integrated
				jsat_upper_outer_small_integrated = np.array(jsat_upper_outer_small_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_small_integrated'] = jsat_upper_outer_small_integrated
				jsat_upper_inner_mid_integrated = np.array(jsat_upper_inner_mid_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_mid_integrated'] = jsat_upper_inner_mid_integrated
				jsat_upper_outer_mid_integrated = np.array(jsat_upper_outer_mid_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_integrated'] = jsat_upper_outer_mid_integrated
				jsat_upper_inner_large_integrated = np.array(jsat_upper_inner_large_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_large_integrated'] = jsat_upper_inner_large_integrated
				jsat_upper_outer_large_integrated = np.array(jsat_upper_outer_large_integrated)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_large_integrated'] = jsat_upper_outer_large_integrated

				jsat_lower_inner_small_integrated_sigma = np.array(jsat_lower_inner_small_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_small_integrated_sigma'] = jsat_lower_inner_small_integrated_sigma
				jsat_lower_outer_small_integrated_sigma = np.array(jsat_lower_outer_small_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_small_integrated_sigma'] = jsat_lower_outer_small_integrated_sigma
				jsat_lower_inner_mid_integrated_sigma = np.array(jsat_lower_inner_mid_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_mid_integrated_sigma'] = jsat_lower_inner_mid_integrated_sigma
				jsat_lower_outer_mid_integrated_sigma = np.array(jsat_lower_outer_mid_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_integrated_sigma'] = jsat_lower_outer_mid_integrated_sigma
				jsat_lower_inner_large_integrated_sigma = np.array(jsat_lower_inner_large_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_large_integrated_sigma'] = jsat_lower_inner_large_integrated_sigma
				jsat_lower_outer_large_integrated_sigma = np.array(jsat_lower_outer_large_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_large_integrated_sigma'] = jsat_lower_outer_large_integrated_sigma
				jsat_upper_inner_small_integrated_sigma = np.array(jsat_upper_inner_small_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_small_integrated_sigma'] = jsat_upper_inner_small_integrated_sigma
				jsat_upper_outer_small_integrated_sigma = np.array(jsat_upper_outer_small_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_small_integrated_sigma'] = jsat_upper_outer_small_integrated_sigma
				jsat_upper_inner_mid_integrated_sigma = np.array(jsat_upper_inner_mid_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_mid_integrated_sigma'] = jsat_upper_inner_mid_integrated_sigma
				jsat_upper_outer_mid_integrated_sigma = np.array(jsat_upper_outer_mid_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_integrated_sigma'] = jsat_upper_outer_mid_integrated_sigma
				jsat_upper_inner_large_integrated_sigma = np.array(jsat_upper_inner_large_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_large_integrated_sigma'] = jsat_upper_inner_large_integrated_sigma
				jsat_upper_outer_large_integrated_sigma = np.array(jsat_upper_outer_large_integrated_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_large_integrated_sigma'] = jsat_upper_outer_large_integrated_sigma

				jsat_lower_inner_small_max_sigma = np.array(jsat_lower_inner_small_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_small_max_sigma'] = jsat_lower_inner_small_max_sigma
				jsat_lower_outer_small_max_sigma = np.array(jsat_lower_outer_small_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_small_max_sigma'] = jsat_lower_outer_small_max_sigma
				jsat_lower_inner_mid_max_sigma = np.array(jsat_lower_inner_mid_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_mid_max_sigma'] = jsat_lower_inner_mid_max_sigma
				jsat_lower_outer_mid_max_sigma = np.array(jsat_lower_outer_mid_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_max_sigma'] = jsat_lower_outer_mid_max_sigma
				jsat_lower_inner_large_max_sigma = np.array(jsat_lower_inner_large_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_large_max_sigma'] = jsat_lower_inner_large_max_sigma
				jsat_lower_outer_large_max_sigma = np.array(jsat_lower_outer_large_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_large_max_sigma'] = jsat_lower_outer_large_max_sigma
				jsat_upper_inner_small_max_sigma = np.array(jsat_upper_inner_small_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_small_max_sigma'] = jsat_upper_inner_small_max_sigma
				jsat_upper_outer_small_max_sigma = np.array(jsat_upper_outer_small_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_small_max_sigma'] = jsat_upper_outer_small_max_sigma
				jsat_upper_inner_mid_max_sigma = np.array(jsat_upper_inner_mid_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_mid_max_sigma'] = jsat_upper_inner_mid_max_sigma
				jsat_upper_outer_mid_max_sigma = np.array(jsat_upper_outer_mid_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_max_sigma'] = jsat_upper_outer_mid_max_sigma
				jsat_upper_inner_large_max_sigma = np.array(jsat_upper_inner_large_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_large_max_sigma'] = jsat_upper_inner_large_max_sigma
				jsat_upper_outer_large_max_sigma = np.array(jsat_upper_outer_large_max_sigma)
				full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_large_max_sigma'] = jsat_upper_outer_large_max_sigma
			else:	# old method without Peter Ryan functions

				for probe_size,probe_type in [[9.69975e-06,'small'],[1.35553e-05,'mid'],[4.56942e-05,'large']]:
					try:
						data_s10 = lp_data.s10_lower_data.jsat[:,lp_data.s10_lower_data.physical_area==probe_size]
					except:
						data_s10 = lp_data.s10_lower_data.jsat_tile[:,lp_data.s10_lower_data.physical_area==probe_size]
					time_orig_s10 = lp_data.s10_lower_data.time
					s_orig_s10 = lp_data.s10_lower_data.s[lp_data.s10_lower_data.physical_area==probe_size]
					r_orig_s10 = lp_data.s10_lower_data.r[lp_data.s10_lower_data.physical_area==probe_size]
					tiles_covered_s10 = lp_data.s10_lower_data.tiles_covered
					try:
						data_s4 = lp_data.s4_lower_data.jsat[:,lp_data.s4_lower_data.physical_area==probe_size]
					except:
						data_s4 = lp_data.s4_lower_data.jsat_tile[:,lp_data.s4_lower_data.physical_area==probe_size]
					time_orig_s4 = lp_data.s4_lower_data.time
					s_orig_s4 = lp_data.s4_lower_data.s[lp_data.s4_lower_data.physical_area==probe_size]
					r_orig_s4 = lp_data.s4_lower_data.r[lp_data.s4_lower_data.physical_area==probe_size]
					coordinate = np.concatenate([r_orig_s4,r_orig_s10])
					jsat = np.concatenate([data_s4.T,data_s10.T]).T
					tiles_covered_s4 = lp_data.s10_lower_data.tiles_covered
					inner_max = []
					outer_max = []
					for i_time,time in enumerate(time_orig_s10):
						i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
						mid_strike_point = np.mean(efit_reconstruction.strikepointR[i_efit_time][:2])
						inner_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate<=mid_strike_point]])))
						outer_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate>mid_strike_point]])))
					if probe_type=='small':
						jsat_lower_inner_small_max = np.array(inner_max)	#central column
						jsat_lower_outer_small_max = np.array(outer_max)	#central column
					elif probe_type=='mid':
						jsat_lower_inner_mid_max = np.array(inner_max)
						jsat_lower_outer_mid_max = np.array(outer_max)
					elif probe_type=='large':
						jsat_lower_inner_large_max = np.array(inner_max)
						jsat_lower_outer_large_max = np.array(outer_max)


				for probe_size,probe_type in [[9.69975e-06,'small'],[1.35553e-05,'mid'],[4.56942e-05,'large']]:
					try:
						data_s10 = lp_data.s10_upper_data.jsat[:,lp_data.s10_upper_data.physical_area==probe_size]
					except:
						data_s10 = lp_data.s10_upper_data.jsat_tile[:,lp_data.s10_upper_data.physical_area==probe_size]
					time_orig_s10 = lp_data.s10_upper_data.time
					s_orig_s10 = lp_data.s10_upper_data.s[lp_data.s10_upper_data.physical_area==probe_size]
					r_orig_s10 = lp_data.s10_upper_data.r[lp_data.s10_upper_data.physical_area==probe_size]
					tiles_covered_s10 = lp_data.s10_upper_data.tiles_covered
					try:
						data_s4 = lp_data.s4_upper_data.jsat[:,lp_data.s4_upper_data.physical_area==probe_size]
					except:
						data_s4 = lp_data.s4_upper_data.jsat_tile[:,lp_data.s4_upper_data.physical_area==probe_size]
					time_orig_s4 = lp_data.s4_upper_data.time
					s_orig_s4 = lp_data.s4_upper_data.s[lp_data.s4_upper_data.physical_area==probe_size]
					r_orig_s4 = lp_data.s4_upper_data.r[lp_data.s4_upper_data.physical_area==probe_size]
					coordinate = np.concatenate([r_orig_s4,r_orig_s10])
					jsat = np.concatenate([data_s4.T,data_s10.T]).T
					tiles_covered_s4 = lp_data.s10_upper_data.tiles_covered
					inner_max = []
					outer_max = []
					for i_time,time in enumerate(time_orig_s10):
						i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
						mid_strike_point = np.mean(efit_reconstruction.strikepointR[i_efit_time][:2])
						inner_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate<=mid_strike_point]])))
						outer_max.append(np.nanmax(np.concatenate([[0],jsat[i_time][coordinate>mid_strike_point]])))
					if probe_type=='small':
						jsat_upper_inner_small_max = np.array(inner_max)
						jsat_upper_outer_small_max = np.array(outer_max)
					elif probe_type=='mid':
						jsat_upper_inner_mid_max = np.array(inner_max)
						jsat_upper_outer_mid_max = np.array(outer_max)
					elif probe_type=='large':
						jsat_upper_inner_large_max = np.array(inner_max)	# T5
						jsat_upper_outer_large_max = np.array(outer_max)	# T5
				# time,r = np.meshgrid(time_orig_s4,coordinate)
				# plt.figure(figsize=(10,10))
				# plt.pcolor(r,time,jsat.T,norm=LogNorm())#,vmin=np.nanmax(data)*1e-3)
				# plt.colorbar().set_label('jsat')
				# for __i in range(np.shape(efit_reconstruction.strikepointR)[1]):
				# 	plt.plot(efit_reconstruction.strikepointR[:,__i],efit_reconstruction.time,'--r')
				# plt.grid()
				# plt.xlim(left=0.5)
				# plt.ylabel('time [s]')
				# plt.xlabel('R [m]')
				# # plt.title(str(shot)+'\n'+str(tiles_covered))
				# plt.pause(0.01)
				jsat_time = cp.deepcopy(time_orig_s10)
				jsat_read = True

			full_saved_file_dict_FAST['multi_instrument']['jsat_time'] = jsat_time
			jsat_lower_inner_small_max = np.array(jsat_lower_inner_small_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_small_max'] = jsat_lower_inner_small_max
			jsat_lower_outer_small_max = np.array(jsat_lower_outer_small_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_small_max'] = jsat_lower_outer_small_max
			jsat_lower_inner_mid_max = np.array(jsat_lower_inner_mid_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_mid_max'] = jsat_lower_inner_mid_max
			jsat_lower_outer_mid_max = np.array(jsat_lower_outer_mid_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_max'] = jsat_lower_outer_mid_max
			jsat_lower_inner_large_max = np.array(jsat_lower_inner_large_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_lower_inner_large_max'] = jsat_lower_inner_large_max
			jsat_lower_outer_large_max = np.array(jsat_lower_outer_large_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_large_max'] = jsat_lower_outer_large_max
			jsat_upper_inner_small_max = np.array(jsat_upper_inner_small_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_small_max'] = jsat_upper_inner_small_max
			jsat_upper_outer_small_max = np.array(jsat_upper_outer_small_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_small_max'] = jsat_upper_outer_small_max
			jsat_upper_inner_mid_max = np.array(jsat_upper_inner_mid_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_mid_max'] = jsat_upper_inner_mid_max
			jsat_upper_outer_mid_max = np.array(jsat_upper_outer_mid_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_max'] = jsat_upper_outer_mid_max
			jsat_upper_inner_large_max = np.array(jsat_upper_inner_large_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_upper_inner_large_max'] = jsat_upper_inner_large_max
			jsat_upper_outer_large_max = np.array(jsat_upper_outer_large_max)
			full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_large_max'] = jsat_upper_outer_large_max
			if lambda_q_determination:
				# full_saved_file_dict_FAST['multi_instrument']['LP_lambdaq_time'] = np.array(LP_lambdaq_time)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_10_lower'] = np.array(lambda_q_10_lower)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_10_lower_sigma'] = np.array(lambda_q_10_lower_sigma)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_4_lower'] = np.array(lambda_q_4_lower)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_4_lower_sigma'] = np.array(lambda_q_4_lower_sigma)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_4_upper'] = np.array(lambda_q_4_upper)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_4_upper_sigma'] = np.array(lambda_q_4_upper_sigma)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_10_upper'] = np.array(lambda_q_10_upper)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_10_upper_sigma'] = np.array(lambda_q_10_upper_sigma)
			print('marker LP done')
		except Exception as e:
			logging.exception('with error: ' + str(e))
			jsat_read = False
			lambda_q_determination = False
			print('LP skipped')

		from mastu_exhaust_analysis.read_efit import read_epm
		fdir = coleval.uda_transfer(shotnumber,'epm')
		efit_data = read_epm(fdir,calc_bfield=True)
		os.remove(fdir)
		dWdt = calc_w_dot(efit_data = efit_data)
		pohm = calc_pohm(efit_data = efit_data)
		dWdt['data'][~np.isfinite(dWdt['data'])] = 0.0
		pohm['pohm'][~np.isfinite(pohm['pohm'])] = 0.0
		smooth_dt = 0.015
		window_size = int(smooth_dt / np.median(np.gradient(efit_data['t'])))
		if window_size % 2 == 0:
			window_size = window_size + 1
		poly_order = np.min([3, window_size-1])

		dWdt = scipy.signal.savgol_filter(dWdt['data'], window_size, poly_order)
		dWdt = np.interp(time_full_binned_crop,pohm['t'],dWdt,right=0.,left=0.)
		output_pohm = scipy.signal.savgol_filter(pohm['pohm'], window_size, poly_order)
		output_pohm = np.interp(time_full_binned_crop,pohm['t'],output_pohm,right=0.,left=0.)
		# output_pdw_dt = scipy.signal.savgol_filter(wdot['data'], window_size, poly_order)
		ss_absorption = 0.8
		sw_absorption = 0.4
		try:
			data = client.get('/XNB/SS/BEAMPOWER',laser_to_analyse[-9:-4])
			data_time = coleval.shift_beam_to_pulse_time(data.time.data, 'SS', laser_to_analyse[-9:-4])
			SS_BEAMPOWER = np.interp(efit_data['t'],data_time,data.data,right=0.,left=0.)*1e6	# W
		except:
			SS_BEAMPOWER = np.zeros_like(efit_data['t'])
		try:
			data = client.get('/XNB/SW/BEAMPOWER',laser_to_analyse[-9:-4])
			data_time = coleval.shift_beam_to_pulse_time(data.time.data, 'SW', laser_to_analyse[-9:-4])
			SW_BEAMPOWER = np.interp(efit_data['t'],data_time,data.data,right=0.,left=0.)*1e6	# W
		except:
			SW_BEAMPOWER = np.zeros_like(efit_data['t'])
		SS_BEAMPOWER *= ss_absorption
		SW_BEAMPOWER *= sw_absorption
		SS_BEAMPOWER = scipy.signal.savgol_filter(SS_BEAMPOWER, window_size, poly_order)
		SW_BEAMPOWER = scipy.signal.savgol_filter(SW_BEAMPOWER, window_size, poly_order)
		SS_BEAMPOWER = np.interp(time_full_binned_crop,efit_data['t'],SS_BEAMPOWER,right=0.,left=0.)
		SW_BEAMPOWER = np.interp(time_full_binned_crop,efit_data['t'],SW_BEAMPOWER,right=0.,left=0.)
		stored_energy = efit_data['stored_energy']
		stored_energy = scipy.signal.savgol_filter(stored_energy, window_size, poly_order)
		stored_energy = np.interp(time_full_binned_crop,efit_data['t'],stored_energy,right=0.,left=0.)	# J
		P_heat = output_pohm + SS_BEAMPOWER + SW_BEAMPOWER
		P_loss = P_heat-dWdt
		energy_confinement_time = stored_energy/P_loss	# s
		energy_confinement_time[energy_confinement_time<0] = np.nan
		# I add this to check if using transp there is any difference
		power_balance = calc_psol(shot=shotnumber,smooth_dt =0.015)	# in W
		pnbi = np.interp(time_full_binned_crop,power_balance['t'],power_balance['pnbi'],right=0.,left=0.)
		P_heat_transp = output_pohm + pnbi
		P_loss_transp = P_heat_transp-dWdt
		energy_confinement_time_transp = stored_energy/P_loss_transp	# s
		energy_confinement_time_transp[energy_confinement_time_transp<0] = np.nan
		full_saved_file_dict_FAST['multi_instrument']['BEAMPOWER_time'] = time_full_binned_crop
		full_saved_file_dict_FAST['multi_instrument']['SW_BEAMPOWER'] = SW_BEAMPOWER
		full_saved_file_dict_FAST['multi_instrument']['SS_BEAMPOWER'] = SS_BEAMPOWER
		full_saved_file_dict_FAST['multi_instrument']['ss_absorption'] = ss_absorption
		full_saved_file_dict_FAST['multi_instrument']['sw_absorption'] = sw_absorption
		full_saved_file_dict_FAST['multi_instrument']['stored_energy'] = stored_energy
		full_saved_file_dict_FAST['multi_instrument']['power_balance_pohm'] = output_pohm
		full_saved_file_dict_FAST['multi_instrument']['power_balance_pdw_dt'] = dWdt
		full_saved_file_dict_FAST['multi_instrument']['P_heat'] = P_heat
		full_saved_file_dict_FAST['multi_instrument']['P_loss'] = P_loss
		full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time'] = energy_confinement_time
		full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_transp'] = energy_confinement_time_transp
		kappa = efit_data['kappa']
		kappa = scipy.signal.savgol_filter(kappa, window_size, poly_order)
		kappa = np.interp(time_full_binned_crop,efit_data['t'],kappa,right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['kappa'] = kappa
		major_R = efit_reconstruction.mag_axis_r
		major_R = np.interp(time_full_binned_crop,efit_reconstruction.time,major_R,right=0.,left=0.)	# m
		full_saved_file_dict_FAST['multi_instrument']['major_R'] = major_R
		minor_radius = efit_data['minor_radius']
		minor_radius = scipy.signal.savgol_filter(minor_radius, window_size, poly_order)
		minor_radius = np.interp(time_full_binned_crop,efit_data['t'],minor_radius,right=0.,left=0.)	# m
		full_saved_file_dict_FAST['multi_instrument']['minor_radius'] = minor_radius
		plasma_current = np.interp(time_full_binned_crop,efit_reconstruction.time,efit_reconstruction.cpasma)	# A
		full_saved_file_dict_FAST['multi_instrument']['plasma_current'] = plasma_current
		try:
			data = client.get('/EPM/OUTPUT/GLOBALPARAMETERS/BPHIRMAG',laser_to_analyse[-9:-4])	# Toroidal magnetic field at magnetic axis
			toroidal_field_good = True
		except:
			print('Toroidal magnetic field at magnetic axis'+' has been flagged as bad data')
			client.set_property(pyuda.Properties.GET_BAD, True)	# this enables to collect data that was flagged as bad
			data = client.get('/EPM/OUTPUT/GLOBALPARAMETERS/BPHIRMAG',laser_to_analyse[-9:-4])	# Toroidal magnetic field at magnetic axis
			client.set_property(pyuda.Properties.GET_BAD, False)
			toroidal_field_good = False
		toroidal_field = -np.interp(efit_data['t'],data.time.data,data.data,right=0.,left=0.)	# T
		toroidal_field = scipy.signal.savgol_filter(toroidal_field, window_size, poly_order)
		toroidal_field = np.interp(time_full_binned_crop,efit_data['t'],toroidal_field,right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['toroidal_field'] = toroidal_field
		full_saved_file_dict_FAST['multi_instrument']['toroidal_field_good'] = toroidal_field_good

		vert_displacement = np.interp(time_full_binned_crop,efit_reconstruction.time,efit_reconstruction.mag_axis_z,right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['vert_displacement'] = vert_displacement
		radius_outer_separatrix = np.interp(time_full_binned_crop,efit_data['t'],efit_data['rmidplaneOut'],right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['radius_outer_separatrix'] = radius_outer_separatrix
		radius_inner_separatrix = np.interp(time_full_binned_crop,efit_data['t'],efit_data['rmidplaneIn'],right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['radius_inner_separatrix'] = radius_inner_separatrix

		if not density_data_missing:
			energy_confinement_time_98y2 = 0.0562 * ((plasma_current*1e-6)**0.93) * ((toroidal_field*1)**0.15) * ((ne_bar*1e-19)**0.41) * ((P_loss*1e-6)**-0.69) * ((major_R*1e0)**1.97) * (((minor_radius/major_R)*1e0)**0.58)  * ((kappa*1e0)**0.78) * ((2*1e0)**0.19)
			full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_98y2'] = energy_confinement_time_98y2
			energy_confinement_time_97P = 0.037 * ((plasma_current*1e-6)**0.74) * ((toroidal_field*1)**0.2) * ((ne_bar*1e-19)**0.24) * ((P_loss*1e-6)**-0.75) * ((major_R*1e0)**1.69) * (((minor_radius/major_R)*1e0)**0.31)  * ((kappa*1e0)**0.67) * ((2*1e0)**0.26)
			full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_97P'] = energy_confinement_time_97P
			energy_confinement_time_HST = 0.066 * ((plasma_current*1e-6)**0.53) * ((toroidal_field*1)**1.05) * ((ne_bar*1e-19)**0.65) * ((P_heat*1e-6)**-0.58) * ((major_R*1e0)**2.66) * ((kappa*1e0)**0.78)
			full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_HST'] = energy_confinement_time_HST
			energy_confinement_time_LST = 0.153 * ((plasma_current*1e-6)**1.01) * ((toroidal_field*1)**0.7) * ((ne_bar*1e-19)**-0.07) * ((P_loss*1e-6)**-0.37)
			full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_LST'] = energy_confinement_time_LST


		data = client.get('/XIM/DA/HM10/T',laser_to_analyse[-9:-4])
		Dalpha = data.data
		Dalpha_time = data.time.data
		full_saved_file_dict_FAST['multi_instrument']['Dalpha'] = Dalpha
		full_saved_file_dict_FAST['multi_instrument']['Dalpha_time'] = Dalpha_time

		from mastu_exhaust_analysis.read_efit import read_epm
		fdir = coleval.uda_transfer(shotnumber,'epm')
		efit_data = read_epm(fdir,calc_bfield=True)
		os.remove(fdir)
		psiN_peak_inner_all = []
		psiN_core_inner_side_baricenter_all = []
		from scipy.interpolate import interp1d,interp2d
		for i_t in range(len(time_full_binned_crop)):
			i_efit_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
			psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,efit_data['psiN'][i_efit_time])
			psiN = psi_interpolator(inversion_R,inversion_Z)
			select = np.logical_and(inversion_Z<-0.6,inversion_Z>efit_reconstruction.lower_xpoint_z[i_efit_time])
			if inner_emissivity_peak_all[i_t][1]<efit_reconstruction.lower_xpoint_z[i_efit_time]:
				psiN_peak_inner_all.append([np.nan])
				# print(i_t)
			else:
				psiN_peak_inner_all.append( psi_interpolator(inner_emissivity_peak_all[i_t][0],inner_emissivity_peak_all[i_t][1]))
			psiN_core_inner_side_baricenter_all.append(np.nansum(((psiN*np.logical_and(psiN>0.9,psiN<1.1)).T*(inverted_data[i_t]))[:][:,select]) / np.nansum((inverted_data[i_t]*np.logical_and(psiN.T>0.9,psiN.T<1.1))[:][:,select]))
		psiN_peak_inner_all = np.array(psiN_peak_inner_all).flatten()
		psiN_core_inner_side_baricenter_all = np.array(psiN_core_inner_side_baricenter_all)
		inverted_dict[str(grid_resolution)]['psiN_peak_inner_all'] = psiN_peak_inner_all
		inverted_dict[str(grid_resolution)]['psiN_core_inner_side_baricenter_all'] = psiN_core_inner_side_baricenter_all

		# here I calculate the upstream density
		try:
			from mastu_exhaust_analysis import divertor_geometry
			from mastu_exhaust_analysis import Thomson
			TS_data = Thomson(shot=laser_to_analyse[-9:-4])
			# TS_data.ne = median_filter(TS_data.ne,size=[3,1])
			# TS_data.Te = median_filter(TS_data.Te,size=[3,1])
			tu_cowley = []
			tu_labombard = []
			tu_stangeby = []
			nu_cowley = []
			nu_labombard = []
			nu_stangeby = []
			nu_mean = []
			tu_EFIT = []
			nu_EFIT = []
			tu_EFIT_smoothing = []
			nu_EFIT_smoothing = []
			tu_EFIT_spatial_uncertainty = []
			nu_EFIT_spatial_uncertainty = []
			tu_EFIT_smoothing_spatial_uncertainty = []
			nu_EFIT_smoothing_spatial_uncertainty = []
			TS_time_smoothing = 0.05	# s
			TS_time_smoothing_min = np.mean(np.diff(time_full_binned_crop))

			client=pyuda.Client()
			try:
				recv_seg01_r = client.get('/xdc/reconstruction/s/recv_seg01_r',laser_to_analyse[-9:-4])
				recv_seg01_r = np.interp(time_full_binned_crop,recv_seg01_r.time.data,recv_seg01_r.data)
			except:
				recv_seg01_r = np.ones_like(time_full_binned_crop)*np.nan
			try:
				epm_rmidplaneout = client.get('/epm/output/separatrixgeometry/rmidplaneout',laser_to_analyse[-9:-4])
				epm_rmidplaneout = np.interp(time_full_binned_crop,epm_rmidplaneout.time.data,epm_rmidplaneout.data)
			except:
				epm_rmidplaneout = np.ones_like(time_full_binned_crop)*np.nan
			try:
				epq_rmidplaneout = client.get('/epq/output/separatrixgeometry/rmidplaneout',laser_to_analyse[-9:-4])
				epq_rmidplaneout = np.interp(time_full_binned_crop,epq_rmidplaneout.time.data,epq_rmidplaneout.data)
			except:
				epq_rmidplaneout = np.ones_like(time_full_binned_crop)*np.nan
			coleval.reset_connection(client)
			del client
			R_separatrix_OMP_uncertainty = 0.015/2	# m  1.5cm is already a lot, and this way it's split half in and half out of the separatrix
			# the following is from a chat with Lucy Colgan 17/02/2025
			R_separatrix_OMP_uncertainty_down = 0.01	# m  1.5cm is already a lot, and the uncertainty is more on the inboard side. it might be too much, I take 1cm
			R_separatrix_OMP_uncertainty_up = 0.00	# m  1.5cm is already a lot, and the uncertainty is more on the inboard side. it might be too much, I take 1cm

			for time in time_full_binned_crop:
				try:
					temp = divertor_geometry(shot=shotnumber,time=time)
					tu_cowley.append(temp.tu_cowley)
					tu_labombard.append(temp.tu_labombard)
					tu_stangeby.append(temp.tu_stangeby)
				except:
					tu_cowley.append(np.nan)
					tu_labombard.append(np.nan)
					tu_stangeby.append(np.nan)
				try:
					try:
						# temp = np.abs(TS_data.time.data-time).argmin()
						temp = np.abs(TS_data.time.data-time)<=TS_time_smoothing/2
						ne = TS_data.ne.data[temp]
						R_TS = TS_data.R.data[temp]
						Te = TS_data.Te.data[temp]
					except:
						# temp = np.abs(TS_data.time-time).argmin()
						temp = np.abs(TS_data.time-time)<=TS_time_smoothing/2
						ne = TS_data.ne[temp]
						R_TS = TS_data.R[temp]
						Te = TS_data.Te[temp]
					ne_smoothing = np.nanmedian(ne,axis=0)
					R_TS_smoothing = np.nanmedian(R_TS,axis=0)
					Te_smoothing = np.nanmedian(Te,axis=0)
					ne_smoothing = ne_smoothing[np.isfinite(Te_smoothing)]
					R_TS_smoothing = R_TS_smoothing[np.isfinite(Te_smoothing)]
					Te_smoothing = Te_smoothing[np.isfinite(Te_smoothing)]
					temp = Te_smoothing[:-10].argmax()
					ne_smoothing = ne_smoothing[temp:]
					R_TS_smoothing = R_TS_smoothing[temp:]
					Te_smoothing = Te_smoothing[temp:]
					Te_smoothing = scipy.signal.savgol_filter(Te_smoothing, 7, 3)
					ne_smoothing = scipy.signal.savgol_filter(ne_smoothing, 7, 3)
					# ineffective. filtered twice for only decreasing signal
					ne_smoothing = ne_smoothing[1:][np.diff(Te_smoothing)<=0]
					R_TS_smoothing = R_TS_smoothing[1:][np.diff(Te_smoothing)<=0]
					Te_smoothing = Te_smoothing[1:][np.diff(Te_smoothing)<=0]
					ne_smoothing = ne_smoothing[1:][np.diff(Te_smoothing)<=0]
					R_TS_smoothing = R_TS_smoothing[1:][np.diff(Te_smoothing)<=0]
					Te_smoothing = Te_smoothing[1:][np.diff(Te_smoothing)<=0]
					interp_ne_Te = interp1d(Te_smoothing[Te_smoothing.argmax():],ne_smoothing[Te_smoothing.argmax():],fill_value='extrapolate')

					try:
						try:
							temp = np.abs(TS_data.time.data-time)<=TS_time_smoothing_min/2
							if np.sum(temp)>0:
								ne = TS_data.ne.data[temp]
								R_TS = TS_data.R.data[temp]
								Te = TS_data.Te.data[temp]
							else:
								temp = np.abs(TS_data.time.data-time).argmin()
								ne = [TS_data.ne.data[temp]]
								R_TS = [TS_data.R.data[temp]]
								Te = [TS_data.Te.data[temp]]
						except:
							temp = np.abs(TS_data.time-time)<=TS_time_smoothing_min/2
							if np.sum(temp)>0:
								ne = TS_data.ne[temp]
								R_TS = TS_data.R[temp]
								Te = TS_data.Te[temp]
							else:
								temp = np.abs(TS_data.time-time).argmin()
								ne = [TS_data.ne[temp]]
								R_TS = [TS_data.R[temp]]
								Te = [TS_data.Te[temp]]
						ne = np.nanmedian(ne,axis=0)
						R_TS = np.nanmedian(R_TS,axis=0)
						Te = np.nanmedian(Te,axis=0)
						ne = ne[np.isfinite(Te)]
						R_TS = R_TS[np.isfinite(Te)]
						Te = Te[np.isfinite(Te)]
						temp = Te[:-10].argmax()
						ne = ne[temp:]
						R_TS = R_TS[temp:]
						Te = Te[temp:]
						Te = scipy.signal.savgol_filter(Te, 7, 3)
						ne = scipy.signal.savgol_filter(ne, 7, 3)
						# ineffective. filtered twice for only decreasing signal
						ne = ne[1:][np.diff(Te)<=0]
						R_TS = R_TS[1:][np.diff(Te)<=0]
						Te = Te[1:][np.diff(Te)<=0]
						ne = ne[1:][np.diff(Te)<=0]
						R_TS = R_TS[1:][np.diff(Te)<=0]
						Te = Te[1:][np.diff(Te)<=0]
					except:
						ne = [np.nan,np.nan]
						R_TS = [np.nan,np.nan]
						Te = [np.nan,np.nan]

					# interp_ne_Te = interp1d(Te[Te.argmax():],ne[Te.argmax():],fill_value='extrapolate')
					nu_cowley.append(interp_ne_Te(tu_cowley[-1]))
					nu_labombard.append(interp_ne_Te(tu_labombard[-1]))
					nu_stangeby.append(interp_ne_Te(tu_stangeby[-1]))
					nu_mean.append(np.nanmean([nu_cowley[-1],nu_labombard[-1],nu_stangeby[-1]]))
					# I want the upstream parameters according to EFIT
					try:
						i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
						mag_axis_z = efit_reconstruction.mag_axis_z[i_efit_time]
						psidat = efit_reconstruction.psidat[i_efit_time]
						psidat = psidat[np.abs(efit_reconstruction.z - mag_axis_z).argmin()]
						R_ = efit_reconstruction.R
						R_ = R_[psidat.argmax():psidat.argmin()]
						psidat = psidat[psidat.argmax():psidat.argmin()]
						R_separatrix_OMP = np.interp(efit_reconstruction.psi_bnd[i_efit_time],np.flip(psidat,axis=0),np.flip(R_,axis=0))

						# I wanted to try using different methods of calculating the upstream radius to estimate the uncertainty, but id does not seem to really work
						# R_separatrix_OMP-recv_seg01_r[i_time],R_separatrix_OMP-epm_rmidplaneout[i_time],R_separatrix_OMP-epq_rmidplaneout[i_time]

						tu_EFIT_smoothing_spatial_uncertainty.append(max(0,np.abs(np.diff(np.interp([R_separatrix_OMP-R_separatrix_OMP_uncertainty_down,R_separatrix_OMP+R_separatrix_OMP_uncertainty_up],R_TS_smoothing,Te_smoothing)))[0] ))
						nu_EFIT_smoothing_spatial_uncertainty.append(np.abs(np.diff(np.interp([R_separatrix_OMP-R_separatrix_OMP_uncertainty_down,R_separatrix_OMP+R_separatrix_OMP_uncertainty_up],R_TS_smoothing,ne_smoothing)))[0] )
						tu_EFIT_spatial_uncertainty.append(np.abs(np.diff(np.interp([R_separatrix_OMP-R_separatrix_OMP_uncertainty_down,R_separatrix_OMP+R_separatrix_OMP_uncertainty_up],R_TS,Te)))[0] )
						nu_EFIT_spatial_uncertainty.append(np.abs(np.diff(np.interp([R_separatrix_OMP-R_separatrix_OMP_uncertainty_down,R_separatrix_OMP+R_separatrix_OMP_uncertainty_up],R_TS,ne)))[0] )
						tu_EFIT_smoothing.append(max(0,np.interp(R_separatrix_OMP,R_TS_smoothing,Te_smoothing)))
						nu_EFIT_smoothing.append(np.interp(R_separatrix_OMP,R_TS_smoothing,ne_smoothing))
						tu_EFIT.append(np.interp(R_separatrix_OMP,R_TS,Te))
						nu_EFIT.append(np.interp(R_separatrix_OMP,R_TS,ne))
					except:
						tu_EFIT_smoothing.append(np.nan)
						nu_EFIT_smoothing.append(np.nan)
						tu_EFIT.append(np.nan)
						nu_EFIT.append(np.nan)
						tu_EFIT_smoothing_spatial_uncertainty.append(np.nan)
						nu_EFIT_smoothing_spatial_uncertainty.append(np.nan)
						tu_EFIT_spatial_uncertainty.append(np.nan)
						nu_EFIT_spatial_uncertainty.append(np.nan)
				except:
					nu_cowley.append(np.nan)
					nu_labombard.append(np.nan)
					nu_stangeby.append(np.nan)
					nu_mean.append(np.nan)
					tu_EFIT_smoothing.append(np.nan)
					nu_EFIT_smoothing.append(np.nan)
					tu_EFIT.append(np.nan)
					nu_EFIT.append(np.nan)
					tu_EFIT_smoothing_spatial_uncertainty.append(np.nan)
					nu_EFIT_smoothing_spatial_uncertainty.append(np.nan)
					tu_EFIT_spatial_uncertainty.append(np.nan)
					nu_EFIT_spatial_uncertainty.append(np.nan)
			tu_cowley = np.array(tu_cowley)
			tu_labombard = np.array(tu_labombard)
			tu_stangeby = np.array(tu_stangeby)
			nu_cowley = np.array(nu_cowley)
			nu_labombard = np.array(nu_labombard)
			nu_stangeby = np.array(nu_stangeby)
			nu_mean = np.array(nu_mean)
			tu_EFIT = np.array(tu_EFIT)
			nu_EFIT = np.array(nu_EFIT)
			tu_EFIT_smoothing = np.array(tu_EFIT_smoothing)
			nu_EFIT_smoothing = np.array(nu_EFIT_smoothing)
			tu_EFIT_spatial_uncertainty = np.array(tu_EFIT_spatial_uncertainty).flatten()
			nu_EFIT_spatial_uncertainty = np.array(nu_EFIT_spatial_uncertainty)
			tu_EFIT_smoothing_spatial_uncertainty = np.array(tu_EFIT_smoothing_spatial_uncertainty)
			nu_EFIT_smoothing_spatial_uncertainty = np.array(nu_EFIT_smoothing_spatial_uncertainty)
			full_saved_file_dict_FAST['multi_instrument']['tu_cowley'] = tu_cowley
			full_saved_file_dict_FAST['multi_instrument']['tu_labombard'] = tu_labombard
			full_saved_file_dict_FAST['multi_instrument']['tu_stangeby'] = tu_stangeby
			full_saved_file_dict_FAST['multi_instrument']['nu_cowley'] = nu_cowley
			full_saved_file_dict_FAST['multi_instrument']['nu_labombard'] = nu_labombard
			full_saved_file_dict_FAST['multi_instrument']['nu_stangeby'] = nu_stangeby
			full_saved_file_dict_FAST['multi_instrument']['nu_mean'] = nu_mean
			full_saved_file_dict_FAST['multi_instrument']['tu_EFIT'] = tu_EFIT
			full_saved_file_dict_FAST['multi_instrument']['nu_EFIT'] = nu_EFIT
			full_saved_file_dict_FAST['multi_instrument']['tu_EFIT_smoothing'] = tu_EFIT_smoothing
			full_saved_file_dict_FAST['multi_instrument']['nu_EFIT_smoothing'] = nu_EFIT_smoothing
			full_saved_file_dict_FAST['multi_instrument']['tu_EFIT_spatial_uncertainty'] = tu_EFIT_spatial_uncertainty
			full_saved_file_dict_FAST['multi_instrument']['nu_EFIT_spatial_uncertainty'] = nu_EFIT_spatial_uncertainty
			full_saved_file_dict_FAST['multi_instrument']['tu_EFIT_smoothing_spatial_uncertainty'] = tu_EFIT_smoothing_spatial_uncertainty
			full_saved_file_dict_FAST['multi_instrument']['nu_EFIT_smoothing_spatial_uncertainty'] = nu_EFIT_smoothing_spatial_uncertainty
			pu_cowley = nu_cowley*tu_cowley*11604*1.380649E-23
			full_saved_file_dict_FAST['multi_instrument']['pu_cowley'] = pu_cowley
			pu_labombard = nu_labombard*tu_labombard*11604*1.380649E-23
			full_saved_file_dict_FAST['multi_instrument']['pu_labombard'] = pu_labombard
			pu_stangeby = nu_stangeby*tu_stangeby*11604*1.380649E-23
			full_saved_file_dict_FAST['multi_instrument']['pu_stangeby'] = pu_stangeby
			pu_EFIT_smoothing = nu_EFIT_smoothing*tu_EFIT_smoothing*11604*1.380649E-23
			full_saved_file_dict_FAST['multi_instrument']['pu_EFIT'] = pu_EFIT_smoothing
			pu_EFIT = nu_EFIT*tu_EFIT*11604*1.380649E-23
			full_saved_file_dict_FAST['multi_instrument']['pu_EFIT'] = pu_EFIT

			try:
				time_uncertainty_down = full_saved_file_dict_FAST['first_pass']['time_uncertainty_down']
			except:
				time_uncertainty_down = full_saved_file_dict_FAST['first_pass'].all()['time_uncertainty_down']
			try:
				time_uncertainty_up = full_saved_file_dict_FAST['first_pass']['time_uncertainty_up']
			except:
				time_uncertainty_up = full_saved_file_dict_FAST['first_pass'].all()['time_uncertainty_up']
			def calc_time_related_uncertainty(time,quantity,time_uncertainty_down,time_uncertainty_up):
				temp =[np.linspace(value-time_uncertainty_down,value+time_uncertainty_up,num=50) for value in time]	# 50 steps is completely arbitrary
				temp = np.interp(temp,time,quantity)
				diff = np.nanmax(temp,axis=1) - np.nanmin(temp,axis=1)
				# diff = np.abs(np.diff(np.interp(np.array([time-time_uncertainty_down] + [time+time_uncertainty_up]).T,time,quantity),axis=1).flatten())
				return diff
			tu_EFIT_time_uncertainty = calc_time_related_uncertainty(time_full_binned_crop,tu_EFIT,time_uncertainty_down,time_uncertainty_up)
			nu_EFIT_time_uncertainty = calc_time_related_uncertainty(time_full_binned_crop,nu_EFIT,time_uncertainty_down,time_uncertainty_up)
			tu_EFIT_smoothing_time_uncertainty = calc_time_related_uncertainty(time_full_binned_crop,tu_EFIT_smoothing,time_uncertainty_down,time_uncertainty_up)
			nu_EFIT_smoothing_time_uncertainty = calc_time_related_uncertainty(time_full_binned_crop,nu_EFIT_smoothing,time_uncertainty_down,time_uncertainty_up)
			full_saved_file_dict_FAST['multi_instrument']['tu_EFIT_time_uncertainty'] = tu_EFIT_time_uncertainty
			full_saved_file_dict_FAST['multi_instrument']['nu_EFIT_time_uncertainty'] = nu_EFIT_time_uncertainty
			full_saved_file_dict_FAST['multi_instrument']['tu_EFIT_smoothing_time_uncertainty'] = tu_EFIT_smoothing_time_uncertainty
			full_saved_file_dict_FAST['multi_instrument']['nu_EFIT_smoothing_time_uncertainty'] = nu_EFIT_smoothing_time_uncertainty
			TS_reading_success = True

		except Exception as e:
			print('TS reading failed')
			logging.exception('with error: ' + str(e))
			TS_reading_success = False

		# plt.figure()
		# plt.plot(time_full_binned_crop,nu_labombard,label='nu_labombard')
		# plt.plot(time_full_binned_crop,nu_stangeby,label='nu_stangeby')
		# plt.plot(time_full_binned_crop,nu_cowley,label='nu_cowley')
		# plt.plot(time_full_binned_crop,nu_mean,label='nu_mean')
		# plt.plot(time_nesep,nesep,'--',label='nesep')
		# plt.legend()


		try:
			from scipy.signal import find_peaks, peak_prominences as get_proms

			try:
				gna26 = client.get('/xbm/core/F26/AMP', laser_to_analyse[-9:-4])
				gna27 = client.get('/xbm/core/F27/AMP', laser_to_analyse[-9:-4])
				try:
					smoothing=1000
					grad26 = scipy.signal.savgol_filter(gna26.data,smoothing//4+1,5)	# I need the +1 otherwise savgol_filter fails in python3.7
					grad26 = scipy.signal.savgol_filter(grad26,smoothing//4+1,3)
					grad26 = scipy.signal.savgol_filter(grad26,smoothing//2+1,3)
					grad27 = scipy.signal.savgol_filter(gna27.data,smoothing//4+1,5)
					grad27 = scipy.signal.savgol_filter(grad27,smoothing//4+1,3)
					grad27 = scipy.signal.savgol_filter(grad27,smoothing//2+1,3)

					smooth26=scipy.signal.savgol_filter(np.gradient(grad26,gna26.time.data),smoothing//2+1,5)
					smooth26=scipy.signal.savgol_filter(smooth26,smoothing//2+1,2)
					smooth27=scipy.signal.savgol_filter(np.gradient(grad27,gna27.time.data),smoothing//2+1,5)
					smooth27=scipy.signal.savgol_filter(smooth27,smoothing//2+1,2)

					# new_MARFE_marker = (smooth27-smooth26)/(smooth26[np.logical_and(gna26.time.data>0,gna26.time.data<time_full_binned_crop.max()-0.1)].min())	# -0.1 because when there is a final disrupotion there is a sudden final spike down, and I want to avoid it
					ratio = smooth27/smooth26
					ratio[smooth27<0] = 1
					new_MARFE_marker = median_filter(ratio,size=int(smoothing//1.5))

					if False:
						# gna = client.get('/xbm/core/F28/AMP', laser_to_analyse[-9:-4]).data.T
						# time_res_bolo = client.get('/xbm/core/F28', laser_to_analyse[-9:-4]).data
						gna26 = client.get('/xbm/core/F26/AMP', laser_to_analyse[-9:-4])
						gna27 = client.get('/xbm/core/F27/AMP', laser_to_analyse[-9:-4])
						smoothing=1000
						grad26 = scipy.signal.savgol_filter(gna26.data,smoothing//4+1,5)
						grad26 = scipy.signal.savgol_filter(grad26,smoothing//4+1,3)
						grad26 = scipy.signal.savgol_filter(grad26,smoothing//2+1,3)
						grad27 = scipy.signal.savgol_filter(gna27.data,smoothing//4+1,5)
						grad27 = scipy.signal.savgol_filter(grad27,smoothing//4+1,3)
						grad27 = scipy.signal.savgol_filter(grad27,smoothing//2+1,3)
						plt.figure()
						a = plt.plot(gna26.time.data,gna26.data)
						plt.plot(gna26.time.data,grad26,'--',color='k')
						a = plt.plot(gna27.time.data,gna27.data)
						plt.plot(gna27.time.data,grad27,'--',color='k')
						plt.xlim(left=0,right=1)
						plt.grid()

						# smooth26 = smooth26 - smooth26[np.logical_and(gna26.time.data>0,gna26.time.data<2)].max()
						# smooth27 = smooth27 - smooth27[np.logical_and(gna27.time.data>0,gna27.time.data<2)].max()
						plt.figure()
						plt.plot(gna26.time.data,np.gradient(grad26,gna26.time.data))
						plt.plot(gna26.time.data,smooth26,'--k')
						plt.plot(gna27.time.data,np.gradient(grad27,gna27.time.data))
						plt.plot(gna27.time.data,smooth27,'--k')
						# plt.plot(gna27.time.data,smooth27/smooth26)
						# plt.plot(gna27.time.data,median_filter(smooth27/smooth26,size=smoothing//3))
						# plt.ylim(bottom=-5,top=10)
						plt.xlim(left=0,right=1)
						plt.grid()


						plt.figure()
						plt.plot(gna27.time.data,smooth27/smooth26)
						plt.plot(gna27.time.data,median_filter(smooth27/smooth26,size=smoothing//3+1))
						plt.plot(gna27.time.data,(smooth27-smooth26)/(smooth26[np.logical_and(gna26.time.data>0,gna26.time.data<time_full_binned_crop.max()-0.1)].min()))
						plt.ylim(bottom=-5,top=10)
						plt.xlim(left=0,right=1)
						plt.grid()
				except Exception as e:
					logging.exception('with error: ' + str(e))
					new_MARFE_marker = np.ones_like(gna26.time.data)
				new_MARFE_marker_time = gna26.time.data
			except:
				new_MARFE_marker = np.zeros_like(time_full_binned_crop)
				new_MARFE_marker_time = time_full_binned_crop
				pass
			full_saved_file_dict_FAST['multi_instrument']['new_MARFE_marker'] = new_MARFE_marker
			full_saved_file_dict_FAST['multi_instrument']['new_MARFE_marker_time'] = new_MARFE_marker_time


			def peak_prominences_as_area(data,peaks):
				# The area is defined as the area above the line that unites the two neibouring trough around the peak
				data = np.array(data)
				peaks = np.array(peaks)
				num_peaks = len(peaks)
				prominence = []
				if num_peaks==0:
					return prominence
				elif num_peaks==1:
					start = data[:peaks[0]].argmin()
					end = peaks[0]+data[peaks[0]:].argmin()
					prominence.append(np.trapz(data[start:end]-data[start:end].min()) - (end-start)*(np.abs(data[start]-data[end]))/2)
				elif num_peaks>0:
					start = data[:peaks[0]].argmin()
					end = peaks[0]+data[peaks[0]:peaks[1]].argmin()
					prominence.append(np.trapz(data[start:end]-data[start:end].min()) - (end-start)*(np.abs(data[start]-data[end]))/2)
				if num_peaks>2:
					for i in np.arange(1,num_peaks-1):
						start = peaks[i-1]+data[peaks[i-1]:peaks[i]].argmin()
						end = peaks[i]+data[peaks[i]:peaks[i+1]].argmin()
						prominence.append(np.trapz(data[start:end]-data[start:end].min()) - (end-start)*(np.abs(data[start]-data[end]))/2)
				if num_peaks>1:
					start = peaks[-2]+data[peaks[-2]:peaks[-1]].argmin()
					end = peaks[-1]+data[peaks[-1]:].argmin()
					prominence.append(np.trapz(data[start:end]-data[start:end].min()) - (end-start)*(np.abs(data[start]-data[end]))/2)
				return np.array(prominence)

			brightness_res_bolo = client.get('/abm/core/brightness', laser_to_analyse[-9:-4]).data.T
			good_res_bolo = client.get('/abm/core/good', laser_to_analyse[-9:-4]).data
			time_res_bolo = client.get('/abm/core/time', laser_to_analyse[-9:-4]).data
			channel_res_bolo = client.get('/abm/core/channel', laser_to_analyse[-9:-4]).data
			# select only usefull time
			brightness_res_bolo = brightness_res_bolo[:,np.logical_and(time_res_bolo>-0.2,time_res_bolo<time_full_binned_crop.max()+0.1)]
			time_res_bolo = time_res_bolo[np.logical_and(time_res_bolo>-0.2,time_res_bolo<time_full_binned_crop.max()+0.1)]
			running_average_time = int(0.2/(np.median(np.diff(time_res_bolo))))

			CH27 = median_filter(brightness_res_bolo[channel_res_bolo==27][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))
			CH27_derivative = generic_filter(np.gradient(CH27,time_res_bolo),np.mean,running_average_time)
			peaks = find_peaks(CH27_derivative,distance=int(running_average_time/2))[0]
			proms = peak_prominences_as_area(CH27_derivative,peaks)
			peaks = peaks[np.abs(proms)>np.abs(proms).max()*1e-2]
			proms = peak_prominences_as_area(CH27_derivative,peaks)
			anti_peaks = find_peaks(-CH27_derivative,distance=int(running_average_time/2))[0]
			anti_proms = peak_prominences_as_area(-CH27_derivative,anti_peaks)
			anti_peaks = anti_peaks[np.abs(anti_proms)>np.abs(anti_proms).max()*1e-2]
			anti_proms = peak_prominences_as_area(-CH27_derivative,anti_peaks)
			CH27_rise_centre = peaks[time_res_bolo[peaks]>0.1][proms[time_res_bolo[peaks]>0.1].argmax()]
			# CH27_start = anti_peaks[anti_peaks<peaks[proms.argmax()]].max()
			# CH27_end = anti_peaks[anti_peaks>peaks[proms.argmax()]].min()
			if CH27_derivative[CH27_rise_centre]>1e5:	# arbitrary limit to exclude non existing cases
				CH27_derivative_limit = CH27_derivative.max()*4e-1
				CH27_derivative = generic_filter(np.gradient(CH27,time_res_bolo),np.mean,int(running_average_time/10))
				temp = np.arange(len(time_res_bolo))<CH27_rise_centre
				CH27_start = np.arange(len(time_res_bolo))[temp][np.logical_and(np.abs(CH27_derivative[temp])<CH27_derivative_limit,generic_filter(np.gradient(CH27_derivative[temp],time_res_bolo[temp]),np.mean,int(running_average_time/10))>0)].max()
				temp = np.arange(len(time_res_bolo))>CH27_rise_centre
				CH27_end = np.arange(len(time_res_bolo))[temp][np.logical_and(np.abs(CH27_derivative[temp])<CH27_derivative_limit,generic_filter(np.gradient(CH27_derivative[temp],time_res_bolo[temp]),np.mean,int(running_average_time/10))<0)].min()
				CH27_angle = (CH27[CH27_end]-CH27[CH27_start])/(time_res_bolo[CH27_end]-time_res_bolo[CH27_start])
				# CH27_start_of_raise = time_res_bolo[CH27_derivative[:CH27.argmax()].argmax()] - CH27[CH27_derivative[:CH27.argmax()].argmax()]/(CH27_derivative[:CH27.argmax()].max())
				CH27_start_of_raise = time_res_bolo[CH27_start]
				CH27_end_of_raise = time_res_bolo[CH27_end]
			else:
				CH27_angle = 0
				CH27_start_of_raise = np.inf
				CH27_end_of_raise = np.inf


			select = (np.logical_or(channel_res_bolo == 8,channel_res_bolo == 9) * good_res_bolo).astype(bool)
			CH8_9 = median_filter(brightness_res_bolo[select][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))
			CH8_9_derivative = generic_filter(np.gradient(CH8_9,time_res_bolo),np.mean,running_average_time)
			peaks = find_peaks(CH8_9_derivative,distance=int(running_average_time/2))[0]
			proms = peak_prominences_as_area(CH8_9_derivative,peaks)
			peaks = peaks[np.abs(proms)>np.abs(proms).max()*1e-2]
			proms = peak_prominences_as_area(CH8_9_derivative,peaks)
			anti_peaks = find_peaks(-CH8_9_derivative,distance=int(running_average_time/2))[0]
			anti_proms = peak_prominences_as_area(-CH8_9_derivative,anti_peaks)
			anti_peaks = anti_peaks[np.abs(anti_proms)>np.abs(anti_proms).max()*1e-2]
			anti_proms = peak_prominences_as_area(-CH8_9_derivative,anti_peaks)
			CH8_9_rise_centre = peaks[time_res_bolo[peaks]>0.1][proms[time_res_bolo[peaks]>0.1].argmax()]
			# CH8_9_start = anti_peaks[anti_peaks<peaks[proms.argmax()]].max()
			# CH8_9_end = anti_peaks[anti_peaks>peaks[proms.argmax()]].min()
			if CH8_9_derivative[CH8_9_rise_centre]>1e5:	# arbitrary limit to exclude non existing cases
				CH8_9_derivative_limit = CH8_9_derivative.max()*4e-1
				CH8_9_derivative = generic_filter(np.gradient(CH8_9,time_res_bolo),np.mean,int(running_average_time/10))
				temp = np.arange(len(time_res_bolo))<CH8_9_rise_centre
				CH8_9_start = np.arange(len(time_res_bolo))[temp][np.logical_and(np.abs(CH8_9_derivative[temp])<CH8_9_derivative_limit,generic_filter(np.gradient(CH8_9_derivative[temp],time_res_bolo[temp]),np.mean,int(running_average_time/10))>0)].max()
				temp = np.arange(len(time_res_bolo))>CH8_9_rise_centre
				CH8_9_end = np.arange(len(time_res_bolo))[temp][np.logical_and(np.abs(CH8_9_derivative[temp])<CH8_9_derivative_limit,generic_filter(np.gradient(CH8_9_derivative[temp],time_res_bolo[temp]),np.mean,int(running_average_time/10))<0)].min()
				CH8_9_angle = (CH8_9[CH8_9_end]-CH8_9[CH8_9_start])/(time_res_bolo[CH8_9_end]-time_res_bolo[CH8_9_start])
				# CH8_9_start_of_raise = time_res_bolo[CH8_9_derivative[:CH8_9.argmax()].argmax()] - CH8_9[CH8_9_derivative[:CH8_9.argmax()].argmax()]/(CH8_9_derivative[:CH8_9.argmax()].max())
				CH8_9_start_of_raise = time_res_bolo[CH8_9_start]
				CH8_9_end_of_raise = time_res_bolo[CH8_9_end]
			else:
				CH8_9_angle = 0
				CH8_9_start_of_raise = np.inf
				CH8_9_end_of_raise = np.inf
			# time_start_MARFE = max(CH27_start_of_raise,CH8_9_start_of_raise)

			CH25 = median_filter(brightness_res_bolo[channel_res_bolo==25][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))
			CH25 = CH25*20/18.4	# scaling to compensatefor the different integration length in the core plasma
			CH26 = median_filter(brightness_res_bolo[channel_res_bolo==26][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))
			CH26 = CH26*13.7/13.3	# scaling to compensatefor the different integration length in the core plasma
			CH27 = median_filter(brightness_res_bolo[channel_res_bolo==27][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))
			temp = CH27-CH26
			temp[temp<0] = 0
			temp += CH26
			temp = temp/CH26
			temp[np.isnan(temp)]
			temp[time_res_bolo<0.1] = 1
			CH27_26 = median_filter(temp,size=int(0.05/(np.median(np.diff(time_res_bolo)))))
			CH27_26 = generic_filter(CH27_26,np.mean,int(running_average_time/10))
			if np.sum(CH27_26>3)>0:
				time_active_MARFE = time_res_bolo[(CH27_26>3).argmax()]
			else:
				time_active_MARFE = None

			if np.sum(CH27_26[time_res_bolo>0.2]>1.5)>0:
				add_time = 0.05	# arbitrary
				def check(args):
					c0,c1,t0=args
					out = np.sum((CH27_26[np.logical_and(time_res_bolo<=t0+add_time,time_res_bolo>=t0-2*add_time)]-c0-c1*np.maximum(0,time_res_bolo[np.logical_and(time_res_bolo<=t0+add_time,time_res_bolo>=t0-2*add_time)]-t0))**2)
					# print(out)
					return out
				guess = [1,3/0.1,time_res_bolo[(np.diff(CH27_26)[:CH27_26.argmax()]>0).argmax()]]
				bds = [[0.9999,1.000001],[0,np.inf],[min(guess[-1],0.2),time_full_binned_crop.max()]]
				fit_bolo, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(check, x0=guess,approx_grad=True,bounds = bds,iprint=-1,epsilon=0.001, factr=1e0) #,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				time_start_MARFE = fit_bolo[-1]
			else:
				fit_bolo = [1,0,np.inf]
				time_start_MARFE = np.inf

			full_saved_file_dict_FAST['multi_instrument']['time_res_bolo'] = time_res_bolo
			full_saved_file_dict_FAST['multi_instrument']['CH25'] = CH25
			full_saved_file_dict_FAST['multi_instrument']['CH26'] = CH26
			full_saved_file_dict_FAST['multi_instrument']['CH27'] = CH27
			full_saved_file_dict_FAST['multi_instrument']['CH27_angle'] = CH27_angle
			full_saved_file_dict_FAST['multi_instrument']['CH27_start_of_raise'] = CH27_start_of_raise
			full_saved_file_dict_FAST['multi_instrument']['CH27_end_of_raise'] = CH27_end_of_raise
			full_saved_file_dict_FAST['multi_instrument']['CH8_9'] = CH8_9
			full_saved_file_dict_FAST['multi_instrument']['CH8_9_angle'] = CH8_9_angle
			full_saved_file_dict_FAST['multi_instrument']['CH8_9_start_of_raise'] = CH8_9_start_of_raise
			full_saved_file_dict_FAST['multi_instrument']['CH8_9_end_of_raise'] = CH8_9_end_of_raise
			full_saved_file_dict_FAST['multi_instrument']['CH27_26'] = CH27_26

			# plt.figure()
			# plt.plot(time_res_bolo,temp,':')
			# plt.plot(time_res_bolo,CH27_26)
			# plt.axhline(y=3,linestyle='--',color='k')
			# plt.title(name)
			# plt.grid()
			# plt.pause(0.01)
			# plt.plot(time_res_bolo,fit_bolo[0]+fit_bolo[1]*np.maximum(0,time_res_bolo-fit_bolo[2]),'--')
			# plt.pause(0.01)


			# plt.figure()
			# plt.plot(time_res_bolo,CH27,'r',label='core27')
			# plt.plot([CH27_start_of_raise,CH27_end_of_raise],[CH27[CH27_start],CH27[CH27_end]],'--r')
			# plt.axvline(x=CH27_start_of_raise,linestyle='--',color='r')
			# plt.plot(time_res_bolo,CH8_9,'b',label='core8/9')
			# plt.plot([CH8_9_start_of_raise,CH8_9_end_of_raise],[CH8_9[CH8_9_start],CH8_9[CH8_9_end]],'--b')
			# plt.axvline(x=CH8_9_start_of_raise,linestyle='--',color='b')
			# plt.legend(loc='best', fontsize='xx-small')
			# plt.ylabel('Brightness [W/m2]')
			# plt.grid()
			# plt.pause(0.01)
			# #
			# plt.figure()
			# plt.plot(time_res_bolo,CH27_derivative,'r')
			# plt.plot(time_res_bolo,CH8_9_derivative,'b')
			# plt.grid()
			# plt.pause(0.01)


		except:
			time_start_MARFE = None
			time_active_MARFE = None
		full_saved_file_dict_FAST['multi_instrument']['time_start_MARFE'] = time_start_MARFE
		full_saved_file_dict_FAST['multi_instrument']['time_active_MARFE'] = time_active_MARFE

		# I add the triangularity and other measures
		client=pyuda.Client()
		try:
			betan = client.get('/epm/output/globalparameters/betan', shotnumber)
			betan_time = betan.time.data
			betan = betan.data
			elongation = client.get('/epm/output/separatrixgeometry/elongation', shotnumber)
			elongation = elongation.data
			temp = client.get('/epm/output/separatrixgeometry/lowertriangularity', shotnumber)
			if len(np.shape(temp.data))!=1 and pass_number>0:
				try:
					lowertriangularity = full_saved_file_dict_FAST['multi_instrument']['lowertriangularity']
				except:
					lowertriangularity = np.ones_like(betan.time.data)*np.nan
			else:
				lowertriangularity = cp.deepcopy(temp.data)
			temp = client.get('/epm/output/separatrixgeometry/uppertriangularity', shotnumber)
			if len(np.shape(temp.data))!=1 and pass_number>0:	# I have to do this stupid thing because sometimes a txnmxn array is read instead of a t one
				try:
					uppertriangularity = full_saved_file_dict_FAST['multi_instrument']['uppertriangularity']
				except:
					uppertriangularity = np.ones_like(betan.time.data)*np.nan
			else:
				uppertriangularity = cp.deepcopy(temp.data)
			adimensional_quantities_ok = True
			full_saved_file_dict_FAST['multi_instrument']['lowertriangularity'] = lowertriangularity
			full_saved_file_dict_FAST['multi_instrument']['uppertriangularity'] = uppertriangularity
			full_saved_file_dict_FAST['multi_instrument']['elongation'] = elongation
			full_saved_file_dict_FAST['multi_instrument']['betan'] = betan
			full_saved_file_dict_FAST['multi_instrument']['betan_time'] = betan_time
		except:
			adimensional_quantities_ok = False
			pass
		coleval.reset_connection(client)
		del client

		fig, ax = plt.subplots( 2,1,figsize=(12, 12), squeeze=False,sharex=True)
		if pass_number ==0:
			fig.suptitle('shot '+str(shotnumber)+', '+scenario+' , '+experiment+'\nfirst pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		elif pass_number ==1:
			fig.suptitle('shot '+str(shotnumber)+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		elif pass_number ==2:
			fig.suptitle('shot '+str(shotnumber)+', '+scenario+' , '+experiment+'\nthird pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		for i in [0,1]:
			ax[i,0].plot(time_full_binned_crop,inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all,'r-',label='inner_L_poloidal_peak/x-point')
			ax[i,0].plot(time_full_binned_crop,inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all,'r--',label='inner_L_poloidal_baricentre/x-point')
			ax[i,0].plot(time_full_binned_crop,outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all,'b-',label='outer_L_poloidal_peak/x-point')
			ax[i,0].plot(time_full_binned_crop,outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all,'b--',label='outer_L_poloidal_baricentre/x-point')
			if not density_data_missing:
				ax[i,0].plot(time_full_binned_crop,ne_bar/np.nanmax(ne_bar),label='relative ne_bar',color=color[-1])
				ax[i,0].plot(time_full_binned_crop,ne_bar/greenwald_density,'--',label='ne_bar/greenwald_density',color=color[-1])
				ax[i,0].plot(time_full_binned_crop,greenwald_density/np.nanmax(greenwald_density[np.isfinite(greenwald_density)]),'--',label='relative greenwald_density',color=color[-2])
			ax[i,0].plot(time_full_binned_crop,dr_sep_in*10+2,'-',label='dr_sep_in*10+2 [m]',color=color[14])
			ax[i,0].plot(time_full_binned_crop,np.ones_like(dr_sep_in)*2,'--',color=color[14])
			ax[i,0].plot(time_full_binned_crop,dr_sep_out*10+2.5,'-',label='dr_sep_out*10+2.5 [m]',color=color[13])
			ax[i,0].plot(time_full_binned_crop,np.ones_like(dr_sep_in)*2.5,'--',color=color[13])
			try:
				ax[i,0].plot(gas_time,gas_inner/np.nanmax(gas_all),label='inner gas flow/total\n'+str(gas_inner_valves),color=color[15])
			except:
				pass
			try:
				ax[i,0].plot(gas_time,gas_outer/np.nanmax(gas_all),label='outer gas flow/total\n'+str(gas_outer_valves),color=color[4])
			except:
				pass
			try:
				ax[i,0].plot(gas_time,gas_div/np.nanmax(gas_all),label='divertor gas flow/total\n'+str(gas_div_valves),color=color[9])
			except:
				pass
			ax[i,0].plot(gas_time,gas_all/np.nanmax(gas_all),'--',label='relative total gas flow',color=color[6],dashes=(3, 6))

			try:
				if i==0:
					power_balance = calc_psol(shot=shotnumber,smooth_dt =0.015)	# in W
					SS_BEAMPOWER = np.interp(power_balance['t'],time_full_binned_crop,SS_BEAMPOWER,right=0.,left=0.)
					SW_BEAMPOWER = np.interp(power_balance['t'],time_full_binned_crop,SW_BEAMPOWER,right=0.,left=0.)
					output_pohm = power_balance['pohm']
					dWdt = power_balance['pdw_dt']
					input_power = np.interp(time_full_binned_crop,power_balance['t'],output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt)
				# stored_energy = np.interp(power_balance['t'],efit_data['t'],stored_energy,right=0.,left=0.)
				# P_heat = output_pohm + SS_BEAMPOWER + SW_BEAMPOWER
				# P_loss = P_heat-dWdt
				# energy_confinement_time = stored_energy/P_loss	# s
				# full_saved_file_dict_FAST['multi_instrument']['SW_BEAMPOWER'] = SW_BEAMPOWER
				# full_saved_file_dict_FAST['multi_instrument']['SS_BEAMPOWER'] = SS_BEAMPOWER
				# full_saved_file_dict_FAST['multi_instrument']['stored_energy'] = stored_energy
				# full_saved_file_dict_FAST['multi_instrument']['P_heat'] = P_heat
				# full_saved_file_dict_FAST['multi_instrument']['P_loss'] = P_loss
				# full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time'] = energy_confinement_time

				ax[i,0].plot(power_balance['t'],(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt)/np.nanmax(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),label='relative input power (ohm+beams-dW/dt)',color=color[5])
				ax[i,0].plot(power_balance['t'],(output_pohm-dWdt)/np.nanmax(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),'--',label='relative ohmic input power (ohm-dW/dt)',color=color[5])
				ax[i,0].plot(power_balance['t'],power_balance['prad_core']/(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),label='prad_core/input power',color=color[7])
				temp = inverted_data[:,:,inversion_Z<0]
				temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
				# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
				ax[i,0].plot(time_full_binned_crop,2*temp/input_power,label='total IRVB power (Z<0 x2)/input power',color=color[2])
				full_saved_file_dict_FAST['multi_instrument']['power_balance_t'] = power_balance['t']
				full_saved_file_dict_FAST['multi_instrument']['power_balance_pohm'] = output_pohm
				full_saved_file_dict_FAST['multi_instrument']['power_balance_pdw_dt'] = dWdt
				full_saved_file_dict_FAST['multi_instrument']['power_balance_prad_core'] = power_balance['prad_core']
			except:
				input_power = output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt

				ax[i,0].plot(time_full_binned_crop,(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt)/np.nanmax(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),label='relative input power (ohm+beams-dW/dt)',color=color[5])
				ax[i,0].plot(time_full_binned_crop,(output_pohm-dWdt)/np.nanmax(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),'--',label='relative ohmic input power (ohm-dW/dt)',color=color[5])
				temp = inverted_data[:,:,inversion_Z<0]
				temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
				# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
				ax[i,0].plot(time_full_binned_crop,2*temp/input_power,label='total IRVB power (Z<0 x2)/input power',color=color[2])
			if jsat_read:
				if True:	# if using Peter Ryan function
					temp = np.max([jsat_lower_inner_small_max,jsat_lower_outer_small_max,jsat_lower_inner_mid_max,jsat_lower_outer_mid_max,jsat_lower_inner_large_max,jsat_lower_outer_large_max],axis=0)
					ax[i,0].plot(jsat_time,np.nanmax([jsat_lower_inner_small_max,jsat_lower_outer_small_max,jsat_lower_inner_mid_max,jsat_lower_outer_mid_max,jsat_lower_inner_large_max,jsat_lower_outer_large_max],axis=0)/np.nanmax(temp),label='relative jsat lower max',color=color[8],alpha=0.4)
					temp = np.max([jsat_upper_inner_small_max,jsat_upper_outer_small_max,jsat_upper_inner_mid_max,jsat_upper_outer_mid_max,jsat_upper_inner_large_max,jsat_upper_outer_large_max],axis=0)
					ax[i,0].plot(jsat_time,np.nanmax([jsat_upper_inner_small_max,jsat_upper_outer_small_max,jsat_upper_inner_mid_max,jsat_upper_outer_mid_max,jsat_upper_inner_large_max,jsat_upper_outer_large_max],axis=0)/np.nanmax(temp),'--',label='relative jsat upper max',color=color[1],alpha=0.4)
					temp = jsat_lower_inner_small_integrated+jsat_lower_outer_small_integrated+jsat_lower_inner_mid_integrated+jsat_lower_outer_mid_integrated+jsat_lower_inner_large_integrated+jsat_lower_outer_large_integrated
					ax[i,0].plot(jsat_time,temp/np.nanmax(temp),':',label='relative jsat lower integrated',color=color[8],alpha=0.4)
					temp = jsat_upper_inner_small_integrated+jsat_upper_outer_small_integrated+jsat_upper_inner_mid_integrated+jsat_upper_outer_mid_integrated+jsat_upper_inner_large_integrated+jsat_upper_outer_large_integrated
					ax[i,0].plot(jsat_time,temp/np.nanmax(temp),':',label='relative jsat upper integrated',color=color[1],alpha=0.4)
				else:
					temp = np.concatenate([median_filter(jsat_lower_inner_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_inner_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_inner_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11)])
					ax[i,0].plot(jsat_time,np.nanmax([jsat_lower_inner_small_max,jsat_lower_outer_small_max,jsat_lower_inner_mid_max,jsat_lower_outer_mid_max,jsat_lower_inner_large_max,jsat_lower_outer_large_max],axis=0)/np.nanmax(temp),label='relative jsat lower max',color=color[8],alpha=0.4)
					temp = np.concatenate([median_filter(jsat_upper_inner_small_max,size=11),median_filter(jsat_upper_outer_small_max,size=11),median_filter(jsat_upper_inner_mid_max,size=11),median_filter(jsat_upper_outer_mid_max,size=11),median_filter(jsat_upper_inner_large_max,size=11),median_filter(jsat_upper_outer_large_max,size=11)])
					ax[i,0].plot(jsat_time,np.nanmax([jsat_upper_inner_small_max,jsat_upper_outer_small_max,jsat_upper_inner_mid_max,jsat_upper_outer_mid_max,jsat_upper_inner_large_max,jsat_upper_outer_large_max],axis=0)/np.nanmax(temp),'--',label='relative jsat upper max',color=color[1],alpha=0.4)
			ax[i,0].grid()
		ax[0,0].legend(loc='best', fontsize='x-small',ncol=3)
		ax[0,0].set_ylim(bottom=1,top=4)
		ax[1,0].set_ylim(bottom=-0.1,top=1)
		ax[1,0].set_xlabel('time [s]')
		plt.subplots_adjust(wspace=0, hspace=0)
		# plt.pause(0.01)
		plt.savefig(filename_root+filename_root_add+'_all_variables.eps')
		plt.close()



		# plot of absolute quantities
		fig, ax = plt.subplots( 13,1,figsize=(12, 50), squeeze=False,sharex=False)
		if pass_number ==0:
			fig.suptitle('shot '+str(shotnumber)+', '+scenario+' , '+experiment+'\nfirst pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		elif pass_number ==1:
			fig.suptitle('shot '+str(shotnumber)+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		elif pass_number ==2:
			fig.suptitle('shot '+str(shotnumber)+', '+scenario+' , '+experiment+'\nthird pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		ax[0,0].axhline(y=1,color='k',linestyle='--')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_midplane_all/inner_L_poloidal_x_point_all,'m--')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all,'r-',label=r'$IN_{peak}$')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all,'r:',label=r'$IN_{peak\;only\;leg}$')
		# ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_baricentre_only_leg_all/inner_L_poloidal_x_point_all,'r',label=r'$IN_{bari\;only\;leg}$')
		inner_half_peak_L_pol_all = np.array(inner_half_peak_L_pol_all)
		temp = np.abs(inner_half_peak_L_pol_all.T-inner_half_peak_L_pol_all[:,1]).T
		ax[0,0].errorbar(time_full_binned_crop,inner_half_peak_L_pol_all[:,1]/inner_L_poloidal_x_point_all,yerr=[temp[:,0]/inner_L_poloidal_x_point_all,temp[:,2]/inner_L_poloidal_x_point_all],color='r')
		ax[0,0].plot(time_full_binned_crop,inner_half_peak_L_pol_all[:,1]/inner_L_poloidal_x_point_all,'r+',label=r'$IN_{0.5\;front\;InLine}$')
		inner_half_peak_divertor_L_pol_all = np.array(inner_half_peak_divertor_L_pol_all)
		temp = np.abs(inner_half_peak_divertor_L_pol_all.T-inner_half_peak_divertor_L_pol_all[:,1]).T
		ax[0,0].errorbar(time_full_binned_crop,inner_half_peak_divertor_L_pol_all[:,1]/inner_L_poloidal_x_point_all,yerr=[temp[:,0]/inner_L_poloidal_x_point_all,temp[:,2]/inner_L_poloidal_x_point_all],color='r')
		ax[0,0].plot(time_full_binned_crop,inner_half_peak_divertor_L_pol_all[:,1]/inner_L_poloidal_x_point_all,'ro',fillstyle='none',label=r'$IN_{0.5\;front\;InLineDiv}$')
		ax[0,0].plot(time_full_binned_crop,movement_local_inner_leg_mean_emissivity,'r')
		ax[0,0].plot(time_full_binned_crop,movement_local_inner_leg_mean_emissivity,'rx',label=r'$IN_{0.5\;front\;binned}$')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all,'r--',label=r'$IN_{baricentre}$')
		ax[0,0].plot(time_full_binned_crop,np.array([inner_local_L_poloidal_all[i_][i] for i_,i in zip(np.arange(len(inner_local_mean_emis_all)),np.argmax(inner_local_mean_emis_all,axis=1))])/inner_L_poloidal_x_point_all,'r-.',label=r'$IN_{loc\;emissivity}$')
		ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all,'b-',label=r'$OUT_{peak}$')
		ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_peak_only_leg_all/outer_L_poloidal_x_point_all,'b:',label=r'$OUT_{peak\;only\;leg}$')
		# ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_baricentre_only_leg_all/outer_L_poloidal_x_point_all,'b',label=r'$OUT_{bari\;only\;leg}$')
		outer_half_peak_L_pol_all = np.array(outer_half_peak_L_pol_all)
		temp = np.abs(outer_half_peak_L_pol_all.T-outer_half_peak_L_pol_all[:,1]).T
		ax[0,0].errorbar(time_full_binned_crop,outer_half_peak_L_pol_all[:,1]/outer_L_poloidal_x_point_all,yerr=[temp[:,0]/outer_L_poloidal_x_point_all,temp[:,2]/outer_L_poloidal_x_point_all],color='b')
		ax[0,0].plot(time_full_binned_crop,outer_half_peak_L_pol_all[:,1]/outer_L_poloidal_x_point_all,'b+',label=r'$OUT_{0.5\;front\;InLine}$')
		outer_half_peak_divertor_L_pol_all = np.array(outer_half_peak_divertor_L_pol_all)
		temp = np.abs(outer_half_peak_divertor_L_pol_all.T-outer_half_peak_divertor_L_pol_all[:,1]).T
		ax[0,0].errorbar(time_full_binned_crop,outer_half_peak_divertor_L_pol_all[:,1]/outer_L_poloidal_x_point_all,yerr=[temp[:,0]/outer_L_poloidal_x_point_all,temp[:,2]/outer_L_poloidal_x_point_all],color='b')
		ax[0,0].plot(time_full_binned_crop,outer_half_peak_divertor_L_pol_all[:,1]/outer_L_poloidal_x_point_all,'bo',fillstyle='none',label=r'$OUT_{0.5\;front\;InLineDiv}$')
		ax[0,0].plot(time_full_binned_crop,movement_local_outer_leg_mean_emissivity,'b')
		ax[0,0].plot(time_full_binned_crop,movement_local_outer_leg_mean_emissivity,'bx',label=r'$OUT_{0.5\;front\;binned}$')
		ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all,'b--',label=r'$OUT_{baricentre}$')
		ax[0,0].plot(time_full_binned_crop,np.array([outer_local_L_poloidal_all[i_][i] for i_,i in zip(np.arange(len(outer_local_mean_emis_all)),np.argmax(outer_local_mean_emis_all,axis=1))])/outer_L_poloidal_x_point_all,'b-.',label=r'$OUT_{loc\;emissivity}$')
		ax[0,0].set_ylim(bottom=-0.1,top=4)
		ax[0,0].grid()
		ax[0,0].set_ylabel('Lpoloidal/Lpol x-pt [au]')
		ax0 = ax[0,0].twinx()  # instantiate a second axes that shares the same x-axis
		# ax2.spines["right"].set_position(("axes", 1.1125))
		ax0.spines["right"].set_visible(True)
		a0a, = ax0.plot(time_full_binned_crop,psiN_peak_inner_all,label=r'$IN\psi_{N\;peak}$',color='g')
		a0b, = ax0.plot(time_full_binned_crop,psiN_core_inner_side_baricenter_all,':',label=r'$IN\psi_{N\;inner\;core\;baric}$',color='g')
		ax0.axhline(y=1,color='g',linestyle='--')
		ax0.set_ylabel('psi N [au]', color='g')  # we already handled the x-label with ax1
		# ax2.tick_params(axis='y', labelcolor=a2a.get_color())
		ax0.set_ylim(bottom=max(0.9,min(0.95,np.nanmin([psiN_peak_inner_all,psiN_core_inner_side_baricenter_all]))),top=min(1.2,max(1.05,np.nanmax([psiN_peak_inner_all,psiN_core_inner_side_baricenter_all]))))
		handles, labels = ax[0,0].get_legend_handles_labels()
		handles.append(a0a)
		labels.append(a0a.get_label())
		handles.append(a0b)
		labels.append(a0b.get_label())
		ax0.legend(handles=handles, labels=labels, loc='upper left', fontsize='xx-small',ncol=3)
		if time_start_MARFE!=None:
			ax[0,0].axvline(x=time_start_MARFE,linestyle='--',color='k',label='MARFE start from bolo')
		if time_active_MARFE!=None:
			ax[0,0].axvline(x=time_active_MARFE,linestyle='-',color='k',label='MARFE active from bolo')
		# ax[0,0].legend(loc='best', fontsize='xx-small')
		if False:	# moved together with Tu
			ax[1,0].plot(time_full_binned_crop,nu_mean,'--',label='ne,up mean',color=color[-2])
			ax[1,0].plot(time_full_binned_crop,nu_cowley,':',label='ne,up Cowley',color=color[-3])
			ax[1,0].plot(time_full_binned_crop,nu_EFIT,'+',label='ne,up EFIT',color=color[-5])
			ax[1,0].plot(time_full_binned_crop,nu_EFIT_smoothing,'-',label='ne,up EFIT smooth',color=color[-5])
		if not density_data_missing:
			ax[1,0].plot(time_full_binned_crop,core_density,label='core line int ne',color=color[0])
			ax[1,0].plot(time_full_binned_crop,ne_bar,label='core line averaged ne',color=color[-1])
			ax[1,0].plot(time_full_binned_crop,greenwald_density,'--',label='greenwald_density',color=color[-4])

			ax1 = ax[1,0].twinx()  # instantiate a second axes that shares the same x-axis
			# ax1.spines["right"].set_position(("axes", 1.1125))
			ax1.spines["right"].set_visible(True)
			a1a, = ax1.plot(time_full_binned_crop,ne_bar/greenwald_density,label='greenwald fraction',color=color[1])
			ax1.set_ylabel('relative [ua]', color=a1a.get_color())  # we already handled the x-label with ax1
			ax1.tick_params(axis='y', labelcolor=a1a.get_color())
			ax1.set_ylim(bottom=0)
			handles, labels = ax[1,0].get_legend_handles_labels()
			handles.append(a1a)
			labels.append(a1a.get_label())
			ax[1,0].legend(handles=handles, labels=labels, loc='best', fontsize='xx-small')
		# ax[1,0].legend(loc='best', fontsize='xx-small')
		ax[1,0].grid()
		ax[1,0].set_ylabel('ne [#/m3]')
		# ax[1,0].semilogy()
		ax[2,0].plot(time_full_binned_crop,dr_sep_in,'-',label='dr_sep_in (<-)',color=color[14])
		ax[2,0].plot(time_full_binned_crop,dr_sep_out,'-',label='dr_sep_out (<-)',color=color[13])
		# ax[2,0].plot(time_full_binned_crop,radius_inner_separatrix-0.2608,'-',label='inner gap',color=color[12])
		ax[2,0].grid()
		ax[2,0].set_ylabel('dr sep [m]')
		if lambda_q_determination:
			ax[2,0].errorbar(time_full_binned_crop,lambda_q_4_lower,yerr=lambda_q_4_lower_sigma,label=r'$\lambda_q$'+'_4_lower (<-)')
			ax[2,0].errorbar(time_full_binned_crop,lambda_q_10_lower,yerr=lambda_q_10_lower_sigma,label=r'$\lambda_q$'+'_10_lower (<-)')
			ax[2,0].errorbar(time_full_binned_crop,lambda_q_4_upper,yerr=lambda_q_4_upper_sigma,label=r'$\lambda_q$'+'_4_upper (<-)')
			ax[2,0].errorbar(time_full_binned_crop,lambda_q_10_upper,yerr=lambda_q_10_upper_sigma,label=r'$\lambda_q$'+'_10_upper (<-)')
			temp = median_filter(dr_sep_in,size=[int(max(1,0.1/(np.diff(time_full_binned_crop).mean())))],mode='constant',cval=0).tolist() + \
			median_filter(dr_sep_out,size=[int(max(1,0.1/(np.diff(time_full_binned_crop).mean())))],mode='constant',cval=0).tolist() + \
			median_filter(lambda_q_10_lower,size=[int(max(1,0.1/(np.diff(time_full_binned_crop).mean())))],mode='constant',cval=0).tolist() + \
			median_filter(lambda_q_4_lower,size=[int(max(1,0.1/(np.diff(time_full_binned_crop).mean())))],mode='constant',cval=0).tolist() + \
			median_filter(lambda_q_4_upper,size=[int(max(1,0.1/(np.diff(time_full_binned_crop).mean())))],mode='constant',cval=0).tolist() + \
			median_filter(lambda_q_10_upper,size=[int(max(1,0.1/(np.diff(time_full_binned_crop).mean())))],mode='constant',cval=0).tolist()
			ax[2,0].set_ylim(bottom =  np.nanmin(temp), top = np.nanmax(temp))

		ax2 = ax[2,0].twinx()  # instantiate a second axes that shares the same x-axis
		# ax2.spines["right"].set_position(("axes", 1.1125))
		ax2.spines["right"].set_visible(True)
		a2a, = ax2.plot(time_full_binned_crop,vert_displacement,label='vert\n displacement (->)',color='r')
		a2b, = ax2.plot(time_full_binned_crop,radius_inner_separatrix-0.2608,'-',label='inner gap (->)',color=color[12])
		ax2.set_ylabel('vert disp, gap [m]', color='r')  # we already handled the x-label with ax1
		# ax2.tick_params(axis='y', labelcolor=a2a.get_color())
		ax2.set_ylim(bottom=min(0,np.nanmin(np.concatenate([vert_displacement[time_full_binned_crop>0.05][:-4],(radius_inner_separatrix-0.2608)[time_full_binned_crop>0.05][:-4]]))),top=np.nanmax(np.concatenate([vert_displacement[time_full_binned_crop>0.05][:-4],(radius_inner_separatrix-0.2608)[time_full_binned_crop>0.05][:-4]])))
		handles, labels = ax[2,0].get_legend_handles_labels()
		handles.append(a2a)
		handles.append(a2b)
		labels.append(a2a.get_label())
		labels.append(a2b.get_label())
		ax2.legend(handles=handles, labels=labels, loc='best', fontsize='xx-small')
		# ax[2,0].legend(loc='best', fontsize='xx-small')
		try:
			ax[3,0].plot(gas_time,gas_inner,label='inner gas flow'+str(gas_inner_valves),color=color[15])
		except:
			pass
		try:
			ax[3,0].plot(gas_time,gas_outer,label='outer gas flow'+str(gas_outer_valves),color=color[4])
		except:
			pass
		try:
			ax[3,0].plot(gas_time,gas_div,label='divertor gas flow'+str(gas_div_valves),color=color[9])
		except:
			pass
		ax[3,0].plot(gas_time,gas_all,'--',label='total gas flow',color=color[6],dashes=(3, 6))
		ax[3,0].grid()
		ax[3,0].set_ylabel('fueling [#mol/s]')
		ax[3,0].legend(loc='best', fontsize='xx-small')

		ax[4,0].errorbar(time_full_binned_crop,1e-6*equivalent_res_bolo_view_all*2,yerr=1e-6*equivalent_res_bolo_view_sigma_all*2,label='= res bolo',color=color[1])
		try:
			power_balance = calc_psol(shot=shotnumber,smooth_dt =0.015)	# in W
			input_power = np.interp(time_full_binned_crop,power_balance['t'],power_balance['pohm'] + SW_BEAMPOWER + SS_BEAMPOWER-power_balance['pdw_dt'])

			ax[4,0].plot(power_balance['t'],1e-6*(power_balance['pohm'] + SW_BEAMPOWER + SS_BEAMPOWER-power_balance['pdw_dt']),label='input power (ohm+beams-dW/dt)',color=color[5])
			ax[4,0].plot(power_balance['t'],1e-6*(power_balance['pohm']-power_balance['pdw_dt']),'--',label='ohmic input power (ohm-dW/dt)',color=color[5])
			ax[4,0].plot(power_balance['t'],1e-6*power_balance['psol'],label='Psol',color=color[0])
			ax[4,0].plot(power_balance['t'],1e-6*power_balance['pdw_dt'],label='pdw_dt',color=color[6])
			ax[4,0].plot(power_balance['t'],1e-6*power_balance['prad_core'],label='prad_core res bolo',color=color[7])
			temp = inverted_data[:,:,inversion_Z<0]
			temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
			# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
			ax[4,0].plot(time_full_binned_crop,1e-6*2*temp,label='total IRVB power (Z<0 x2)',color=color[2])
			ax[4,0].plot(power_balance['t'],1e-6*SW_BEAMPOWER,label='SW beam x '+str(sw_absorption),color=color[8])
			ax[4,0].plot(power_balance['t'],1e-6*SS_BEAMPOWER,label='SS beam x '+str(ss_absorption),color=color[9])
			temp = power_balance['pohm'] + SW_BEAMPOWER + SS_BEAMPOWER-power_balance['pdw_dt']
			try:
				ax[4,0].set_ylim(bottom=0,top=1e-6*1.2*np.nanmax(median_filter(temp[np.isfinite(temp)],size=21)[np.logical_and(power_balance['t'][np.isfinite(temp)]>0,power_balance['t'][np.isfinite(temp)]<time_full_binned_crop[-5])]))
			except:
				pass

			# here I want to calculate lmda q from the scaling in Harrisom 2013
			try:
				from mastu_exhaust_analysis.read_efit import read_uda
				efit_data = read_uda(shot=shotnumber)
				from scipy.interpolate import RegularGridInterpolator
				interpolator = RegularGridInterpolator((efit_data['t'], efit_data['z'], efit_data['r']), efit_data['Bpol'],bounds_error=False,fill_value=0)
				bpol = np.interp(time_full_binned_crop,efit_data['t'],interpolator(np.array([efit_data['t'],efit_data['z_axis'],efit_data['r_axis']]).T))
				Psol = np.interp(time_full_binned_crop,power_balance['t'],1e-6*power_balance['psol'])
				lambda_q_Lmode_scaling = 0.012 * Psol**0.14 * (1e-6*plasma_current)**-0.36 * (1e-19*nu_EFIT_smoothing)**0.65 * bpol**-0.09
				lambda_q_Hmode_scaling = 0.0045 * Psol**0.11 * (1e-6*plasma_current)**-1.05 * (1e-19*nu_EFIT_smoothing)**0.76 * bpol**-0.07
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_Lmode_scaling'] = np.array(lambda_q_Lmode_scaling)
				full_saved_file_dict_FAST['multi_instrument']['lambda_q_Hmode_scaling'] = np.array(lambda_q_Hmode_scaling)
				handles, labels = ax2.get_legend_handles_labels()
				handles1, labels1 = ax[2,0].get_legend_handles_labels()
				handles = handles +handles1
				labels = labels + labels1
				a2a, = ax[2,0].plot(time_full_binned_crop,lambda_q_Lmode_scaling,'--',label=r'$\lambda_q$'+'_L-mode scaling (<-)')
				a2b, = ax[2,0].plot(time_full_binned_crop,lambda_q_Hmode_scaling,'--',label=r'$\lambda_q$'+'_H-mode scaling (<-)')
				handles.append(a2a)
				handles.append(a2b)
				labels.append(a2a.get_label())
				labels.append(a2b.get_label())
				ax2.legend(handles=handles, labels=labels, loc='center left', fontsize='xx-small')
			except:
				pass

		except:
			input_power = output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt

			ax[4,0].plot(time_full_binned_crop,1e-6*(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),label='input power (ohm+beams-dW/dt)',color=color[5])
			ax[4,0].plot(time_full_binned_crop,1e-6*(output_pohm-dWdt),'--',label='ohmic input power (ohm-dW/dt)',color=color[5])
			temp = inverted_data[:,:,inversion_Z<0]
			temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
			ax[4,0].plot(time_full_binned_crop,1e-6*2*temp,label='total IRVB power (Z<0 x2)',color=color[2])
			ax[4,0].plot(time_full_binned_crop,1e-6*SW_BEAMPOWER,label='SW beam x '+str(sw_absorption),color=color[8])
			ax[4,0].plot(time_full_binned_crop,1e-6*SS_BEAMPOWER,label='SS beam x '+str(ss_absorption),color=color[9])
			ax[4,0].set_ylim(bottom=0)
		ax[4,0].plot(time_full_binned_crop,1e-6*real_core_radiation_all*2,label='core_radiation',color=color[3])
		ax[4,0].plot(time_full_binned_crop,1e-6*real_non_core_radiation_all*2,label='non_core_radiation',color=color[4])
		ax[4,0].grid()
		ax[4,0].set_ylabel('power [MW]')
		ax[4,0].legend(loc='upper left', fontsize='xx-small')
		ax5 = ax[5,0].twinx()  # instantiate a second axes that shares the same x-axis
		if jsat_read:
			a5a = ax5.errorbar(jsat_time,jsat_lower_inner_small_max,yerr=jsat_lower_inner_small_max_sigma,capsize=5,linestyle='--',label='lower_inner_small',color=color[0])
			# ax5.plot(jsat_time,jsat_lower_inner_small_max,'+',color=color[0])
			ax[5,0].errorbar(jsat_time,jsat_lower_outer_small_max,yerr=jsat_lower_outer_small_max_sigma,capsize=5,linestyle='--',label='lower_outer_small',color=color[1])
			# ax[5,0].plot(jsat_time,jsat_lower_outer_small_max,'+',color=color[1])
			a5b = ax5.errorbar(jsat_time,jsat_lower_inner_mid_max,yerr=jsat_lower_inner_mid_max_sigma,capsize=5,linestyle='-',label='lower_inner_mid',color=color[0])
			# ax5.plot(jsat_time,jsat_lower_inner_mid_max,'+',color=color[0])
			ax[5,0].errorbar(jsat_time,jsat_lower_outer_mid_max,yerr=jsat_lower_outer_mid_max_sigma,capsize=5,linestyle='-',label='lower_outer_mid',color=color[1])
			# ax[5,0].plot(jsat_time,jsat_lower_outer_mid_max,'+',color=color[1])
			a5c = ax5.errorbar(jsat_time,jsat_lower_inner_large_max,yerr=jsat_lower_inner_large_max_sigma,capsize=5,linestyle=':',label='lower_inner_large',color=color[0])
			# ax5.plot(jsat_time,jsat_lower_inner_large_max,'+',color=color[0])
			ax[5,0].errorbar(jsat_time,jsat_lower_outer_large_max,yerr=jsat_lower_outer_large_max_sigma,capsize=5,linestyle=':',label='lower_outer_large',color=color[1])
			# ax[5,0].plot(jsat_time,jsat_lower_outer_large_max,'+',color=color[1])
			a5d = ax5.errorbar(jsat_time,jsat_upper_inner_small_max,yerr=jsat_upper_inner_small_max_sigma,capsize=5,linestyle='--',label='upper_inner_small',color=color[2])
			# ax5.plot(jsat_time,jsat_upper_inner_small_max,'+',color=color[2])
			ax[5,0].errorbar(jsat_time,jsat_upper_outer_small_max,yerr=jsat_upper_outer_small_max_sigma,capsize=5,linestyle='--',label='upper_outer_small',color=color[3])
			# ax[5,0].plot(jsat_time,jsat_upper_outer_small_max,'+',color=color[3])
			a5e = ax5.errorbar(jsat_time,jsat_upper_inner_mid_max,yerr=jsat_upper_inner_mid_max_sigma,capsize=5,linestyle='-',label='upper_inner_mid',color=color[2])
			# ax5.plot(jsat_time,jsat_upper_inner_mid_max,'+',color=color[2])
			ax[5,0].errorbar(jsat_time,jsat_upper_outer_mid_max,yerr=jsat_upper_outer_mid_max_sigma,capsize=5,linestyle='-',label='upper_outer_mid',color=color[3])
			# ax[5,0].plot(jsat_time,jsat_upper_outer_mid_max,'+',color=color[3])
			a5f = ax5.errorbar(jsat_time,jsat_upper_inner_large_max,yerr=jsat_upper_inner_large_max_sigma,capsize=5,linestyle=':',label='upper_inner_large',color=color[2])
			# ax5.plot(jsat_time,jsat_upper_inner_large_max,'+',color=color[2])
			ax[5,0].errorbar(jsat_time,jsat_upper_outer_large_max,yerr=jsat_upper_outer_large_max_sigma,capsize=5,linestyle=':',label='upper_outer_large',color=color[3])
			# ax[5,0].plot(jsat_time,jsat_upper_outer_large_max,'+',color=color[3])
			# temp = np.nanmax([median_filter(jsat_lower_inner_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_inner_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_inner_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11) , median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_outer_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11)],axis=0)
			# ax[5,0].set_ylim(bottom=0,top=np.nanmax(temp[np.logical_and(jsat_time>0,jsat_time<time_full_binned_crop[-2])]))
			handles, labels = ax[5,0].get_legend_handles_labels()
			handles.append(a5a)
			handles.append(a5b)
			handles.append(a5c)
			handles.append(a5d)
			handles.append(a5e)
			handles.append(a5f)
			labels.append(a5a.get_label())
			labels.append(a5b.get_label())
			labels.append(a5c.get_label())
			labels.append(a5d.get_label())
			labels.append(a5e.get_label())
			labels.append(a5f.get_label())
			ax[5,0].legend(handles=handles, labels=labels, loc='best', fontsize='xx-small', ncol=2)
		ax[5,0].grid()
		ax[5,0].set_ylabel(r'$j_{sat\;max\;outer}$'+' [A/m2]')
		ax5.spines["right"].set_visible(True)
		ax5.set_ylabel(r'$j_{sat\;max\;inner}$'+' [A/m2]', color='r')  # we already handled the x-label with ax1
		# ax5.tick_params(axis='y', labelcolor=a5a.get_color())
		# ax5.set_ylim(bottom=0)

		ax6 = ax[6,0].twinx()  # instantiate a second axes that shares the same x-axis
		if jsat_read:
			a6a = ax6.errorbar(jsat_time,jsat_lower_inner_small_integrated,yerr=jsat_lower_inner_small_integrated_sigma,capsize=5,linestyle='--',label='lower_inner_small',color=color[0])
			# ax6.plot(jsat_time,jsat_lower_inner_small_integrated,'+',color=color[0])
			ax[6,0].errorbar(jsat_time,jsat_lower_outer_small_integrated,yerr=jsat_lower_outer_small_integrated_sigma,capsize=5,linestyle='--',label='lower_outer_small',color=color[1])
			# ax[6,0].plot(jsat_time,jsat_lower_outer_small_integrated,'+',color=color[1])
			a6b = ax6.errorbar(jsat_time,jsat_lower_inner_mid_integrated,yerr=jsat_lower_inner_mid_integrated_sigma,capsize=5,linestyle='-',label='lower_inner_mid',color=color[0])
			# ax6.plot(jsat_time,jsat_lower_inner_mid_integrated,'+',color=color[0])
			ax[6,0].errorbar(jsat_time,jsat_lower_outer_mid_integrated,yerr=jsat_lower_outer_mid_integrated_sigma,capsize=5,linestyle='-',label='lower_outer_mid',color=color[1])
			# ax[6,0].plot(jsat_time,jsat_lower_outer_mid_integrated,'+',color=color[1])
			a6c = ax6.errorbar(jsat_time,jsat_lower_inner_large_integrated,yerr=jsat_lower_inner_large_integrated_sigma,capsize=5,linestyle=':',label='lower_inner_large',color=color[0])
			# ax6.plot(jsat_time,jsat_lower_inner_large_integrated,'+',color=color[0])
			ax[6,0].errorbar(jsat_time,jsat_lower_outer_large_integrated,yerr=jsat_lower_outer_large_integrated_sigma,capsize=5,linestyle=':',label='lower_outer_large',color=color[1])
			# ax[6,0].plot(jsat_time,jsat_lower_outer_large_integrated,'+',color=color[1])
			a6d = ax6.errorbar(jsat_time,jsat_upper_inner_small_integrated,yerr=jsat_upper_inner_small_integrated_sigma,capsize=5,linestyle='--',label='upper_inner_small',color=color[2])
			# ax6.plot(jsat_time,jsat_upper_inner_small_integrated,'+',color=color[2])
			ax[6,0].errorbar(jsat_time,jsat_upper_outer_small_integrated,yerr=jsat_upper_outer_small_integrated_sigma,capsize=5,linestyle='--',label='upper_outer_small',color=color[3])
			# ax[6,0].plot(jsat_time,jsat_upper_outer_small_integrated,'+',color=color[3])
			a6e = ax6.errorbar(jsat_time,jsat_upper_inner_mid_integrated,yerr=jsat_upper_inner_mid_integrated_sigma,capsize=5,linestyle='-',label='upper_inner_mid',color=color[2])
			# ax6.plot(jsat_time,jsat_upper_inner_mid_integrated,'+',color=color[2])
			ax[6,0].errorbar(jsat_time,jsat_upper_outer_mid_integrated,yerr=jsat_upper_outer_mid_integrated_sigma,capsize=5,linestyle='-',label='upper_outer_mid',color=color[3])
			# ax[6,0].plot(jsat_time,jsat_upper_outer_mid_integrated,'+',color=color[3])
			a6f = ax6.errorbar(jsat_time,jsat_upper_inner_large_integrated,yerr=jsat_upper_inner_large_integrated_sigma,capsize=5,linestyle=':',label='upper_inner_large',color=color[2])
			# ax6.plot(jsat_time,jsat_upper_inner_large_integrated,'+',color=color[2])
			ax[6,0].errorbar(jsat_time,jsat_upper_outer_large_integrated,yerr=jsat_upper_outer_large_integrated_sigma,capsize=5,linestyle=':',label='upper_outer_large',color=color[3])
			# ax[6,0].plot(jsat_time,jsat_upper_outer_large_integrated,'+',color=color[3])
		ax[6,0].grid()
		ax[6,0].set_ylabel(r'$\int j_{sat\;outer}$'+' [A]')
		ax6.spines["right"].set_visible(True)
		ax6.set_ylabel(r'$\int j_{sat\;inner}$'+' [A]', color='r')  # we already handled the x-label with ax1
		# ax5.tick_params(axis='y', labelcolor=a5a.get_color())
		# ax5.set_ylim(bottom=0)
		# handles, labels = ax[6,0].get_legend_handles_labels()
		# handles.append(a6a)
		# handles.append(a6b)
		# handles.append(a6c)
		# handles.append(a6d)
		# handles.append(a6e)
		# handles.append(a6f)
		# labels.append(a6a.get_label())
		# labels.append(a6b.get_label())
		# labels.append(a6c.get_label())
		# labels.append(a6d.get_label())
		# labels.append(a6e.get_label())
		# labels.append(a6f.get_label())
		# ax[5,0].legend(handles=handles, labels=labels, loc='best', fontsize='xx-small')

		if not toroidal_field_good:
			ax[7,0].plot([],[],'+k',label='Toroidal magnetic\nfield at magnetic axis\nflagged as bad data')
		if np.abs(np.nanmedian((energy_confinement_time_transp-energy_confinement_time)/energy_confinement_time))>0.01:	# I want it plotted only if there is a significant difference
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time,label='confinement time NBI approx',color=color[0])
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time_transp,':',label='confinement time transp',color=color[0])
		else:
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time,label='confinement time',color=color[0])
		if not density_data_missing:
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time_98y2,label='98y2',color=color[3],alpha=0.3)
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time_97P,label='97P',color=color[2],alpha=0.3)
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time_HST,label='HST',color=color[1])
			ax[7,0].plot(time_full_binned_crop,energy_confinement_time_LST,label='LST',color=color[5])
		ax[7,0].grid()
		ax[7,0].set_ylim(bottom=0)
		ax[7,0].set_ylabel('confinement time [s]')

		ax7 = ax[7,0].twinx()  # instantiate a second axes that shares the same x-axis
		# ax7.spines["right"].set_position(("axes", 1.1125))
		ax7.spines["right"].set_visible(True)
		if not density_data_missing:
			a7a, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_98y2,'--',color=color[3],alpha=0.3)
			a7b, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_97P,'--',color=color[2],alpha=0.3)
			a7c, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_HST,'--',color=color[1])
			a7d, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_LST,'--',color=color[5])
		ax7.set_ylabel('--=relative time [au]', color='r')  # we already handled the x-label with ax1
		# ax7.tick_params(axis='y', labelcolor=a7a.get_color())
		ax7.set_ylim(bottom=0, top=1.2)
		handles, labels = ax[7,0].get_legend_handles_labels()
		# handles.append(a7a)
		# handles.append(a7b)
		# handles.append(a7c)
		# handles.append(a7d)
		# labels.append(a7a.get_label())
		# labels.append(a7b.get_label())
		# labels.append(a7c.get_label())
		# labels.append(a7d.get_label())
		ax[7,0].legend(handles=handles, labels=labels, loc='best', fontsize='xx-small')

		ax[8,0].plot(Dalpha_time,Dalpha)
		ax[8,0].set_ylim(top=median_filter(Dalpha,size=2000).max())
		ax[8,0].set_ylabel('Dalpha [V]')

		# if time_start_MARFE != None:
		# 	ax[9,0].plot(time_res_bolo,CH27,'g',label='core27')
		# 	ax[9,0].plot(time_res_bolo,CH8_9,'b',label='core8/9')
		# 	if np.isfinite(time_start_MARFE):
		# 		ax[9,0].plot([CH27_start_of_raise,CH27_end_of_raise],[CH27[CH27_start],CH27[CH27_end]],'--g')
		# 		ax[9,0].axvline(x=CH27_start_of_raise,linestyle='--',color='g')
		# 		ax[9,0].plot([CH8_9_start_of_raise,CH8_9_end_of_raise],[CH8_9[CH8_9_start],CH8_9[CH8_9_end]],'--b')
		# 		ax[9,0].axvline(x=CH8_9_start_of_raise,linestyle='--',color='b')
		# 	ax[9,0].legend(loc='best', fontsize='xx-small')
		# 	ax[9,0].set_ylabel('Brightness [W/m2]')
		# 	ax[9,0].grid()
		# 	ax[9,0].set_xlabel('time [s]')
		# else:
		# 	ax[8,0].set_xlabel('time [s]')

		ax8 = ax[8,0].twinx()  # instantiate a second axes that shares the same x-axis
		# ax7.spines["right"].set_position(("axes", 1.1125))
		ax8.spines["right"].set_visible(True)
		if time_start_MARFE != None:
			ax8.plot(time_res_bolo,CH27_26,'r')
			ax8.plot(time_res_bolo,fit_bolo[0]+fit_bolo[1]*np.maximum(0,time_res_bolo-fit_bolo[2]),'--')
			ax8.axvline(x=time_start_MARFE,linestyle='--',color='b')
			ax8.set_ylim(bottom=min(0.5,new_MARFE_marker[np.logical_and(new_MARFE_marker_time<time_full_binned_crop.max(),new_MARFE_marker_time>time_full_binned_crop.min())].min()),top=np.max([3,CH27_26.max(),new_MARFE_marker[np.logical_and(new_MARFE_marker_time<time_full_binned_crop.max(),new_MARFE_marker_time>time_full_binned_crop.min())].max()]))
			# ax8.set_xlabel('time [s]')
			ax8.set_ylabel('Brigtness\nCH27/CH26 [au]', color='r')
		elif time_active_MARFE == None:
			# ax8.set_xlabel('time [s]')
			ax8.set_ylim(bottom=new_MARFE_marker[np.logical_and(new_MARFE_marker_time<time_full_binned_crop.max(),new_MARFE_marker_time>time_full_binned_crop.min())].min(),top=np.max([3,new_MARFE_marker[np.logical_and(new_MARFE_marker_time<time_full_binned_crop.max(),new_MARFE_marker_time>time_full_binned_crop.min())].max()]))
			pass
		if time_active_MARFE != None:
			# ax9 = ax8.twinx()  # instantiate a second axes that shares the same x-axis
			# ax9.spines["right"].set_visible(True)
			ax8.set_ylabel('Brigtness\nCH27/CH26 [au]', color='r')
			ax8.axvline(x=time_active_MARFE,linestyle='--',color='g')
			ax8.axhline(y=3,linestyle='--',color='r')
			# ax8.set_xlabel('time [s]')
		elif time_start_MARFE == None:
			# ax8.set_xlabel('time [s]')
			pass
		ax8.plot(new_MARFE_marker_time,new_MARFE_marker,'--r')
		ax8.grid()

		ax[9,0].set_ylabel('estimated pu [Pa]')
		if TS_reading_success:
			ax[9,0].plot(time_full_binned_crop,pu_cowley,label='cowley')
			ax[9,0].plot(time_full_binned_crop,pu_labombard,label='labombard')
			ax[9,0].plot(time_full_binned_crop,pu_stangeby,label='stangeby')
			temp, = ax[9,0].plot(time_full_binned_crop,pu_EFIT,'+',label='EFIT')
			ax[9,0].plot(time_full_binned_crop,pu_EFIT_smoothing,color=temp.get_color(),label='EFIT smooth')
			ax[9,0].grid()
			ax[9,0].legend(loc='best', fontsize='xx-small')
			ax[9,0].set_ylim(bottom=0,top=np.nanmax([pu_cowley,pu_labombard,pu_stangeby])*1.1)
			# ax[9,0].set_ylim(bottom=0,top=np.nanmax([pu_EFIT])*1.1)

		ax[10,0].set_ylabel('estimated Tu [eV]'+' (-,+)')
		ax10 = ax[10,0].twinx()  # instantiate a second axes that shares the same x-axis
		ax10.spines["right"].set_visible(True)
		ax10.set_ylabel('estimated nu '+r'$[10^{19}\frac{\#}{m^3}]$'+' (- -,o)')
		if TS_reading_success:
			temp, = ax[10,0].plot(time_full_binned_crop,tu_cowley,label='cowley')
			ax10.plot(time_full_binned_crop,nu_cowley*1e-19,'--',color=temp.get_color())
			temp, = ax[10,0].plot(time_full_binned_crop,tu_labombard,label='labombard')
			ax10.plot(time_full_binned_crop,nu_labombard*1e-19,'--',color=temp.get_color())
			temp, = ax[10,0].plot(time_full_binned_crop,tu_stangeby,label='stangeby')
			ax10.plot(time_full_binned_crop,nu_stangeby*1e-19,'--',color=temp.get_color())
			# temp, = ax[10,0].plot(time_full_binned_crop,tu_EFIT,'+',label='tu EFIT')
			# ax[10,0].plot(time_full_binned_crop,tu_EFIT_smoothing,color=temp.get_color(),label='EFIT smooth')
			# ax10.plot(time_full_binned_crop,nu_EFIT*1e-19,marker='o',fillstyle='none',color=temp.get_color())
			# ax10.plot(time_full_binned_crop,nu_EFIT_smoothing*1e-19,'--',color=temp.get_color())
			temp, = ax[10,0].plot(time_full_binned_crop,tu_EFIT,'+',label='tu EFIT')
			ax[10,0].errorbar(time_full_binned_crop,tu_EFIT_smoothing,yerr=tu_EFIT_smoothing_spatial_uncertainty,color=temp.get_color(),label='tu EFIT smooth',capsize=5)
			temp, = ax10.plot(time_full_binned_crop,nu_EFIT*1e-19,marker='o',fillstyle='none',color='C6')
			temp = ax10.errorbar(time_full_binned_crop,nu_EFIT_smoothing*1e-19,yerr=nu_EFIT_smoothing_spatial_uncertainty*1e-19,linestyle='--',color=temp.get_color(),capsize=5,label='nu EFIT smooth')
			ax[10,0].grid()
			# ax[10,0].legend(loc='best', fontsize='xx-small')
			handles, labels = ax[10,0].get_legend_handles_labels()
			handles.append(temp)
			labels.append(temp.get_label())
			ax[10,0].legend(handles=handles, labels=labels, loc='best', fontsize='xx-small')

			select = time_full_binned_crop<time_full_binned_crop.max()-0.1
			ax[10,0].set_ylim(bottom=0,top=np.nanmax([tu_cowley[select],tu_labombard[select],tu_stangeby[select],tu_EFIT_smoothing[select]])*1.1)
			ax10.set_ylim(bottom=0,top=np.nanmax([nu_cowley[select],nu_labombard[select],nu_stangeby[select],nu_EFIT_smoothing[select]])*1.1*1e-19)

		ax[11,0].axhline(y=1,color='k',linestyle='--')
		ax[11,0].axhline(y=1.15,color='k',linestyle='--',label=r'$\psi_{N}=1.15$')
		ax[11,0].plot(time_full_binned_crop,psiN_peak_inner_all,label=r'$IN\psi_{N\;peak}$')
		ax[11,0].plot(time_full_binned_crop,psiN_min_lower_baffle_all,label=r'$\psi_{N\;lower\;baffle}$')
		ax[11,0].plot(time_full_binned_crop,psiN_min_upper_baffle_all,'--',label=r'$\psi_{N\;upper\;baffle}$')
		ax[11,0].plot(time_full_binned_crop,psiN_min_lower_target_all,label=r'$\psi_{N\;lower\;target}$')
		ax[11,0].plot(time_full_binned_crop,psiN_min_upper_target_all,'--',label=r'$\psi_{N\;upper\;target}$')
		ax[11,0].plot(time_full_binned_crop,psiN_min_central_column_all,label=r'$\psi_{N\;central\;column}$')
		ax[11,0].grid()
		ax[11,0].legend(loc='best', fontsize='xx-small')
		ax[11,0].set_ylim(bottom=0.9,top=1.3)
		ax[11,0].set_ylabel(r'$\psi_{N}$')

		if adimensional_quantities_ok:
			try:
				print(lowertriangularity)
				ax[12,0].plot(betan_time,lowertriangularity,label='lowertriangularity')
				print(uppertriangularity)
				ax[12,0].plot(betan_time,uppertriangularity,label='uppertriangularity')
				ax[12,0].plot(betan_time,elongation,label='elongation')
				ax[12,0].plot(betan_time,betan,label='betan')
			except:
				pass
			ax[12,0].grid()
			ax[12,0].legend(loc='best', fontsize='xx-small')

		ax[0,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[1,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[2,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[3,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[4,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[5,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[6,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[7,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[8,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[9,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[10,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[11,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[12,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())

		# plt.subplots_adjust(wspace=0, hspace=0)
		# plt.pause(0.01)
		plt.savefig(filename_root+filename_root_add+'_all_variables_absolute.eps')
		plt.close()

		coleval.reset_connection(client)
		del client


		if pass_number ==0:
			full_saved_file_dict_FAST['first_pass']['inverted_dict'] = inverted_dict
		elif pass_number ==1:
			full_saved_file_dict_FAST['second_pass']['inverted_dict'] = inverted_dict
		elif pass_number ==2:
			full_saved_file_dict_FAST['third_pass']['inverted_dict'] = inverted_dict
		# np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
		coleval.savez_protocol4(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
		print('DONE '+laser_to_analyse)

	except Exception as e:
		print('FAILED ' + laser_to_analyse)
		logging.exception('with error: ' + str(e))
	return full_saved_file_dict_FAST

full_saved_file_dict_FAST = temp_function(full_saved_file_dict_FAST)

##########
