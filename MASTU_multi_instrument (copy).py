from mastu_exhaust_analysis.calc_pohm import calc_pohm

def temp_function(full_saved_file_dict_FAST):
	try:
		try:
			trash = full_saved_file_dict_FAST['multi_instrument']
			full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
		except:
			full_saved_file_dict_FAST['multi_instrument'] = dict([])
		grid_resolution = 2	# cm
		time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
		inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
		inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
		binning_type = inverted_dict[str(grid_resolution)]['binning_type']

		EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
		efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
		inversion_R = inverted_dict[str(grid_resolution)]['geometry']['R']
		inversion_Z = inverted_dict[str(grid_resolution)]['geometry']['Z']


		client=pyuda.Client()
		try:
			ne_bar_dict=calc_ne_bar(efit_reconstruction.shotnumber, efit_data = None)
			greenwald_density = np.interp(time_full_binned_crop,ne_bar_dict['t'],ne_bar_dict['greenwald_density'])
			ne_bar = np.interp(time_full_binned_crop,ne_bar_dict['t'],ne_bar_dict['data'])	# this should be the line averaged density
			density_data_missing = False
			full_saved_file_dict_FAST['multi_instrument']['greenwald_density'] = greenwald_density
			full_saved_file_dict_FAST['multi_instrument']['ne_bar'] = ne_bar
			data = client.get('/ANE/DENSITY',efit_reconstruction.shotnumber)
			# core_density = np.array([data.data[np.abs(data.time.data-time).argmin()] for time in time_full_binned_crop])
			core_density = np.interp(time_full_binned_crop,data.time.data,data.data)	# this is core line integrated density
			full_saved_file_dict_FAST['multi_instrument']['line_integrated_density'] = core_density
		except:
			density_data_missing = True


		outer_local_mean_emis_all,outer_local_power_all,outer_local_L_poloidal_all,outer_leg_length_interval_all,outer_leg_length_all,outer_data_length,outer_leg_resolution,outer_emissivity_baricentre_all,outer_emissivity_peak_all,outer_L_poloidal_baricentre_all,outer_L_poloidal_peak_all,trash,dr_sep_in,dr_sep_out,outer_L_poloidal_x_point_all = coleval.baricentre_outer_separatrix_radiation(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)
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
		full_saved_file_dict_FAST['multi_instrument']['time_full_binned_crop'] = time_full_binned_crop
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_in'] = dr_sep_in
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_out'] = dr_sep_out
		inverted_dict[str(grid_resolution)]['outer_L_poloidal_x_point_all'] = outer_L_poloidal_x_point_all
		inner_local_mean_emis_all,inner_local_power_all,inner_local_L_poloidal_all,inner_leg_length_interval_all,inner_leg_length_all,inner_data_length,inner_leg_resolution,inner_emissivity_baricentre_all,inner_emissivity_peak_all,inner_L_poloidal_baricentre_all,inner_L_poloidal_peak_all,inner_L_poloidal_peak_only_leg_all,trash,dr_sep_in,dr_sep_out,inner_L_poloidal_x_point_all = coleval.baricentre_inner_separatrix_radiation(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)
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
		full_saved_file_dict_FAST['multi_instrument']['time_full_binned_crop'] = time_full_binned_crop
		full_saved_file_dict_FAST['multi_instrument']['greenwald_density'] = greenwald_density
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_in'] = dr_sep_in
		full_saved_file_dict_FAST['multi_instrument']['dr_sep_out'] = dr_sep_out
		inverted_dict[str(grid_resolution)]['inner_L_poloidal_x_point_all'] = inner_L_poloidal_x_point_all

		real_core_radiation_all,real_non_core_radiation_all = coleval.inside_vs_outside_separatrix_radiation(inverted_data,inverted_data_sigma,inversion_R,inversion_Z,time_full_binned_crop,efit_reconstruction)
		inverted_dict[str(grid_resolution)]['real_core_radiation_all'] = real_core_radiation_all
		inverted_dict[str(grid_resolution)]['real_non_core_radiation_all'] = real_non_core_radiation_all

		print('marker pre gas')
		gas = get_gas_info(efit_reconstruction.shotnumber)
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
			elif valve['valve'][:3] == 'pfr':
				gas_div += valve['flux'][select]
				gas_div_valves.append(valve['valve'])
			elif valve['valve'][:3] == 'lfs':
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
		try:
			try:
				fdir = uda_transfer(efit_reconstruction.shotnumber,'elp',extra_path='/0'+str(efit_reconstruction.shotnumber)[:2])
				lp_data,output_contour1 = coleval.read_LP_data(efit_reconstruction.shotnumber,path = os.path.split(fdir)[0])
				os.remove(fdir)
			except:
				lp_data,output_contour1 = coleval.read_LP_data(efit_reconstruction.shotnumber)
			if True:	# use ot Peter Ryan function to combine data within time slice and to filter when strike point is on a dead channel
				temp = np.logical_and(time_full_binned_crop>output_contour1['time'][0].min(),time_full_binned_crop<output_contour1['time'][0].max())
				trange = tuple(map(list, np.array([time_full_binned_crop[temp] - np.mean(np.diff(time_full_binned_crop))/2,time_full_binned_crop[temp] + np.mean(np.diff(time_full_binned_crop))/2]).T))
				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10 ,log_data=False, quantity = 'isat', coordinate='r',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
				s10_lower_good_probes = np.nanmax(median_filter((output_contour1['y'][0]>0.005),size=[1,5]),axis=1)	# threshold for broken probes
				s10_lower_s = output_contour1['s'][0]
				s10_lower_r = output_contour1['r'][0]
				output,Eich=compare_shots(version='new',filepath='/home/ffederic/work/irvb/from_pryan_LP'+'/',shot=[efit_reconstruction.shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=10, quantity = 'jsat_tile', coordinate='r',tiles=['C5','C6','T2','T3','T4','T5'],combine=True,show=False)
				s10_lower_jsat = []
				s10_lower_jsat_r = []
				s10_lower_jsat_s = []
				for i in range(len(output)):
					s10_lower_jsat.append(output[i]['y_data'][0])	# here there are only standard probes
					s10_lower_jsat_r.append(output[i]['x_data'][0])	# here there are only standard probes
					s10_lower_jsat_s.append(np.array([output_contour1['s'][0][np.abs(value-output_contour1['r'][0]).argmin()] for value in output[i]['x_data'][0]]))
				plt.close('all')

				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=4 ,log_data=False, quantity = 'isat', coordinate='r',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
				s4_lower_good_probes = np.nanmax(median_filter((output_contour1['y'][0]>0.005),size=[1,5]),axis=1)	# threshold for broken probes
				s4_lower_s = output_contour1['s'][0]
				s4_lower_r = output_contour1['r'][0]
				s4_lower_z = output_contour1['z'][0]
				output,Eich=compare_shots(version='new',filepath='/home/ffederic/work/irvb/from_pryan_LP'+'/',shot=[efit_reconstruction.shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='lower', sectors=4, quantity = 'jsat_tile', coordinate='r',tiles=['C5','C6','T2','T3','T4','T5'],combine=True,show=False)
				s4_lower_jsat = []
				s4_lower_jsat_r = []
				s4_lower_jsat_s = []
				for i in range(len(output)):
					s4_lower_jsat.append(output[i]['y_data'][0])	# here there are only small probes
					s4_lower_jsat_r.append(output[i]['x_data'][0])	# here there are only small probes
					s4_lower_jsat_s.append(np.array([output_contour1['s'][0][np.abs(value-output_contour1['r'][0]).argmin()] for value in output[i]['x_data'][0]]))
				plt.close('all')

				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=4 ,log_data=False, quantity = 'isat', coordinate='r',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
				s4_upper_good_probes = np.nanmax(median_filter((output_contour1['y'][0]>0.005),size=[1,5]),axis=1)	# threshold for broken probes
				s4_upper_s = output_contour1['s'][0]
				s4_upper_r = output_contour1['r'][0]
				output,Eich=compare_shots(version='new',filepath='/home/ffederic/work/irvb/from_pryan_LP'+'/',shot=[efit_reconstruction.shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=4, quantity = 'jsat_tile', coordinate='r',tiles=['C5','C6','T2','T3','T4','T5'],combine=True,show=False)
				s4_upper_jsat = []
				s4_upper_jsat_r = []
				s4_upper_jsat_s = []
				for i in range(len(output)):
					s4_upper_jsat.append(output[i]['y_data'][0])	# here there are only standard probes
					s4_upper_jsat_r.append(output[i]['x_data'][0])	# here there are only standard probes
					s4_upper_jsat_s.append(np.array([output_contour1['s'][0][np.abs(value-output_contour1['r'][0]).argmin()] for value in output[i]['x_data'][0]]))
				plt.close('all')

				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=10 ,log_data=False, quantity = 'isat', coordinate='r',tiles=['C5','C6','T2','T3','T4'],show=False)
				s10_upper_std_good_probes = np.nanmax(median_filter((output_contour1['y'][0]>0.005),size=[1,5]),axis=1)	# threshold for broken probes
				s10_upper_std_s = output_contour1['s'][0]
				s10_upper_std_r = output_contour1['r'][0]
				output,Eich=compare_shots(version='new',filepath='/home/ffederic/work/irvb/from_pryan_LP'+'/',shot=[efit_reconstruction.shotnumber]*len(trange),bin_x_step=1e-4,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='r',tiles=['C5','C6','T2','T3','T4'],combine=True,show=False)
				s10_upper_std_jsat = []
				s10_upper_std_jsat_r = []
				s10_upper_std_jsat_s = []
				for i in range(len(output)):
					s10_upper_std_jsat.append(output[i]['y_data'][0])	# here there are only standard probes
					s10_upper_std_jsat_r.append(output[i]['x_data'][0])	# here there are only standard probes
					s10_upper_std_jsat_s.append(np.array([output_contour1['s'][0][np.abs(value-output_contour1['r'][0]).argmin()] for value in output[i]['x_data'][0]]))
				plt.close('all')

				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='upper', sectors=10 ,log_data=False, quantity = 'isat', coordinate='r',tiles=['T5'],show=False)
				s10_upper_large_good_probes = np.nanmax(median_filter((output_contour1['y'][0]>0.005),size=[1,5]),axis=1)	# threshold for broken probes
				s10_upper_large_s = output_contour1['s'][0]
				s10_upper_large_r = output_contour1['r'][0]
				output,Eich=compare_shots(version='new',filepath='/home/ffederic/work/irvb/from_pryan_LP'+'/',shot=[efit_reconstruction.shotnumber]*len(trange),bin_x_step=1e-3,bad_probes=None,trange=trange,divertor='upper', sectors=10, quantity = 'jsat_tile', coordinate='r',tiles=['T5'],combine=True,show=False)
				s10_upper_large_jsat = []
				s10_upper_large_jsat_r = []
				s10_upper_large_jsat_s = []
				for i in range(len(output)):
					s10_upper_large_jsat.append(output[i]['y_data'][0])	# here there are only standard probes
					s10_upper_large_jsat_r.append(output[i]['x_data'][0])	# here there are only standard probes
					s10_upper_large_jsat_s.append(np.array([output_contour1['s'][0][np.abs(value-output_contour1['r'][0]).argmin()] for value in output[i]['x_data'][0]]))
				plt.close('all')

				closeness_limit_to_dead_channels = 0.01	# m
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

				skipped = 0
				for i_time,time in enumerate(time_full_binned_crop):
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
						skipped += 1
						continue

					i_time -= skipped

					i_efit_time = np.abs(efit_reconstruction.time-time).argmin()
					inner_strike_point = [np.min(efit_reconstruction.strikepointR[i_efit_time][:2]),0]
					inner_strike_point[1] = -efit_reconstruction.strikepointZ[i_efit_time][np.abs(efit_reconstruction.strikepointR[i_efit_time]-inner_strike_point[0]).argmin()]
					outer_strike_point = [np.max(efit_reconstruction.strikepointR[i_efit_time][:2]),0]
					outer_strike_point[1] = -efit_reconstruction.strikepointZ[i_efit_time][np.abs(efit_reconstruction.strikepointR[i_efit_time]-outer_strike_point[0]).argmin()]
					mid_strike_point = [np.mean(efit_reconstruction.strikepointR[i_efit_time][:2]),-np.mean(efit_reconstruction.strikepointZ[i_efit_time][:2])]

					# upper
					s4_upper_inner_test = np.sum(np.logical_not(s4_upper_good_probes)[np.abs(s4_upper_r-inner_strike_point[0])<closeness_limit_to_dead_channels])==0
					s10_upper_std_inner_test = np.sum(np.logical_not(s10_upper_std_good_probes)[np.abs(s10_upper_std_r-inner_strike_point[0])<closeness_limit_to_dead_channels])==0
					s10_upper_large_inner_test = np.sum(np.logical_not(s10_upper_large_good_probes)[np.abs(s10_upper_large_r-inner_strike_point[0])<closeness_limit_to_dead_channels])==0
					s4_upper_outer_test = np.sum(np.logical_not(s4_upper_good_probes)[np.abs(s4_upper_r-outer_strike_point[0])<closeness_limit_to_dead_channels])==0
					s10_upper_std_outer_test = np.sum(np.logical_not(s10_upper_std_good_probes)[np.abs(s10_upper_std_r-outer_strike_point[0])<closeness_limit_to_dead_channels])==0
					s10_upper_large_outer_test = np.sum(np.logical_not(s10_upper_large_good_probes)[np.abs(s10_upper_large_r-outer_strike_point[0])<closeness_limit_to_dead_channels])==0

					jsat_upper_inner_small_max.append(0)
					jsat_upper_outer_small_max.append(0)
					jsat_upper_inner_small_integrated.append(0)
					jsat_upper_outer_small_integrated.append(0)
					temp = [np.nan]
					if s4_upper_inner_test and s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp = np.concatenate([[0],temp,s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]])
						temp = np.concatenate([[0],temp,s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_upper_inner_mid_max.append(np.nanmax(temp))
					temp = 0
					if s4_upper_inner_test and s10_upper_std_inner_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp += max(np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]),np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]))
					# elif s4_upper_inner_test:
					# 	temp += np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]<mid_strike_point[0]])
					# elif s10_upper_std_inner_test:
					# 	temp += np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_upper_inner_mid_integrated.append(temp)
					temp = [np.nan]
					if s4_upper_outer_test and s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp = np.concatenate([[0],temp,s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]])
						temp = np.concatenate([[0],temp,s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_upper_outer_mid_max.append(np.nanmax(temp))
					temp = 0
					if s4_upper_outer_test and s10_upper_std_outer_test:	# I should consider them independently, but s10 upper has too many dead channes and it's not trustworthy on its own
						temp += max(np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]),np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]))
					# elif s4_upper_outer_test:
					# 	temp += np.trapz(s4_upper_jsat[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_upper_jsat_r[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]],x=s4_upper_jsat_s[i_time][s4_upper_jsat_r[i_time]>mid_strike_point[0]])
					# elif s10_upper_std_outer_test:
					# 	temp += np.trapz(s10_upper_std_jsat[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_std_jsat_r[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_std_jsat_s[i_time][s10_upper_std_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_upper_outer_mid_integrated.append(temp)

					temp = [np.nan]
					if s10_upper_large_inner_test:
						temp = np.concatenate([[0],temp,s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_upper_inner_large_max.append(np.nanmax(temp))
					temp = 0
					if s10_upper_large_inner_test:
						temp += np.trapz(s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]],x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_upper_inner_large_integrated.append(temp)
					temp = [np.nan]
					if s10_upper_large_outer_test:
						temp = np.concatenate([[0],temp,s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_upper_outer_large_max.append(np.nanmax(temp))
					temp = 0
					if s10_upper_large_outer_test:
						temp += np.trapz(s10_upper_large_jsat[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_upper_large_jsat_r[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]],x=s10_upper_large_jsat_s[i_time][s10_upper_large_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_upper_outer_large_integrated.append(temp)


					# lower
					s4_lower_inner_test = np.sum(np.logical_not(s4_lower_good_probes)[np.abs(s4_lower_z-inner_strike_point[1])<closeness_limit_to_dead_channels])==0
					s10_lower_inner_test = np.sum(np.logical_not(s10_lower_good_probes)[np.abs(s10_lower_r-inner_strike_point[0])<closeness_limit_to_dead_channels])==0
					s4_lower_outer_test = np.sum(np.logical_not(s4_lower_good_probes)[np.abs(s4_lower_z-outer_strike_point[1])<closeness_limit_to_dead_channels])==0
					s10_lower_outer_test = np.sum(np.logical_not(s10_lower_good_probes)[np.abs(s10_lower_r-outer_strike_point[0])<closeness_limit_to_dead_channels])==0

					temp = [np.nan]
					if s4_lower_inner_test:
						temp = np.concatenate([[0],temp,s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_lower_inner_small_max.append(np.nanmax(temp))
					temp = 0
					if s4_lower_inner_test:
						temp += np.trapz(s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]],x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_lower_inner_small_integrated.append(-temp)
					temp = [np.nan]
					if s4_lower_outer_test:
						temp = np.concatenate([[0],temp,s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_lower_outer_small_max.append(np.nanmax(temp))
					temp = 0
					if s4_lower_outer_test:
						temp += np.trapz(s4_lower_jsat[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s4_lower_jsat_r[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]],x=s4_lower_jsat_s[i_time][s4_lower_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_lower_outer_small_integrated.append(-temp)
					temp = [np.nan]
					if s10_lower_inner_test:
						temp = np.concatenate([[0],temp,s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]])
					jsat_lower_inner_mid_max.append(np.nanmax(temp))
					temp = 0
					if s10_lower_inner_test:
						temp += np.trapz(s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]],x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]<mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_lower_inner_mid_integrated.append(-temp)
					temp = [np.nan]
					if s10_lower_outer_test:
						temp = np.concatenate([[0],temp,s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]])
					jsat_lower_outer_mid_max.append(np.nanmax(temp))
					temp = 0
					if s10_lower_outer_test:
						temp += np.trapz(s10_lower_jsat[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]]*2*np.pi*s10_lower_jsat_r[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]],x=s10_lower_jsat_s[i_time][s10_lower_jsat_r[i_time]>mid_strike_point[0]])
					else:
						temp = np.nan
					jsat_lower_outer_mid_integrated.append(-temp)

					jsat_lower_inner_large_max.append(0)
					jsat_lower_outer_large_max.append(0)
					jsat_lower_inner_large_integrated.append(0)
					jsat_lower_outer_large_integrated.append(0)

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
		except:
			jsat_read = False

		print('marker LP done')

		shot_list = get_data(path+'shot_list2.ods')
		temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
		scenario = ''
		experiment = ''
		useful = ''
		Preshot = ''
		Postshot = ''
		SC = ''
		SL = ''
		for i in range(1,len(shot_list['Sheet1'])):
			if shot_list['Sheet1'][i][temp1] == efit_reconstruction.shotnumber:
				scenario = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Scenario').argmax()]
				experiment = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax()]
				useful = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Useful').argmax()]
				Preshot = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Preshot').argmax()]
				Postshot = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='Postshot').argmax()]
				SC = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='SC').argmax()]
				SL = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='SL').argmax()]
				break
		full_saved_file_dict_FAST['multi_instrument']['scenario'] = scenario
		full_saved_file_dict_FAST['multi_instrument']['experiment'] = experiment
		full_saved_file_dict_FAST['multi_instrument']['useful'] = useful
		full_saved_file_dict_FAST['multi_instrument']['Preshot'] = Preshot
		full_saved_file_dict_FAST['multi_instrument']['Postshot'] = Postshot
		full_saved_file_dict_FAST['multi_instrument']['SC'] = SC
		full_saved_file_dict_FAST['multi_instrument']['SL'] = SL

		from mastu_exhaust_analysis.read_efit import read_epm
		fdir = coleval.uda_transfer(efit_reconstruction.shotnumber,'epm')
		efit_data = read_epm(fdir,calc_bfield=True)
		os.remove(fdir)
		dWdt = calc_w_dot(efit_data = efit_data)
		pohm = calc_pohm(efit_data = efit_data)
		dWdt['data'][~np.isfinite(dWdt['data'])] = 0.0
		pohm['pohm'][~np.isfinite(pohm['pohm'])] = 0.0
		smooth_dt = 0.015
		window_size = np.int(smooth_dt / np.median(np.gradient(efit_data['t'])))
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
		full_saved_file_dict_FAST['multi_instrument']['SW_BEAMPOWER'] = SW_BEAMPOWER
		full_saved_file_dict_FAST['multi_instrument']['SS_BEAMPOWER'] = SS_BEAMPOWER
		full_saved_file_dict_FAST['multi_instrument']['stored_energy'] = stored_energy
		full_saved_file_dict_FAST['multi_instrument']['power_balance_pohm'] = output_pohm
		full_saved_file_dict_FAST['multi_instrument']['power_balance_pdw_dt'] = dWdt
		full_saved_file_dict_FAST['multi_instrument']['P_heat'] = P_heat
		full_saved_file_dict_FAST['multi_instrument']['P_loss'] = P_loss
		full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time'] = energy_confinement_time
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
		data = client.get('/EPM/OUTPUT/GLOBALPARAMETERS/BPHIRMAG',laser_to_analyse[-9:-4])
		toroidal_field = -np.interp(efit_data['t'],data.time.data,data.data,right=0.,left=0.)	# T
		toroidal_field = scipy.signal.savgol_filter(toroidal_field, window_size, poly_order)
		toroidal_field = np.interp(time_full_binned_crop,efit_data['t'],toroidal_field,right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['toroidal_field'] = toroidal_field

		vert_displacement = np.interp(time_full_binned_crop,efit_reconstruction.time,efit_reconstruction.mag_axis_z,right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['vert_displacement'] = vert_displacement
		radius_outer_separatrix = np.interp(time_full_binned_crop,efit_data['t'],efit_data['rmidplaneOut'],right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['radius_outer_separatrix'] = radius_outer_separatrix
		radius_inner_separatrix = np.interp(time_full_binned_crop,efit_data['t'],efit_data['rmidplaneIn'],right=0.,left=0.)
		full_saved_file_dict_FAST['multi_instrument']['radius_inner_separatrix'] = radius_inner_separatrix

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

		psiN_peak_inner_all = []
		from scipy.interpolate import interp1d,interp2d
		for i_t in range(len(time_full_binned_crop)):
			i_efit_time = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
			if inner_emissivity_peak_all[i_t][1]<efit_reconstruction.lower_xpoint_z[i_efit_time]:
				psiN_peak_inner_all.append(np.nan)
				continue
			gna = efit_data['psiN'][i_efit_time]
			psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,gna)
			psiN_peak_inner_all.append( psi_interpolator(inner_emissivity_peak_all[i_t][0],inner_emissivity_peak_all[i_t][1]))
		psiN_peak_inner_all = np.array(psiN_peak_inner_all)
		full_saved_file_dict_FAST['multi_instrument']['psiN_peak_inner_all'] = psiN_peak_inner_all

		fig, ax = plt.subplots( 2,1,figsize=(12, 12), squeeze=False,sharex=True)
		if pass_number ==0:
			fig.suptitle('shot '+str(efit_reconstruction.shotnumber)+', '+scenario+' , '+experiment+'\nfirst pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		elif pass_number ==1:
			fig.suptitle('shot '+str(efit_reconstruction.shotnumber)+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
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
					power_balance = calc_psol(shot=efit_reconstruction.shotnumber,smooth_dt =0.015)	# in W
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
		plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables.png')
		plt.close()

		# plot of absolute quantities
		fig, ax = plt.subplots( 9,1,figsize=(12, 35), squeeze=False,sharex=False)
		if pass_number ==0:
			fig.suptitle('shot '+str(efit_reconstruction.shotnumber)+', '+scenario+' , '+experiment+'\nfirst pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		elif pass_number ==1:
			fig.suptitle('shot '+str(efit_reconstruction.shotnumber)+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all,'r-',label='inner_peak')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all,'r:',label='inner_peak only leg')
		ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all,'r--',label='inner_baricentre')
		ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all,'b-',label='outer_peak only leg')
		ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all,'b--',label='outer_baricentre only leg')
		ax[0,0].set_ylim(bottom=-0.1,top=4)
		ax[0,0].grid()
		ax[0,0].set_ylabel('Lpoloidal/Lpol x-pt [au]')
		ax0 = ax[0,0].twinx()  # instantiate a second axes that shares the same x-axis
		# ax2.spines["right"].set_position(("axes", 1.1125))
		ax0.spines["right"].set_visible(True)
		a0a, = ax0.plot(time_full_binned_crop,psiN_peak_inner_all,label='psiN_peak_inner_all',color='g')
		ax0.axhline(y=1,color='g',linestyle='--')
		ax0.set_ylabel('psi N [au]', color='g')  # we already handled the x-label with ax1
		# ax2.tick_params(axis='y', labelcolor=a2a.get_color())
		ax0.set_ylim(bottom=max(0.9,np.nanmin(psiN_peak_inner_all)),top=min(1.2,np.nanmax(psiN_peak_inner_all)))
		handles, labels = ax[0,0].get_legend_handles_labels()
		handles.append(a0a)
		labels.append(a0a.get_label())
		ax0.legend(handles=handles, labels=labels, loc='best', fontsize='xx-small')
		# ax[0,0].legend(loc='best', fontsize='xx-small')
		if  not density_data_missing:
			ax[1,0].plot(time_full_binned_crop,core_density,label='core_density',color=color[0])
			ax[1,0].plot(time_full_binned_crop,ne_bar,label='ne_bar',color=color[-1])
			ax[1,0].plot(time_full_binned_crop,greenwald_density,'--',label='greenwald_density',color=color[-2])

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
		ax[2,0].plot(time_full_binned_crop,dr_sep_in,'-',label='dr_sep_in',color=color[14])
		ax[2,0].plot(time_full_binned_crop,dr_sep_out,'-',label='dr_sep_out',color=color[13])
		ax[2,0].plot(time_full_binned_crop,radius_inner_separatrix-0.2608,'-',label='inner gap',color=color[12])
		ax[2,0].grid()
		ax[2,0].set_ylabel('dr sep [m]')

		ax2 = ax[2,0].twinx()  # instantiate a second axes that shares the same x-axis
		# ax2.spines["right"].set_position(("axes", 1.1125))
		ax2.spines["right"].set_visible(True)
		a2a, = ax2.plot(time_full_binned_crop,vert_displacement,label='vertical displacement',color='r')
		a2b, = ax2.plot(time_full_binned_crop,radius_inner_separatrix-0.2608,'-',label='inner gap-Rf',color=color[12])
		ax2.set_ylabel('vert disp [m]', color='r')  # we already handled the x-label with ax1
		# ax2.tick_params(axis='y', labelcolor=a2a.get_color())
		# ax2.set_ylim(bottom=0)
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

		try:
			power_balance = calc_psol(shot=efit_reconstruction.shotnumber,smooth_dt =0.015)	# in W
			input_power = np.interp(time_full_binned_crop,power_balance['t'],power_balance['pohm'] + SW_BEAMPOWER + SS_BEAMPOWER-power_balance['pdw_dt'])

			ax[4,0].plot(power_balance['t'],1e-6*(power_balance['pohm'] + SW_BEAMPOWER + SS_BEAMPOWER-power_balance['pdw_dt']),label='input power (ohm+beams-dW/dt)',color=color[5])
			ax[4,0].plot(power_balance['t'],1e-6*(power_balance['pohm']-power_balance['pdw_dt']),'--',label='ohmic input power (ohm-dW/dt)',color=color[5])
			ax[4,0].plot(power_balance['t'],1e-6*power_balance['prad_core'],label='prad_core',color=color[7])
			temp = inverted_data[:,:,inversion_Z<0]
			temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
			# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
			ax[4,0].plot(time_full_binned_crop,1e-6*2*temp,label='total IRVB power (Z<0 x2)',color=color[2])
			ax[4,0].plot(power_balance['t'],1e-6*SW_BEAMPOWER,label='SW beam x '+str(sw_absorption),color=color[8])
			ax[4,0].plot(power_balance['t'],1e-6*SS_BEAMPOWER,label='SS beam x '+str(ss_absorption),color=color[9])
			temp = power_balance['pohm'] + SW_BEAMPOWER + SS_BEAMPOWER-power_balance['pdw_dt']
			ax[4,0].set_ylim(bottom=0,top=1e-6*1.2*np.nanmax(median_filter(temp[np.isfinite(temp)],size=21)[np.logical_and(power_balance['t'][np.isfinite(temp)]>0,power_balance['t'][np.isfinite(temp)]<time_full_binned_crop[-5])]))
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
		ax[4,0].legend(loc='best', fontsize='xx-small')
		ax5 = ax[5,0].twinx()  # instantiate a second axes that shares the same x-axis
		if jsat_read:
			a5a, = ax5.plot(jsat_time,jsat_lower_inner_small_max,'--',label='lower_inner_small',color=color[0])
			ax5.plot(jsat_time,jsat_lower_inner_small_max,'+',color=color[0])
			ax[5,0].plot(jsat_time,jsat_lower_outer_small_max,'--',label='lower_outer_small',color=color[1])
			ax[5,0].plot(jsat_time,jsat_lower_outer_small_max,'+',color=color[1])
			a5b, = ax5.plot(jsat_time,jsat_lower_inner_mid_max,'-',label='lower_inner_mid',color=color[0])
			ax5.plot(jsat_time,jsat_lower_inner_mid_max,'+',color=color[0])
			ax[5,0].plot(jsat_time,jsat_lower_outer_mid_max,'-',label='lower_outer_mid',color=color[1])
			ax[5,0].plot(jsat_time,jsat_lower_outer_mid_max,'+',color=color[1])
			a5c, = ax5.plot(jsat_time,jsat_lower_inner_large_max,':',label='lower_inner_large',color=color[0])
			ax5.plot(jsat_time,jsat_lower_inner_large_max,'+',color=color[0])
			ax[5,0].plot(jsat_time,jsat_lower_outer_large_max,':',label='lower_outer_large',color=color[1])
			ax[5,0].plot(jsat_time,jsat_lower_outer_large_max,'+',color=color[1])
			a5d, = ax5.plot(jsat_time,jsat_upper_inner_small_max,'--',label='upper_inner_small',color=color[2])
			ax5.plot(jsat_time,jsat_upper_inner_small_max,'+',color=color[2])
			ax[5,0].plot(jsat_time,jsat_upper_outer_small_max,'--',label='upper_outer_small',color=color[3])
			ax[5,0].plot(jsat_time,jsat_upper_outer_small_max,'+',color=color[3])
			a5e, = ax5.plot(jsat_time,jsat_upper_inner_mid_max,'-',label='upper_inner_mid',color=color[2])
			ax5.plot(jsat_time,jsat_upper_inner_mid_max,'+',color=color[2])
			ax[5,0].plot(jsat_time,jsat_upper_outer_mid_max,'-',label='upper_outer_mid',color=color[3])
			ax[5,0].plot(jsat_time,jsat_upper_outer_mid_max,'+',color=color[3])
			a5f, = ax5.plot(jsat_time,jsat_upper_inner_large_max,':',label='upper_inner_large',color=color[2])
			ax5.plot(jsat_time,jsat_upper_inner_large_max,'+',color=color[2])
			ax[5,0].plot(jsat_time,jsat_upper_outer_large_max,':',label='upper_outer_large',color=color[3])
			ax[5,0].plot(jsat_time,jsat_upper_outer_large_max,'+',color=color[3])
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
		ax[5,0].set_ylabel('jsat max outer [A/m2]')
		ax5.spines["right"].set_visible(True)
		ax5.set_ylabel('jsat max inner [A/m2]', color='r')  # we already handled the x-label with ax1
		# ax5.tick_params(axis='y', labelcolor=a5a.get_color())
		# ax5.set_ylim(bottom=0)

		ax6 = ax[6,0].twinx()  # instantiate a second axes that shares the same x-axis
		if jsat_read:
			a6a, = ax6.plot(jsat_time,jsat_lower_inner_small_integrated,'--',label='lower_inner_small',color=color[0])
			ax6.plot(jsat_time,jsat_lower_inner_small_integrated,'+',color=color[0])
			ax[6,0].plot(jsat_time,jsat_lower_outer_small_integrated,'--',label='lower_outer_small',color=color[1])
			ax[6,0].plot(jsat_time,jsat_lower_outer_small_integrated,'+',color=color[1])
			a6b, = ax6.plot(jsat_time,jsat_lower_inner_mid_integrated,'-',label='lower_inner_mid',color=color[0])
			ax6.plot(jsat_time,jsat_lower_inner_mid_integrated,'+',color=color[0])
			ax[6,0].plot(jsat_time,jsat_lower_outer_mid_integrated,'-',label='lower_outer_mid',color=color[1])
			ax[6,0].plot(jsat_time,jsat_lower_outer_mid_integrated,'+',color=color[1])
			a6c, = ax6.plot(jsat_time,jsat_lower_inner_large_integrated,':',label='lower_inner_large',color=color[0])
			ax6.plot(jsat_time,jsat_lower_inner_large_integrated,'+',color=color[0])
			ax[6,0].plot(jsat_time,jsat_lower_outer_large_integrated,':',label='lower_outer_large',color=color[1])
			ax[6,0].plot(jsat_time,jsat_lower_outer_large_integrated,'+',color=color[1])
			a6d, = ax6.plot(jsat_time,jsat_upper_inner_small_integrated,'--',label='upper_inner_small',color=color[2])
			ax6.plot(jsat_time,jsat_upper_inner_small_integrated,'+',color=color[2])
			ax[6,0].plot(jsat_time,jsat_upper_outer_small_integrated,'--',label='upper_outer_small',color=color[3])
			ax[6,0].plot(jsat_time,jsat_upper_outer_small_integrated,'+',color=color[3])
			a6e, = ax6.plot(jsat_time,jsat_upper_inner_mid_integrated,'-',label='upper_inner_mid',color=color[2])
			ax6.plot(jsat_time,jsat_upper_inner_mid_integrated,'+',color=color[2])
			ax[6,0].plot(jsat_time,jsat_upper_outer_mid_integrated,'-',label='upper_outer_mid',color=color[3])
			ax[6,0].plot(jsat_time,jsat_upper_outer_mid_integrated,'+',color=color[3])
			a6f, = ax6.plot(jsat_time,jsat_upper_inner_large_integrated,':',label='upper_inner_large',color=color[2])
			ax6.plot(jsat_time,jsat_upper_inner_large_integrated,'+',color=color[2])
			ax[6,0].plot(jsat_time,jsat_upper_outer_large_integrated,':',label='upper_outer_large',color=color[3])
			ax[6,0].plot(jsat_time,jsat_upper_outer_large_integrated,'+',color=color[3])
		ax[6,0].grid()
		ax[6,0].set_ylabel('integrated current outer [A]')
		ax6.spines["right"].set_visible(True)
		ax6.set_ylabel('integrated current inner [A]', color='r')  # we already handled the x-label with ax1
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

		ax[7,0].plot(time_full_binned_crop,energy_confinement_time,label='confinement time',color=color[0])
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
		a7a, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_98y2,'--',color=color[3],alpha=0.3)
		a7b, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_97P,'--',color=color[2],alpha=0.3)
		a7c, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_HST,'--',color=color[1])
		a7d, = ax7.plot(time_full_binned_crop,energy_confinement_time/energy_confinement_time_LST,'--',color=color[5])
		ax7.set_ylabel('--=relative time [au]', color='r')  # we already handled the x-label with ax1
		# ax7.tick_params(axis='y', labelcolor=a7a.get_color())
		ax7.set_ylim(bottom=0)
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

		ax[8,0].set_xlabel('time [s]')

		ax[8,0].plot(Dalpha_time,Dalpha)
		ax[8,0].set_ylim(top=median_filter(Dalpha,size=2000).max())
		ax[8,0].set_ylabel('Dalpha [V]')

		ax[0,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[1,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[2,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[3,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[4,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[5,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[6,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[7,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
		ax[8,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())

		# plt.subplots_adjust(wspace=0, hspace=0)
		# plt.pause(0.01)
		plt.savefig('/home/ffederic/work/irvb/MAST-U/FAST_results/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute.png')
		plt.close()

		if pass_number ==0:
			full_saved_file_dict_FAST['first_pass']['inverted_dict'] = inverted_dict
		elif pass_number ==1:
			full_saved_file_dict_FAST['second_pass']['inverted_dict'] = inverted_dict
		np.savez_compressed(laser_to_analyse[:-4]+'_FAST',**full_saved_file_dict_FAST)
		print('DONE '+laser_to_analyse)

	except Exception as e:
		print('FAILED ' + laser_to_analyse)
		logging.exception('with error: ' + str(e))
	return full_saved_file_dict_FAST

full_saved_file_dict_FAST = temp_function(full_saved_file_dict_FAST)

##########