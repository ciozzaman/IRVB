# Created 13/01/2020
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

# added to reat the .ptw
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
import pyradi.ryptw as ryptw

# just to import _MASTU_CORE_GRID_POLYGON
calculate_tangency_angle_for_poloidal_section=coleval.calculate_tangency_angle_for_poloidal_section
exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())

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

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'teal', 'olive','blueviolet','tan','skyblue','brown','hotpink']
markers = ['o','+','*','v','^','<','>']
path = '/home/ffederic/work/irvb/MAST-U/'
shot_list = get_data(path+'shot_list2.ods')


if False:	# plot for the papers
	to_do = []
	for i in range(1,len(shot_list['Sheet1'])):
		try:
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ][-2:] in ['OH']:# ['SS','SW','OH']:#,'2B']:
			# 	continue
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ][-6:-3] in ['-CD']:#['SXD','-SF']
			# 	continue
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ] in ['DN-600-CD-OH','DN-450-CD-OH','DN-750-CD-OH']:# ['DN-750-CD-2B']:#['SXD','-SF']
			# 	continue
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ] in ['DN-600-SXD-OH','DN-450-SXD-OH','DN-750-SXD-OH']:#['DN-750-CD-2B']:#['SXD','-SF']
			# 	continue
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ] in ['DN-450-CD-1BSS','DN-450-CD-1BSS','DN-600-CD-1BSS','DN-600-CD-1BSS','DN-600-CD-1BSW','DN-600-CD-1BSW','DN-600-CD-2B','DN-600-CD-2B','DN-750-CD-1BSS','DN-750-CD-1BSS','DN-750-CD-1BSW','DN-750-CD-1BSW','DN-750-CD-2B','DN-750-CD-2B']:
			# 	continue
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ] in ['DN-600-SXD-OH']:#,'DN-450-SXD-OH','DN-750-SXD-OH']:#['DN-750-CD-2B']:#['SXD','-SF']
			# 	continue
			# try:
			# 	if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='fuelling LFSD_BOT_L0102').argmax() ] == 'X':
			# 		continue
			# except:
			# 	pass
			# try:
			# 	if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='fuelling LFSD_TOP_U0506').argmax() ] == 'X':
			# 		continue
			# except:
			# 	pass
			# if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Useful').argmax() ] == 'No':
			# 	continue
			# if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='IRVB Useful').argmax() ] == 'No':
			# 	continue
			# if str(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='date').argmax() ].date()) == '2021-10-27':
			# 	continue
			# if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] != 'MU01-EXH-01':
			# 	continue
			if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] == 'MU01-EXH-06':	# too high drsep
				continue
			if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] == 'MU01-EXH-13':	# almost super-x on outer leg
				continue
			# if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] == 'MU01-MHD-01':	# not sure what it is done here
			# 	continue
			if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] == 'MU01-MHD-02':	# not sure what it is done here
				continue
			# if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45473,45470,45302]:	# gas imput sometimes at 0 and with ramp up
			# 	continue
			# if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45468,45469,45293,45466,45327,45326,45325,45324,45323,45322,45320]:	# always some imput gas wayform
			# 	continue
			# if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45303]:	# always some imput gas wayform, but weirder
			# 	continue
			if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45063,45302,45320]:	# 45063: strike point too high to be seen, 45302: central column limited, 45320: central column limited
				continue
			if not int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45468,45469,45470,45473]:	# CD plot for thesis
				continue
			# if not int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45463,45464,45462,45461,45459,45443,45444,45446]:	# SXD plot for thesis
			# 	continue
			# if not int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45243,45244,45245,45241,45143,45080,45071,45060,45059,45058,45057,45056,45046,45047,45048]:	# always some imput gas wayform, but weirder
			# 	continue
			if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) < 45200:
				continue
			# if not int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45401,45399]:
			# 	continue
			to_do.append('IRVB-MASTU_shot-'+str(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ])+'.ptw')
		except:
			pass



	fig, ax = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
	fig1, ax1 = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
	fig1a, ax1a = plt.subplots( 2,2,figsize=(16, 10), squeeze=False,sharex=True)
	fig2, ax2 = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
	fig3, ax3 = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
	fig4, ax4 = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
	shot_list = get_data(path+'shot_list2.ods')
	n_shot_added = 0
	for name in np.flip(to_do,axis=0):
		try:
			temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
			for i in range(1,len(shot_list['Sheet1'])):
				if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
					date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
					break
			i_day,day = 0,str(date.date())
			laser_to_analyse=path+day+'/'+name

			full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
			full_saved_file_dict_FAST.allow_pickle=True
			full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
			full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
			try:
				# gna=gne
				full_saved_file_dict_FAST['second_pass'] = full_saved_file_dict_FAST['second_pass'].all()
				inverted_dict = full_saved_file_dict_FAST['second_pass']['inverted_dict']
				print(str(laser_to_analyse[-9:-4])+' second pass')
			except:
				full_saved_file_dict_FAST['first_pass'] = full_saved_file_dict_FAST['first_pass'].all()
				inverted_dict = full_saved_file_dict_FAST['first_pass']['inverted_dict']
				print(str(laser_to_analyse[-9:-4])+' first pass')
			grid_resolution = 2	# cm
			time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
			inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
			inverted_data_sigma = inverted_dict[str(grid_resolution)]['inverted_data_sigma']
			binning_type = inverted_dict[str(grid_resolution)]['binning_type']
			# nu_cowley = np.array(full_saved_file_dict_FAST['multi_instrument']['nu_cowley'])*1e-19
			# nu_cowley = np.array(full_saved_file_dict_FAST['multi_instrument']['nu_labombard'])*1e-19
			nu_cowley = np.array(full_saved_file_dict_FAST['multi_instrument']['nu_stangeby'])*1e-19

			greenwald_density = full_saved_file_dict_FAST['multi_instrument']['greenwald_density']
			ne_bar = full_saved_file_dict_FAST['multi_instrument']['ne_bar']

			if 'CD' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
				t_min = 0.30
			elif 'SF' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
				t_min = 0.4
			elif 'SXD' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
				t_min = 0.5
			t_end = -4

			# valid only for the 4 CD shots in the thesis (45473, 45470, 45469, 45468)
			try:
				pair = []
				with open(os.path.join('/home/ffederic/work/analysis_scripts/scripts/from_DavidMoulton','nesep_'+name[-9:-4])) as csv_file:
					csv_reader = csv.reader(csv_file, delimiter=',')
					for row in csv_reader:
						pair.append(row)
				pair = np.array(pair).astype(float)
				time_nesep = pair[:,0]
				nesep = pair[:,1]
				nsep_interpolator = interp1d(time_nesep,nesep,bounds_error=False, fill_value='extrapolate')
			except:
				pass




			# if '2B' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
			# 	data = client.get('/XNB/SS/BEAMPOWER',laser_to_analyse[-9:-4])
			# 	# SS_BEAMPOWER = np.array([data.data[np.abs(data.time.data-time).argmin()] for time in time_full_binned_crop])
			# 	SS_BEAMPOWER = np.interp(time_full_binned_crop,data.time.data,data.data,right=0.,left=0.)
			# 	data = client.get('/XNB/SW/BEAMPOWER',laser_to_analyse[-9:-4])
			# 	# SW_BEAMPOWER = np.array([data.data[np.abs(data.time.data-time).argmin()] for time in time_full_binned_crop])
			# 	SW_BEAMPOWER = np.interp(time_full_binned_crop,data.time.data,data.data,right=0.,left=0.)
			# 	t_min = max(t_min,time_full_binned_crop[SS_BEAMPOWER+SW_BEAMPOWER>2.5].min()+0.07)
			# 	t_end = min(t_end,np.arange(len(time_full_binned_crop))[SS_BEAMPOWER+SW_BEAMPOWER>2.5].max()-len(time_full_binned_crop))
			# elif '1BSS' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
			# 	data = client.get('/XNB/SS/BEAMPOWER',laser_to_analyse[-9:-4])
			# 	# SS_BEAMPOWER = np.array([data.data[np.abs(data.time.data-time).argmin()] for time in time_full_binned_crop])
			# 	SS_BEAMPOWER = np.interp(time_full_binned_crop,data.time.data,data.data,right=0.,left=0.)
			# 	SW_BEAMPOWER = 0
			# 	t_min = max(t_min,time_full_binned_crop[SS_BEAMPOWER>2.5].min()+0.07)
			# 	t_end = min(t_end,np.arange(len(time_full_binned_crop))[SS_BEAMPOWER>1.25].max()-len(time_full_binned_crop))
			# elif '1BSW' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
			# 	data = client.get('/XNB/SW/BEAMPOWER',laser_to_analyse[-9:-4])
			# 	# SW_BEAMPOWER = np.array([data.data[np.abs(data.time.data-time).argmin()] for time in time_full_binned_crop])
			# 	SW_BEAMPOWER = np.interp(time_full_binned_crop,data.time.data,data.data,right=0.,left=0.)
			# 	SS_BEAMPOWER = 0
			# 	t_min = max(t_min,time_full_binned_crop[SW_BEAMPOWER>2.5].min()+0.07)
			# 	t_end = min(t_end,np.arange(len(time_full_binned_crop))[SW_BEAMPOWER>1.25].max()-len(time_full_binned_crop))

			if np.sum((time_full_binned_crop>t_min)[:t_end])==0:
				fdsaf =dfdasf

			outer_L_poloidal_baricentre_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_baricentre_all']
			outer_L_poloidal_peak_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_all']
			outer_L_poloidal_peak_only_leg_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_only_leg_all']
			outer_L_poloidal_x_point_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_x_point_all']
			inner_L_poloidal_baricentre_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_baricentre_all']
			inner_L_poloidal_peak_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_all']
			inner_L_poloidal_peak_only_leg_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_only_leg_all']
			inner_L_poloidal_x_point_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_x_point_all']
			energy_confinement_time = full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time']
			energy_confinement_time_98y2 = full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_98y2']
			energy_confinement_time_97P = full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_97P']
			energy_confinement_time_HST = full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_HST']
			energy_confinement_time_LST = full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time_LST']
			psiN_peak_inner_all = full_saved_file_dict_FAST['multi_instrument']['psiN_peak_inner_all']
			core_density = full_saved_file_dict_FAST['multi_instrument']['line_integrated_density']
			try:
				jsat_lower_outer_mid_integrated = full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_integrated']
				jsat_upper_outer_mid_integrated = full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_integrated']
				jsat_lower_outer_mid_integrated_sigma = full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_integrated_sigma']
				jsat_upper_outer_mid_integrated_sigma = full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_integrated_sigma']
				ax[1,0].plot((core_density)[time_full_binned_crop>t_min][:t_end],(jsat_lower_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
				ax[1,0].plot((core_density)[time_full_binned_crop>t_min][:t_end],(jsat_upper_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end],markers[(n_shot_added+len(color))//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
				ax1[0,0].errorbar((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(jsat_lower_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end],yerr=(jsat_lower_outer_mid_integrated_sigma)[time_full_binned_crop>t_min][:t_end],fmt='+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5,capsize=5)
				ax1[1,0].errorbar((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(jsat_upper_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end],yerr=(jsat_upper_outer_mid_integrated_sigma)[time_full_binned_crop>t_min][:t_end],fmt='+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5,capsize=5)
				ax1a[1,0].errorbar((nu_cowley)[time_full_binned_crop>t_min][:t_end],(jsat_lower_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end]/1.6e-19*1e-22,yerr=(jsat_lower_outer_mid_integrated_sigma)[time_full_binned_crop>t_min][:t_end]/1.6e-19*1e-22,fmt='+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5,capsize=5)
				ax1a[0,0].errorbar((nu_cowley)[time_full_binned_crop>t_min][:t_end],(jsat_upper_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end]/1.6e-19*1e-22,yerr=(jsat_upper_outer_mid_integrated_sigma)[time_full_binned_crop>t_min][:t_end]/1.6e-19*1e-22,fmt='+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5,capsize=5)
				ax4[0,0].errorbar(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(jsat_lower_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end],yerr=(jsat_lower_outer_mid_integrated_sigma)[time_full_binned_crop>t_min][:t_end],fmt='+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5,capsize=5)
				ax4[1,0].errorbar(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(jsat_upper_outer_mid_integrated)[time_full_binned_crop>t_min][:t_end],yerr=(jsat_upper_outer_mid_integrated_sigma)[time_full_binned_crop>t_min][:t_end],fmt='+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5,capsize=5)
			except:
				pass

			try:
				time_start_MARFE = full_saved_file_dict_FAST['multi_instrument']['time_start_MARFE']
				time_active_MARFE = full_saved_file_dict_FAST['multi_instrument']['time_active_MARFE']
				if time_start_MARFE==None:
					time_start_MARFE = np.inf
				if time_active_MARFE==None:
					time_active_MARFE = np.inf
			except:
				time_start_MARFE = np.inf
				time_active_MARFE = np.inf

			true_outer_target_position = outer_L_poloidal_peak_only_leg_all
			if name == 'IRVB-MASTU_shot-45468.ptw':
				pass
			elif name == 'IRVB-MASTU_shot-45469.ptw':
				true_outer_target_position = outer_L_poloidal_peak_all
			elif name == 'IRVB-MASTU_shot-45470.ptw':
				true_outer_target_position[time_full_binned_crop>0.65] = outer_L_poloidal_peak_all[time_full_binned_crop>0.65]
			elif name == 'IRVB-MASTU_shot-45473.ptw':
				true_outer_target_position[time_full_binned_crop>0.68] = outer_L_poloidal_peak_all[time_full_binned_crop>0.68]



			# if np.nanmax(median_filter(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all,size=1)) >0.9:
			# 	inner_L_poloidal_peak_only_leg_all[np.nanargmax(median_filter(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all,size=1)):] = np.nan
			# if np.nanmax(median_filter(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all,size=1)) >0.9:
			# 	outer_L_poloidal_peak_all[np.nanargmax(median_filter(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all,size=1)):] = np.nan

			ax[0,0].plot(core_density[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax[0,1].plot(core_density[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax[0,2].plot(core_density[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax[1,1].plot(core_density[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax[1,2].plot(core_density[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)


			ax1[0,1].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax1[0,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			ax1[0,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			ax1[0,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			ax1[0,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			# ax1[0,1].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax1[0,2].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			# ax1[1,1].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax1[1,1].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(true_outer_target_position/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax1[1,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			ax1[1,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			ax1[1,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			ax1[1,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			ax1[1,2].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)

			ax1a[0,1].plot((nu_cowley)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			# ax1a[0,0].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			if time_start_MARFE>t_min:
				ax1a[0,1].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax1a[0,0].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			# ax1a[0,1].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			ax1a[1,1].plot((nu_cowley)[time_full_binned_crop>t_min][:t_end],(true_outer_target_position/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			# ax1a[1,0].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax1a[1,1].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax1a[1,0].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			# ax1a[1,1].axvline(x=(nu_cowley)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])

			ax2[0,0].plot((energy_confinement_time)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax2[0,1].plot((energy_confinement_time)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax2[0,2].plot((energy_confinement_time)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax2[1,1].plot((energy_confinement_time)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax2[1,2].plot((energy_confinement_time)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)

			ax3[0,0].plot((energy_confinement_time/energy_confinement_time_LST)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax3[0,1].plot((energy_confinement_time/energy_confinement_time_LST)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax3[0,2].plot((energy_confinement_time/energy_confinement_time_LST)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax3[1,1].plot((energy_confinement_time/energy_confinement_time_LST)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax3[1,2].plot((energy_confinement_time/energy_confinement_time_LST)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)

			# ax4[0,0].plot((energy_confinement_time/energy_confinement_time_LST)[time_full_binned_crop>t_min][:t_end],(psiN_peak_inner_all[:,0])[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax4[0,1].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			# ax4[0,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax4[0,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax4[0,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			# ax4[0,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			ax4[0,2].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			# ax4[1,1].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(true_outer_target_position/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax4[1,1].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],'o',fillstyle='none',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax4[1,1].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(outer_L_poloidal_peak_only_leg_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],'+',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax4[1,1].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(true_outer_target_position/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],'s',fillstyle='none',color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			# ax4[1,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax4[1,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_start_MARFE).argmin()],linestyle='--',color=color[n_shot_added%len(color)])
			# ax4[1,0].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			# ax4[1,1].axvline(x=(ne_bar/greenwald_density)[np.abs(time_full_binned_crop-time_active_MARFE).argmin()],linestyle='-',color=color[n_shot_added%len(color)])
			ax4[1,2].plot(nsep_interpolator(time_full_binned_crop[time_full_binned_crop>t_min][:t_end]),(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)


			print('included '+str(laser_to_analyse[-9:-4]))
			n_shot_added +=1
		except:
			print('failed '+str(laser_to_analyse[-9:-4]))
			pass
	# fig.suptitle('L poloidal peak or baricentre / L poloidal x-point\nDN-600-SXD-OH, shot number < 45366, Experiment Tags!=MU01-EXH-06 (large drsep)\nt>'+str(t_min*1e3)+'ms,t<t end-150ms')
	fig1.suptitle('L poloidal peak or baricentre / L poloidal x-point\nDN-450/600/750-CD-OH, shot number < 45366, Experiment Tags!=MU01-EXH-01 (large drsep)\nt>'+str(t_min*1e3)+'ms,t<t end-150ms')
	fig.suptitle('L poloidal peak or baricentre / L poloidal x-point\nDN-450/600/750-CD-OH, shot number < 45366, Experiment Tags!=MU01-EXH-01 (large drsep)\nt>'+str(t_min*1e3)+'ms,t<t end-150ms')
	ax[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	# ax[0,1].legend(loc='best', fontsize='xx-small',ncol=2)
	# ax[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[0,0].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	ax[0,1].set_ylabel('inner leg peak\nL poloidal/L poloidal x-point')
	ax[0,2].set_ylabel('inner separatrix baricentre\nL poloidal/L poloidal x-point')
	ax[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax[1,2].set_ylabel('outer leg baricentre\nL poloidal/L poloidal x-point')
	ax[1,1].set_xlabel('line averaged ne [#/m2]')
	ax[1,2].set_xlabel('line averaged ne [#/m2]')
	ax[0,0].grid()
	ax[0,1].grid()
	ax[0,2].grid()
	ax[1,0].grid()
	ax[1,1].grid()
	ax[1,2].grid()
	ax[0,0].axhline(y=1,linestyle='--',color='k')
	ax[0,1].axhline(y=1,linestyle='--',color='k')
	ax[0,2].axhline(y=1,linestyle='--',color='k')
	ax[1,1].axhline(y=1,linestyle='--',color='k')
	ax[1,2].axhline(y=1,linestyle='--',color='k')
	ax1[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	# ax1[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
	ax1[0,1].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	# ax1[0,1].set_ylabel('inner leg peak\nL poloidal/L poloidal x-point')
	ax1[0,2].set_ylabel('inner separatrix baricentre\nL poloidal/L poloidal x-point')
	ax1[0,0].set_ylabel('integradet lower outer jsat')
	ax1[1,0].set_ylabel('integradet upper outer jsat')
	ax1[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax1[1,2].set_ylabel('outer leg baricentre\nL poloidal/L poloidal x-point')
	ax1[0,0].grid()
	ax1[0,1].grid()
	ax1[0,2].grid()
	ax1[1,0].grid()
	ax1[1,1].grid()
	ax1[1,2].grid()
	ax1[0,1].axhline(y=1,linestyle='--',color='k')
	ax1[0,2].axhline(y=1,linestyle='--',color='k')
	ax1[1,1].axhline(y=1,linestyle='--',color='k')
	ax1[1,2].axhline(y=1,linestyle='--',color='k')
	ax1[1,1].set_xlabel('greenwald fraction [au]')
	ax1[1,2].set_xlabel('greenwald fraction [au]')
	# ax1[1,1].set_xlim(right=0.27)
	# ax1[1,1].set_xlim(right=0.27)

	ax1a[0,0].legend(loc='best', fontsize='small',ncol=2)
	ax1a[0,1].set_ylabel(r'${\hat{L}}_{peak}$ inner separatrix')
	ax1a[0,0].set_ylabel('upper outer target\n'+r'particle flux [$10^{22}\#/s$]')
	ax1a[1,0].set_ylabel('lower outer target\n'+r'particle flux [$10^{22}\#/s$]')
	ax1a[1,1].set_ylabel(r'${\hat{L}}_{peak}$ outer separatrix')
	ax1a[0,0].grid()
	ax1a[0,1].grid()
	ax1a[1,0].grid()
	ax1a[1,1].grid()
	ax1a[0,1].axhline(y=1,linestyle='--',color='k')
	ax1a[1,1].axhline(y=1,linestyle='--',color='k')
	ax1a[1,0].set_xlabel(r'upstream density [$10^{19}\#/m^3$]')
	ax1a[1,1].set_xlabel(r'upstream density [$10^{19}\#/m^3$]')

	ax2[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	# ax2[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
	ax2[0,0].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	ax2[0,1].set_ylabel('inner leg peak\nL poloidal/L poloidal x-point')
	ax2[0,2].set_ylabel('inner separatrix baricentre\nL poloidal/L poloidal x-point')
	ax2[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax2[1,2].set_ylabel('outer leg baricentre\nL poloidal/L poloidal x-point')
	ax2[0,0].grid()
	ax2[0,1].grid()
	ax2[0,2].grid()
	ax2[1,1].grid()
	ax2[1,2].grid()
	ax2[0,1].axhline(y=1,linestyle='--',color='k')
	ax2[0,2].axhline(y=1,linestyle='--',color='k')
	ax2[1,1].axhline(y=1,linestyle='--',color='k')
	ax2[1,2].axhline(y=1,linestyle='--',color='k')
	ax2[1,1].set_xlabel('energy confinement time [s]')
	ax2[1,2].set_xlabel('energy confinement time [s]')

	ax3[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	# ax3[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
	ax3[0,0].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	ax3[0,1].set_ylabel('inner leg peak\nL poloidal/L poloidal x-point')
	ax3[0,2].set_ylabel('inner separatrix baricentre\nL poloidal/L poloidal x-point')
	ax3[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax3[1,2].set_ylabel('outer leg baricentre\nL poloidal/L poloidal x-point')
	ax3[0,0].grid()
	ax3[0,1].grid()
	ax3[0,2].grid()
	ax3[1,1].grid()
	ax3[1,2].grid()
	ax3[0,0].axhline(y=1,linestyle='--',color='k')
	ax3[0,2].axhline(y=1,linestyle='--',color='k')
	ax3[1,1].axhline(y=1,linestyle='--',color='k')
	ax3[1,2].axhline(y=1,linestyle='--',color='k')
	ax3[1,1].set_xlabel('energy confinement/lmode ref [au]')
	ax3[1,2].set_xlabel('energy confinement/lmode ref [au]')

	ax4[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax4[0,1].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	ax4[0,0].set_ylabel('integradet lower outer jsat')
	ax4[1,0].set_ylabel('integradet upper outer jsat')
	ax4[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax4[0,0].grid()
	ax4[0,1].grid()
	ax4[1,0].grid()
	ax4[1,1].grid()
	ax4[0,1].axhline(y=1,linestyle='--',color='k')
	ax4[1,1].axhline(y=1,linestyle='--',color='k')
	ax4[1,1].set_xlabel('upstream density [#/m3]')

	fig1a.savefig('/home/ffederic/work/irvb/0__outputs/CD_OH_MU01_compare.png')
	plt.close('all')


	plt.pause(0.01)
else:
	pass


















	# name = 'IRVB-MASTU_shot-45473.ptw'
	# name = 'IRVB-MASTU_shot-44879.ptw'
	name = 'IRVB-MASTU_shot-45401.ptw'
	# name = 'IRVB-MASTU_shot-45371.ptw'
	temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
	for i in range(1,len(shot_list['Sheet1'])):
		if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
			date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
			break
	i_day,day = 0,str(date.date())
	laser_to_analyse=path+day+'/'+name

	full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
	full_saved_file_dict_FAST.allow_pickle=True
	full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
	full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
	pass_number = 1
	if pass_number==0:
		full_saved_file_dict_FAST['first_pass'] = full_saved_file_dict_FAST['first_pass'].all()
		inverted_dict = full_saved_file_dict_FAST['first_pass']['inverted_dict']
	else:
		full_saved_file_dict_FAST['second_pass'] = full_saved_file_dict_FAST['second_pass'].all()
		inverted_dict = full_saved_file_dict_FAST['second_pass']['inverted_dict']
	grid_resolution = 2	# cm
	time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
	binning_type = inverted_dict[str(grid_resolution)]['binning_type']

	outer_L_poloidal_baricentre_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_baricentre_all']
	outer_L_poloidal_peak_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_all']
	outer_L_poloidal_peak_only_leg_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_only_leg_all']
	outer_L_poloidal_x_point_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_x_point_all']
	inner_L_poloidal_baricentre_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_baricentre_all']
	inner_L_poloidal_peak_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_all']
	inner_L_poloidal_peak_only_leg_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_only_leg_all']
	inner_L_poloidal_x_point_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_x_point_all']
	outer_local_L_poloidal_all = inverted_dict[str(grid_resolution)]['outer_local_L_poloidal_all']
	inner_local_L_poloidal_all = inverted_dict[str(grid_resolution)]['inner_local_L_poloidal_all']
	outer_local_mean_emis_all = inverted_dict[str(grid_resolution)]['outer_local_mean_emis_all']
	inner_local_mean_emis_all = inverted_dict[str(grid_resolution)]['inner_local_mean_emis_all']
	nu_cowley = np.array(full_saved_file_dict_FAST['multi_instrument']['nu_cowley'])

	inverted_data = inverted_dict[str(grid_resolution)]['inverted_data']
	inversion_R = inverted_dict[str(grid_resolution)]['geometry']['R']
	inversion_Z = inverted_dict[str(grid_resolution)]['geometry']['Z']
	outer_emissivity_peak_all = inverted_dict[str(grid_resolution)]['outer_emissivity_peak_all']
	outer_L_poloidal_peak_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_peak_all']
	inner_emissivity_peak_all = inverted_dict[str(grid_resolution)]['inner_emissivity_peak_all']
	inner_L_poloidal_peak_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_peak_all']
	greenwald_density = full_saved_file_dict_FAST['multi_instrument']['greenwald_density']
	ne_bar = full_saved_file_dict_FAST['multi_instrument']['ne_bar']

	jsat_time = full_saved_file_dict_FAST['multi_instrument']['jsat_time']
	jsat_lower_outer_mid_integrated = full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_integrated']
	jsat_upper_outer_mid_integrated = full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_integrated']
	jsat_lower_outer_mid_integrated_sigma = full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_integrated_sigma']
	jsat_upper_outer_mid_integrated_sigma = full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_integrated_sigma']
	jsat_lower_outer_mid_max = full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_max']
	jsat_upper_outer_mid_max = full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_max']
	scenario = full_saved_file_dict_FAST['multi_instrument']['scenario']
	experiment = full_saved_file_dict_FAST['multi_instrument']['experiment']

	SW_BEAMPOWER = full_saved_file_dict_FAST['multi_instrument']['SW_BEAMPOWER']
	SS_BEAMPOWER = full_saved_file_dict_FAST['multi_instrument']['SS_BEAMPOWER']
	stored_energy = full_saved_file_dict_FAST['multi_instrument']['stored_energy']
	output_pohm = full_saved_file_dict_FAST['multi_instrument']['power_balance_pohm']
	dWdt = full_saved_file_dict_FAST['multi_instrument']['power_balance_pdw_dt']
	P_heat = full_saved_file_dict_FAST['multi_instrument']['P_heat']
	P_loss = full_saved_file_dict_FAST['multi_instrument']['P_loss']
	energy_confinement_time = full_saved_file_dict_FAST['multi_instrument']['energy_confinement_time']
	power_balance = dict([])
	power_balance['t'] = full_saved_file_dict_FAST['multi_instrument']['power_balance_t']
	power_balance['prad_core'] = full_saved_file_dict_FAST['multi_instrument']['power_balance_prad_core']
	SS_BEAMPOWER = np.interp(power_balance['t'],time_full_binned_crop,SS_BEAMPOWER,right=0.,left=0.)
	SW_BEAMPOWER = np.interp(power_balance['t'],time_full_binned_crop,SW_BEAMPOWER,right=0.,left=0.)
	ss_absorption = 0.8
	sw_absorption = 0.4
	real_core_radiation_all = inverted_dict[str(grid_resolution)]['real_core_radiation_all']
	real_core_radiation_sigma_all = inverted_dict[str(grid_resolution)]['real_core_radiation_sigma_all']
	real_non_core_radiation_all = inverted_dict[str(grid_resolution)]['real_non_core_radiation_all']
	real_non_core_radiation_sigma_all = inverted_dict[str(grid_resolution)]['real_non_core_radiation_sigma_all']
	x_point_tot_rad_power_all = inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_all']
	x_point_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['x_point_tot_rad_power_sigma_all']
	outer_leg_reliable_power_all = inverted_dict[str(grid_resolution)]['outer_leg_reliable_power_all']
	outer_leg_reliable_power_sigma_all = inverted_dict[str(grid_resolution)]['outer_leg_reliable_power_sigma_all']
	inner_leg_reliable_power_all = inverted_dict[str(grid_resolution)]['inner_leg_reliable_power_all']
	inner_leg_reliable_power_sigma_all = inverted_dict[str(grid_resolution)]['inner_leg_reliable_power_sigma_all']
	sxd_tot_rad_power_all = inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_all']
	sxd_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['sxd_tot_rad_power_sigma_all']
	divertor_tot_rad_power_all = inverted_dict[str(grid_resolution)]['divertor_tot_rad_power_all']
	divertor_tot_rad_power_sigma_all = inverted_dict[str(grid_resolution)]['divertor_tot_rad_power_sigma_all']
	all_lower_volume_radiation_all = inverted_dict[str(grid_resolution)]['all_lower_volume_radiation_all']
	all_lower_volume_radiation_sigma_all = inverted_dict[str(grid_resolution)]['all_lower_volume_radiation_sigma_all']
	inner_SOL_leg_all = inverted_dict[str(grid_resolution)]['inner_SOL_leg_all']
	inner_SOL_leg_sigma_all = inverted_dict[str(grid_resolution)]['inner_SOL_leg_sigma_all']
	inner_SOL_all = inverted_dict[str(grid_resolution)]['inner_SOL_all']
	inner_SOL_sigma_all = inverted_dict[str(grid_resolution)]['inner_SOL_sigma_all']
	outer_SOL_leg_all = inverted_dict[str(grid_resolution)]['outer_SOL_leg_all']
	outer_SOL_leg_sigma_all = inverted_dict[str(grid_resolution)]['outer_SOL_leg_sigma_all']
	outer_SOL_all = inverted_dict[str(grid_resolution)]['outer_SOL_all']
	outer_SOL_sigma_all = inverted_dict[str(grid_resolution)]['outer_SOL_sigma_all']
	equivalent_res_bolo_view = inner_SOL_all+real_core_radiation_all+outer_SOL_all
	equivalent_res_bolo_view_sigma = (inner_SOL_sigma_all**2+real_core_radiation_sigma_all**2+outer_SOL_sigma_all**2)**0.5
	approx_divertor = inner_SOL_leg_all+outer_SOL_leg_all
	approx_divertor_sigma = inner_SOL_leg_sigma_all+outer_SOL_leg_sigma_all

	peak_emissivity_inner = []
	peak_emissivity_outer = []
	for i_time,time in enumerate(time_full_binned_crop):
		peak_emissivity_inner.append(inverted_data[i_time][np.abs(inner_emissivity_peak_all[i_time,0]-inversion_R).argmin(),np.abs(inner_emissivity_peak_all[i_time,1]-inversion_Z).argmin()])
		peak_emissivity_outer.append(inverted_data[i_time][np.abs(outer_emissivity_peak_all[i_time,0]-inversion_R).argmin(),np.abs(outer_emissivity_peak_all[i_time,1]-inversion_Z).argmin()])

	time_res_bolo = full_saved_file_dict_FAST['multi_instrument']['time_res_bolo']
	CH25 = full_saved_file_dict_FAST['multi_instrument']['CH25']
	CH26 = full_saved_file_dict_FAST['multi_instrument']['CH26']
	CH27 = full_saved_file_dict_FAST['multi_instrument']['CH27']
	CH27_angle = full_saved_file_dict_FAST['multi_instrument']['CH27_angle']
	CH27_start_of_raise = full_saved_file_dict_FAST['multi_instrument']['CH27_start_of_raise']
	CH27_end_of_raise = full_saved_file_dict_FAST['multi_instrument']['CH27_end_of_raise']
	CH8_9 = full_saved_file_dict_FAST['multi_instrument']['CH8_9']
	CH8_9_angle = full_saved_file_dict_FAST['multi_instrument']['CH8_9_angle']
	CH8_9_start_of_raise = full_saved_file_dict_FAST['multi_instrument']['CH8_9_start_of_raise']
	CH8_9_end_of_raise = full_saved_file_dict_FAST['multi_instrument']['CH8_9_end_of_raise']
	CH27_26 = full_saved_file_dict_FAST['multi_instrument']['CH27_26']
	time_start_MARFE = full_saved_file_dict_FAST['multi_instrument']['time_start_MARFE']
	time_active_MARFE = full_saved_file_dict_FAST['multi_instrument']['time_active_MARFE']
	CH27_start = np.abs(time_res_bolo-CH27_start_of_raise).argmin()
	CH27_end = np.abs(time_res_bolo-CH27_end_of_raise).argmin()
	CH8_9_start = np.abs(time_res_bolo-CH8_9_start_of_raise).argmin()
	CH8_9_end = np.abs(time_res_bolo-CH8_9_end_of_raise).argmin()

	Dalpha = full_saved_file_dict_FAST['multi_instrument']['Dalpha']
	Dalpha_time = full_saved_file_dict_FAST['multi_instrument']['Dalpha_time']

	true_outer_target_position = outer_L_poloidal_peak_only_leg_all
	if name == 'IRVB-MASTU_shot-45468.ptw':
		pass
	elif name == 'IRVB-MASTU_shot-45469.ptw':
		true_outer_target_position = outer_L_poloidal_peak_all
	elif name == 'IRVB-MASTU_shot-45470.ptw':
		true_outer_target_position[time_full_binned_crop>0.65] = outer_L_poloidal_peak_all[time_full_binned_crop>0.65]
	elif name == 'IRVB-MASTU_shot-45473.ptw':
		true_outer_target_position[time_full_binned_crop>0.68] = outer_L_poloidal_peak_all[time_full_binned_crop>0.68]
		inner_L_poloidal_peak_all[-4:] = np.nan
		real_core_radiation_all[-4:] = np.nan
		x_point_tot_rad_power_all[-4:] = np.nan
		divertor_tot_rad_power_all[-4:] = np.nan
		all_lower_volume_radiation_all[-4:] = np.nan
	elif name == 'IRVB-MASTU_shot-45401.ptw':
		pass
		# true_outer_target_position[time_full_binned_crop>0.6] = outer_L_poloidal_peak_all[time_full_binned_crop>0.6]

		import pyuda
		client=pyuda.Client()
		TS_Te = client.get('ayc/T_e', laser_to_analyse[-9:-4]).data
		TS_ne = client.get('ayc/n_e', laser_to_analyse[-9:-4]).data
		TS_time = client.get('ayc/T_e', laser_to_analyse[-9:-4]).time.data
		TS_R = client.get('ayc/R', laser_to_analyse[-9:-4]).data
		min_psi = 0.6
		max_psi = 1.2
		# plt.figure()
		fig, ax = plt.subplots( 3,1,figsize=(8, 9), squeeze=False,sharex=False, gridspec_kw={'height_ratios': [1,1, 0.5]})
		# fig.subplots_adjust(hspace=0.3)
		ax[2,0].plot(Dalpha_time*1e3,Dalpha,'k')
		ax[2,0].set_xlim(left=220,right=235)
		ax[2,0].set_ylim(bottom=(Dalpha[np.logical_and(Dalpha_time>0.220,Dalpha_time<0.235)]).min()*0.9,top=(Dalpha[np.logical_and(Dalpha_time>0.220,Dalpha_time<0.235)]).max()*1.1)
		ax[2,0].set_xlabel('time [ms]')
		ax[2,0].set_ylabel(r'$D_{\alpha}$ [V]')
		for i___,time_ in enumerate([0.222,0.2275,0.2335]):
			i__ = np.abs(TS_time-time_).argmin()
			Te = TS_Te[i__]
			ne = TS_ne[i__]/1e19
			R = TS_R[i__]
			a = ax[2,0].axvline(x=TS_time[i__]*1e3,linestyle='--',color='C'+str(i___))
			R = R[np.isfinite(ne)]
			ne = ne[np.isfinite(ne)]
			Te = Te[np.isfinite(Te)]
			# gna = efit_reconstruction.psidat[i_]
			i_ = np.abs(efit_reconstruction.time-time_).argmin()
			gna=(efit_reconstruction.psidat[i_]-efit_reconstruction.psi_axis[i_])/(efit_reconstruction.psi_bnd[i_]-efit_reconstruction.psi_axis[i_])
			psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,gna)
			psi = np.diag(psi_interpolator(R,np.zeros_like(R)))
			ax[1,0].plot(psi[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],ne[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],color=a.get_color())#,label='%.3gms' %(TS_time[i__]*1e3))
			ax[1,0].plot(psi[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],ne[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],'+',color=a.get_color())
			ax[0,0].plot(psi[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],Te[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],color=a.get_color(),label='%.3gms' %(TS_time[i__]*1e3))
			ax[0,0].plot(psi[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],Te[R>efit_reconstruction.mag_axis_r[i_]][np.logical_and(psi[R>efit_reconstruction.mag_axis_r[i_]]>min_psi,psi[R>efit_reconstruction.mag_axis_r[i_]]<max_psi)],'+',color=a.get_color())
		ax[1,0].set_xlabel(r'$\rho$ [au]')
		ax[1,0].set_ylabel(r'$n_e$ [$10^{19}$]')
		ax[1,0].axvline(x=1,linestyle='--',color='k')
		ax[1,0].grid()

		ax[0,0].legend(loc='best')
		# ax[0,0].set_xlabel(r'$\rho$ [au]')
		ax[0,0].set_ylabel(r'$T_e$ [eV]')
		ax[0,0].axvline(x=1,linestyle='--',color='k')
		ax[0,0].grid()
		ax[0,0].get_shared_x_axes().join(ax[0,0],ax[1,0])
		ax[0,0].set_xticklabels([])

		# line to custom move the socond plot
		ax[1,0].set_position(ax[0,0].get_position().translated(0, -1 * ax[0,0].get_position().height
		# ax[0,0].set_ymargin(2)
		plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small7.png')
		plt.close()

	if name == 'IRVB-MASTU_shot-45401.ptw':
		client=pyuda.Client()
		brightness_res_bolo = client.get('/abm/core/brightness', laser_to_analyse[-9:-4]).data.T
		good_res_bolo = client.get('/abm/core/good', laser_to_analyse[-9:-4]).data
		time_res_bolo = client.get('/abm/core/time', laser_to_analyse[-9:-4]).data
		channel_res_bolo = client.get('/abm/core/channel', laser_to_analyse[-9:-4]).data
		# select only usefull time
		brightness_res_bolo = brightness_res_bolo[:,np.logical_and(time_res_bolo>-0.2,time_res_bolo<time_full_binned_crop.max()+0.1)]
		time_res_bolo = time_res_bolo[np.logical_and(time_res_bolo>-0.2,time_res_bolo<time_full_binned_crop.max()+0.1)]
		running_average_time = int(0.2/(np.median(np.diff(time_res_bolo))))
		CH4 = median_filter(brightness_res_bolo[channel_res_bolo==4][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))
		CH13 = median_filter(brightness_res_bolo[channel_res_bolo==13][0],size=int(0.005/(np.median(np.diff(time_res_bolo)))))

	gas_time = full_saved_file_dict_FAST['multi_instrument']['gas_time']
	gas_all = full_saved_file_dict_FAST['multi_instrument']['gas_all']
	gas_all_valves = full_saved_file_dict_FAST['multi_instrument']['gas_all_valves']
	gas_core = full_saved_file_dict_FAST['multi_instrument']['gas_core']
	gas_core_valves = full_saved_file_dict_FAST['multi_instrument']['gas_core_valves']
	gas_div = full_saved_file_dict_FAST['multi_instrument']['gas_div']
	gas_div_valves = full_saved_file_dict_FAST['multi_instrument']['gas_div_valves']


if False:	# explore the change i profile, abandoned
		fig, ax = plt.subplots( 2,1,figsize=(12, 12), squeeze=False,sharex=True)
		for i_time,time in enumerate([0.417,0.48,0.542,0.605,0.699,0.73]):
			i_t = np.abs(time_full_binned_crop-time).argmin()
			ax[0,0].plot(np.array(inner_local_L_poloidal_all[i_t])/inner_L_poloidal_x_point_all[i_t],inner_local_mean_emis_all[i_t],label=str(int(time_full_binned_crop[i_t]*1e3))+'ms',color=color[i_time])
			ax[1,0].plot(np.array(outer_local_L_poloidal_all[i_t])/outer_L_poloidal_x_point_all[i_t],outer_local_mean_emis_all[i_t],label=str(int(time_full_binned_crop[i_t]*1e3))+'ms',color=color[i_time])
			ax[0,0].plot((inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[i_t],peak_emissivity_inner[i_t],'o',color=color[i_time])
			ax[1,0].plot((outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[i_t],peak_emissivity_outer[i_t],'o',color=color[i_time])
		ax[0,0].legend(loc='best', fontsize='xx-small')
		ax[0,0].grid()
		ax[0,0].axvline(x=1,linestyle='--',color='k')
		ax[0,0].set_ylabel('emissivity inner\nseparatrix [W/m3]')
		# ax[0,0].set_xlabel('L poloidal/ L poloidal x-point [au]')
		ax[1,0].axvline(x=1,linestyle='--',color='k')
		ax[1,0].grid()
		ax[1,0].set_ylabel('emissivity outer\nseparatrix [W/m3]')
		ax[1,0].set_xlabel('L poloidal/ L poloidal x-point [au]')

		plt.figure()
		plt.plot(time_full_binned_crop,ne_bar/greenwald_density,'k')
		for i_time,time in enumerate([0.417,0.48,0.542,0.605,0.699,0.73]):
		# for i_time,time in enumerate([0.351,0.680,0.826]):
			plt.axvline(x=time,linestyle='--',color=color[i_time],linewidth=2.0)
		plt.xlabel('time [s]')
		plt.ylabel('greenwald fraction [au]')
		plt.grid()
		plt.pause(0.01)
elif False:
	fig, ax = plt.subplots( 2,2,figsize=(18, 10), squeeze=False,sharex=True)
	fig.suptitle('shot '+str(name[-9:-4])+', '+scenario+' , '+experiment+'\npass'+str(pass_number)+', '+', grid resolution '+str(grid_resolution)+'cm')
	ax[0,0].plot(time_full_binned_crop,inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all,'r-',label='inner_peak')
	ax[0,0].plot(time_full_binned_crop,outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all,'b-',label='outer_peak')
	ax[0,0].axhline(y=1,linestyle='--',color='k')
	ax[0,0].set_ylim(bottom=-0.1,top=4)
	ax[0,0].grid()
	ax[0,0].set_ylabel('Lpoloidal/Lpol x-pt [au]')
	ax[0,0].legend(loc='best', fontsize='small')
	ax[1,0].plot(time_full_binned_crop,ne_bar/greenwald_density,label='greenwald fraction',color='k')
	ax[1,0].set_ylabel('greenwald fraction [ua]')
	ax[1,0].grid()
	ax[1,0].set_xlabel('time [s]')
	if False:
		ax[0,1].plot(jsat_time,jsat_lower_outer_mid_max,'-',label='lower outer leg',color=color[1])
		ax[0,1].plot(jsat_time,jsat_lower_outer_mid_max,'+',color=color[1])
		ax[0,1].plot(jsat_time,jsat_upper_outer_mid_max,'-',label='upper outer leg',color=color[3])
		ax[0,1].plot(jsat_time,jsat_upper_outer_mid_max,'+',color=color[3])
		# temp = np.nanmax([median_filter(jsat_lower_inner_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_inner_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_inner_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11) , median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_outer_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11)],axis=0)
		# ax[5,0].set_ylim(bottom=0,top=np.nanmax(temp[np.logical_and(jsat_time>0,jsat_time<time_full_binned_crop[-2])]))
		ax[0,1].grid()
		ax[0,1].set_ylabel('jsat max [A/m2]')
		ax[0,1].legend(loc='best', fontsize='small',ncol=2)
	else:
		ax[0,1].plot(power_balance['t'],1e-6*(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt),label='input power (ohm+beams-dW/dt)',color=color[5])
		ax[0,1].plot(power_balance['t'],1e-6*(output_pohm-dWdt),'--',label='ohmic input power (ohm-dW/dt)',color=color[5])
		ax[0,1].plot(power_balance['t'],1e-6*power_balance['prad_core'],label='prad_core res bolo',color=color[7])
		temp = inverted_data[:,:,inversion_Z<0]
		temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
		# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
		ax[0,1].plot(time_full_binned_crop,1e-6*2*temp,label='total IRVB power (Z<0 x2)',color=color[2])
		ax[0,1].plot(power_balance['t'],1e-6*SW_BEAMPOWER,label='SW beam x '+str(sw_absorption),color=color[8])
		ax[0,1].plot(power_balance['t'],1e-6*SS_BEAMPOWER,label='SS beam x '+str(ss_absorption),color=color[9])
		temp = output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt
		ax[0,1].set_ylim(bottom=0,top=1e-6*1.2*np.nanmax(median_filter(temp[np.isfinite(temp)],size=21)[np.logical_and(power_balance['t'][np.isfinite(temp)]>0,power_balance['t'][np.isfinite(temp)]<time_full_binned_crop[-5])]))

		ax[0,1].plot(time_full_binned_crop,1e-6*real_core_radiation_all*2,label='core_radiation',color=color[3])
		ax[0,1].plot(time_full_binned_crop,1e-6*real_non_core_radiation_all*2,label='non_core_radiation',color=color[4])
		ax[0,1].grid()
		ax[0,1].set_ylabel('power [MW]')
		ax[0,1].legend(loc='best', fontsize='xx-small')

	ax[1,1].plot(jsat_time,jsat_lower_outer_mid_integrated,'-',label='lower outer leg',color=color[1])
	ax[1,1].plot(jsat_time,jsat_lower_outer_mid_integrated,'+',color=color[1])
	ax[1,1].plot(jsat_time,jsat_upper_outer_mid_integrated,'-',label='upper outer leg',color=color[3])
	ax[1,1].plot(jsat_time,jsat_upper_outer_mid_integrated,'+',color=color[3])
	ax[1,1].grid()
	ax[1,1].set_ylabel('integrated current [A]')
	# ax[6,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[1,1].set_xlabel('time [s]')
	plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute.png')
	plt.close()
elif False:	#this is the correct approach, density on the x axis
	select_time = time_full_binned_crop>0.30
	select_time[time_full_binned_crop>0.9]=False
	select_time[-3:] = False
	fig, ax = plt.subplots( 4,1,figsize=(10, 13), squeeze=False,sharex=True)
	fig.suptitle('shot '+str(name[-9:-4])+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
	ax[0,0].plot((nu_cowley)[select_time],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[select_time],'r-',label='inner separatrix')
	ax[0,0].plot((nu_cowley)[select_time],(true_outer_target_position/outer_L_poloidal_x_point_all)[select_time],'b-',label='outer separatrix')
	ax[0,0].axhline(y=1,linestyle='--',color='k')
	ax[0,0].axvline(x=nu_cowley[np.abs(0.386-time_full_binned_crop).argmin()],linestyle='--',color='g')
	ax[0,0].axvline(x=nu_cowley[np.abs(0.511-time_full_binned_crop).argmin()],linestyle='--',color='g')
	ax[0,0].axvline(x=nu_cowley[np.abs(0.605-time_full_binned_crop).argmin()],linestyle='--',color='g')
	ax[0,0].axvline(x=nu_cowley[np.abs(0.762-time_full_binned_crop).argmin()],linestyle='--',color='g')
	# ax[0,0].set_ylim(bottom=-0.1,top=4)
	ax[0,0].grid()
	# ax[0,0].set_ylabel('Lpoloidal/Lpol\nx-pt [au]')
	ax[0,0].set_ylabel(r'${\hat{L}}_{peak}$ [au]')
	ax[0,0].legend(loc='best', fontsize='x-small')
	# ax[0,0].set_xlabel('greenwald fraction [ua]')

	select_time2 = power_balance['t']>0.3
	select_time2[power_balance['t']>time_full_binned_crop[select_time].max()] = False
	ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,nu_cowley),1e-6*(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt)[select_time2],label='input',color=color[5])
	# ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,ne_bar/greenwald_density),1e-6*(output_pohm-dWdt)[select_time2],'--',label='ohmic input power (ohm-dW/dt)',color=color[5])
	ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,nu_cowley),1e-6*power_balance['prad_core'][select_time2],label='prad_core res bolo',color=color[7])
	# temp = inverted_data[:,:,inversion_Z<0]
	# temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
	# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
	# ax[1,0].plot((nu_cowley)[select_time],1e-6*2*temp[select_time],label='total IRVB power (Z<0 x2)',color=color[2])
	# ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,ne_bar/greenwald_density),1e-6*SW_BEAMPOWER[select_time2],label='SW beam x '+str(sw_absorption),color=color[8])
	# ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,ne_bar/greenwald_density),1e-6*SS_BEAMPOWER[select_time2],label='SS beam x '+str(ss_absorption),color=color[9])
	temp = output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt
	ax[1,0].set_ylim(bottom=0,top=1e-6*1.2*np.nanmax(median_filter(temp[np.isfinite(temp)],size=21)[np.logical_and(power_balance['t'][np.isfinite(temp)]>0,power_balance['t'][np.isfinite(temp)]<time_full_binned_crop[-5])]))

	# ax[1,0].plot((nu_cowley)[select_time],1e-6*real_core_radiation_all[select_time]*2,label='core_radiation',color=color[3])
	# ax[1,0].plot((nu_cowley)[select_time],1e-6*real_non_core_radiation_all[select_time]*2,label='non_core_radiation',color=color[4])
	ax[1,0].errorbar((nu_cowley)[select_time],1e-6*all_lower_volume_radiation_all[select_time]*2,yerr=1e-6*all_lower_volume_radiation_sigma_all[select_time]*2,capsize=5,linestyle='-',label='total IRVB (Z<0 x2)',color=color[2])
	ax[1,0].errorbar((nu_cowley)[select_time],1e-6*real_core_radiation_all[select_time]*2,yerr=1e-6*real_core_radiation_sigma_all[select_time]*2,capsize=5,linestyle='-',label='core IRVB',color=color[3])
	ax[1,0].errorbar((nu_cowley)[select_time],1e-6*x_point_tot_rad_power_all[select_time]*2,yerr=1e-6*x_point_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='x-point IRVB',color=color[4])
	ax[1,0].errorbar((nu_cowley)[select_time],1e-6*divertor_tot_rad_power_all[select_time]*2,yerr=1e-6*divertor_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='divertor IRVB',color=color[8])
	ax[1,0].grid()
	ax[1,0].set_ylabel('power [MW]')
	ax[1,0].legend(loc='best', fontsize='x-small')


	# ax[2,0].plot((nu_cowley)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,'-',label='lower outer leg',color=color[1])
	# ax[2,0].plot((nu_cowley)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,'+',color=color[1])
	ax[2,0].errorbar((nu_cowley)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,yerr=jsat_lower_outer_mid_integrated_sigma[select_time]/1.6e-19,capsize=5,linestyle='-',label='lower outer',color=color[1])
	# ax[2,0].plot((nu_cowley)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,'-',label='upper outer leg',color=color[3])
	# ax[2,0].plot((nu_cowley)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,'+',color=color[3])
	ax[2,0].errorbar((nu_cowley)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,yerr=jsat_upper_outer_mid_integrated_sigma[select_time]/1.6e-19,capsize=5,linestyle='-',label='upper outer',color=color[3])
	ax[2,0].grid()
	ax[2,0].set_ylabel('target particle\nflux [#/s]')
	# ax[6,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[2,0].legend(loc='best', fontsize='x-small',ncol=2)

	select_time3 = time_res_bolo>0.30
	select_time3[time_res_bolo>0.9]=False
	select_time3[time_res_bolo>time_full_binned_crop[select_time].max()] = False
	# greenwald_fraction_interp = interp1d(time_full_binned_crop,(ne_bar/greenwald_density),bounds_error=False,fill_value='extrapolate')
	greenwald_fraction_interp = interp1d(time_full_binned_crop,nu_cowley,bounds_error=False,fill_value='extrapolate')
	ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH26[select_time3]*1e-3,'b',label='core26')
	ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH27[select_time3]*1e-3,'g',label='core27')
	# ax[3,0].plot(greenwald_fraction_interp(np.array([CH27_start_of_raise,CH27_end_of_raise])),[CH27[CH27_start],CH27[CH27_end]],'--g')
	# ax[3,0].axvline(x=greenwald_fraction_interp(CH27_start_of_raise),linestyle='--',color='g')
	# ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH8_9[select_time3]*1e-3,'b',label='core8/9')
	# ax[3,0].plot(greenwald_fraction_interp(np.array([CH8_9_start_of_raise,CH8_9_end_of_raise])),[CH8_9[CH8_9_start],CH8_9[CH8_9_end]],'--b')
	# ax[3,0].axvline(x=greenwald_fraction_interp(CH8_9_start_of_raise),linestyle='--',color='b')
	ax[3,0].legend(loc='best', fontsize='x-small')
	ax[3,0].set_ylabel('Brightness [kW/m2]')
	ax[3,0].grid()
	ax3 = ax[3,0].twinx()  # instantiate a second axes that shares the same x-axis
	ax3.spines["right"].set_visible(True)
	ax3.set_ylabel('Brigtness\nCH27/CH26 [au]', color='r')  # we already handled the x-label with ax1
	ax3.plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH27_26[select_time3],'r')
	# ax3.axhline(y=3,linestyle='--',color='r')
	# ax3.axvline(x=greenwald_fraction_interp(time_active_MARFE),linestyle='-',color='r')
	ax3.axvline(x=greenwald_fraction_interp(time_start_MARFE),linestyle='-',color='r')
	# ax3.set_ylim(top=4)

	# ax[3,0].set_xlabel('greenwald fraction [ua]')
	ax[3,0].set_xlabel('upstream density [#/m^3]')
	plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small2.png')
	plt.close()
elif False:	# same but time is the axis
	select_time = time_full_binned_crop>0.
	if name == 'IRVB-MASTU_shot-45473.ptw':
		select_time[time_full_binned_crop>0.9]=False
		select_time[-3:] = False
		fig, ax = plt.subplots( 5,1,figsize=(10, 18), squeeze=False,sharex=True)
	elif name == 'IRVB-MASTU_shot-45401.ptw':
		select_time[time_full_binned_crop>1.1]=False
		select_time[-3:] = False
		fig, ax = plt.subplots( 5,1,figsize=(10, 18), squeeze=False,sharex=True)
	elif name == 'IRVB-MASTU_shot-45371.ptw':
		select_time = time_full_binned_crop>0.
		select_time[time_full_binned_crop>0.89]=False
		select_time[-1:] = False
		fig, ax = plt.subplots( 5,1,figsize=(10, 19), squeeze=False,sharex=True, gridspec_kw={'height_ratios': [1, 2, 2,1.5,1.5]})
	fig.suptitle('shot '+str(name[-9:-4])+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
	ax[0,0].plot((time_full_binned_crop)[select_time],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[select_time],'r-',label='inner separatrix')
	ax[0,0].plot((time_full_binned_crop)[select_time],(true_outer_target_position/outer_L_poloidal_x_point_all)[select_time],'b-',label='outer separatrix')
	ax[0,0].axhline(y=1,linestyle='--',color='k')
	if name == 'IRVB-MASTU_shot-45473.ptw':
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.386-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.511-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.605-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.699-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.762-time_full_binned_crop).argmin()],linestyle='--',color='g')
	elif name == 'IRVB-MASTU_shot-45401.ptw':
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.315-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.607-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.717-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(0.826-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[0,0].axvline(x=time_full_binned_crop[np.abs(1.009-time_full_binned_crop).argmin()],linestyle='--',color='g')
	ax[0,0].set_ylim(top=5)
	ax[0,0].grid()
	# ax[0,0].set_ylabel('Lpoloidal/Lpol\nx-pt [au]')
	ax[0,0].set_ylabel(r'${\hat{L}}_{peak}$ [au]')
	ax[0,0].legend(loc='best', fontsize='x-small')
	# ax[0,0].set_xlabel('greenwald fraction [ua]')

	select_time2 = power_balance['t']>0.
	select_time2[power_balance['t']>time_full_binned_crop[select_time].max()] = False
	ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt)[select_time2],label='input',color=color[5])
	if np.nanmax(SW_BEAMPOWER)>0:
		ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*(SW_BEAMPOWER)[select_time2],label='SW NBI',color=color[1])
	if np.nanmax(SS_BEAMPOWER)>0:
		ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*(SS_BEAMPOWER)[select_time2],label='SS NBI',color=color[6])

	# ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,ne_bar/greenwald_density),1e-6*(output_pohm-dWdt)[select_time2],'--',label='ohmic input power (ohm-dW/dt)',color=color[5])
	ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*power_balance['prad_core'][select_time2],label='prad_core\nres bolo',color=color[7])
	# temp = inverted_data[:,:,inversion_Z<0]
	# temp = np.nansum(np.nansum(temp,axis=-1)*inversion_R*(np.mean(np.diff(inversion_R))**2)*2*np.pi,axis=1)
	# plt.plot(time_full_binned_crop,temp/np.nanmax(temp),label='relative total power')
	# ax[1,0].plot((nu_cowley)[select_time],1e-6*2*temp[select_time],label='total IRVB power (Z<0 x2)',color=color[2])
	# ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,ne_bar/greenwald_density),1e-6*SW_BEAMPOWER[select_time2],label='SW beam x '+str(sw_absorption),color=color[8])
	# ax[1,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,ne_bar/greenwald_density),1e-6*SS_BEAMPOWER[select_time2],label='SS beam x '+str(ss_absorption),color=color[9])
	temp = output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt
	ax[1,0].set_ylim(bottom=0,top=1e-6*1.2*np.nanmax(median_filter(temp[np.isfinite(temp)],size=21)[np.logical_and(power_balance['t'][np.isfinite(temp)]>0,power_balance['t'][np.isfinite(temp)]<time_full_binned_crop[-5])]))

	# ax[1,0].plot((nu_cowley)[select_time],1e-6*real_core_radiation_all[select_time]*2,label='core_radiation',color=color[3])
	# ax[1,0].plot((nu_cowley)[select_time],1e-6*real_non_core_radiation_all[select_time]*2,label='non_core_radiation',color=color[4])
	ax[1,0].grid()
	ax[1,0].set_ylabel('power [MW]')
	if name != 'IRVB-MASTU_shot-45371.ptw':
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*all_lower_volume_radiation_all[select_time]*2,yerr=1e-6*all_lower_volume_radiation_sigma_all[select_time]*2,capsize=5,linestyle='-',label='total IRVB',color=color[2])
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*real_core_radiation_all[select_time]*2,yerr=1e-6*real_core_radiation_sigma_all[select_time]*2,capsize=5,linestyle='-',label='core+SOL IRVB',color=color[3])
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*x_point_tot_rad_power_all[select_time]*2,yerr=1e-6*x_point_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='x-point IRVB',color=color[4])
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*approx_divertor[select_time]*2,yerr=1e-6*approx_divertor_sigma[select_time]*2,capsize=5,linestyle='-',label='divertor IRVB',color=color[8])
		ax[1,0].legend(loc='best', fontsize='x-small',ncol=2)
	else:
		ax[2,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*(output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt)[select_time2],label='input',color=color[5])
		if np.nanmax(SW_BEAMPOWER)>0:
			ax[2,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*(SW_BEAMPOWER)[select_time2],label='SW NBI',color=color[1])
		if np.nanmax(SS_BEAMPOWER)>0:
			ax[2,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*(SS_BEAMPOWER)[select_time2],label='SS NBI',color=color[6])
		ax[2,0].plot(np.interp(power_balance['t'][select_time2],time_full_binned_crop,time_full_binned_crop),1e-6*power_balance['prad_core'][select_time2],label='prad_core\nres bolo',color=color[7])
		temp = output_pohm + SW_BEAMPOWER + SS_BEAMPOWER-dWdt
		ax[2,0].set_ylim(bottom=0,top=1e-6*1.2*np.nanmax(median_filter(temp[np.isfinite(temp)],size=21)[np.logical_and(power_balance['t'][np.isfinite(temp)]>0,power_balance['t'][np.isfinite(temp)]<time_full_binned_crop[-5])]))

		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*all_lower_volume_radiation_all[select_time]*2,yerr=1e-6*all_lower_volume_radiation_sigma_all[select_time]*2,capsize=5,linestyle='-',label='total IRVB',color=color[2])
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*x_point_tot_rad_power_all[select_time]*2,yerr=1e-6*x_point_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='x-point IRVB',color=color[3])
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*sxd_tot_rad_power_all[select_time]*2,yerr=1e-6*sxd_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='SXD IRVB',color=color[4])
		ax[1,0].errorbar((time_full_binned_crop)[select_time],1e-6*(outer_SOL_leg_all-sxd_tot_rad_power_all)[select_time]*2,yerr=1e-6*((sxd_tot_rad_power_sigma_all**2 + outer_SOL_leg_sigma_all**2)**0.5)[select_time]*2,capsize=5,linestyle='-',label='outer leg\n+SOL -SXD IRVB',color=color[8])
		# ax[1,0].set_xlabel('time [s]')
		ax[2,0].errorbar((time_full_binned_crop)[select_time],1e-6*all_lower_volume_radiation_all[select_time]*2,yerr=1e-6*all_lower_volume_radiation_sigma_all[select_time]*2,capsize=5,linestyle='-',label='total IRVB',color=color[2])
		ax[2,0].errorbar((time_full_binned_crop)[select_time],1e-6*x_point_tot_rad_power_all[select_time]*2,yerr=1e-6*x_point_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='x-point IRVB',color=color[3])
		ax[2,0].errorbar((time_full_binned_crop)[select_time],1e-6*sxd_tot_rad_power_all[select_time]*2,yerr=1e-6*sxd_tot_rad_power_sigma_all[select_time]*2,capsize=5,linestyle='-',label='SXD IRVB',color=color[4])
		ax[2,0].errorbar((time_full_binned_crop)[select_time],1e-6*(outer_SOL_leg_all-sxd_tot_rad_power_all)[select_time]*2,yerr=1e-6*((sxd_tot_rad_power_sigma_all**2 + outer_SOL_leg_sigma_all**2)**0.5)[select_time]*2,capsize=5,linestyle='-',label='outer leg\n+SOL -SXD IRVB',color=color[8])
		ax[2,0].grid()
		# ax[2,0].set_xlabel('time [s]')
		# ax[1,0].semilogy()
		ax[1,0].legend(loc='best', fontsize='x-small',ncol=2)
		ax[1,0].set_ylim(bottom=0.2)
		ax[2,0].set_ylim(top=0.2)

		ax[1,0].spines['bottom'].set_visible(False)
		ax[2,0].spines['top'].set_visible(False)
		ax[1,0].xaxis.tick_top()
		ax[1,0].tick_params(labeltop=False)  # don't put tick labels at the top
		ax[2,0].xaxis.tick_bottom()

		ax[1,0].axvline(x=time_full_binned_crop[np.abs(0.462-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[1,0].axvline(x=time_full_binned_crop[np.abs(0.572-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[1,0].axvline(x=time_full_binned_crop[np.abs(0.754-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[1,0].axvline(x=time_full_binned_crop[np.abs(0.828-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[2,0].axvline(x=time_full_binned_crop[np.abs(0.462-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[2,0].axvline(x=time_full_binned_crop[np.abs(0.572-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[2,0].axvline(x=time_full_binned_crop[np.abs(0.754-time_full_binned_crop).argmin()],linestyle='--',color='g')
		ax[2,0].axvline(x=time_full_binned_crop[np.abs(0.828-time_full_binned_crop).argmin()],linestyle='--',color='g')


		d = .01  # how big to make the diagonal lines in axes coordinates
		# arguments to pass to plot, just so we don't keep repeating them
		kwargs = dict(transform=ax[1,0].transAxes, color='k', clip_on=False)
		ax[1,0].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
		ax[1,0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

		kwargs.update(transform=ax[2,0].transAxes)  # switch to the bottom axes
		ax[2,0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
		ax[2,0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

		ax[3,0].plot(time_full_binned_crop,ne_bar*1e-19,label=r'$\overline{n_e}$',color=color[-1])
		ax[3,0].plot(time_full_binned_crop,nu_cowley*1e-19,label=r'$n_u$',color=color[-3])
		ax[3,0].grid()
		ax[3,0].set_ylabel(r'Density [$10^{19}\#/m^3$]')
		ax[3,0].legend(loc='best', fontsize='x-small')
		ax4 = ax[3,0].twinx()  # instantiate a second axes that shares the same x-axis
		ax4.spines["right"].set_visible(True)
		ax4.set_ylabel('Greenwald fraction [au]', color='r')  # we already handled the x-label with ax1
		ax4.plot(time_full_binned_crop,ne_bar/greenwald_density,'r')

		try:
			ax[4,0].plot(gas_time,gas_inner*1e-21,label='inner gas flow'+str(gas_inner_valves),color=color[15])
		except:
			pass
		try:
			ax[4,0].plot(gas_time,gas_outer*1e-21,label='outer gas flow'+str(gas_outer_valves),color=color[4])
		except:
			pass
		try:
			ax[4,0].plot(gas_time,gas_div*1e-21,label='divertor gas flow'+str(gas_div_valves),color=color[9])
		except:
			pass
		# ax[3,0].plot(gas_time,gas_all,'--',label='total gas flow',color=color[6],dashes=(3, 6))
		ax[4,0].grid()
		ax[4,0].set_ylabel(r'fueling [$10^{21}mol/s$]')
		ax[4,0].legend(loc='best', fontsize='xx-small')

		ax[4,0].set_xlabel('time [s]')
		ax[4,0].set_xlim(left=0.2-0.005,right=time_full_binned_crop[select_time].max()+0.005)

		plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small4.png')
		plt.close()


	# ax[2,0].plot((time_full_binned_crop)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,'-',label='lower outer leg',color=color[1])
	# ax[2,0].plot((time_full_binned_crop)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,'+',color=color[1])
	ax[2,0].errorbar((time_full_binned_crop)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19*1e-22,yerr=jsat_lower_outer_mid_integrated_sigma[select_time]/1.6e-19*1e-22,capsize=5,linestyle='-',label='lower outer',color=color[1])
	# ax[2,0].plot((time_full_binned_crop)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,'-',label='upper outer leg',color=color[3])
	# ax[2,0].plot((time_full_binned_crop)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,'+',color=color[3])
	ax[2,0].errorbar((time_full_binned_crop)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19*1e-22,yerr=jsat_upper_outer_mid_integrated_sigma[select_time]/1.6e-19*1e-22,capsize=5,linestyle='-',label='upper outer',color=color[3])
	ax[2,0].grid()
	ax[2,0].set_ylabel('target particle\nflux '+r'[$10^{22}\#/s$]')
	# ax[6,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[2,0].legend(loc='best', fontsize='x-small',ncol=2)

	select_time3 = time_res_bolo>0.
	select_time3[time_res_bolo>time_full_binned_crop[select_time].max()] = False
	# greenwald_fraction_interp = interp1d(time_full_binned_crop,(ne_bar/greenwald_density),bounds_error=False,fill_value='extrapolate')
	greenwald_fraction_interp = interp1d(time_full_binned_crop,time_full_binned_crop,bounds_error=False,fill_value='extrapolate')
	ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH25[select_time3]*1e-3,'y',label='tang CH25')
	ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH26[select_time3]*1e-3,'b',label='tang CH26')
	ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH27[select_time3]*1e-3,'g',label='tang CH27')
	# ax[3,0].plot(greenwald_fraction_interp(np.array([CH27_start_of_raise,CH27_end_of_raise])),[CH27[CH27_start],CH27[CH27_end]],'--g')
	# ax[3,0].axvline(x=greenwald_fraction_interp(CH27_start_of_raise),linestyle='--',color='g')
	# ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH8_9[select_time3]*1e-3,'b',label='core8/9')
	# ax[3,0].plot(greenwald_fraction_interp(np.array([CH8_9_start_of_raise,CH8_9_end_of_raise])),[CH8_9[CH8_9_start],CH8_9[CH8_9_end]],'--b')
	# ax[3,0].axvline(x=greenwald_fraction_interp(CH8_9_start_of_raise),linestyle='--',color='b')
	ax[3,0].set_ylabel(r'Brightness [$kW/m^2$]')
	ax[3,0].grid()
	ax3 = ax[3,0].twinx()  # instantiate a second axes that shares the same x-axis
	ax3.spines["right"].set_visible(True)
	ax3.set_ylabel('Brigtness\nCH27/CH26 [au]', color='r')  # we already handled the x-label with ax1
	ax3.plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH27_26[select_time3],'r')
	# ax3.axhline(y=3,linestyle='--',color='r')
	# ax3.axvline(x=greenwald_fraction_interp(time_active_MARFE),linestyle='-',color='r')
	ax3.axvline(x=greenwald_fraction_interp(time_start_MARFE),linestyle='--',color='r')
	ax3.set_ylim(top=5)

	if name == 'IRVB-MASTU_shot-45401.ptw':
		ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH4[select_time3]*1e-3,'m',label='poloid CH4')
		ax[3,0].plot(greenwald_fraction_interp(time_res_bolo[select_time3]),CH13[select_time3]*1e-3,'c',label='poloid CH13')
	ax[3,0].legend(loc='best', fontsize='x-small')

	ax[4,0].plot(time_full_binned_crop,ne_bar*1e-19,label=r'$\overline{n_e}$',color=color[-1])
	ax[4,0].plot(time_full_binned_crop,nu_cowley*1e-19,label=r'$n_u$',color=color[-3])
	ax[4,0].grid()
	ax[4,0].set_ylabel(r'Density [$10^{19}\#/m^3$]')
	ax[4,0].legend(loc='best', fontsize='x-small')
	ax4 = ax[4,0].twinx()  # instantiate a second axes that shares the same x-axis
	ax4.spines["right"].set_visible(True)
	ax4.set_ylabel('Greenwald fraction [au]', color='r')  # we already handled the x-label with ax1
	ax4.plot(time_full_binned_crop,ne_bar/greenwald_density,'r')

	# ax[3,0].set_xlabel('greenwald fraction [ua]')
	ax[4,0].set_xlabel('time [s]')
	if name == 'IRVB-MASTU_shot-45401.ptw':
		ax[0,0].set_xlim(left=0.1-0.005,right=time_full_binned_crop[select_time].max()+0.005)
	if name == 'IRVB-MASTU_shot-45473.ptw':
		ax[0,0].set_xlim(left=0.3-0.005,right=time_full_binned_crop[select_time].max()+0.005)

	plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small3.png')
	plt.close()


	# section for 45401 to check the type of ELM regime by looking at Dalpha

	select_dalpha = Dalpha_time<time_full_binned_crop[select_time].max()
	select_dalpha = np.logical_and(Dalpha_time>time_full_binned_crop[select_time].min(),select_dalpha)
	fig, ax = plt.subplots( 1,1,figsize=(12, 6), squeeze=False,sharex=True)
	fig.suptitle('shot '+str(name[-9:-4])+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
	ax[0,0].plot(Dalpha_time[select_dalpha],Dalpha[select_dalpha])
	ax[0,0].grid()
	# ax[0,0].set_ylabel('Lpoloidal/Lpol\nx-pt [au]')
	ax[0,0].set_ylabel(r'$D_{\alpha}$ [V]')
	ax[0,0].set_xlabel('time [s]')
	# ax[0,0].set_xlabel('greenwald fraction [ua]')
	if name == 'IRVB-MASTU_shot-45401.ptw':
		ax[0,0].set_xlim(left=0.1-0.005,right=time_full_binned_crop[select_time].max()+0.005)

		ax2 = plt.axes([.18, .53, .13, .29], facecolor='w',yscale='linear')
		# ax[0,0].plot(np.array([0.264,0.268,0.268,0.264,0.264]),[0.02,0.02,0.09,0.09,0.02],'r--',label='small ELMs')	# last ELM
		# ax2.set_xlim(left=0.264,right=0.268)
		# ax2.set_ylim(bottom=0.02,top=0.09)
		ax[0,0].plot(np.array([0.2255,0.2285,0.2285,0.2255,0.2255]),[0.015,0.015,0.075,0.075,0.016],'r--',label='small ELMs')	# good ELM for TS
		ax2.set_xlim(left=0.2255,right=0.2285)
		ax2.set_ylim(bottom=0.015,top=0.075)
		ax2.plot(Dalpha_time[select_dalpha],Dalpha[select_dalpha],'r')
		ax2.set_title('small ELM')
		ax2.grid()

		ax3 = plt.axes([.38, .53, .27, .29], facecolor='w',yscale='linear')
		ax[0,0].plot(np.array([0.951,0.96,0.96,0.951,0.951]),[0.1,0.1,0.25,0.25,0.1],'g--',label='dithering')
		ax3.plot(Dalpha_time[select_dalpha],Dalpha[select_dalpha],'g')
		ax3.set_xlim(left=0.951,right=0.96)
		ax3.set_ylim(bottom=0.1,top=0.25)
		ax3.set_title('dithering')
		ax3.grid()

	plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small6.png')
	plt.close()


	# this bit is for the MWI data comparison with IRVB SXD
	MWI_common = '/home/ffederic/work/analysis_scripts/scripts/from_Tijs'
	cam3_B62 = np.load(MWI_common+'/Power_cam3.npy')
	cam3_B62_time = np.load(MWI_common+'/time_cam3.npy')
	cam6_B52 = np.load(MWI_common+'/Power_cam6.npy')
	cam6_B52_time = np.load(MWI_common+'/time_cam6.npy')
	cam7_CIII = np.load(MWI_common+'/Power_cam7.npy')
	cam7_CIII_time = np.load(MWI_common+'/time_cam7.npy')
	cam9_B32 = np.load(MWI_common+'/Power_cam9.npy')
	cam9_B32_time = np.load(MWI_common+'/time_cam9.npy')
	cam10_F = np.load(MWI_common+'/Power_cam10.npy')
	cam10_F_time = np.load(MWI_common+'/time_cam10.npy')

	dt = np.mean(np.diff(time_full_binned_crop))
	cam3_B62 = generic_filter(cam3_B62[np.logical_and(cam3_B62_time>0.3,cam3_B62_time<0.8)],np.mean,size=int(np.mean(dt/np.diff(cam3_B62_time))))
	cam3_B62_time = cam3_B62_time[np.logical_and(cam3_B62_time>0.3,cam3_B62_time<0.8)]
	cam6_B52 = generic_filter(cam6_B52[np.logical_and(cam6_B52_time>0.3,cam6_B52_time<0.8)],np.mean,size=int(np.mean(dt/np.diff(cam6_B52_time))))
	cam6_B52_time = cam6_B52_time[np.logical_and(cam6_B52_time>0.3,cam6_B52_time<0.8)]
	cam7_CIII = generic_filter(cam7_CIII[np.logical_and(cam7_CIII_time>0.3,cam7_CIII_time<0.8)],np.mean,size=int(np.mean(dt/np.diff(cam7_CIII_time))))
	cam7_CIII_time = cam7_CIII_time[np.logical_and(cam7_CIII_time>0.3,cam7_CIII_time<0.8)]
	cam9_B32 = generic_filter(cam9_B32[np.logical_and(cam9_B32_time>0.3,cam9_B32_time<0.8)],np.mean,size=int(np.mean(dt/np.diff(cam9_B32_time))))
	cam9_B32_time = cam9_B32_time[np.logical_and(cam9_B32_time>0.3,cam9_B32_time<0.8)]
	cam10_F = generic_filter(cam10_F[np.logical_and(cam10_F_time>0.3,cam10_F_time<0.8)],np.mean,size=int(np.mean(dt/np.diff(cam10_F_time))))
	cam10_F_time = cam10_F_time[np.logical_and(cam10_F_time>0.3,cam10_F_time<0.8)]
	select_time = np.logical_and(time_full_binned_crop>0.3,time_full_binned_crop<0.8)

	fig, ax = plt.subplots( 1,1,figsize=(12, 6), squeeze=True,sharex=True)
	# plt.figure(figsize=(12, 6))
	ax.plot(cam7_CIII_time,cam7_CIII/(cam7_CIII.max()),color=color[0],label=r'$\leftarrow$ CIII')
	ax.plot(cam10_F_time,cam10_F/(cam10_F.max()),color=color[1],label=r'$\leftarrow$ Fulcher Band')
	ax.plot(cam9_B32_time,cam9_B32/(cam9_B32.max()),color=color[3],label=r'$\leftarrow$ Balmer $3 \rightarrow 2$')
	ax.plot(cam6_B52_time,cam6_B52/(cam6_B52.max()),color=color[2],label=r'$\leftarrow$ Balmer $5 \rightarrow 2$')
	ax.plot(cam3_B62_time,cam3_B62/(cam3_B62.max()),color=color[5],label=r'$\leftarrow$ Balmer $6 \rightarrow 2$')
	# ax.set_ylabel(r'$\frac{\langle ph/s \rangle}{{\langle ph/s \rangle}_{max}}$ [MW]')
	ax.set_ylabel(r'$\langle ph/s \rangle / {\langle ph/s \rangle}_{max}$ [au]')
	ax.set_xlabel('time [s]')
	ax4 = ax.twinx()  # instantiate a second axes that shares the same x-axis
	ax4.spines["right"].set_visible(True)
	ax4.set_ylabel('Power [MW]')  # we already handled the x-label with ax1
	a1 = ax4.errorbar((time_full_binned_crop)[select_time],2*1e-6*sxd_tot_rad_power_all[select_time],yerr=2*1e-6*sxd_tot_rad_power_sigma_all[select_time],capsize=5,linestyle='-',label=r'$\rightarrow$ SXD IRVB',color=color[4])
	a2 = ax4.errorbar((time_full_binned_crop)[select_time],1e-6*(outer_SOL_leg_all-sxd_tot_rad_power_all)[select_time]*2,yerr=1e-6*((sxd_tot_rad_power_sigma_all**2 + outer_SOL_leg_sigma_all**2)**0.5)[select_time]*2,capsize=5,linestyle='-',label=r'$\rightarrow$ outer leg'+'\n+SOL -SXD IRVB',color=color[8])
	ax4.set_ylim(bottom=0)
	ax4.tick_params(axis='y')
	ax.set_xlim(left=0.3-0.005,right=0.8+0.005)
	ax.set_ylim(bottom=0,top=1.22)
	ax.grid()
	handles, labels = ax.get_legend_handles_labels()
	handles.append(a1)
	labels.append(a1.get_label())
	handles.append(a2)
	labels.append(a2.get_label())
	ax.legend(handles=handles, labels=labels, loc='lower center', fontsize='x-small')
	# plt.pause(0.01)
	plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small5.png')
	plt.close()

	import h5py
	MWI_filename = '/home/ffederic/work/analysis_scripts/scripts/from_Tijs/'+'MWI_inv_data_sh45371_cam7_v1.h5'
	f = h5py.File(MWI_filename, "r")

	r = f['grid']['vertices'][:,0]
	z = f['grid']['vertices'][:,1]
	r = np.mean(r[f['grid']['cells'].value],axis=-1)
	z = np.mean(z[f['grid']['cells'].value],axis=-1)
	MWI_emissivity = f['inversion']['emissivity'].value
	MWI_time = f['inversion']['times'].value

	EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
	efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse[-9:-4]+'.nc',pulse_ID=laser_to_analyse[-9:-4])
	all_time_sep_r,all_time_sep_z,r_fine,z_fine = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
	all_time_separatrix = coleval.return_all_time_separatrix_radial(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)


	time_target = 0.5
	plt.figure()
	plt.tricontourf(r,z,MWI_emissivity[:,np.abs(MWI_time-time_target).argmin()])
	plt.colorbar()
	plt.xlabel('R [m]')
	plt.ylabel('Z [m]')
	plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	i_time = np.abs(time_target-efit_reconstruction.time).argmin()
	for __i in range(4):
		plt.plot(all_time_separatrix[i_time][__i][0],all_time_separatrix[i_time][__i][1],'b')
	plt.axes().set_aspect('equal')
	plt.pause(0.01)

if True:	# this bit is to check the fast camera data
	def high_speed_visible_load_cih(filename):
		"""
		Load HSV metadata.

		source: https://github.com/ladisk/pyMRAW

		:param int/str shot: MAST-U shot number.
		:param str fpath_data: Path to data directory, see above.
		:return: dict of metadata.
		"""

		cih = dict()
		with open(filename, 'r') as f:
			for line in f:
				if line == '\n':  # end of cif header
					break
				line_sp = line.replace('\n', '').split(' : ')
				if len(line_sp) == 2:
					key, value = line_sp
					try:
						if '.' in value:
							value = float(value)
						else:
							value = int(value)
						cih[key] = value
					except:
						cih[key] = value
		return cih

	# data_dir = '/home/ffederic/work/Collaboratory/test/experimental_data/45473/C001H001S0001/'
	# data_dir = '/home/ffederic/work/Collaboratory/test/experimental_data/45351/C001H001S0001/'
	# T0=-0.1	# for 45351

	# data_dir = '/home/ffederic/work/Collaboratory/test/experimental_data/44647_HD/C001H001S0001/'
	# T0=0.0	# for 44647

	# data_dir = '/home/ffederic/work/Collaboratory/test/experimental_data/45295/C001H001S0001/'
	# T0=-0.1	# for 45295

	data_dir = '/home/ffederic/work/Collaboratory/test/experimental_data/45401/C001H001S0001/'
	T0=-0.1	# for 45401

	filename = 'C001H001S0001'

	def high_speed_visible_mraw_to_numpy(data_dir,filename,T0,downsample=1):
		full_filename = data_dir+filename+'.mraw'
		fname, ext = full_filename.split('.')
		cih = high_speed_visible_load_cih(fname+'.cih')
		num, denom = cih['Shutter Speed(s)'].split('/')
		cih['Shutter Speed(s)'] = float(num) / float(denom)
		nframes = cih['Total Frame']
		height = cih['Image Height']
		width = cih['Image Width']
		fps = cih['Record Rate(fps)']

		HEIGHT_FULL=1024
		WIDTH_FULL=1024


		xpix = np.arange(width) - width // 2 + WIDTH_FULL // 2
		ypix = np.arange(height)

		times = np.arange(nframes) * 1 / fps + T0
		with open(full_filename, 'rb') as f:
			data = np.frombuffer(f.read(), dtype=np.uint8)

		# optimised loading of 12-bit uint data from binary file into numpy array
		# source: https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files
		fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).T
		fst_uint12 = (fst_uint8.astype(np.uint16) << 4) + (mid_uint8.astype(np.uint16) >> 4)
		snd_uint12 = ((mid_uint8.astype(np.uint16) % 16) << 8) + lst_uint8.astype(np.uint16)
		data = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
		data = data.reshape([nframes, height, width, ])
		return data,times

	data,times = high_speed_visible_mraw_to_numpy(data_dir,filename,T0)
	# np.savez_compressed(data_dir+filename,dict=([('data',data),('times',times)]))
	np.save(data_dir+filename+'_data',data)
	np.save(data_dir+filename+'_times',times)

	time=np.abs(times-0.95).argmin()
	plt.figure()
	plt.imshow(data[time],'rainbow')
	plt.colorbar()
	plt.pause(0.01)


	plt.figure()
	plt.plot(times,np.mean(data,axis=(1,2)))
	plt.plot(times,np.mean(data[:,513-2:525+2,521-2:533+2],axis=(1,2)),'--')
	plt.grid()
	plt.pause(0.01)
