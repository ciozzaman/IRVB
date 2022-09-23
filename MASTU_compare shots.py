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


if False:	# plot for the papers
	to_do = []
	shot_list = get_data(path+'shot_list2.ods')
	for i in range(1,len(shot_list['Sheet1'])):
		try:
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ][-2:] in ['OH']:# ['SS','SW','OH']:#,'2B']:
			# 	continue
			# if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ][-6:-3] in ['-CD']:#['SXD','-SF']
			# 	continue
			if not shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Scenario').argmax() ] in ['DN-600-CD-OH','DN-450-CD-OH','DN-750-CD-OH']:#['DN-750-CD-2B']:#['SXD','-SF']
				continue
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
			if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] == 'MU01-MHD-01':	# not sure what it is done here
				continue
			if shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='Experiment Tags').argmax() ] == 'MU01-MHD-02':	# not sure what it is done here
				continue
			# if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45473,45470,45302]:	# gas imput sometimes at 0 and with ramp up
			# 	continue
			# if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45468,45469,45293,45466,45327,45326,45325,45324,45323,45322,45320]:	# always some imput gas wayform
			# 	continue
			# if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45303]:	# always some imput gas wayform, but weirder
			# 	continue
			if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45063]:	# strike point too high to be seen
				continue
			# if not int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) in [45243,45244,45245,45241,45143,45080,45071,45060,45059,45058,45057,45056,45046,45047,45048]:	# always some imput gas wayform, but weirder
			# 	continue
			if int(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ]) < 45300:
				continue
			to_do.append('IRVB-MASTU_shot-'+str(shot_list['Sheet1'][i][ (np.array(shot_list['Sheet1'][0])=='shot number').argmax() ])+'.ptw')
		except:
			pass




	fig, ax = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
	fig1, ax1 = plt.subplots( 2,3,figsize=(22, 25), squeeze=False,sharex=True)
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
			i_day,day = 0,str(date.date())
			laser_to_analyse=path+day+'/'+name

			full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
			full_saved_file_dict_FAST.allow_pickle=True
			full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
			full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
			try:
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

			greenwald_density = full_saved_file_dict_FAST['multi_instrument']['greenwald_density']
			ne_bar = full_saved_file_dict_FAST['multi_instrument']['ne_bar']

			if 'CD' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
				t_min = 0.35
			elif 'SF' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
				t_min = 0.4
			elif 'SXD' in full_saved_file_dict_FAST['multi_instrument']['scenario']:
				t_min = 0.5
			t_end = -4


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

			inner_L_poloidal_peak_only_leg_all[median_filter(inner_L_poloidal_peak_only_leg_all,size=3).argmax():] = np.nan
			outer_L_poloidal_peak_all[median_filter(outer_L_poloidal_peak_all,size=3).argmax():] = np.nan

			ax[0,0].plot(core_density[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax[0,1].plot(core_density[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax[0,2].plot(core_density[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax[1,1].plot(core_density[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax[1,2].plot(core_density[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)

			ax1[0,0].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax1[0,1].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_peak_only_leg_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4],alpha=0.5)
			ax1[0,2].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(inner_L_poloidal_baricentre_all/inner_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax1[1,1].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)
			ax1[1,2].plot((ne_bar/greenwald_density)[time_full_binned_crop>t_min][:t_end],(outer_L_poloidal_baricentre_all/outer_L_poloidal_x_point_all)[time_full_binned_crop>t_min][:t_end],markers[n_shot_added//len(color)],color=color[n_shot_added%len(color)],label=laser_to_analyse[-9:-4]+' '+full_saved_file_dict_FAST['multi_instrument']['scenario']+' '+full_saved_file_dict_FAST['multi_instrument']['experiment'],alpha=0.5)

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

			print('included '+str(laser_to_analyse[-9:-4]))
			n_shot_added +=1
		except:
			print('failed '+str(laser_to_analyse[-9:-4]))
			pass
	fig.suptitle('L poloidal peak or baricentre / L poloidal x-point\nDN-600-SXD-OH, shot number < 45366, Experiment Tags!=MU01-EXH-06 (large drsep)\nt>'+str(t_min*1e3)+'ms,t<t end-150ms')
	fig1.suptitle('L poloidal peak or baricentre / L poloidal x-point\nDN-600-SXD-OH, shot number < 45366, Experiment Tags!=MU01-EXH-06 (large drsep)\nt>'+str(t_min*1e3)+'ms,t<t end-150ms')
	ax[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[0,1].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[0,0].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	ax[0,1].set_ylabel('inner leg peak\nL poloidal/L poloidal x-point')
	ax[0,2].set_ylabel('inner separatrix baricentre\nL poloidal/L poloidal x-point')
	ax[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax[1,2].set_ylabel('outer leg baricentre\nL poloidal/L poloidal x-point')
	ax[1,1].set_xlabel('line averaged ne [#/m2]')
	ax[1,2].set_xlabel('line averaged ne [#/m2]')
	ax[0,0].grid()
	ax[0,1].grid()
	ax[0,1].grid()
	ax[0,2].grid()
	ax[1,1].grid()
	ax[1,2].grid()
	ax[0,0].axhline(y=1,linestyle='--',color='k')
	ax[0,1].axhline(y=1,linestyle='--',color='k')
	ax[0,2].axhline(y=1,linestyle='--',color='k')
	ax[1,1].axhline(y=1,linestyle='--',color='k')
	ax[1,2].axhline(y=1,linestyle='--',color='k')
	ax1[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax1[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
	ax1[0,0].set_ylabel('inner separatrix peak\nL poloidal/L poloidal x-point')
	ax1[0,1].set_ylabel('inner leg peak\nL poloidal/L poloidal x-point')
	ax1[0,2].set_ylabel('inner separatrix baricentre\nL poloidal/L poloidal x-point')
	ax1[1,1].set_ylabel('outer leg peak\nL poloidal/L poloidal x-point')
	ax1[1,2].set_ylabel('outer leg baricentre\nL poloidal/L poloidal x-point')
	ax1[0,0].grid()
	ax1[0,1].grid()
	ax1[0,2].grid()
	ax1[1,1].grid()
	ax1[1,2].grid()
	ax1[0,0].axhline(y=1,linestyle='--',color='k')
	ax1[0,2].axhline(y=1,linestyle='--',color='k')
	ax1[1,1].axhline(y=1,linestyle='--',color='k')
	ax1[1,2].axhline(y=1,linestyle='--',color='k')
	ax1[1,1].set_xlabel('greenwald fraction [au]')
	ax1[1,2].set_xlabel('greenwald fraction [au]')
	# ax1[1,1].set_xlim(right=0.27)
	# ax1[1,1].set_xlim(right=0.27)

	ax2[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax2[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
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
	ax2[0,0].axhline(y=1,linestyle='--',color='k')
	ax2[0,2].axhline(y=1,linestyle='--',color='k')
	ax2[1,1].axhline(y=1,linestyle='--',color='k')
	ax2[1,2].axhline(y=1,linestyle='--',color='k')
	ax2[1,1].set_xlabel('energy confinement time [s]')
	ax2[1,2].set_xlabel('energy confinement time [s]')

	ax3[0,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax3[1,2].legend(loc='best', fontsize='xx-small',ncol=2)
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

	ax4[0,0].set_xlabel('energy confinement/lmode ref [au]')
	ax4[0,0].set_ylabel('psiN peak radiaton inner sep [au]')

	plt.pause(0.01)



















	name = 'IRVB-MASTU_shot-45473.ptw'
	# name = 'IRVB-MASTU_shot-45401.ptw'
	temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
	for i in range(1,len(shot_list['Sheet1'])):
		if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
			date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
	i_day,day = 0,str(date.date())
	laser_to_analyse=path+day+'/'+name

	full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
	full_saved_file_dict_FAST.allow_pickle=True
	full_saved_file_dict_FAST = dict(full_saved_file_dict_FAST)
	full_saved_file_dict_FAST['multi_instrument'] = full_saved_file_dict_FAST['multi_instrument'].all()
	pass_number = 1
	full_saved_file_dict_FAST['second_pass'] = full_saved_file_dict_FAST['second_pass'].all()
	inverted_dict = full_saved_file_dict_FAST['second_pass']['inverted_dict']
	grid_resolution = 2	# cm
	time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']
	outer_local_mean_emis_all = inverted_dict[str(grid_resolution)]['outer_local_mean_emis_all']
	outer_local_L_poloidal_all = inverted_dict[str(grid_resolution)]['outer_local_L_poloidal_all']
	outer_L_poloidal_x_point_all = inverted_dict[str(grid_resolution)]['outer_L_poloidal_x_point_all']
	inner_local_mean_emis_all = inverted_dict[str(grid_resolution)]['inner_local_mean_emis_all']
	inner_local_L_poloidal_all = inverted_dict[str(grid_resolution)]['inner_local_L_poloidal_all']
	inner_L_poloidal_x_point_all = inverted_dict[str(grid_resolution)]['inner_L_poloidal_x_point_all']

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
	jsat_lower_outer_mid_max = full_saved_file_dict_FAST['multi_instrument']['jsat_lower_outer_mid_max']
	jsat_upper_outer_mid_max = full_saved_file_dict_FAST['multi_instrument']['jsat_upper_outer_mid_max']
	scenario = full_saved_file_dict_FAST['multi_instrument']['scenario']
	experiment = full_saved_file_dict_FAST['multi_instrument']['experiment']


	peak_emissivity_inner = []
	peak_emissivity_outer = []
	for i_time,time in enumerate(time_full_binned_crop):
		peak_emissivity_inner.append(inverted_data[i_time][np.abs(inner_emissivity_peak_all[i_time,0]-inversion_R).argmin(),np.abs(inner_emissivity_peak_all[i_time,1]-inversion_Z).argmin()])
		peak_emissivity_outer.append(inverted_data[i_time][np.abs(outer_emissivity_peak_all[i_time,0]-inversion_R).argmin(),np.abs(outer_emissivity_peak_all[i_time,1]-inversion_Z).argmin()])

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


if False:
	fig, ax = plt.subplots( 2,2,figsize=(18, 10), squeeze=False,sharex=True)
	fig.suptitle('shot '+str(name[-9:-4])+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
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
	ax[0,1].plot(jsat_time,jsat_lower_outer_mid_max,'-',label='lower outer leg',color=color[1])
	ax[0,1].plot(jsat_time,jsat_lower_outer_mid_max,'+',color=color[1])
	ax[0,1].plot(jsat_time,jsat_upper_outer_mid_max,'-',label='upper outer leg',color=color[3])
	ax[0,1].plot(jsat_time,jsat_upper_outer_mid_max,'+',color=color[3])
	# temp = np.nanmax([median_filter(jsat_lower_inner_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_inner_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_inner_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11) , median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_outer_small_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_outer_mid_max,size=11),median_filter(jsat_lower_outer_large_max,size=11),median_filter(jsat_lower_outer_large_max,size=11)],axis=0)
	# ax[5,0].set_ylim(bottom=0,top=np.nanmax(temp[np.logical_and(jsat_time>0,jsat_time<time_full_binned_crop[-2])]))
	ax[0,1].grid()
	ax[0,1].set_ylabel('jsat max [A/m2]')
	ax[0,1].legend(loc='best', fontsize='small',ncol=2)
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
elif False:	#this is the correct approach
	select_time = time_full_binned_crop>0.35
	select_time[-4:] = False
	fig, ax = plt.subplots( 2,1,figsize=(10, 10), squeeze=False,sharex=True)
	fig.suptitle('shot '+str(name[-9:-4])+', '+scenario+' , '+experiment+'\nsecond pass, '+binning_type+', grid resolution '+str(grid_resolution)+'cm')
	ax[0,0].plot((ne_bar/greenwald_density)[select_time],(inner_L_poloidal_peak_all/inner_L_poloidal_x_point_all)[select_time],'r-',label='inner separatrix')
	ax[0,0].plot((ne_bar/greenwald_density)[select_time],(outer_L_poloidal_peak_all/outer_L_poloidal_x_point_all)[select_time],'b-',label='outer separatrix')
	ax[0,0].axhline(y=1,linestyle='--',color='k')
	# ax[0,0].set_ylim(bottom=-0.1,top=4)
	ax[0,0].grid()
	ax[0,0].set_ylabel('Lpoloidal/Lpol x-pt [au]')
	ax[0,0].legend(loc='best', fontsize='small')
	# ax[0,0].set_xlabel('greenwald fraction [ua]')
	ax[1,0].plot((ne_bar/greenwald_density)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,'-',label='lower outer leg',color=color[1])
	ax[1,0].plot((ne_bar/greenwald_density)[select_time],jsat_lower_outer_mid_integrated[select_time]/1.6e-19,'+',color=color[1])
	ax[1,0].plot((ne_bar/greenwald_density)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,'-',label='upper outer leg',color=color[3])
	ax[1,0].plot((ne_bar/greenwald_density)[select_time],jsat_upper_outer_mid_integrated[select_time]/1.6e-19,'+',color=color[3])
	ax[1,0].grid()
	ax[1,0].set_ylabel('target particle flux [#/s]')
	# ax[6,0].legend(loc='best', fontsize='xx-small',ncol=2)
	ax[1,0].set_xlabel('greenwald fraction [ua]')
	ax[1,0].legend(loc='best', fontsize='small',ncol=2)
	plt.savefig('/home/ffederic/work/irvb/0__outputs/'+os.path.split(laser_to_analyse[:-4])[1]+'_pass'+str(pass_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_all_variables_absolute_small.png')
	plt.close()


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

	data_dir = '/home/ffederic/work/Collaboratory/test/experimental_data/45295/C001H001S0001/'
	T0=-0.1	# for 45295

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

	time=np.abs(times-0.4).argmin()
	plt.figure()
	plt.imshow(data[time])
	plt.colorbar()
	plt.pause(0.01)


	plt.figure()
	plt.plot(times,np.mean(data,axis=(1,2)))
	plt.plot(times,np.mean(data[:,513-2:525+2,521-2:533+2],axis=(1,2)),'--')
	plt.grid()
	plt.pause(0.01)
