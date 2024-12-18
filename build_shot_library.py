# Created 01/12/2021
# Fabio Federici


from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe, probe_array, compare_shots
import pyuda
client=pyuda.Client()
# import matplotlib.pyplot as plt

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

from pyexcel_ods import save_data,get_data
path = '/home/ffederic/work/irvb/MAST-U/'

shot_list = get_data(path+'shot_list2.ods')
max_length = 0
for i in range(len(shot_list['Sheet1'])):
	max_length = max(max_length,len(shot_list['Sheet1'][i]))
for i in range(len(shot_list['Sheet1'])):
	shot_list['Sheet1'][i].extend(['']*(max_length-len(shot_list['Sheet1'][i])))

fuelling_prefix = '/XDC/GAS/T/'
fuelling_location_list_orig = shot_list['Sheet1'][0][25:72]
fuelling_location_list = [string.replace('fuelling ',fuelling_prefix) for string in fuelling_location_list_orig]


if True:
	shot_list = get_data(path+'shot_list2.ods')
	for i in range(1,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[np.array(shot_list['Sheet1'][0]) == 'shot number'])
		# if shot<48900:
		# 	continue
		print(shot)
		try:
			data_all = client.get_batch(fuelling_location_list,shot)
			for i_data,data in enumerate(data_all):
				if np.nanmax(data.data[data.time.data>0.2])>0:
					print(fuelling_location_list[i_data]+' set')
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == fuelling_location_list_orig[i_data]).argmax()] = 'X'
				else:
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == fuelling_location_list_orig[i_data]).argmax()] = ''
			continue
		except:
			pass
		for i_signal_name,signal_name in enumerate(fuelling_location_list):
			try:
				data = client.get(signal_name,shot,timefirst=0.2,time_last=False)
				if np.nanmax(data.data)>0:#[data.time.data>0.2])>0:
					print(signal_name+' set')
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == fuelling_location_list_orig[i_signal_name]).argmax()] = 'X'
				else:
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == fuelling_location_list_orig[i_signal_name]).argmax()] = ''
			except:
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == fuelling_location_list_orig[i_signal_name]).argmax()] = ''
				pass
	save_data(path+'shot_list2.ods',shot_list)
	print('done')


	shot_list = get_data(path+'shot_list2.ods')
	beams_prefix = '/XNB/'
	beams_affix = '/BEAMPOWER'
	for i in range(2,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[(np.array(shot_list['Sheet1'][0]) == 'shot number').argmax()])
		# if shot<48900:
		# 	continue
		try:
			data = client.get('/AMB/CTIME',shot)
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'SW beam').argmax()] = 'N'
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'SS beam').argmax()] = 'N'

			signal_name = beams_prefix + 'SW' + beams_affix
			try:
				data = client.get(signal_name,shot,timefirst=0.2,time_last=False)
				if np.nanmax(np.abs(data.data))>0:
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'SW beam').argmax()] = 'Y'
					print(shot+'SW')
			except:
				pass
			try:
				signal_name = beams_prefix + 'SS' + beams_affix
				data = client.get(signal_name,shot)
				if np.nanmax(np.abs(data.data))>0:
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'SS beam').argmax()] = 'Y'
					print(shot+'SS')
			except:
				pass
		except:
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'SS beam').argmax()] = ''
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'SW beam').argmax()] = ''

	save_data(path+'shot_list2.ods',shot_list)
	print('done')
	# exit()


	# I want to at least get the date automatically so I can run my analysis
	if False:	# ATTENTION, this DOES NOT work. afterwards you need to open the file and convert from text to date
		f = []
		for dirnames in os.listdir('/home/ffederic/work/irvb/MAST-U'):
			if os.path.isdir('/home/ffederic/work/irvb/MAST-U/'+dirnames) and len(dirnames)==10:
				f.append(dirnames)

		for dirname in f:
			for filename in os.listdir('/home/ffederic/work/irvb/MAST-U/'+dirname):
				if filename[:16] == 'IRVB-MASTU_shot-' and filename[-4:] == '.ptw':
					shot = filename[16:16+5]
					select = (np.array(shot_list['Sheet1'])[:,np.array(shot_list['Sheet1'][0]) == 'shot number'].astype(str) == shot).flatten().argmax()
					if shot_list['Sheet1'][select][(np.array(shot_list['Sheet1'][0]) == 'date').argmax()] == '':
						shot_list['Sheet1'][select][(np.array(shot_list['Sheet1'][0]) == 'date').argmax()] = dirname+' 00:00:00' # datetime(int(dirname[:4]),int(dirname[5:7]),int(dirname[8:]),00,00)
						print(shot)

	else:	# this actually works, saving the data as a text
		import pyuda
		client=pyuda.Client()
		from datetime import datetime
		shot_list = get_data(path+'shot_list2.ods')
		for i in range(2,len(shot_list['Sheet1'])):
			shot = int(np.array(shot_list['Sheet1'][i])[(np.array(shot_list['Sheet1'][0]) == 'shot number').argmax()])
			try:
				date_time = client.get_shot_date_time(shot)[0]+' '+client.get_shot_date_time(shot)[1][:8]
				# date_format = datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()] = date_time
				print(shot)
			except:
				pass
	save_data(path+'shot_list2.ods',shot_list)
	print('done')
	exit()

	# I want to add the peak greenwald fraction
	from mastu_exhaust_analysis.calc_ne_bar import calc_ne_bar
	for i in range(2,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[(np.array(shot_list['Sheet1'][0]) == 'shot number').argmax()])
		try:
			ne_bar_dict=calc_ne_bar(shot, efit_data = None)
			greenwald_fraction_max = np.nanmax(ne_bar_dict['greenwald_fraction'])
			if greenwald_fraction_max<10:
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Greenwald fraction max').argmax()] = str(np.round(greenwald_fraction_max,decimals=3))
				print(shot)
			else:
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Greenwald fraction max').argmax()] = 'too large'
		except:
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Greenwald fraction max').argmax()] = ''
			pass
	save_data(path+'shot_list2.ods',shot_list)
	print('done')
	exit()

	signal_name = '/EPM/OUTPUT/GLOBALPARAMETERS/MAGNETICAXIS/Z'
	for i in range(2,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[(np.array(shot_list['Sheet1'][0]) == 'shot number').argmax()])
		# if shot<48900:
		# 	continue
		try:
			data = client.get(signal_name,shot)
			# data = client.get(signal_name,shot,timefirst=0.2,time_last=False)
			if np.nanmin(data.data[-10:])<-0.5:	# arbitrary threshold of 0.5m
				if not (shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] in ['yes']):
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] = 'maybe'
					print(str(shot)+' Ends with lower VDE')
				else:
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] = 'no'
			elif shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] == '':
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] = 'no'
		except:
			if shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] != 'yes':
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Ends with lower VDE').argmax()] = 'no'
			pass
	save_data(path+'shot_list2.ods',shot_list)
	print('done')
	exit()

	signal_name = '/AMC/Plasma_current'
	for i in range(2,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[(np.array(shot_list['Sheet1'][0]) == 'shot number').argmax()])
		# if shot<48900:
		# 	continue
		try:
			data = client.get(signal_name,shot)
			# data = client.get(signal_name,shot,timefirst=0.2,time_last=False)
			if np.nanmax(data.data[:])<1:	# arbitrary threshold of 1kA to establish an aborted shot
				if not (shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Abort').argmax()] in ['yes','no']):
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Abort').argmax()] = 'effectively yes'
					print(str(shot)+' aborted')
		except:
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == 'Abort').argmax()] = 'yes'
			print(str(shot)+' aborted')
			pass
	save_data(path+'shot_list2.ods',shot_list)
	print('done')
	exit()



else:
	pass

###############################################
# once the upper section is done this can be used for plots

from pycpf import pycpf

shot=45460
tend=coleval.get_tend(shot) +0.1
fig, ax = plt.subplots( 4,1,figsize=(12, 20), squeeze=False, sharex=True)
# plt.figure(figsize=(15,12))
shot_array = np.array(shot_list['Sheet1'])
i = (shot_array[:,2].astype(str) == str(shot)).argmax()
for i_signal_name,signal_name in enumerate(fuelling_location_list):
	if shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0]) == fuelling_location_list_orig[i_signal_name]).argmax()] == 'X':
		data = client.get(signal_name,shot)
		ax[0,0].plot(data.time.data,data.data,label=signal_name)
ax[0,0].set_ylabel('valve voltage [V]')
ax[0,0].set_title(str(shot))
ax[0,0].grid()
ax[0,0].legend(loc='best', fontsize='x-small')
ax[0,0].set_xlim(left=-0.1,right=tend)
signal_name = '/ANE/DENSITY'
data = client.get(signal_name,shot)
ax[1,0].plot(data.time.data,data.data,label=signal_name)
ax[1,0].set_ylabel('core line int ne [m-2]')
ax[1,0].grid()
ax[1,0].set_xlim(left=-0.1,right=tend)
signal_name = '/XIM/DA/HM10/T'
data = client.get(signal_name,shot)
ax[2,0].plot(data.time.data,data.data,label=signal_name)
ax[2,0].set_yscale('log')
ax[2,0].set_ylabel('core Da [V]')
# ax[2,0].set_xlabel('time [s]')
ax[2,0].grid()
ax[2,0].set_xlim(left=-0.1,right=tend)
try:
	path = '/home/ffederic/work/irvb/MAST-U/'
	day = client.get_shot_date_time(shot)[0]
	name='IRVB-MASTU_shot-'+str(shot)+'.ptw'
	laser_to_analyse=path+day+'/'+name
	full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST.npz')
	full_saved_file_dict_FAST.allow_pickle=True
	inverted_dict = full_saved_file_dict_FAST['first_pass'].all()['inverted_dict']
	grid_resolution = 2
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
	time_full_binned_crop = inverted_dict[str(grid_resolution)]['time_full_binned_crop']

	ax[3,0].errorbar(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,yerr=outer_leg_tot_rad_power_sigma_all/1e3,label='outer_leg')
	ax[3,0].errorbar(time_full_binned_crop,sxd_tot_rad_power_all/1e3,yerr=sxd_tot_rad_power_sigma_all/1e3,label='sxd')
	ax[3,0].errorbar(time_full_binned_crop,inner_leg_tot_rad_power_all/1e3,yerr=inner_leg_tot_rad_power_sigma_all/1e3,label='inner_leg')
	ax[3,0].errorbar(time_full_binned_crop,core_tot_rad_power_all/1e3,yerr=core_tot_rad_power_sigma_all/1e3,label='core')
	ax[3,0].errorbar(time_full_binned_crop,x_point_tot_rad_power_all/1e3,yerr=x_point_tot_rad_power_sigma_all/1e3,label='x_point')
	ax[3,0].errorbar(time_full_binned_crop,(outer_leg_tot_rad_power_all+inner_leg_tot_rad_power_all+core_tot_rad_power_all)/1e3,yerr=((outer_leg_tot_rad_power_sigma_all**2+inner_leg_tot_rad_power_sigma_all**2+core_tot_rad_power_sigma_all**2)**0.5)/1e3,label='tot')
	ax[3,0].legend(loc='best', fontsize='x-small')
	ax[3,0].set_xlabel('time [s]')
	ax[3,0].set_ylabel('power [kW]')
	ax[3,0].grid()
	ax[3,0].set_xlim(left=-0.1,right=tend)
except:
	print('read inverted data failed')
	pass
ax[3,0].set_title('radiated power in the lower half of the machine')
plt.savefig(laser_to_analyse[:-4]+'_manual_plot.eps', bbox_inches='tight')
plt.close()

plt.pause(0.01)
