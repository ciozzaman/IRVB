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
fuelling_location_list_orig = shot_list['Sheet1'][1][20:67]
fuelling_location_list = [string.replace('fuelling ',fuelling_prefix) for string in fuelling_location_list_orig]


if True:
	for i in range(2,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[np.array(shot_list['Sheet1'][1]) == 'shot number'])
		for i_signal_name,signal_name in enumerate(fuelling_location_list):
			# try:
			data = client.get(signal_name,shot)
			print(np.nanmax(np.abs(data.data)))
			if np.nanmax(np.abs(data.data))>0:
				print('set')
					shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][1]) == fuelling_location_list_orig[i_signal_name]).argmax()] = 'X'
			except:
				pass

	beams_prefix = '/XNB/'
	beams_affix = '/BEAMPOWER'
	for i in range(2,len(shot_list['Sheet1'])):
		shot = int(np.array(shot_list['Sheet1'][i])[np.array(shot_list['Sheet1'][1]) == 'shot number'])
		signal_name = beams_prefix + 'SW' + beams_affix
		try:
			data = client.get(signal_name,shot)
			if np.nanmax(np.abs(data.data))>0:
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][1]) == 'SW beam').argmax()] = 'X'
		except:
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][1]) == 'SW beam').argmax()] = ''
			pass
		try:
			signal_name = beams_prefix + 'SS' + beams_affix
			data = client.get(signal_name,shot)
			if np.nanmax(np.abs(data.data))>0:
				shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][1]) == 'SS beam').argmax()] = 'X'
		except:
			shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][1]) == 'SS beam').argmax()] = ''
			pass

	save_data(path+'shot_list2.ods',shot_list)
	print('done')
	exit()
else:
	pass

###############################################
# once the upper section is done this can be used for plots



shot=44607

plt.figure(figsize=(15,12))
shot_array = np.array(shot_list['Sheet1'])
i = (shot_array[:,2].astype(str) == str(shot)).argmax()
for i_signal_name,signal_name in enumerate(fuelling_location_list):
	if shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][1]) == fuelling_location_list_orig[i_signal_name]).argmax()] == 'X':
		data = client.get(signal_name,shot)
		plt.plot(data.time.data,data.data,label=signal_name)
plt.xlabel('time [s]')
plt.ylabel('valve voltage [V]')
plt.title(str(shot))
plt.grid()
plt.legend(loc='best', fontsize='x-small')
plt.pause(0.01)
