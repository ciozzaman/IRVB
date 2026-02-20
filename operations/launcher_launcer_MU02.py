import os
import time as tm
import numpy as np

# microscopit code tat loops indefinitely only to launch the launcer
while True:
	last_pulse_path = '/home/ffederic/mudiagbkup_MU02/rbv/'
	last_pulse_dict = dict(np.load(last_pulse_path+'last_pulse.npz'))

	done_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
	try:
		done_pulse_dict = dict(np.load(done_pulse_path+'done_pulse.npz'))
		done_pulse_dict['location'] = list(done_pulse_dict['location'])
		done_pulse_dict['filename'] = list(done_pulse_dict['filename'])
	except:
		done_pulse_dict = dict([])
		done_pulse_dict['location'] = ['null']
		done_pulse_dict['filename'] = ['null']
		np.savez_compressed(done_pulse_path+'done_pulse',**done_pulse_dict)

	selected_index = 0
	all_days = last_pulse_dict['location']
	for i in range(len(all_days)):
		if done_pulse_dict['location'][-1] == all_days[i]:
			selected_index = i+1

	if selected_index == len(all_days):
		print('no pulse to analyse')
	else:
		launcher = os.popen('llsubmit /home/ffederic/work/analysis_scripts/scripts/operations/command.cmd')
		launcher.read()
		print('shot analysis launched')

	tm.sleep(60*2)	# seconds
