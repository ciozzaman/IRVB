import os
import time as tm
import numpy as np
import subprocess

# this code is the launcher to reprocess old shots (?)

# microscopic code tat loops indefinitely only to launch the launcer
last_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
done_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
while True:

	# this is quite funny. if it doesn't work it launches itself!
	try:
		last_pulse_dict = dict(np.load(last_pulse_path+'done_pulse.npz'))
	except:
		launcher = os.popen('llsubmit /home/ffederic/work/analysis_scripts/scripts/operations/command_launcher_2.cmd')
		launcher.read()
		tm.sleep(60*1)	# seconds
		exit()

	try:
		done_pulse_dict = dict(np.load(done_pulse_path+'really_done_pulse.npz'))
		done_pulse_dict['location'] = list(done_pulse_dict['location'])
		done_pulse_dict['filename'] = list(done_pulse_dict['filename'])
	except:
		done_pulse_dict = dict([])
		done_pulse_dict['location'] = ['null']
		done_pulse_dict['filename'] = ['null']
		np.savez_compressed(done_pulse_path+'really_done_pulse',**done_pulse_dict)

	selected_index = 0
	all_days = last_pulse_dict['location']
	for i in range(1,len(done_pulse_dict['location'])+1):
		if not (all_days[-i] in done_pulse_dict['location']):
			selected_index = -i
			break

	if selected_index == 0:
		print('no pulse to analyse')
	else:
		result = subprocess.run(['bash', '/home/ffederic/work/analysis_scripts/scripts/operations/detect_list_of_processes.sh'], stdout=subprocess.PIPE, text=True)
		temp = [f for f in result.stdout.splitlines() if 'MASTU_2_AutoProc' in f ]
		if len(temp)>2:
			print('paused, too many second preliminary analysis running')
			tm.sleep(60*0)	# seconds
		else:
			launcher = os.popen('llsubmit /home/ffederic/work/analysis_scripts/scripts/operations/command_2.cmd')
			launcher.read()
			print(tm.strftime("%Y/%m/%d %H:%M:%S", tm.localtime()))
			print('shot analysis launched')
			tm.sleep(60*0)	# seconds

	tm.sleep(60*2)	# seconds
