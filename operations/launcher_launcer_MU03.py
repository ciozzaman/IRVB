import os
import time as tm
import numpy as np
import subprocess

import tempfile	# added to have the shot number visible as the shot being analysed
import textwrap	# added to have the shot number visible as the shot being analysed

# this code is the launcher to analyse the live MASTU shots
max_number_of_MASTU_AutoProc = 6

# microscopic code tat loops indefinitely only to launch the launcer
last_pulse_path = '/home/ffederic/mudiagbkup_MU02/rbv/'
done_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
while True:
	if not os.path.exists(last_pulse_path):	# this is just to launch a freia job and restore the connection to the mackup location
		print('the connection to mudiagbkup_MU02 was lost')
		launcher = os.popen('llsubmit /home/ffederic/work/analysis_scripts/scripts/operations/command_trash.cmd')
		launcher.read()
		tm.sleep(60*1)	# seconds
		print(tm.strftime("%Y/%m/%d %H:%M:%S", tm.localtime()))
		print('shot analysis launched')
		print('should be good now?')

	# this is quite funny. if it doesn't work it launches itself!
	try:
		last_pulse_dict = dict(np.load(last_pulse_path+'last_pulse.npz'))
	except:
		print('launching a new launcher_launcher_ job')
		launcher = os.popen('llsubmit /home/ffederic/work/analysis_scripts/scripts/operations/command_launcher.cmd')
		launcher.read()
		print('done')
		tm.sleep(60*1)	# seconds
		exit()

	# checking the already processed pulses
	try:
		done_pulse_dict = dict(np.load(done_pulse_path+'done_pulse.npz'))
		done_pulse_dict['location'] = list(done_pulse_dict['location'])
		done_pulse_dict['filename'] = list(done_pulse_dict['filename'])
	except:
		done_pulse_dict = dict([])
		done_pulse_dict['location'] = ['null']
		done_pulse_dict['filename'] = ['null']
		np.savez_compressed(done_pulse_path+'done_pulse',**done_pulse_dict)

	# figuring out what the latest puse is
	selected_index = 0
	all_days = last_pulse_dict['location']
	for i in range(1,len(last_pulse_dict['location'])):
		if not (all_days[-i] in done_pulse_dict['location']):
			selected_index = -i
			shot_to_do = all_days[-i]
			break

	date_to_do = [i for i, c in enumerate(shot_to_do) if c == "_"]
	date_to_do = shot_to_do[date_to_do[0]+1:date_to_do[1]]

	if selected_index == 0:
		print('no pulse to analyse')
	else:
		result = subprocess.run(['bash', '/home/ffederic/work/analysis_scripts/scripts/operations/detect_list_of_processes.sh'], stdout=subprocess.PIPE, text=True)
		temp = [f for f in result.stdout.splitlines() if 'Auto_' in f ]
		print(str(len(temp)) + ' auto processes running')
		if len(temp)>=max_number_of_MASTU_AutoProc:
			print('paused, too many preliminary analysis running')
			tm.sleep(60*0)	# seconds
		elif len(temp)==(max_number_of_MASTU_AutoProc-1) and date_to_do!=tm.strftime("%Y%m%d"):
			print('paused, I could have 1 more proc running, but I want to keep the quese free from a process from today')
			tm.sleep(60*0)	# seconds
		else:
			if False:
				launcher = os.popen('llsubmit /home/ffederic/work/analysis_scripts/scripts/operations/command.cmd')
			else:	# 2026-01-22 I want the shot number to be visible
				cmd_file = tempfile.NamedTemporaryFile(mode="w", delete=False,prefix="command_", suffix=".cmd",dir='/home/ffederic/work/analysis_scripts/scripts/operations')
				text = textwrap.dedent(f"""
				# @ executable = python3.9
				# @ arguments = /home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process_launcer_MU03_3.py {shot_to_do}
				# @ output = /home/ffederic/work/analysis_scripts/scripts/MASTU_AutoProc_{shot_to_do[3:]}.out
				# @ error =  /home/ffederic/work/analysis_scripts/scripts/MASTU_AutoProc_{shot_to_do[3:]}.err
				# @ initialdir = /home/ffederic/work/analysis_scripts/scripts/
				# @ jobtype = std
				# @ max_processors = 1
				# @ min_processors = 1
				# @ job_name = Auto_{shot_to_do[3:]}
				# @ notify_user = ffederic
				# @ notification = complete
				# @ queue
				""")
				text = text[1:]
				text = text.replace('\n\n','\n')
				cmd_file.write(text)
				cmd_file.close()

				launcher = os.popen('llsubmit '+cmd_file.name)

			launcher.read()
			print(tm.strftime("%Y/%m/%d %H:%M:%S", tm.localtime()))
			tm.sleep(60*0)	# seconds
			if False:
				print('shot analysis launched')
			else:	# 2026-01-22 I want the shot number to be visible
				print('analysis '+'Auto_'+shot_to_do[3:]+' launched')
				os.remove(cmd_file.name)


	tm.sleep(60*1)	# seconds
