# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8
import sys	# 2026-01-22 I want the shot number to be visible
shot_to_do = sys.argv[1]
shot_to_do = shot_to_do.replace(":", ":\\")
print("Running job with suffix:", shot_to_do)

if False:
	# # I want to stop warnings to clog my error file
	# import warnings
	# warnings.filterwarnings("ignore")
	try:
		import argparse
		parser = argparse.ArgumentParser(
		prog='ProgramName',
		description='What the program does',
		epilog='Text at the bottom of help')
		parser.add_argument('ext_sequencer',type=int)
		parser.add_argument('MU_campaign',type=int)
		parser.add_argument('ext_number_of_passes',type=int)
		# parser.add_argument('--debug',action='store_true')

		ext_sequencer = vars(parser.parse_args())['ext_sequencer']
		MU_campaign = vars(parser.parse_args())['MU_campaign']
		ext_number_of_passes = vars(parser.parse_args())['ext_number_of_passes']
		# print(ext_sequencer)
		print('running sequential shot number '+str(ext_sequencer))

		# note
		# to use these feature the typical command to launch is this
		# qsub -V -cwd -o /home/ffederic/work/analysis_scripts/scripts/MASTU_prelim_data_examination3.out -e /home/ffederic/work/analysis_scripts/scripts/MASTU_prelim_data_examination3.err -N MASTU_prelim_data_examination3 ./command_single_core.sh 0 1000000 3 1
	except:
		# print('error: ' + str(e))
		print('no sequential ID provided')
		ext_sequencer = None
		MU_campaign = 0
		ext_number_of_passes = 0
	# exit()

	print('ext_sequencer')
	print(ext_sequencer)
	print('MU_campaign')
	print(MU_campaign)
	print('ext_number_of_passes')
	print(ext_number_of_passes)

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

# added to reat the .ptw
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
# import pyradi.ryptw as ryptw

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
# pathparams_BB='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
pathparams_BB='/home/ffederic/work/irvb/2022-12-07_multiple_search_for_parameters'	# original NUC experiments, calibration with wavewlength_top=5.1,wavelength_bottom=1.5
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

pathparams_BB_X6980='/home/ffederic/work/irvb/2024-10-28_FLIRX6980'	# new BB source before the camera broke, calibration with wavewlength_top=5.1,wavelength_bottom=1.5
f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams_BB_X6980):
	f.append(dirnames)
parameters_available_BB_X6980 = f[0]
parameters_available_int_time_BB_X6980 = []
parameters_available_framerate_BB_X6980 = []
for path in parameters_available_BB_X6980:
	parameters_available_int_time_BB_X6980.append(float(path[:path.find('ms')]))
	parameters_available_framerate_BB_X6980.append(float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time_BB_X6980 = np.array(parameters_available_int_time_BB_X6980)
parameters_available_framerate_BB_X6980 = np.array(parameters_available_framerate_BB_X6980)

pathparams_BB_X6980_2='/home/ffederic/work/irvb/2025-04-27_FLIRX6980'	# new BB source after the camera broke, calibration with wavewlength_top=5.1,wavelength_bottom=1.5
f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams_BB_X6980_2):
	f.append(dirnames)
parameters_available_BB_X6980_2 = f[0]
parameters_available_int_time_BB_X6980_2 = []
parameters_available_framerate_BB_X6980_2 = []
for path in parameters_available_BB_X6980_2:
	parameters_available_int_time_BB_X6980_2.append(float(path[:path.find('ms')]))
	parameters_available_framerate_BB_X6980_2.append(float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time_BB_X6980_2 = np.array(parameters_available_int_time_BB_X6980_2)
parameters_available_framerate_BB_X6980_2 = np.array(parameters_available_framerate_BB_X6980_2)

pathparams_BB_X6980_3='/home/ffederic/work/irvb/2026-01-25_FLIRX6980'	# new BB source after the camera broke, calibration with wavewlength_top=5.1,wavelength_bottom=1.5
f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams_BB_X6980_3):
	f.append(dirnames)
parameters_available_BB_X6980_3 = f[0]
parameters_available_int_time_BB_X6980_3 = []
parameters_available_framerate_BB_X6980_3 = []
for path in parameters_available_BB_X6980_3:
	parameters_available_int_time_BB_X6980_3.append(float(path[:path.find('ms')]))
	parameters_available_framerate_BB_X6980_3.append(float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time_BB_X6980_3 = np.array(parameters_available_int_time_BB_X6980_3)
parameters_available_framerate_BB_X6980_3 = np.array(parameters_available_framerate_BB_X6980_3)

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']

path = '/home/ffederic/work/irvb/MAST-U/'



seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s


# here is the section that retrives which file has to be processed
# last_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
# last_pulse_dict = dict(np.load(last_pulse_path+'done_pulse.npz'))
last_pulse_path = '/home/ffederic/mudiagbkup_MU02/rbv/'
last_pulse_dict = dict(np.load(last_pulse_path+'last_pulse.npz'))

done_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
try:
	done_pulse_dict = dict(np.load(done_pulse_path+'done_pulse.npz'))
	done_pulse_dict['location'] = list(done_pulse_dict['location'])
	done_pulse_dict['filename'] = list(done_pulse_dict['filename'])
except Exception as e:
	if False: #	this exact thing is already done in the previous loop. why do i t again? I risk doing more damage
		done_pulse_dict = dict([])
		done_pulse_dict['location'] = ['null']
		done_pulse_dict['filename'] = ['null']
		np.savez_compressed(done_pulse_path+'done_pulse',**done_pulse_dict)
	else:
		print('there was an error in reading '+done_pulse_path+'done_pulse.npz')
		print(e)
		print('process terminated')
		exit()

if False: # with this cycle i select the last shot that happened but is not yet processed and i go backwards
	selected_index=0
	all_days = last_pulse_dict['location']
	for i in range(1,len(done_pulse_dict['location'])+1):
		if not (all_days[-i] in done_pulse_dict['location']):
			selected_index = -i
			break
else:	# the shot to do was already assigned by the job launcher
	selected_index = (last_pulse_dict['location'] == shot_to_do).argmax()


if selected_index == 0:
	print('no pulse to analyse')
	exit()
else:
	done_pulse_dict['location'].append(last_pulse_dict['location'][selected_index])
	done_pulse_dict['filename'].append(last_pulse_dict['filename'][selected_index])
	np.savez_compressed(done_pulse_path+'done_pulse',**done_pulse_dict)

full_day = last_pulse_dict['location'][selected_index]
full_day = full_day[full_day.find('\\')+1:]
day = str(last_pulse_dict['location'][selected_index])
day = day[day.find('_')+1:]
day = day[:day.find('_')]
day = day[:4] + '-' + day[4:6] + '-' + day[6:]
name = str(last_pulse_dict['filename'][selected_index])

# to copy the files
import shutil

if not os.path.exists(done_pulse_path + day):
	os.mkdir(done_pulse_path + day)
if not os.path.isfile(done_pulse_path + day + '/' + name):
	shutil.copyfile(last_pulse_path + full_day + '/' + name, done_pulse_path + day + '/' + name)
	print('copied '+done_pulse_path + day + '/' + name)
else:
	if os.path.exists(path + 'FAST_results' + '/' + name[-9:-4]):
		files_in_folder = [f for f in os.listdir(path + 'FAST_results' + '/' + name[-9:-4]) if name[:-4]+'_pass0_' in f ]
		if len(files_in_folder)>0:
			print('This shot was already processed')
			exit()
# here we go back to the normal code



continue_after_FAST = False
override_FAST_analysis = True
do_inversions = False

every_pixel_independent = False
overwrite_oscillation_filter = True
overwrite_binning = True
skip_second_pass = True
override_second_pass = False
skip_third_pass = False
override_third_pass = False

if overwrite_oscillation_filter:
	overwrite_binning = True

only_plot_brightness = False


path = '/home/ffederic/work/irvb/MAST-U/'
# shot_list = get_data(path+'shot_list2.ods')
# temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
# for i in range(1,len(shot_list['Sheet1'])):
# 	if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
# 		date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
# 		break
# # i_day,day = 0,str(date.date())
# # i_day,day = 0,'2022-12-08'
# try:
# 	i_day,day = 0,str(date.date())
# except:
# 	i_day,day = 0,str(date)
i_day,day = 0,coleval.retrive_shot_date_and_time(name[-9:-4])[0]
print(name)
print(path+day)

to_do = name
laser_to_analyse=path+day+'/'+name

exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())

print('all done')
