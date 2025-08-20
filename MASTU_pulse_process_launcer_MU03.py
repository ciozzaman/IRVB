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
# pathparams_BB='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
pathparams_BB='/home/ffederic/work/irvb/2022-12-07_multiple_search_for_parameters'
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

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']

path = '/home/ffederic/work/irvb/MAST-U/'


seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s

every_pixel_independent = False


# last_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
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

# with this cycle i select the last shot that happened but is not yet processed and i go backwards
selected_index=0
all_days = last_pulse_dict['location']
for i in range(1,len(done_pulse_dict['location'])+1):
	if not (all_days[-i] in done_pulse_dict['location']):
		selected_index = -i
		break

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
else:
	if os.path.exists(path + 'FAST_results' + '/' + name[-9:-4]):
		files_in_folder = [f for f in os.listdir(path + 'FAST_results' + '/' + name[-9:-4]) if name[:-4]+'_pass0_' in f ]
		if len(files_in_folder)>0:
			print('This shot was already processed')
			exit()


# laser_to_analyse=done_pulse_path+day+'/'+name

overwrite_oscillation_filter = False

# only for now because there is not enough room in FREIA
continue_after_FAST = False
override_FAST_analysis = True
do_inversions = False

every_pixel_independent = False
overwrite_oscillation_filter = True
overwrite_binning = True
skip_second_pass = True
override_second_pass = True
skip_third_pass = True
override_third_pass = False

if overwrite_oscillation_filter:
	overwrite_binning = True

only_plot_brightness = True

# i_day = 0
i_day,day = 0,coleval.retrive_shot_date_and_time(name[-9:-4])[0]
laser_to_analyse=path+day+'/'+name
to_do = [name]

# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process.py").read())
exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
