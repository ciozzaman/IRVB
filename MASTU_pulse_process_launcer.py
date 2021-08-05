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
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive','blueviolet','tan','skyblue','brown','dimgray','hotpink']
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


seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s

every_pixel_independent = False


last_pulse_path = '/home/ffederic/work/irvb/MAST-U/'
last_pulse_dict = dict(np.load(last_pulse_path+'last_pulse.npz'))

day = str(last_pulse_dict['location'][0])
if len(last_pulse_dict['location'])>1:
	last_pulse_dict['location'] = last_pulse_dict['location'][1:]
elif len(last_pulse_dict['location'])==0:
	print('no pulse to analyse')
	exit()
else:
	last_pulse_dict['location'] = []
name = str(last_pulse_dict['filename'][0])
if len(last_pulse_dict['filename'])>1:
	last_pulse_dict['filename'] = last_pulse_dict['filename'][1:]
else:
	last_pulse_dict['filename'] = []
np.savez_compressed(last_pulse_path+'last_pulse',**last_pulse_dict)

laser_to_analyse=last_pulse_path+day+'/'+name
path = last_pulse_path

# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process.py").read())
exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2.py").read())
