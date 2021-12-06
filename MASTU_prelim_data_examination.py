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
pathparams='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
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


path = '/home/ffederic/work/irvb/MAST-U/'
# to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25','2021-06-29','2021-06-30','2021-07-01','2021-07-06','2021-07-08','2021-07-09','2021-07-15','2021-07-27','2021-07-28','2021-07-29','2021-08-04','2021-08-05','2021-08-06','2021-08-11','2021-08-12','2021-08-13','2021-08-17','2021-08-18','2021-08-19','2021-08-20','2021-08-24','2021-08-25','2021-08-26','2021-08-27','2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04','2021-10-05','2021-10-06','2021-10-07','2021-10-08','2021-10-11','2021-10-12','2021-10-13','2021-10-14','2021-10-15','2021-10-19','2021-10-20','2021-10-21','2021-10-22']
# to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22']
# to_do = ['2021-09-08','2021-06-17','2021-08-05']
# to_do = ['2021-10-15','2021-10-12','2021-10-01','2021-09-30']
to_do = ['2021-08-26']
# to_do = ['2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04']
to_do = np.flip(to_do,axis=0)
# to_do = ['2021-06-03']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']

seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s


f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.append(dirnames)
days_available = f[0]
shot_available = []
for i_day,day in enumerate(to_do):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path+day+'/'):
		f.append(filenames)
	shot_available.append([])
	for name in f[0]:
		if name[-3:]=='ats' or name[-3:]=='ptw':
			shot_available[i_day].append(name)


# to_do = ['2021-10-15','2021-10-12','2021-10-01','2021-09-30']
# shot_available = [['IRVB-MASTU_shot-45314.ptw','IRVB-MASTU_shot-45315.ptw'],['IRVB-MASTU_shot-45248.ptw'],['IRVB-MASTU_shot-45125.ptw'],['IRVB-MASTU_shot-45100.ptw','IRVB-MASTU_shot-45099.ptw']]
# to_do = ['2021-10-13','2021-10-22','2021-10-26']
# shot_available = [['IRVB-MASTU_shot-45272.ptw'],['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-45399.ptw'],['IRVB-MASTU_shot-45414.ptw','IRVB-MASTU_shot-45420.ptw','IRVB-MASTU_shot-45415.ptw','IRVB-MASTU_shot-45416.ptw']]
# to_do = ['2021-10-14','2021-10-21','2021-10-12','2021-09-28']
# shot_available = [['IRVB-MASTU_shot-45293.ptw','IRVB-MASTU_shot-45296.ptw'],['IRVB-MASTU_shot-45382.ptw'],['IRVB-MASTU_shot-45242.ptw'],['IRVB-MASTU_shot-45069.ptw']]

every_pixel_independent = False
continue_after_FAST = False
overwrite_oscillation_filter = True
if True:
	for i_day,day in enumerate(to_do):
	# for i_day,day in enumerate(np.flip(to_do,axis=0)):
		# for name in shot_available[i_day]:
		for name in np.flip(shot_available[i_day],axis=0):
			laser_to_analyse=path+day+'/'+name

			# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process.py").read())
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2.py").read())
			# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3.py").read())
else:
	# i_day,day = 0,'2021-07-29'
	# name='IRVB-MASTU_shot-44578.ptw'
	# i_day,day = 0,'2021-08-13'
	# name='IRVB-MASTU_shot-44677.ptw'
	# i_day,day = 0,'2021-06-24'
	# name='IRVB-MASTU_shot-44308.ptw'
	# i_day,day = 0,'2021-09-01'
	# name='IRVB-MASTU_shot-44863.ptw'
	# i_day,day = 0,'2021-09-28'
	# name='IRVB-MASTU_shot-45071.ptw'
	# i_day,day = 0,'2021-10-05'
	# name='IRVB-MASTU_shot-45156.ptw'
	i_day,day = 0,'2021-10-12'
	name='IRVB-MASTU_shot-45240.ptw'
	laser_to_analyse=path+day+'/'+name
	exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2.py").read())
	# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3.py").read())
