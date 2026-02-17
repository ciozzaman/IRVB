# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

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
if MU_campaign==1:
	# do all of them from MU01
	# to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25','2021-06-29','2021-06-30','2021-07-01','2021-07-06','2021-07-08','2021-07-09','2021-07-15','2021-07-27','2021-07-28','2021-07-29','2021-08-04','2021-08-05','2021-08-06','2021-08-11','2021-08-12','2021-08-13','2021-08-17','2021-08-18','2021-08-19','2021-08-20','2021-08-24','2021-08-25','2021-08-26','2021-08-27','2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04','2021-10-05','2021-10-06','2021-10-07','2021-10-08','2021-10-11','2021-10-12','2021-10-13','2021-10-14','2021-10-15','2021-10-19','2021-10-20','2021-10-21','2021-10-22','2021-10-25','2021-10-26','2021-10-27','2021-10-28']
	# to save memory I won't analyse the senario developement and commissioning shots
	to_do = ['2021-07-27','2021-07-28','2021-07-29','2021-08-04','2021-08-05','2021-08-06','2021-08-11','2021-08-12','2021-08-13','2021-08-17','2021-08-18','2021-08-19','2021-08-20','2021-08-24','2021-08-25','2021-08-26','2021-08-27','2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04','2021-10-05','2021-10-06','2021-10-07','2021-10-08','2021-10-11','2021-10-12','2021-10-13','2021-10-14','2021-10-15','2021-10-19','2021-10-20','2021-10-21','2021-10-22','2021-10-25','2021-10-26','2021-10-27','2021-10-28']
elif MU_campaign==2:
	# # # do all of them from MU02
	# to_do = ['2022-10-26', '2022-10-27', '2022-10-28', '2022-10-31', '2022-11-01', '2022-11-02', '2022-11-03', '2022-11-04', '2022-11-07', '2022-11-08', '2022-11-09', '2022-11-10', '2022-11-11', '2022-11-17', '2022-11-18', '2022-11-22', '2022-11-23', '2022-11-24', '2022-11-25', '2022-11-28', '2022-11-29', '2022-11-30', '2022-12-01', '2022-12-02', '2022-12-05', '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09', '2022-12-12', '2022-12-13', '2022-12-14', '2022-12-15', '2022-12-16', '2022-12-20', '2022-12-21', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-21', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-31', '2023-01-30']
	# to save memory I won't analyse the senario developement and commissioning shots
	to_do = ['2022-11-22', '2022-11-23', '2022-11-24', '2022-11-25', '2022-11-28', '2022-11-29', '2022-11-30', '2022-12-01', '2022-12-02', '2022-12-05', '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09', '2022-12-12', '2022-12-13', '2022-12-14', '2022-12-15', '2022-12-16', '2022-12-20', '2022-12-21', '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-21', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-31', '2023-01-30']
elif MU_campaign==3:
	# # do all of them from MU03
	# to_do = ['2023-07-04','2023-07-05','2023-07-06','2023-07-07','2023-07-11','2023-07-12','2023-07-13','2023-07-18','2023-07-19','2023-07-20','2023-07-21','2023-07-25','2023-07-26','2023-07-27','2023-07-28','2023-08-01','2023-08-02','2023-08-03','2023-08-04','2023-08-08','2023-08-09','2023-08-10','2023-08-11','2023-08-15','2023-08-16','2023-08-17','2023-08-18','2023-08-22','2023-08-23','2023-08-24','2023-09-13','2023-09-14','2023-09-15','2023-10-24','2023-10-25','2023-10-26','2023-10-27','2023-10-31','2023-11-01','2023-11-02','2023-11-03','2023-11-07','2023-11-08','2023-11-09','2023-11-10','2023-11-14','2023-11-15','2023-11-16','2023-11-17','2023-11-23','2023-11-24','2023-12-05','2023-12-06','2023-12-07','2023-12-08','2023-12-12','2023-12-13','2023-12-14','2023-12-15','2023-12-18','2023-12-19','2023-12-20','2024-01-09','2024-01-10','2024-01-11','2024-01-12','2024-01-15']
	# to save memory I won't analyse the senario developement and commissioning shots
	to_do = ['2023-07-18','2023-07-19','2023-07-20','2023-07-21','2023-07-25','2023-07-26','2023-07-27','2023-07-28','2023-08-01','2023-08-02','2023-08-03','2023-08-04','2023-08-08','2023-08-09','2023-08-10','2023-08-11','2023-08-15','2023-08-16','2023-08-17','2023-08-18','2023-08-22','2023-08-23','2023-08-24','2023-09-13','2023-09-14','2023-09-15','2023-10-24','2023-10-25','2023-10-26','2023-10-27','2023-10-31','2023-11-01','2023-11-02','2023-11-03','2023-11-07','2023-11-08','2023-11-09','2023-11-10','2023-11-14','2023-11-15','2023-11-16','2023-11-17','2023-11-23','2023-11-24','2023-12-05','2023-12-06','2023-12-07','2023-12-08','2023-12-12','2023-12-13','2023-12-14','2023-12-15','2023-12-18','2023-12-19','2023-12-20','2024-01-09','2024-01-10','2024-01-11','2024-01-12','2024-01-15','2024-01-16','2024-01-17','2024-01-18','2024-01-19','2024-01-22','2024-01-23','2024-01-24','2024-01-25']
	to_do = np.flip(to_do,axis=0)
elif MU_campaign==4:
	# # do all of them from MU04
	# to_do = ['2024-06-18','2024-06-19','2024-06-20','2024-06-21','2024-06-24','2024-06-25','2024-06-26','2024-06-27','2024-07-01','2024-07-02','2024-07-03','2024-07-04','2024-07-08','2024-07-09','2024-07-10','2024-07-11','2024-07-12','2024-07-15','2024-07-16','2024-07-17','2024-07-22','2024-07-23','2024-07-24','2024-07-25','2024-07-26','2024-07-30','2024-07-31','2024-08-01','2024-08-02','2024-08-05','2024-08-06','2024-08-07','2024-08-08','2024-08-09','2024-08-13','2024-08-14','2024-08-15','2024-08-16','2024-08-20','2024-08-21','2024-08-22','2024-09-05','2024-09-06','2024-09-13','2024-09-19','2024-09-20','2024-09-23','2024-09-24','2024-09-25','2024-09-26','2024-09-27','2024-10-01','2024-10-02','2024-10-03','2024-10-04','2024-10-08','2024-10-10','2024-10-11','2024-10-14','2024-10-15','2024-10-16','2024-10-17','2024-10-18','2024-10-21','2024-10-22','2024-10-23','2024-10-24','2024-10-29','2024-10-30','2024-10-31','2024-11-01','2024-11-04','2024-11-05','2024-11-06','2024-11-07','2024-11-08','2024-11-12','2024-11-13','2024-11-14','2024-11-15','2024-11-18','2024-11-19','2024-11-20','2024-11-21','2024-11-22','2024-11-25','2024-11-26','2024-11-27','2024-11-28','2024-11-29','2024-12-03','2024-12-04','2024-12-05','2024-12-06','2024-12-10','2024-12-11','2024-12-12','2024-12-13','2024-12-16','2024-12-17','2025-01-09','2025-01-10','2025-01-15','2025-01-16','2025-01-17','2025-01-21','2025-01-22','2025-01-28','2025-01-31','2025-02-03','2025-02-04','2025-02-05','2025-02-06','2025-02-11','2025-02-12','2025-02-13','2025-03-10','2025-03-11','2025-03-13','2025-03-14','2025-03-18','2025-03-19','2025-03-21','2025-03-14','2025-03-25','2025-03-26','2025-03-27','2025-04-01','2025-04-03','2025-04-04','2025-04-08','2025-04-09','2025-04-10','2025-04-11','2025-04-14','2025-04-15','2025-04-16','2025-04-24','2025-04-29','2025-04-30','2025-05-01','2025-05-02','2025-05-07','2025-05-08','2025-05-09','2025-05-13','2025-05-14','2025-05-15','2025-05-16','2025-05-21','2025-05-23','2025-05-28','2025-05-29','2025-05-30','2025-06-03']
	# to save memory I won't analyse the senario developement and commissioning shots
	to_do = ['2024-10-16','2024-10-17','2024-10-18','2024-10-21','2024-10-22','2024-10-23','2024-10-24','2024-10-29','2024-10-30','2024-10-31','2024-11-01','2024-11-04','2024-11-05','2024-11-06','2024-11-07','2024-11-08','2024-11-12','2024-11-13','2024-11-14','2024-11-15','2024-11-18','2024-11-19','2024-11-20','2024-11-21','2024-11-22','2024-11-25','2024-11-26','2024-11-27','2024-11-28','2024-11-29','2024-12-03','2024-12-04','2024-12-05','2024-12-06','2024-12-10','2024-12-11','2024-12-12','2024-12-13','2024-12-16','2024-12-17','2025-01-09','2025-01-10','2025-01-15','2025-01-16','2025-01-17','2025-01-21','2025-01-22','2025-01-28','2025-01-31','2025-02-03','2025-02-04','2025-02-05','2025-02-06','2025-02-11','2025-02-12','2025-02-13','2025-03-10','2025-03-11','2025-03-13','2025-03-14','2025-03-18','2025-03-19','2025-03-21','2025-03-14','2025-03-25','2025-03-26','2025-03-27','2025-04-01','2025-04-03','2025-04-04','2025-04-08','2025-04-09','2025-04-10','2025-04-11','2025-04-14','2025-04-15','2025-04-16','2025-04-24','2025-04-29','2025-04-30','2025-05-01','2025-05-02','2025-05-07','2025-05-08','2025-05-09','2025-05-13','2025-05-14','2025-05-15','2025-05-16','2025-05-21','2025-05-23','2025-05-28','2025-05-29','2025-05-30','2025-06-03','2025-06-03','2025-06-04','2025-06-05','2025-06-06','2025-06-10','2025-06-11','2025-06-12','2025-06-13','2025-06-18','2025-06-19','2025-06-20','2025-06-23','2025-06-24','2025-06-25','2025-06-26','2025-06-27','2025-07-04','2025-07-10','2025-07-11','2025-07-14','2025-07-15','2025-07-16','2025-07-17','2025-07-18','2025-07-22','2025-07-23','2025-07-24','2025-07-25','2025-07-28','2025-07-29','2025-07-30','2025-07-31','2025-08-01','2025-08-05','2025-08-06','2025-08-07','2025-08-08','2025-08-12','2025-08-13','2025-08-14','2025-08-15','2025-08-18','2025-08-19','2025-08-20','2025-08-21']
	to_do = np.flip(to_do,axis=0)
elif MU_campaign==99:	# just an exeption to have manual runs
	to_do = ['2024-01-19']
else:
	to_do = ['2026-02-12']
# # for Lingyan
# to_do = ['2021-10-27']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']

# to_do = ['2022-10-26','2022-10-27','2022-10-28','2022-10-31']

seconds_for_bad_pixels = 2	# s
seconds_for_reference_frame = 1	# s


if False:	# this method divides the shots per day
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
elif True:	# with this all the shots are in a single array
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.append(dirnames)
	days_available = f[0]
	shot_available = []
	temp = []
	for i_day,day in enumerate(to_do):
		f = []
		for (dirpath, dirnames, filenames) in os.walk(path+day+'/'):
			f.append(filenames)
		# shot_available.append([])
		if len(f)>0:
			for name in f[0]:
				if name[-3:]=='ats' or name[-3:]=='ptw' or (name[-3:]=='npz' and not (name[-8:-4] in ['ance','FAST'])):
					shot_available.append(name)
					temp.append(day)
	to_do = temp


# to_do = ['2021-10-15','2021-10-12','2021-10-01','2021-09-30']
# shot_available = [['IRVB-MASTU_shot-45314.ptw','IRVB-MASTU_shot-45315.ptw'],['IRVB-MASTU_shot-45248.ptw'],['IRVB-MASTU_shot-45125.ptw'],['IRVB-MASTU_shot-45100.ptw','IRVB-MASTU_shot-45099.ptw']]
# to_do = ['2021-10-13','2021-10-22','2021-10-26']
# shot_available = [['IRVB-MASTU_shot-45272.ptw'],['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-45399.ptw'],['IRVB-MASTU_shot-45414.ptw','IRVB-MASTU_shot-45420.ptw','IRVB-MASTU_shot-45415.ptw','IRVB-MASTU_shot-45416.ptw']]
# for Stuart Handerson, RT-14
# to_do = ['2023-12-06']
# shot_available = [['IRVB-MASTU_shot-44891.ptw','IRVB-MASTU_shot-44892.ptw','IRVB-MASTU_shot-44904.ptw','IRVB-MASTU_shot-44905.ptw']]
# these should show a decent evolution from all attached to inner detached, outer detached and marfe
# to_do = ['2021-10-14','2021-10-19']
# shot_available = [['IRVB-MASTU_shot-45295.ptw','IRVB-MASTU_shot-45296.ptw'],['IRVB-MASTU_shot-45324.ptw','IRVB-MASTU_shot-45327.ptw']]
# # for Lingyan
# to_do = ['2021-10-27']
# shot_available = [['IRVB-MASTU_shot-45446.ptw','IRVB-MASTU_shot-45448.ptw','IRVB-MASTU_shot-45450.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45458.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45460.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw']]

# to_do = ['2021-10-27','2021-10-12','2021-10-22','2021-10-13','2021-10-22','2021-10-28','2021-08-05','2021-08-17','2021-08-25','2021-08-26']
# shot_available = [['IRVB-MASTU_shot-45439.ptw','IRVB-MASTU_shot-45443.ptw','IRVB-MASTU_shot-45444.ptw','IRVB-MASTU_shot-45446.ptw','IRVB-MASTU_shot-45448.ptw','IRVB-MASTU_shot-45450.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45458.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45460.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw'],['IRVB-MASTU_shot-45239.ptw'],['IRVB-MASTU_shot-45391.ptw'],['IRVB-MASTU_shot-45272.ptw'],['IRVB-MASTU_shot-45398.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45400.ptw'],['IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw'],['IRVB-MASTU_shot-44607.ptw'],['IRVB-MASTU_shot-44697.ptw','IRVB-MASTU_shot-44699.ptw','IRVB-MASTU_shot-44700.ptw','IRVB-MASTU_shot-44701.ptw'],['IRVB-MASTU_shot-44797.ptw'],['IRVB-MASTU_shot-44822.ptw']]


continue_after_FAST = False
override_FAST_analysis = True
do_inversions = False

every_pixel_independent = False
overwrite_oscillation_filter = True
overwrite_binning = True
override_first_pass = True
skip_second_pass = False
override_second_pass = False
skip_third_pass = False
if skip_second_pass:
	skip_third_pass = True
override_third_pass = False
True_for_inversion_False_for_post_analysis_only = True

if overwrite_oscillation_filter:
	overwrite_binning = True

only_plot_brightness = False

if False or MU_campaign>0:	# section to use when specifying the days
	# for i_day,day in enumerate(to_do):
	to_do = np.flip(to_do,axis=0)
	# shot_available = np.flip(shot_available,axis=0)
	# # for i_day,day in enumerate(np.flip(to_do,axis=0)):
	# 	shot_available[i_day] = np.flip(shot_available[i_day],axis=0)
	# 	for name in shot_available[i_day]:
	if ext_number_of_passes==1:
		skip_second_pass = True
		skip_third_pass = True
	elif ext_number_of_passes==2:
		skip_second_pass = False
		skip_third_pass = True
	elif ext_number_of_passes==3:
		skip_second_pass = False
		skip_third_pass = False

	if ext_sequencer!=None:
		try:
			to_do = [to_do[ext_sequencer]]
			shot_available = [shot_available[ext_sequencer]]
		except:
			print('no shots left to process')
			sys.exit(1)
	for i_day,day in enumerate(to_do):
		name = shot_available[i_day]
		laser_to_analyse=path+day+'/'+name

		try:
			out = coleval.retrive_aborted(name[-9:-4])
		except:
			print('retrive_aborted failed')
			out = False
		if out:	# skip the shot if it was aborted
			print('skipped '+name+ ' because the shot seems to be aborted')
			continue

		try:
			if True:
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
			else:
				pass_number = 0
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
				pass_number = 1
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
		except Exception as e:
			print('FAST FAILED ' + laser_to_analyse)
			logging.exception('with error: ' + str(e))
			pass

elif False:	# section to use when specifying the shots

	# simpler way of starting works
	# to_do = ['45468','45469','45470','45473','45401','45399','45371']#,'45473']
	# to_do = ['45371','45372','45380','45390','45391','45398','45400','45439','45443','45444','45450','45456','45459','45461','45462','45463','45464','45465']
	# to_do = ['45046','45047','45048','45056','45057','45058','45059','45060','45062','45063','45066','45071','45081','45143','45239','45241','45243','45244','45245','45246','45353']	# SXD fuelled from high field side
	# to_do = ['47036','47143','46823']
	# to_do = ['47036','47036','46884','47020','46776','46792']

	# # MU01-EXH-20
	# to_do = ['44816','44817','44818','44819','44820','44821','44822','45391','45397','45398','45399','45400','45401']
	# for the paper
	# to_do = ['44647','45409','45225','45351']
	# to_do = ['45401','44892']
	# to_do = ['45401']
	# shots for the RSI paper
	# to_do = ['44647','45409','45225','45351','45401','44892','45295','45328','45371']

	# # shots for the science paper
	# to_do = []
	# to_do = to_do + ['45371']
	# # shots for the CD radiator location comparison shots
	# to_do = to_do + ['45473','45470','45469','45468','45327','45326','45325','45324','45323','45322','45320','45303','45302','45296','45295','45293','45142','45126','45088','45371']
	# # shots for the CD BEAMS shots for the cross comparison
	# to_do = to_do + ['45401','45399','45311','45310','45309','45306','45304','45299','45286','45272','45271','45270','45268','45267','45266','45264','45262','45261','45252','45251','45237','45236','45212','45193','45194','45175','45170','45167','45132','45125','45097','45091','45006','44969','44968','44967','44960']
	# # shots for David's paper
	# to_do = to_do + ['45443','45444','45446','45456','45459','45461','45462','45463','45464','45465','45468','45469','45470','45473']
	# # shots for Kevin's paper
	# to_do = to_do + ['46860','47958','46866','47079','46702']
	# to_do = to_do + ['49270','49267','49320','49312']
	# to_do = to_do + ['46860','47958','46705','48330']
	# # Stuart THR01
	# to_do = to_do + ['49401']
	# to_do = to_do + ['49392','49394','49396','49397','49400','49401','49404','49405']

	# # shots asked by Jack MU03-DIV-02
	# to_do = ['48590','48592','48596','48597','48599','48690','48692','48696','48765','48766','48767','48768','48769']

	# # MU03-WPTE-05 looking for a cd xpr with nitrogen seeding
	# to_do = ['48594','48595','48597','48598','48599','48602','48603','48605','48609','48617','48618','48619','48620']



	# # MU02 shots for Kevin
	# to_do = []
	# # # # sxd scans and repeats
	# # # to_do = to_do + ['46707','46769','46776','46791','46792','46794','46860','46893','46705','46904','46905']
	# # # # elongated divertor scans and repeats
	# # # to_do = to_do + ['47079','47082','47115','47118','47085']
	# # # # conventional divertor scans and repeats
	# # # to_do = to_do + ['46762','46769','46864','46866','46867','46868','46891','46903','46905','46702','46889']
	# # # # shape scans
	# # # to_do = to_do + ['46889','46895','46890']
	# to_do = to_do + ['49312','49320']

	# # MU03 shots for Kevin
	# to_do = []
	# conventional divertor scans and repeats L mode
	# to_do = to_do + ['47950','47951','47962','48144','48328','48335','48336','49408','49283']
	# conventional divertor scans and repeats H mode
	# to_do = to_do + ['49392','48561','49139']

	# shots for Ivan Paradela Perez
	# to_do = to_do + ['46864','47079','46860']


	# # Stuart shots MU03-DIV-03
	# to_do = []
	# # DDN
	# to_do = to_do + ['48361','48363','48366','48367','48648']	# 48370 data is missing
	# # LSN
	# to_do = to_do + ['48638','48639','48646','48647','48651','48652']

	# checking the effect of valve HFS_MID_L02
	# to_do = ['47218','47228','47401','47402','47448','47570','47580','47826','47869','48063','48239','48290','48325','48376','48377','48378','48381','48382','48576','48583','48629','48633']

	# # trophy shots from mu02
	# to_do = ['46976','46977','46978','46979','46823','48561']

	# # other H mode shots I'm finding
	# to_do = ['48594','48595','48596','48564','48763']

	# # MU03 snowflake by Vlad Soukhanovskii
	# # to_do = ['49463','49464','49465','49466','49467','49468']
	# to_do = ['49465','49467','49468']

	# to_do = []
	# # mu03 DIV-02 by Jack
	# # to_do = to_do + ['48692','46363','49373','49374','48766','49464','49465','49466','49467']
	# # mu03 DIV-03 by Jack
	# # to_do = to_do + ['49213','48651','49220','49259','49262','49404','49198']
	# # mu03 DIV-03 by Vlad Soukhanovskii
	# to_do = to_do + ['49463','49464','49465','49466','49467','49468','48690','48692']


	# data for psi2024 poster
	# to_do = []
	# # ohmic L-mode
	# to_do = to_do + ['45468','45469','45470','45473']
	# to_do = to_do + ['47950','47973','48144','48328','48335']
	# # beam heated L-mode
	# to_do = to_do + ['46866','46864','46867','46868','46889','46891','46903','48336','49408','49283']
	# # H-mode
	# to_do = to_do + ['45401','48596','48597','46977','48599','48561','49139','c','49396','49400','49401']
	# # mix
	# to_do = to_do + ['48690','48692','49463','49464','49465','49466','49467','49468']
	# # IRVB science paper 2025
	# to_do = []
	# to_do = to_do + ['45468','45469','45470','45473']
	# to_do = to_do + ['47950','47973','48144']
	# to_do = to_do + ['46866','46867','46868','46891','48336','49408']

	# # # data for Yasmin Adnrew (female) about LH transition
	# to_do = []
	# to_do = to_do + ['51704','51705','51706','51709','51710','51711','51712','51713','51761','51762','51763','52764','51765']
	# to_do = to_do + ['46631','47012','47078','47094','47889','47909','47916',
	# '47918','48004','48172','48174','48175','48176','48177',
	# '49091','49094','49095']
	# to_do = to_do + ['48004']

	# data for Bob Kool nature 2024 paper
	# to_do = []
	# to_do = to_do + ['47080','47083','47086','47116','47118','47119','49303',
	# '48001','47998','49297','49298']

	# # 2024/11/27 for kevin's experiment
	# to_do = []
	# to_do = to_do + ['50822','50823','50824','50825','50826','50827','50828',
	# '50830','50831','50832','50833']

	# 2025/06/05 Fabio Federici MU04-DIV02 experiments
	to_do = []
	# to_do = to_do + ['51650','51651','51652','51653','51654','51655','51656']	# first session DN
	# to_do = to_do + ['51877','51878','51879','51880','51883','51884','51885','51886','51887']	# second session LSN
	# to_do = to_do + ['52349', '52355']	# first repeat DN cryo high ISP
	# to_do = to_do + ['52420']	# second repeat DN cryo low ISP
	# to_do = to_do + ['52493']	# third repeat DN cryo low ISP
	# to_do = to_do + ['50833','50832']	# additional cases from Kevin with different position of inner leg
	# to_do = to_do + ['50825']	# additional cases from Kevin with density decreasing rather than increasing
	# to_do = to_do + ['52288','52285']	# case from stuart where the outer leg becomes vertical and I could see if that transition is sharp (from the angle of the OSP: open divertor 52288), together with closed divertor reference (52285)
	# to_do = to_do + ['50923','50924','51817','52384','52383']	# Charlie's negative triangularity shots
	# to_do = to_do + ['52612']	# Nocola's CD DN Lmode with N2 seeding
	to_do = to_do + ['47116','49303','47080']	# Nocola's oscillating fueling. NOTE: the fueling of the valve that oscillate is NOT recorded

	# # # 2025/06/05 Stuart XPR experiments
	# to_do = []
	# to_do = to_do + ['51796','51797','51798','51799','51800','51801','51802','51803','51804','51805','51806','51807','52296','52314','52284','52285','52286','52287','52288','52289','52290','52291','52292','52301']
	# to_do = to_do + ['52248','52250','52251','52252','52255','52266','52257','52258']
	# to_do = to_do + ['52356','52315','52354']	# can be interesting as the inner strike point goes from CC to T1, while the rest stays about the same

	# 2025/05/14 Ling yang experiments
	# to_do = []
	# to_do = to_do + ['51782','51777']

	# # 2025-08-18 Charlie's negative triangularity shots
	# to_do = []
	# to_do = to_do + ['50923','50924','51817','52384','52383']

	# # 2025/05/02 Bart EST-01 experiments
	# to_do = []
	# to_do = to_do + ['51685','51692','51691']

	# # 2025/07/04 Rory and Nicola for transients
	# to_do = []
	# to_do = to_do + ['51375','51380','51381','51508','51511','51512','51514','51785','51786','51787','51788','51789','51791','52249','52250']
	# to_do = to_do + ['52506']	# random Hmode added to see if ELMs are visible

	# # 2025/05/02 Nicola for PSI 2026
	# to_do = []
	# to_do = to_do + ['49303','50838','50832','50639']

	# # 2025/08/14 Vlad's snowflake experiments
	# to_do = []
	# to_do = to_do + ['52434','52436','52437','52438','52439','524140','52441','52442']

	to_do = np.flip(to_do,axis=0)

	if ext_sequencer!=None:
		try:
			to_do = [to_do[ext_sequencer]]
		except:
			print('no shots left to process')
			sys.exit(1)
	for name in to_do:
	# for name in np.flip(to_do,axis=0):
		# shot_list = get_data(path+'shot_list2.ods')
		# temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
		# for i in range(1,len(shot_list['Sheet1'])):
		# 	if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
		# 		date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
		# 		break
		# try:
		# 	i_day,day = 0,str(date.date())
		# except:
		# 	i_day,day = 0,str(date)
		if name[:4] == 'IRVB':	# I want to shorten the filenames in to_do
			pass
		else:
			name = 'IRVB-MASTU_shot-' + name + '.ptw'

		try:
			out = coleval.retrive_aborted(name[-9:-4])
		except:
			print('retrive_aborted failed')
			out = False
		if out:	# skip the shot if it was aborted
			print('skipped '+name+ ' because the shot seems to be aborted')
			continue

		i_day,day = 0,coleval.retrive_shot_date_and_time(name[-9:-4])[0]
		laser_to_analyse=path+day+'/'+name

		try:
			if True_for_inversion_False_for_post_analysis_only:
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
			else:
				pass_number = 0
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
				pass_number = 1
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
		except Exception as e:
			print('FAST FAILED ' + laser_to_analyse)
			logging.exception('with error: ' + str(e))
			pass

elif True:	# section to use when analysing only one shot

	# MU03
	name = 'IRVB-MASTU_shot-49445.ptw'	# large VDE: peak fast camera brightness 686ms; peak /XIM/DA/HL02/SXD 689.7ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 689.5ms; IRVB 3185 ms from recording start -2.5 = 685ms
	name = 'IRVB-MASTU_shot-49372.ptw'	# large VDE: peak fast camera brightness 623ms; peak /XIM/DA/HL02/SXD 624.9ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 625.1ms; IRVB 3103 ms from recording start -2.5 = 601ms
	name = 'IRVB-MASTU_shot-48235.ptw'	# large VDE: peak fast camera brightness 552 ms; peak /XIM/DA/HL02/SXD 554 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 554 ms; IRVB 3052 ms from recording start -2.5 = 552 ms
	name = 'IRVB-MASTU_shot-48194.ptw'	# large VDE: peak fast camera brightness 377 ms; peak /XIM/DA/HL02/SXD 371.9 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 378.8 ms; IRVB 2880 ms from recording start -2.5 = 380ms
	name = 'IRVB-MASTU_shot-49148.ptw'	# large VDE: peak fast camera brightness 656ms; peak /XIM/DA/HL02/SXD 649.1ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 658.3ms; IRVB 3130 ms from recording start -2.5 = 630ms
	name = 'IRVB-MASTU_shot-48146.ptw'	# large VDE: peak fast camera brightness 826ms; peak /XIM/DA/HL02/SXD 821.3ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 827.8ms; IRVB 3308 ms from recording start -2.5 = 808ms
	name = 'IRVB-MASTU_shot-48078.ptw'	# large VDE: peak fast camera brightness 370ms; peak /XIM/DA/HL02/SXD 372ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 372ms; IRVB 2867 ms from recording start -2.5 = 367ms
	name = 'IRVB-MASTU_shot-48046.ptw'	# large VDE: peak fast camera brightness MISSING; peak /XIM/DA/HL02/SXD 792.5ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 780.1ms; IRVB 3261 ms from recording start -2.5 = 761ms
	name = 'IRVB-MASTU_shot-47893.ptw'	# large VDE: peak fast camera brightness 775ms; peak /XIM/DA/HL02/SXD 777ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 776ms; IRVB 3258 ms from recording start -2.5 = 758ms
	name = 'IRVB-MASTU_shot-47834.ptw'	# large VDE: peak fast camera brightness 290 ms; peak /XIM/DA/HL02/SXD 292.9 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 292.9 ms; IRVB 2796 ms from recording start -2.5 = 296ms
	name = 'IRVB-MASTU_shot-47829.ptw'	# large VDE: peak fast camera brightness 644ms; peak /XIM/DA/HL02/SXD 645.2 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 645.2 ms; IRVB 3138 ms from recording start -2.5 = 638ms
	# name = 'IRVB-MASTU_shot-47579.ptw'	# large VDE: peak fast camera brightness UNCERTAIN; peak /XIM/DA/HL02/SXD 645.9ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 647.7ms; IRVB MISSING
	# MU02
	name = 'IRVB-MASTU_shot-46256.ptw'	# large VDE: peak fast camera brightness 554ms, secondary at 371; peak /XIM/DA/HL02/SXD 554ms, secondary at 372; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 557.3ms, secondary at 372; IRVB 3036 ms, secondary 2855 ms from recording start -2.5 = 536 ms, secondary 355 ms
	name = 'IRVB-MASTU_shot-46324.ptw'	# large VDE: peak fast camera brightness 443ms; peak /XIM/DA/HL02/SXD 445.3ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 444.5ms; IRVB 2937/27 ms from recording start -2.5 = 437/27 ms
	# name = 'IRVB-MASTU_shot-46448.ptw'	# large VDE: peak fast camera brightness 787ms; peak /XIM/DA/HL02/SXD 777.7ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 788.1ms; IRVB MISSING
	# name = 'IRVB-MASTU_shot-46449.ptw'	# large VDE: peak fast camera brightness XXXms; peak /XIM/DA/HL02/SXD XXXms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP XXXms; IRVB MISSING
	name = 'IRVB-MASTU_shot-46451.ptw'	# large VDE: peak fast camera brightness 195 ms; peak /XIM/DA/HL02/SXD 182.7 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 197.8 ms; IRVB 2694 ms from recording start -2.5 = 194ms
	name = 'IRVB-MASTU_shot-46455.ptw'	# large VDE: peak fast camera brightness 388 ms; peak /XIM/DA/HL02/SXD 392.9 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 392.9 ms; IRVB 2880 ms from recording start -2.5 = 380ms
	name = 'IRVB-MASTU_shot-46458.ptw'	# large VDE: peak fast camera brightness 353 ms; peak /XIM/DA/HL02/SXD 353ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 354.1ms; IRVB 2830/43 ms from recording start -2.5 = 330/43ms
	name = 'IRVB-MASTU_shot-46487.ptw'	# large VDE: peak fast camera brightness 332 ms; peak /XIM/DA/HL02/SXD 332.9ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 333.9ms; IRVB 2830 ms from recording start -2.5 = 330ms
	name = 'IRVB-MASTU_shot-46492.ptw'	# large VDE: peak fast camera brightness 382 ms; peak /XIM/DA/HL02/SXD 385.3 ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 385.3 ms; IRVB 2875 ms from recording start -2.5 = 375 ms
	name = 'IRVB-MASTU_shot-46856.ptw'	# large VDE: peak fast camera brightness 182ms; peak /XIM/DA/HL02/SXD 184.8ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 184.8ms; IRVB 2681 ms from recording start -2.5 = 181ms
	name = 'IRVB-MASTU_shot-46897.ptw'	# large VDE: peak fast camera brightness MISSING; peak /XIM/DA/HL02/SXD 568ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 568ms; IRVB 3063 ms from recording start -2.5 = 563ms
	name = 'IRVB-MASTU_shot-46919.ptw'	# large VDE: peak fast camera brightness 244ms; peak /XIM/DA/HL02/SXD 232ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 247ms; IRVB MISSING
	# name = 'IRVB-MASTU_shot-46941.ptw'	# large VDE: peak fast camera brightness 308ms; peak /XIM/DA/HL02/SXD 310.5ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 310ms; 	NOTE framerate 50Hz, so starting point very wrong
	name = 'IRVB-MASTU_shot-47040.ptw'	# large VDE: peak fast camera brightness 244ms; peak /XIM/DA/HL02/SXD 232ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 247ms; IRVB 2741 ms from recording start -2.5 = 241ms
	name = 'IRVB-MASTU_shot-47041.ptw'	# large VDE: peak fast camera brightness 228ms; peak /XIM/DA/HL02/SXD 230.6ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 229.9ms; IRVB 2734 ms from recording start -2.5 = 234ms
	# MU01
	name = 'IRVB-MASTU_shot-45396.ptw'	# large VDE: peak /XIM/DA/HL02/SXD 467.1ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 466.5ms; IRVB 2958 ms from recording start -2.5 = 458ms
	name = 'IRVB-MASTU_shot-45347.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 696.1ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 695.2ms; IRVB 3167 ms from recording start -2.5 = 667ms
	name = 'IRVB-MASTU_shot-45325.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 604ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 601.4ms; IRVB 3078 ms from recording start -2.5 = 578ms
	name = 'IRVB-MASTU_shot-45313.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 625.3ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 625.3ms; IRVB 3078 ms from recording start -2.5 = 578ms
	name = 'IRVB-MASTU_shot-45306.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 1019ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 1018ms; IRVB 3523 ms from recording start -2.5 = 1023ms
	name = 'IRVB-MASTU_shot-45291.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 404ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 403ms; IRVB 2875 ms from recording start -2.5 = 375ms
	name = 'IRVB-MASTU_shot-45290.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 690.5ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 690.5ms; IRVB 3170 ms from recording start -2.5 = 670ms
	name = 'IRVB-MASTU_shot-45287.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 574ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 574ms; IRVB 3060 ms from recording start -2.5 = 560ms
	name = 'IRVB-MASTU_shot-44904.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 833ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 833ms; IRVB 3313 ms from recording start -2.5 = 813ms
	name = 'IRVB-MASTU_shot-44804.ptw'	# large VDE: peak fast camera brightness MISSING; large VDE: peak /XIM/DA/HL02/SXD 417.9ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 417.3ms; IRVB 2922 ms from recording start -2.5 = 422ms
	# name = 'IRVB-MASTU_shot-45463.ptw'	# large VDE: peak /XIM/DA/HL02/SXD 711.3ms; peak /XIM/DA/HE05/ISP/L and start decrease /XBM/CORE/F15/AMP 710.9ms; IRVB 3019 ms from recording start -2.5 = 519ms	NOTE framerate 50Hz, so starting point very wrong
	# name = 'IRVB-MASTU_shot-47953.ptw'
	# name = 'IRVB-MASTU_shot-52493.ptw'
	# name = 'IRVB-MASTU_shot-52355.ptw'
	name = 'IRVB-MASTU_shot-53520.ptw'

	if name[:4] == 'IRVB':	# I want to shorten the filenames in to_do
		pass
	else:
		name = 'IRVB-MASTU_shot-' + name + '.ptw'

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

	laser_to_analyse=path+day+'/'+name

	if True_for_inversion_False_for_post_analysis_only:
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
	else:
		pass_number = 0
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
		pass_number = 1
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
		pass_number = 2
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
	# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3.py").read())

print('all done')
