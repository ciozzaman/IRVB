# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

try:
	import argparse
	parser = argparse.ArgumentParser(
	prog='ProgramName',
	description='What the program does',
	epilog='Text at the bottom of help')
	parser.add_argument('ext_sequencer',type=int)
	parser.add_argument('MU_campaign',type=int)
	# parser.add_argument('--debug',action='store_true')

	ext_sequencer = vars(parser.parse_args())['ext_sequencer']
	MU_campaign = vars(parser.parse_args())['MU_campaign']
	# print(ext_sequencer)
	print('running sequential shot number '+str(ext_sequencer))

	# note
	# to use these feature the typical command to launch is this
	# qsub -V -cwd -o /home/ffederic/work/analysis_scripts/scripts/MASTU_prelim_data_examination3.out -e /home/ffederic/work/analysis_scripts/scripts/MASTU_prelim_data_examination3.err -N MASTU_prelim_data_examination3 ./command_single_core.sh 0 1000000 3
except:
	print('no sequential ID provided')
	ext_sequencer = None
	MU_campaign = 0
# exit()

print(ext_sequencer)
print(MU_campaign)

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
pathparams_BB='/home/ffederic/work/irvb/2022-12-07_multiple_search_for_parameters'	# original nuc measurements, calibration with wavewlength_top=5.1,wavelength_bottom=1.5
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
elif MU_campaign==99:	# just an exeption to have manual runs
	to_do = ['2024-01-19']
else:
	to_do = ['2024-11-12','2024-11-13','2024-11-27','2024-11-28','2024-11-29']
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
				if name[-3:]=='ats' or name[-3:]=='ptw':
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
skip_second_pass = False
override_second_pass = True
skip_third_pass = True
override_third_pass = False

if overwrite_oscillation_filter:
	overwrite_binning = True

only_plot_brightness = False

if False or MU_campaign>0:	# section to use when specifying the days
	# for i_day,day in enumerate(to_do):
	# to_do = np.flip(to_do,axis=0)
	# shot_available = np.flip(shot_available,axis=0)
	# # for i_day,day in enumerate(np.flip(to_do,axis=0)):
	# 	shot_available[i_day] = np.flip(shot_available[i_day],axis=0)
	# 	for name in shot_available[i_day]:
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

		if coleval.retrive_aborted(name[-9:-4]):	# skip the shot if it was aborted
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
	# to_do = ['IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw','IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45371.ptw']#,'IRVB-MASTU_shot-45473.ptw']
	# to_do = ['IRVB-MASTU_shot-45371.ptw','IRVB-MASTU_shot-45372.ptw','IRVB-MASTU_shot-45380.ptw','IRVB-MASTU_shot-45390.ptw','IRVB-MASTU_shot-45391.ptw','IRVB-MASTU_shot-45398.ptw','IRVB-MASTU_shot-45400.ptw','IRVB-MASTU_shot-45439.ptw','IRVB-MASTU_shot-45443.ptw','IRVB-MASTU_shot-45444.ptw','IRVB-MASTU_shot-45450.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw']
	# to_do = ['IRVB-MASTU_shot-45046.ptw','IRVB-MASTU_shot-45047.ptw','IRVB-MASTU_shot-45048.ptw','IRVB-MASTU_shot-45056.ptw','IRVB-MASTU_shot-45057.ptw','IRVB-MASTU_shot-45058.ptw','IRVB-MASTU_shot-45059.ptw','IRVB-MASTU_shot-45060.ptw','IRVB-MASTU_shot-45062.ptw','IRVB-MASTU_shot-45063.ptw','IRVB-MASTU_shot-45066.ptw','IRVB-MASTU_shot-45071.ptw','IRVB-MASTU_shot-45081.ptw','IRVB-MASTU_shot-45143.ptw','IRVB-MASTU_shot-45239.ptw','IRVB-MASTU_shot-45241.ptw','IRVB-MASTU_shot-45243.ptw','IRVB-MASTU_shot-45244.ptw','IRVB-MASTU_shot-45245.ptw','IRVB-MASTU_shot-45246.ptw','IRVB-MASTU_shot-45353.ptw']	# SXD fuelled from high field side
	# to_do = ['IRVB-MASTU_shot-47036.ptw','IRVB-MASTU_shot-47143.ptw','IRVB-MASTU_shot-46823.ptw']
	# to_do = ['IRVB-MASTU_shot-47036.ptw','IRVB-MASTU_shot-47036.ptw','IRVB-MASTU_shot-46884.ptw','IRVB-MASTU_shot-47020.ptw','IRVB-MASTU_shot-46776.ptw','IRVB-MASTU_shot-46792.ptw']

	# # MU01-EXH-20
	# to_do = ['IRVB-MASTU_shot-44816.ptw','IRVB-MASTU_shot-44817.ptw','IRVB-MASTU_shot-44818.ptw','IRVB-MASTU_shot-44819.ptw','IRVB-MASTU_shot-44820.ptw','IRVB-MASTU_shot-44821.ptw','IRVB-MASTU_shot-44822.ptw','IRVB-MASTU_shot-45391.ptw','IRVB-MASTU_shot-45397.ptw','IRVB-MASTU_shot-45398.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45400.ptw','IRVB-MASTU_shot-45401.ptw']
	# for the paper
	# to_do = ['IRVB-MASTU_shot-44647.ptw','IRVB-MASTU_shot-45409.ptw','IRVB-MASTU_shot-45225.ptw','IRVB-MASTU_shot-45351.ptw']
	# to_do = ['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-44892.ptw']
	# to_do = ['IRVB-MASTU_shot-45401.ptw']
	# shots for the RSI paper
	# to_do = ['IRVB-MASTU_shot-44647.ptw','IRVB-MASTU_shot-45409.ptw','IRVB-MASTU_shot-45225.ptw','IRVB-MASTU_shot-45351.ptw','IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-44892.ptw','IRVB-MASTU_shot-45295.ptw','IRVB-MASTU_shot-45328.ptw','IRVB-MASTU_shot-45371.ptw']

	# shots for the science paper
	# to_do = ['IRVB-MASTU_shot-45371.ptw']
	# shots for the CD radiator location comparison shots
	# to_do = ['IRVB-MASTU_shot-45473.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45327.ptw','IRVB-MASTU_shot-45326.ptw','IRVB-MASTU_shot-45325.ptw','IRVB-MASTU_shot-45324.ptw','IRVB-MASTU_shot-45323.ptw','IRVB-MASTU_shot-45322.ptw','IRVB-MASTU_shot-45320.ptw','IRVB-MASTU_shot-45303.ptw','IRVB-MASTU_shot-45302.ptw','IRVB-MASTU_shot-45296.ptw','IRVB-MASTU_shot-45295.ptw','IRVB-MASTU_shot-45293.ptw','IRVB-MASTU_shot-45142.ptw','IRVB-MASTU_shot-45126.ptw','IRVB-MASTU_shot-45088.ptw','IRVB-MASTU_shot-45371.ptw']
	# shots for the CD BEAMS shots for the cross comparison
	# to_do = ['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45311.ptw','IRVB-MASTU_shot-45310.ptw','IRVB-MASTU_shot-45309.ptw','IRVB-MASTU_shot-45306.ptw','IRVB-MASTU_shot-45304.ptw','IRVB-MASTU_shot-45299.ptw','IRVB-MASTU_shot-45286.ptw','IRVB-MASTU_shot-45272.ptw','IRVB-MASTU_shot-45271.ptw','IRVB-MASTU_shot-45270.ptw','IRVB-MASTU_shot-45268.ptw','IRVB-MASTU_shot-45267.ptw','IRVB-MASTU_shot-45266.ptw','IRVB-MASTU_shot-45264.ptw','IRVB-MASTU_shot-45262.ptw','IRVB-MASTU_shot-45261.ptw','IRVB-MASTU_shot-45252.ptw','IRVB-MASTU_shot-45251.ptw','IRVB-MASTU_shot-45237.ptw','IRVB-MASTU_shot-45236.ptw','IRVB-MASTU_shot-45212.ptw','IRVB-MASTU_shot-45193.ptw','IRVB-MASTU_shot-45194.ptw','IRVB-MASTU_shot-45175.ptw','IRVB-MASTU_shot-45170.ptw','IRVB-MASTU_shot-45167.ptw','IRVB-MASTU_shot-45132.ptw','IRVB-MASTU_shot-45125.ptw','IRVB-MASTU_shot-45097.ptw','IRVB-MASTU_shot-45091.ptw','IRVB-MASTU_shot-45006.ptw','IRVB-MASTU_shot-44969.ptw','IRVB-MASTU_shot-44968.ptw','IRVB-MASTU_shot-44967.ptw','IRVB-MASTU_shot-44960.ptw']
	# shots for David's paper
	# to_do = ['IRVB-MASTU_shot-45443.ptw','IRVB-MASTU_shot-45444.ptw','IRVB-MASTU_shot-45446.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw','IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw']
	# shots for Kevin's paper
	# to_do = ['IRVB-MASTU_shot-46860.ptw','IRVB-MASTU_shot-47958.ptw','IRVB-MASTU_shot-46866.ptw','IRVB-MASTU_shot-47079.ptw','IRVB-MASTU_shot-46702.ptw']
	# to_do = ['IRVB-MASTU_shot-49270.ptw','IRVB-MASTU_shot-49267.ptw','IRVB-MASTU_shot-49320.ptw','IRVB-MASTU_shot-49312.ptw']
	# to_do = ['IRVB-MASTU_shot-46860.ptw','IRVB-MASTU_shot-47958.ptw','IRVB-MASTU_shot-46705.ptw','IRVB-MASTU_shot-48330.ptw']
	# Stuart THR01
	# to_do = ['IRVB-MASTU_shot-49401.ptw']
	# to_do = ['IRVB-MASTU_shot-49392.ptw','IRVB-MASTU_shot-49394.ptw','IRVB-MASTU_shot-49396.ptw','IRVB-MASTU_shot-49397.ptw','IRVB-MASTU_shot-49400.ptw','IRVB-MASTU_shot-49401.ptw','IRVB-MASTU_shot-49404.ptw','IRVB-MASTU_shot-49405.ptw']

	# # shots asked by Jack MU03-DIV-02
	# to_do = ['IRVB-MASTU_shot-48590.ptw','IRVB-MASTU_shot-48592.ptw','IRVB-MASTU_shot-48596.ptw','IRVB-MASTU_shot-48597.ptw','IRVB-MASTU_shot-48599.ptw','IRVB-MASTU_shot-48690.ptw','IRVB-MASTU_shot-48692.ptw','IRVB-MASTU_shot-48696.ptw','IRVB-MASTU_shot-48765.ptw','IRVB-MASTU_shot-48766.ptw','IRVB-MASTU_shot-48767.ptw','IRVB-MASTU_shot-48768.ptw','IRVB-MASTU_shot-48769.ptw']

	# # MU03-WPTE-05 looking for a cd xpr with nitrogen seeding
	# to_do = ['IRVB-MASTU_shot-48594.ptw','IRVB-MASTU_shot-48595.ptw','IRVB-MASTU_shot-48597.ptw','IRVB-MASTU_shot-48598.ptw','IRVB-MASTU_shot-48599.ptw','IRVB-MASTU_shot-48602.ptw','IRVB-MASTU_shot-48603.ptw','IRVB-MASTU_shot-48605.ptw','IRVB-MASTU_shot-48609.ptw','IRVB-MASTU_shot-48617.ptw','IRVB-MASTU_shot-48618.ptw','IRVB-MASTU_shot-48619.ptw','IRVB-MASTU_shot-48620.ptw']



	# # MU02 shots for Kevin
	# to_do = []
	# # sxd scans and repeats
	# to_do = to_do + ['IRVB-MASTU_shot-46707.ptw','IRVB-MASTU_shot-46769.ptw','IRVB-MASTU_shot-46776.ptw','IRVB-MASTU_shot-46791.ptw','IRVB-MASTU_shot-46792.ptw','IRVB-MASTU_shot-46794.ptw','IRVB-MASTU_shot-46860.ptw','IRVB-MASTU_shot-46893.ptw','IRVB-MASTU_shot-46705.ptw','IRVB-MASTU_shot-46904.ptw','IRVB-MASTU_shot-46905.ptw']
	# # elongated divertor scans and repeats
	# to_do = to_do + ['IRVB-MASTU_shot-47079.ptw','IRVB-MASTU_shot-47082.ptw','IRVB-MASTU_shot-47115.ptw','IRVB-MASTU_shot-47118.ptw','IRVB-MASTU_shot-47085.ptw']
	# # conventional divertor scans and repeats
	# to_do = to_do + ['IRVB-MASTU_shot-46762.ptw','IRVB-MASTU_shot-46769.ptw','IRVB-MASTU_shot-46864.ptw','IRVB-MASTU_shot-46866.ptw','IRVB-MASTU_shot-46867.ptw','IRVB-MASTU_shot-46868.ptw','IRVB-MASTU_shot-46891.ptw','IRVB-MASTU_shot-46903.ptw','IRVB-MASTU_shot-46905.ptw','IRVB-MASTU_shot-46702.ptw','IRVB-MASTU_shot-46889.ptw']
	# # shape scans
	# to_do = to_do + ['IRVB-MASTU_shot-46889.ptw','IRVB-MASTU_shot-46895.ptw','IRVB-MASTU_shot-46890.ptw']
	# to_do = to_do + ['IRVB-MASTU_shot-49312.ptw','IRVB-MASTU_shot-49320.ptw']

	# # MU03 shots for Kevin
	# to_do = []
	# conventional divertor scans and repeats L mode
	# to_do = to_do + ['IRVB-MASTU_shot-47950.ptw','IRVB-MASTU_shot-47951.ptw','IRVB-MASTU_shot-47962.ptw','IRVB-MASTU_shot-48144.ptw','IRVB-MASTU_shot-48328.ptw','IRVB-MASTU_shot-48335.ptw','IRVB-MASTU_shot-48336.ptw','IRVB-MASTU_shot-49408.ptw','IRVB-MASTU_shot-49283.ptw']
	# conventional divertor scans and repeats H mode
	# to_do = to_do + ['IRVB-MASTU_shot-49392.ptw','IRVB-MASTU_shot-48561.ptw','IRVB-MASTU_shot-49139.ptw']

	# shots for Ivan Paradela Perez
	# to_do = to_do + ['IRVB-MASTU_shot-46864.ptw','IRVB-MASTU_shot-47079.ptw','IRVB-MASTU_shot-46860.ptw']


	# # Stuart shots MU03-DIV-03
	# to_do = []
	# # DDN
	# to_do = to_do + ['IRVB-MASTU_shot-48361.ptw','IRVB-MASTU_shot-48363.ptw','IRVB-MASTU_shot-48366.ptw','IRVB-MASTU_shot-48367.ptw','IRVB-MASTU_shot-48648.ptw']	# 48370 data is missing
	# # LSN
	# to_do = to_do + ['IRVB-MASTU_shot-48638.ptw','IRVB-MASTU_shot-48639.ptw','IRVB-MASTU_shot-48646.ptw','IRVB-MASTU_shot-48647.ptw','IRVB-MASTU_shot-48651.ptw','IRVB-MASTU_shot-48652.ptw']

	# checking the effect of valve HFS_MID_L02
	# to_do = ['IRVB-MASTU_shot-47218.ptw','IRVB-MASTU_shot-47228.ptw','IRVB-MASTU_shot-47401.ptw','IRVB-MASTU_shot-47402.ptw','IRVB-MASTU_shot-47448.ptw','IRVB-MASTU_shot-47570.ptw','IRVB-MASTU_shot-47580.ptw','IRVB-MASTU_shot-47826.ptw','IRVB-MASTU_shot-47869.ptw','IRVB-MASTU_shot-48063.ptw','IRVB-MASTU_shot-48239.ptw','IRVB-MASTU_shot-48290.ptw','IRVB-MASTU_shot-48325.ptw','IRVB-MASTU_shot-48376.ptw','IRVB-MASTU_shot-48377.ptw','IRVB-MASTU_shot-48378.ptw','IRVB-MASTU_shot-48381.ptw','IRVB-MASTU_shot-48382.ptw','IRVB-MASTU_shot-48576.ptw','IRVB-MASTU_shot-48583.ptw','IRVB-MASTU_shot-48629.ptw','IRVB-MASTU_shot-48633.ptw']

	# # trophy shots from mu02
	# to_do = ['IRVB-MASTU_shot-46976.ptw','IRVB-MASTU_shot-46977.ptw','IRVB-MASTU_shot-46978.ptw','IRVB-MASTU_shot-46979.ptw','IRVB-MASTU_shot-46823.ptw','IRVB-MASTU_shot-48561.ptw']

	# # other H mode shots I'm finding
	# to_do = ['IRVB-MASTU_shot-48594.ptw','IRVB-MASTU_shot-48595.ptw','IRVB-MASTU_shot-48596.ptw','IRVB-MASTU_shot-48564.ptw','IRVB-MASTU_shot-48763.ptw']

	# MU03 snowflake by Vlad Soukhanovskii
	# to_do = ['IRVB-MASTU_shot-49463.ptw','IRVB-MASTU_shot-49464.ptw','IRVB-MASTU_shot-49465.ptw','IRVB-MASTU_shot-49466.ptw','IRVB-MASTU_shot-49467.ptw','IRVB-MASTU_shot-49468.ptw']
	# to_do = ['IRVB-MASTU_shot-49465.ptw','IRVB-MASTU_shot-49467.ptw','IRVB-MASTU_shot-49468.ptw']

	# to_do = []
	# # mu03 DIV-02 by Jack
	# # to_do = to_do + ['IRVB-MASTU_shot-48692.ptw','IRVB-MASTU_shot-46363.ptw','IRVB-MASTU_shot-49373.ptw','IRVB-MASTU_shot-49374.ptw','IRVB-MASTU_shot-48766.ptw','IRVB-MASTU_shot-49464.ptw','IRVB-MASTU_shot-49465.ptw','IRVB-MASTU_shot-49466.ptw','IRVB-MASTU_shot-49467.ptw']
	# # mu03 DIV-03 by Jack
	# # to_do = to_do + ['IRVB-MASTU_shot-49213.ptw','IRVB-MASTU_shot-48651.ptw','IRVB-MASTU_shot-49220.ptw','IRVB-MASTU_shot-49259.ptw','IRVB-MASTU_shot-49262.ptw','IRVB-MASTU_shot-49404.ptw','IRVB-MASTU_shot-49198.ptw']
	# # mu03 DIV-03 by Vlad Soukhanovskii
	# to_do = to_do + ['IRVB-MASTU_shot-49463.ptw','IRVB-MASTU_shot-49464.ptw','IRVB-MASTU_shot-49465.ptw','IRVB-MASTU_shot-49466.ptw','IRVB-MASTU_shot-49467.ptw','IRVB-MASTU_shot-49468.ptw','IRVB-MASTU_shot-48690.ptw','IRVB-MASTU_shot-48692.ptw']


	# data for psi2024 poster
	# to_do = []
	# # ohmic L-mode
	# to_do = to_do + ['IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw']
	# to_do = to_do + ['IRVB-MASTU_shot-47950.ptw','IRVB-MASTU_shot-47973.ptw','IRVB-MASTU_shot-48144.ptw','IRVB-MASTU_shot-48328.ptw','IRVB-MASTU_shot-48335.ptw']
	# # beam heated L-mode
	# to_do = to_do + ['IRVB-MASTU_shot-46866.ptw','IRVB-MASTU_shot-46864.ptw','IRVB-MASTU_shot-46867.ptw','IRVB-MASTU_shot-46868.ptw','IRVB-MASTU_shot-46889.ptw','IRVB-MASTU_shot-46891.ptw','IRVB-MASTU_shot-46903.ptw','IRVB-MASTU_shot-48336.ptw','IRVB-MASTU_shot-49408.ptw','IRVB-MASTU_shot-49283.ptw']
	# # H-mode
	# to_do = to_do + ['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-48596.ptw','IRVB-MASTU_shot-48597.ptw','IRVB-MASTU_shot-46977.ptw','IRVB-MASTU_shot-48599.ptw','IRVB-MASTU_shot-48561.ptw','IRVB-MASTU_shot-49139.ptw','IRVB-MASTU_shot-c.ptw','IRVB-MASTU_shot-49396.ptw','IRVB-MASTU_shot-49400.ptw','IRVB-MASTU_shot-49401.ptw']
	# # mix
	# to_do = to_do + ['IRVB-MASTU_shot-48690.ptw','IRVB-MASTU_shot-48692.ptw','IRVB-MASTU_shot-49463.ptw','IRVB-MASTU_shot-49464.ptw','IRVB-MASTU_shot-49465.ptw','IRVB-MASTU_shot-49466.ptw','IRVB-MASTU_shot-49467.ptw','IRVB-MASTU_shot-49468.ptw']
	# IRVB science paper 2025
	to_do = []
	to_do = to_do + ['IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw']
	to_do = to_do + ['IRVB-MASTU_shot-47950.ptw','IRVB-MASTU_shot-47973.ptw','IRVB-MASTU_shot-48144.ptw']
	to_do = to_do + ['IRVB-MASTU_shot-46866.ptw','IRVB-MASTU_shot-46867.ptw','IRVB-MASTU_shot-46868.ptw','IRVB-MASTU_shot-46891.ptw','IRVB-MASTU_shot-48336.ptw','IRVB-MASTU_shot-49408.ptw']

	# # data for Yasmin Adnrew (female) about LH transition
	# to_do = []
	# # # to_do = to_do + ['IRVB-MASTU_shot-46631.ptw','IRVB-MASTU_shot-47012.ptw','IRVB-MASTU_shot-47078.ptw','IRVB-MASTU_shot-47094.ptw','IRVB-MASTU_shot-47889.ptw','IRVB-MASTU_shot-47909.ptw','IRVB-MASTU_shot-47916.ptw',
	# # # 'IRVB-MASTU_shot-47918.ptw','IRVB-MASTU_shot-48004.ptw','IRVB-MASTU_shot-48172.ptw','IRVB-MASTU_shot-48174.ptw','IRVB-MASTU_shot-48175.ptw','IRVB-MASTU_shot-48176.ptw','IRVB-MASTU_shot-48177.ptw',
	# # # 'IRVB-MASTU_shot-49091.ptw','IRVB-MASTU_shot-49094.ptw','IRVB-MASTU_shot-49095.ptw']
	# to_do = to_do + ['IRVB-MASTU_shot-48004.ptw']

	# data for Bob Kool nature 2024 paper
	# to_do = []
	# to_do = to_do + ['IRVB-MASTU_shot-47080.ptw','IRVB-MASTU_shot-47083.ptw','IRVB-MASTU_shot-47086.ptw','IRVB-MASTU_shot-47116.ptw','IRVB-MASTU_shot-47118.ptw','IRVB-MASTU_shot-47119.ptw','IRVB-MASTU_shot-49303.ptw',
	# 'IRVB-MASTU_shot-48001.ptw','IRVB-MASTU_shot-47998.ptw','IRVB-MASTU_shot-49297.ptw','IRVB-MASTU_shot-49298.ptw']

	# 2024/11/27 for kevin's experiment
	# to_do = []
	# to_do = to_do + ['IRVB-MASTU_shot-50822.ptw','IRVB-MASTU_shot-50823.ptw','IRVB-MASTU_shot-50824.ptw','IRVB-MASTU_shot-50825.ptw','IRVB-MASTU_shot-50826.ptw','IRVB-MASTU_shot-50827.ptw','IRVB-MASTU_shot-50828.ptw',
	# 'IRVB-MASTU_shot-50830.ptw','IRVB-MASTU_shot-50831.ptw','IRVB-MASTU_shot-50832.ptw','IRVB-MASTU_shot-50833.ptw']

	# to_do = np.flip(to_do,axis=0)
	if ext_sequencer!=None:
		try:
			to_do = [to_do[ext_sequencer]]
		except:
			print('no shots left to process')
			sys.exit(1)
	# for name in to_do:
	for name in np.flip(to_do,axis=0):
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
		if coleval.retrive_aborted(name[-9:-4]):	# skip the shot if it was aborted
			continue

		i_day,day = 0,coleval.retrive_shot_date_and_time(name[-9:-4])[0]
		laser_to_analyse=path+day+'/'+name

		try:
			if False:
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

else:	# section to use when analysing only one shot
	# i_day,day = 0,'2021-07-29'
	# name='IRVB-MASTU_shot-44578.ptw'
	# i_day,day = 0,'2021-08-25'
	# name='IRVB-MASTU_shot-44797.ptw'
	# i_day,day = 0,'2021-06-24'
	# name='IRVB-MASTU_shot-44308.ptw'
	# i_day,day = 0,'2021-08-05'
	# name='IRVB-MASTU_shot-44607.ptw'
	# i_day,day = 0,'2021-10-13'
	# name='IRVB-MASTU_shot-45272.ptw'
	# i_day,day = 0,'2021-10-27'
	# name='IRVB-MASTU_shot-45460.ptw'
	# i_day,day = 0,'2021-10-12'
	# name='IRVB-MASTU_shot-47958.ptw'
	# name='IRVB-MASTU_shot-46860.ptw'
	name='IRVB-MASTU_shot-45473.ptw'
	# name='IRVB-MASTU_shot-45371.ptw'
	# name='IRVB-MASTU_shot-49312.ptw'
	# name='IRVB-MASTU_shot-49213.ptw'
	# name='IRVB-MASTU_shot-49312.ptw'
	# name='IRVB-MASTU_shot-49385.ptw'
	# name='IRVB-MASTU_shot-49196.ptw'
	# i_day,day = 0,'2021-10-22'
	# name='IRVB-MASTU_shot-45225.ptw'
	# i_day,day = 0,'2021-10-21'
	# name='IRVB-MASTU_shot-46895.ptw'
	# name='IRVB-MASTU_shot-48324.ptw'
	# name = 'IRVB-MASTU_shot-48636.ptw'
	name = 'IRVB-MASTU_shot-48004.ptw'
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
	# name = 'IRVB-MASTU_shot-45225.ptw'
	name = 'IRVB-MASTU_shot-51422.ptw'

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

	if True:
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
