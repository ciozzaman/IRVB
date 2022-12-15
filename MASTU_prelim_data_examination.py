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
pathparams_BB='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters'
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
# do all of them
to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25','2021-06-29','2021-06-30','2021-07-01','2021-07-06','2021-07-08','2021-07-09','2021-07-15','2021-07-27','2021-07-28','2021-07-29','2021-08-04','2021-08-05','2021-08-06','2021-08-11','2021-08-12','2021-08-13','2021-08-17','2021-08-18','2021-08-19','2021-08-20','2021-08-24','2021-08-25','2021-08-26','2021-08-27','2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04','2021-10-05','2021-10-06','2021-10-07','2021-10-08','2021-10-11','2021-10-12','2021-10-13','2021-10-14','2021-10-15','2021-10-19','2021-10-20','2021-10-21','2021-10-22','2021-10-25','2021-10-26','2021-10-27','2021-10-28']
# MU02
# to_do = ['2022-10-26','2022-10-27','2022-10-28','2022-10-31','2022-11-01','2022-11-02','2022-11-03','2022-11-04','2022-11-07','2022-11-08','2022-11-09','2022-11-10','2022-11-11','2022-11-17','2022-11-18','2022-11-22','2022-11-23']
# to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-f05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22']
# to_do = ['2021-09-08','2021-06-17','2021-08-05']
# to_do = ['2021-10-15','2021-10-12','2021-10-01','2021-09-30']
# to_do = ['2021-08-26']
# to_do = ['2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04']
# to_do = np.flip(to_do,axis=0)
# # for Lingyan
# to_do = ['2021-10-27']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']

# to_do = ['2022-10-26','2022-10-27','2022-10-28','2022-10-31']

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
# for Stuart Handerson, RT-14
# to_do = ['2021-09-09']
# shot_available = [['IRVB-MASTU_shot-44891.ptw','IRVB-MASTU_shot-44892.ptw','IRVB-MASTU_shot-44904.ptw','IRVB-MASTU_shot-44905.ptw']]
# these should show a decent evolution from all attached to inner detached, outer detached and marfe
# to_do = ['2021-10-14','2021-10-19']
# shot_available = [['IRVB-MASTU_shot-45295.ptw','IRVB-MASTU_shot-45296.ptw'],['IRVB-MASTU_shot-45324.ptw','IRVB-MASTU_shot-45327.ptw']]
# # for Lingyan
# to_do = ['2021-10-27']
# shot_available = [['IRVB-MASTU_shot-45446.ptw','IRVB-MASTU_shot-45448.ptw','IRVB-MASTU_shot-45450.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45458.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45460.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw']]

# to_do = ['2021-10-27','2021-10-12','2021-10-22','2021-10-13','2021-10-22','2021-10-28','2021-08-05','2021-08-17','2021-08-25','2021-08-26']
# shot_available = [['IRVB-MASTU_shot-45439.ptw','IRVB-MASTU_shot-45443.ptw','IRVB-MASTU_shot-45444.ptw','IRVB-MASTU_shot-45446.ptw','IRVB-MASTU_shot-45448.ptw','IRVB-MASTU_shot-45450.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45458.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45460.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw'],['IRVB-MASTU_shot-45239.ptw'],['IRVB-MASTU_shot-45391.ptw'],['IRVB-MASTU_shot-45272.ptw'],['IRVB-MASTU_shot-45398.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45400.ptw'],['IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw'],['IRVB-MASTU_shot-44607.ptw'],['IRVB-MASTU_shot-44697.ptw','IRVB-MASTU_shot-44699.ptw','IRVB-MASTU_shot-44700.ptw','IRVB-MASTU_shot-44701.ptw'],['IRVB-MASTU_shot-44797.ptw'],['IRVB-MASTU_shot-44822.ptw']]

# simpler way of starting works
# to_do = ['IRVB-MASTU_shot-45382.ptw','IRVB-MASTU_shot-45466.ptw','IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45473.ptw']
# to_do = ['IRVB-MASTU_shot-45371.ptw','IRVB-MASTU_shot-45372.ptw','IRVB-MASTU_shot-45380.ptw','IRVB-MASTU_shot-45390.ptw','IRVB-MASTU_shot-45391.ptw','IRVB-MASTU_shot-45398.ptw','IRVB-MASTU_shot-45400.ptw','IRVB-MASTU_shot-45439.ptw','IRVB-MASTU_shot-45443.ptw','IRVB-MASTU_shot-45444.ptw','IRVB-MASTU_shot-45450.ptw','IRVB-MASTU_shot-45456.ptw','IRVB-MASTU_shot-45459.ptw','IRVB-MASTU_shot-45461.ptw','IRVB-MASTU_shot-45462.ptw','IRVB-MASTU_shot-45463.ptw','IRVB-MASTU_shot-45464.ptw','IRVB-MASTU_shot-45465.ptw']
# to_do = ['IRVB-MASTU_shot-45046.ptw','IRVB-MASTU_shot-45047.ptw','IRVB-MASTU_shot-45048.ptw','IRVB-MASTU_shot-45056.ptw','IRVB-MASTU_shot-45057.ptw','IRVB-MASTU_shot-45058.ptw','IRVB-MASTU_shot-45059.ptw','IRVB-MASTU_shot-45060.ptw','IRVB-MASTU_shot-45062.ptw','IRVB-MASTU_shot-45063.ptw','IRVB-MASTU_shot-45066.ptw','IRVB-MASTU_shot-45071.ptw','IRVB-MASTU_shot-45081.ptw','IRVB-MASTU_shot-45143.ptw','IRVB-MASTU_shot-45239.ptw','IRVB-MASTU_shot-45241.ptw','IRVB-MASTU_shot-45243.ptw','IRVB-MASTU_shot-45244.ptw','IRVB-MASTU_shot-45245.ptw','IRVB-MASTU_shot-45246.ptw','IRVB-MASTU_shot-45353.ptw']	# SXD fuelled from high field side
# # MU01-EXH-20
# to_do = ['IRVB-MASTU_shot-44816.ptw','IRVB-MASTU_shot-44817.ptw','IRVB-MASTU_shot-44818.ptw','IRVB-MASTU_shot-44819.ptw','IRVB-MASTU_shot-44820.ptw','IRVB-MASTU_shot-44821.ptw','IRVB-MASTU_shot-44822.ptw','IRVB-MASTU_shot-45391.ptw','IRVB-MASTU_shot-45397.ptw','IRVB-MASTU_shot-45398.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45400.ptw','IRVB-MASTU_shot-45401.ptw']
# for the paper
# to_do = ['IRVB-MASTU_shot-44647.ptw','IRVB-MASTU_shot-45409.ptw','IRVB-MASTU_shot-45225.ptw','IRVB-MASTU_shot-45351.ptw']
# to_do = ['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-44892.ptw']
# to_do = ['IRVB-MASTU_shot-45401.ptw']
# shots for the RSI papre
# to_do = ['IRVB-MASTU_shot-44647.ptw','IRVB-MASTU_shot-45409.ptw','IRVB-MASTU_shot-45225.ptw','IRVB-MASTU_shot-45351.ptw','IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-44892.ptw','IRVB-MASTU_shot-45295.ptw','IRVB-MASTU_shot-45328.ptw']
# shots for the science paper
# to_do = ['IRVB-MASTU_shot-45371.ptw']
# shots for the CD radiator location comparison shots
# to_do = ['IRVB-MASTU_shot-45473.ptw','IRVB-MASTU_shot-45470.ptw','IRVB-MASTU_shot-45469.ptw','IRVB-MASTU_shot-45468.ptw','IRVB-MASTU_shot-45327.ptw','IRVB-MASTU_shot-45326.ptw','IRVB-MASTU_shot-45325.ptw','IRVB-MASTU_shot-45324.ptw','IRVB-MASTU_shot-45323.ptw','IRVB-MASTU_shot-45322.ptw','IRVB-MASTU_shot-45320.ptw','IRVB-MASTU_shot-45303.ptw','IRVB-MASTU_shot-45302.ptw','IRVB-MASTU_shot-45296.ptw','IRVB-MASTU_shot-45295.ptw','IRVB-MASTU_shot-45293.ptw','IRVB-MASTU_shot-45142.ptw','IRVB-MASTU_shot-45126.ptw','IRVB-MASTU_shot-45088.ptw','IRVB-MASTU_shot-45371.ptw']
# shots for the CD BEAMS shots for the cross comparison
to_do = ['IRVB-MASTU_shot-45401.ptw','IRVB-MASTU_shot-45399.ptw','IRVB-MASTU_shot-45311.ptw','IRVB-MASTU_shot-45310.ptw','IRVB-MASTU_shot-45309.ptw','IRVB-MASTU_shot-45306.ptw','IRVB-MASTU_shot-45304.ptw','IRVB-MASTU_shot-45299.ptw','IRVB-MASTU_shot-45286.ptw','IRVB-MASTU_shot-45272.ptw','IRVB-MASTU_shot-45271.ptw','IRVB-MASTU_shot-45270.ptw','IRVB-MASTU_shot-45268.ptw','IRVB-MASTU_shot-45267.ptw','IRVB-MASTU_shot-45266.ptw','IRVB-MASTU_shot-45264.ptw','IRVB-MASTU_shot-45262.ptw','IRVB-MASTU_shot-45261.ptw','IRVB-MASTU_shot-45252.ptw','IRVB-MASTU_shot-45251.ptw','IRVB-MASTU_shot-45237.ptw','IRVB-MASTU_shot-45236.ptw','IRVB-MASTU_shot-45212.ptw','IRVB-MASTU_shot-45193.ptw','IRVB-MASTU_shot-45194.ptw','IRVB-MASTU_shot-45175.ptw','IRVB-MASTU_shot-45170.ptw','IRVB-MASTU_shot-45167.ptw','IRVB-MASTU_shot-45132.ptw','IRVB-MASTU_shot-45125.ptw','IRVB-MASTU_shot-45097.ptw','IRVB-MASTU_shot-45091.ptw','IRVB-MASTU_shot-45006.ptw','IRVB-MASTU_shot-44969.ptw','IRVB-MASTU_shot-44968.ptw','IRVB-MASTU_shot-44967.ptw','IRVB-MASTU_shot-44960.ptw']
# to_do = np.flip(to_do,axis=0)

continue_after_FAST = False
override_FAST_analysis = True
do_inversions = False

every_pixel_independent = False
overwrite_oscillation_filter = True
overwrite_binning = True
skip_second_pass = False

if overwrite_oscillation_filter:
	overwrite_binning = True
if False:
	# for i_day,day in enumerate(to_do):
	for i_day,day in enumerate(np.flip(to_do,axis=0)):
		# for name in shot_available[i_day]:
		for name in np.flip(shot_available,axis=0)[i_day]:
			laser_to_analyse=path+day+'/'+name

			if False:
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
			else:
				pass_number = 0
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
				pass_number = 1
				exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())

elif False:
	for name in to_do:
		shot_list = get_data(path+'shot_list2.ods')
		temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
		for i in range(1,len(shot_list['Sheet1'])):
			if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
				date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
				break
		i_day,day = 0,str(date.date())
		laser_to_analyse=path+day+'/'+name

		if True:
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
		else:
			pass_number = 0
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
			pass_number = 1
			exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())

else:
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
	# name='IRVB-MASTU_shot-45239.ptw'
	# i_day,day = 0,'2021-10-22'
	name='IRVB-MASTU_shot-45473.ptw'
	# i_day,day = 0,'2021-10-21'
	# name='IRVB-MASTU_shot-45371.ptw'

	path = '/home/ffederic/work/irvb/MAST-U/'
	shot_list = get_data(path+'shot_list2.ods')
	temp1 = (np.array(shot_list['Sheet1'][0])=='shot number').argmax()
	for i in range(1,len(shot_list['Sheet1'])):
		if shot_list['Sheet1'][i][temp1] == int(name[-9:-4]):
			date = shot_list['Sheet1'][i][(np.array(shot_list['Sheet1'][0])=='date').argmax()]
			break
	i_day,day = 0,str(date.date())
	# i_day,day = 0,'2022-12-08'

	laser_to_analyse=path+day+'/'+name

	if True:
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2_BB.py").read())
	else:
		pass_number = 0
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
		pass_number = 1
		exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_manual_plots.py").read())
	# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3.py").read())

print('all done')
