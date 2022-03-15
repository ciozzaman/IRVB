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



i_day,day = 0,'2021-10-27'
name='IRVB-MASTU_shot-45458.ptw'
laser_to_analyse=path+day+'/'+name

full_saved_file_dict_FAST = np.load(laser_to_analyse[:-4]+'_FAST'+'.npz')
full_saved_file_dict_FAST.allow_pickle=True
inverted_dict = full_saved_file_dict_FAST['inverted_dict']
grid_resolution = 2	# cm
time_full_binned_crop = inverted_dict.all()[str(grid_resolution)]['time_full_binned_crop']
inverted_data = inverted_dict.all()[str(grid_resolution)]['inverted_data']
foil_power = inverted_dict.all()[str(grid_resolution)]['foil_power']
foil_power_std = inverted_dict.all()[str(grid_resolution)]['foil_power_std']
outer_leg_tot_rad_power_all = inverted_dict.all()[str(grid_resolution)]['outer_leg_tot_rad_power_all']
inner_leg_tot_rad_power_all = inverted_dict.all()[str(grid_resolution)]['inner_leg_tot_rad_power_all']
core_tot_rad_power_all = inverted_dict.all()[str(grid_resolution)]['core_tot_rad_power_all']
sxd_tot_rad_power_all = inverted_dict.all()[str(grid_resolution)]['sxd_tot_rad_power_all']
x_point_tot_rad_power_all = inverted_dict.all()[str(grid_resolution)]['x_point_tot_rad_power_all']
outer_leg_tot_rad_power_sigma_all = inverted_dict.all()[str(grid_resolution)]['outer_leg_tot_rad_power_sigma_all']
inner_leg_tot_rad_power_sigma_all = inverted_dict.all()[str(grid_resolution)]['inner_leg_tot_rad_power_sigma_all']
core_tot_rad_power_sigma_all = inverted_dict.all()[str(grid_resolution)]['core_tot_rad_power_sigma_all']
sxd_tot_rad_power_sigma_all = inverted_dict.all()[str(grid_resolution)]['sxd_tot_rad_power_sigma_all']
x_point_tot_rad_power_sigma_all = inverted_dict.all()[str(grid_resolution)]['x_point_tot_rad_power_sigma_all']


plt.figure(figsize=(20, 15))
plt.plot(time_full_binned_crop,outer_leg_tot_rad_power_all/1e3,label='outer_leg')
plt.plot(time_full_binned_crop,sxd_tot_rad_power_all/1e3,label='sxd')
plt.plot(time_full_binned_crop,inner_leg_tot_rad_power_all/1e3,label='inner_leg')
plt.plot(time_full_binned_crop,core_tot_rad_power_all/1e3,label='core')
plt.plot(time_full_binned_crop,x_point_tot_rad_power_all/1e3,label='x_point')
plt.plot(time_full_binned_crop,(outer_leg_tot_rad_power_all+inner_leg_tot_rad_power_all+core_tot_rad_power_all)/1e3,label='tot')
plt.title('radiated power in the lower half of the machine')
plt.legend(loc='best', fontsize='x-small')
plt.xlabel('time [s]')
plt.ylabel('power [kW]')
plt.grid()






##########
