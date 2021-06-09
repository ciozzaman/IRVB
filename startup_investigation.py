# Created 13/01/2020
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 14

# degree of polynomial of choice
n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
# reference frames

laser_to_analyse = '/home/ffederic/work/irvb/startup_investigation/Rec-000030'
try:
	laser_dict = np.load(laser_to_analyse+'.npz')
except:
	print('missing '+laser_to_analyse+'.npz'+' file. rigenerated')
	full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse+'.ats')
	np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
laser_dict = np.load(laser_to_analyse+'.npz')
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
time_of_experiment_s = (time_of_experiment - time_of_experiment[0])*1e-6

fig, ax = plt.subplots( 4,1,figsize=(7, 13), squeeze=False, sharex=True)
plot_index = 0
ax[plot_index,0].plot(time_of_experiment_s,np.mean(data,axis=(1,2)))
fit = np.polyfit(time_of_experiment_s[time_of_experiment_s<7],np.mean(data,axis=(1,2))[time_of_experiment_s<7],1)
ax[plot_index,0].plot(time_of_experiment_s,np.polyval(fit,time_of_experiment_s),'--')
ax[plot_index,0].set_ylabel('mean counts [au]')
ax[plot_index,0].grid()
fig.suptitle('inf/1/13 2ms')
# plt.pause(0.01)

plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_0'])
ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_3'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['DetectorTemp'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(DetectorTemp)')
ax[plot_index,0].grid()
ax[plot_index,0].set_xlabel('time [s]')
plt.savefig(laser_to_analyse+'.eps', bbox_inches='tight')
plt.close('all')



laser_to_analyse = '/home/ffederic/work/irvb/startup_investigation/Rec-000034'
try:
	laser_dict = np.load(laser_to_analyse+'.npz')
except:
	print('missing '+laser_to_analyse+'.npz'+' file. rigenerated')
	full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse+'.ats')
	np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
laser_dict = np.load(laser_to_analyse+'.npz')
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
time_of_experiment_s = (time_of_experiment - time_of_experiment[0])*1e-6

fig, ax = plt.subplots( 4,1,figsize=(7, 13), squeeze=False, sharex=True)
plot_index = 0
ax[plot_index,0].plot(time_of_experiment_s,np.mean(data,axis=(1,2)))
fit = np.polyfit(time_of_experiment_s[time_of_experiment_s<7],np.mean(data,axis=(1,2))[time_of_experiment_s<7],1)
ax[plot_index,0].plot(time_of_experiment_s,np.polyval(fit,time_of_experiment_s),'--')
ax[plot_index,0].set_ylabel('mean counts [au]')
ax[plot_index,0].grid()
fig.suptitle('inf/2/13 2ms')
# plt.pause(0.01)

plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_0'])
ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_3'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['DetectorTemp'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(DetectorTemp)')
ax[plot_index,0].grid()
ax[plot_index,0].set_xlabel('time [s]')
plt.savefig(laser_to_analyse+'.eps', bbox_inches='tight')
plt.close('all')

laser_to_analyse = '/home/ffederic/work/irvb/startup_investigation/Rec-000046'
try:
	laser_dict = np.load(laser_to_analyse+'.npz')
except:
	print('missing '+laser_to_analyse+'.npz'+' file. rigenerated')
	full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse+'.ats')
	np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
laser_dict = np.load(laser_to_analyse+'.npz')
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
time_of_experiment_s = (time_of_experiment - time_of_experiment[0])*1e-6

fig, ax = plt.subplots( 4,1,figsize=(7, 13), squeeze=False, sharex=True)
plot_index = 0
ax[plot_index,0].plot(time_of_experiment_s,np.mean(data,axis=(1,2)))
fit = np.polyfit(time_of_experiment_s[time_of_experiment_s<7],np.mean(data,axis=(1,2))[time_of_experiment_s<7],1)
# ax[plot_index,0].plot(time_of_experiment_s,np.polyval(fit,time_of_experiment_s),'--')
ax[plot_index,0].set_ylabel('mean counts [au]')
ax[plot_index,0].grid()
fig.suptitle('7/1/13 1ms')
# plt.pause(0.01)

plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_0'])
ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_3'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['DetectorTemp'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(DetectorTemp)')
ax[plot_index,0].grid()
ax[plot_index,0].set_xlabel('time [s]')
plt.savefig(laser_to_analyse+'.eps', bbox_inches='tight')
plt.close('all')


laser_to_analyse = '/home/ffederic/work/irvb/startup_investigation/Rec-000052'
try:
	laser_dict = np.load(laser_to_analyse+'.npz')
except:
	print('missing '+laser_to_analyse+'.npz'+' file. rigenerated')
	full_saved_file_dict = coleval.ats_to_dict(laser_to_analyse+'.ats')
	np.savez_compressed(laser_to_analyse,**full_saved_file_dict)
laser_dict = np.load(laser_to_analyse+'.npz')
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
data = laser_dict['data']
time_of_experiment = laser_dict['time_of_measurement']
time_of_experiment_s = (time_of_experiment - time_of_experiment[0])*1e-6

fig, ax = plt.subplots( 4,1,figsize=(7, 13), squeeze=False, sharex=True)
plot_index = 0
ax[plot_index,0].plot(time_of_experiment_s,np.mean(data,axis=(1,2)))
fit = np.polyfit(time_of_experiment_s[time_of_experiment_s<7],np.mean(data,axis=(1,2))[time_of_experiment_s<7],1)
# ax[plot_index,0].plot(time_of_experiment_s,np.polyval(fit,time_of_experiment_s),'--')
ax[plot_index,0].set_ylabel('mean counts [au]')
ax[plot_index,0].grid()
fig.suptitle('15/2/13 2ms')
# plt.pause(0.01)

plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_0'])
ax[plot_index,0].set_ylabel('ambient temp [K]\n(SensorTemp_0)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['SensorTemp_3'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(SensorTemp_3)')
ax[plot_index,0].grid()
plot_index += 1
ax[plot_index,0].plot(time_of_experiment_s,laser_dict['DetectorTemp'])
ax[plot_index,0].set_ylabel('detector temp [K]\n(DetectorTemp)')
ax[plot_index,0].grid()
ax[plot_index,0].set_xlabel('time [s]')
plt.savefig(laser_to_analyse+'.eps', bbox_inches='tight')
plt.close('all')
