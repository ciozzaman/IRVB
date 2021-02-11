# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 14
figure_index=0

# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'

f = []
for (dirpath, dirnames, filenames) in os.walk(pathparams):
	f.append(dirnames)
parameters_available = f[0]
parameters_available_int_time = []
parameters_available_framerate = []
for path in parameters_available:
	parameters_available_int_time.append(np.float(path[:path.find('ms')]))
	parameters_available_framerate.append(np.float(path[path.find('ms')+2:path.find('Hz')]))
parameters_available_int_time = np.array(parameters_available_int_time)
parameters_available_framerate = np.array(parameters_available_framerate)



############


index = 0
collect = np.ones((16,max_ROI[0][1]+1,max_ROI[1][1]+1))*np.nan
for i_index,index in enumerate(range(16)):
	laser_to_analyse = vacuum5[index]
	laser_dict = np.load(laser_to_analyse+'.npz')
	ROI = vacuumROI5[index]
	if ROI=='ff':
		ROI = max_ROI
	data_time_avg_counts = laser_dict['data_time_avg_counts'][0]
	data_time_avg_counts_std = laser_dict['data_time_avg_counts_std'][0]
	collect[i_index][ROI[0][0]:ROI[0][1]+1,ROI[1][0]:ROI[1][1]+1] = data_time_avg_counts


print(np.nanmean(collect[0:10]-collect[0],axis=(1,2)))
print(np.nanstd(collect[0:10]-collect[0],axis=(1,2)))


# OUTCOME
# the behaviour is cetrainly weird. when the ROI is centered things stay the same, as you would ecpect. when it is not they change.
# this might be ok if the deltaT/deltaCounts is the same, but I didn't check that. in 2018 I checked that from 383Hz to 994Hz the coefficients stay the same, and this happens for a centred ROI.


plt.figure(figsize=(20, 10))
plt.imshow(collect[0],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[1],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[2],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[5],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[6],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[5]-collect[6],'rainbow',origin='lower',vmax=100,vmin=-100)
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[3],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)

plt.figure(figsize=(20, 10))
plt.imshow(collect[7],'rainbow',origin='lower',vmax=collect[0].max(),vmin=collect[0].min())
plt.colorbar().set_label('Temp [°C]')
plt.title('index '+str(index))
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.001)



n=3
parameter_index = 3
temp = pathparams+'/'+parameters_available[parameter_index]+'/numcoeff'+str(n)+'/average'
fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(parameters_available_int_time[parameter_index])+'ms.npz')
params_dict=np.load(fullpathparams)
params = params_dict['coeff']
errparams = params_dict['errcoeff']

params1 = np.ones((2,max_ROI[0][1]+1,max_ROI[1][1]+1,n))*np.nan
params1[:,96:64+96] = params

parameter_index = 2
temp = pathparams+'/'+parameters_available[parameter_index]+'/numcoeff'+str(n)+'/average'
fullpathparams=os.path.join(temp,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(parameters_available_int_time[parameter_index])+'ms.npz')
params_dict=np.load(fullpathparams)
params = params_dict['coeff']
errparams = params_dict['errcoeff']

for index in range(3):
	plt.figure(figsize=(20, 10))
	plt.imshow((params[0][:,:,index]-params1[0][:,:,index])/params[0][:,:,index],'rainbow',origin='lower')
	plt.colorbar().set_label('fractional difference [au]')
	plt.title('coefficient '+str(index))
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.pause(0.001)

# You can see here that with even windowing the difference in the parameters is negligible, so I can use the parameters at 383 in all cases on even windowing

#
