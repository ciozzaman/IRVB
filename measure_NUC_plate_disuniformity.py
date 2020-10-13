# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())



# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = files2[-1]
# framerate of the IR camera in Hz
framerate = 50
# integration time of the camera in ms
inttime = 1
# filestype
type = '_stat.npy'
# type='csv'

fullpathparams = os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy')
params = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
filenames = coleval.all_file_names(pathfiles, type)[0]

# datashort=np.load(os.path.join(pathfiles,filenames))
# data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
# data[:,:,64:128,128:]=datashort
basecounts1 = np.array(np.load(os.path.join(pathfiles, filenames))[0])


plt.figure()
plt.title('Counts averaged on 100 frames, ' + str(inttime) + 'ms, '+str(framerate)+'Hz, NUC temp='+str(temperature2[-1])+'°C, from\n' + pathfiles)
plt.imshow(basecounts1, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)


# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = files6[-1]
# framerate of the IR camera in Hz
framerate = 50
# integration time of the camera in ms
inttime = 1
# filestype
type = '_stat.npy'
# type='csv'

fullpathparams = os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy')
params = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
filenames = coleval.all_file_names(pathfiles, type)[0]

# datashort=np.load(os.path.join(pathfiles,filenames))
# data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
# data[:,:,64:128,128:]=datashort
basecounts2 = np.array(np.load(os.path.join(pathfiles, filenames))[0])

plt.figure()
plt.title('Counts averaged on 100 frames, ' + str(inttime) + 'ms, '+str(framerate)+'Hz, NUC temp='+str(temperature6[-1])+'°C, from\n' + pathfiles)
plt.imshow(basecounts2, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)




compare = np.mean(basecounts2,axis=(0,1))/np.mean(basecounts1,axis=(0,1))*basecounts1-basecounts2


plt.figure()
plt.title('Comparison just to check that there is not difference\nin the features of two images at room temperature')
plt.imshow(compare, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)


basecounts = (np.mean(basecounts2,axis=(0,1))/np.mean(basecounts1,axis=(0,1))*basecounts1 + basecounts2 )/2

plt.figure()
plt.title("Reference I'll use")
plt.imshow(basecounts, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)


plt.figure()
type = '_stat.npy'
experiment_index = 2
for experiment_to_check,temperature in [[files2,temperature2],[files3,temperature3],[files4,temperature4],[files5,temperature5],[files6,temperature6]]:
# for experiment_to_check, temperature in [[files2, temperature2], [files6, temperature6]]:
	temp_difference = []
	plate_temperature = []
	filenames = coleval.all_file_names(experiment_to_check[-1], type)[0]
	basecounts = np.array(np.load(os.path.join(experiment_to_check[-1], filenames))[0])
	for index,pathfiles in enumerate(experiment_to_check):
		filenames = coleval.all_file_names(pathfiles, type)[0]
		data = np.array(np.load(os.path.join(pathfiles, filenames))[0])

		datatemp_orig = coleval.count_to_temp_poly2([[data]], params, errparams,averaged_params=True)[0][0,0]
		data_est = basecounts*np.mean(data,axis=(0,1))/np.mean(basecounts,axis=(0,1))
		datatemp_est = coleval.count_to_temp_poly2([[data_est]], params, errparams,averaged_params=True)[0][0,0]
		difference = datatemp_orig-datatemp_est
		average_difference = coleval.average_frame(difference, 4)
		temp_difference.append(np.max(average_difference)-np.min(average_difference))
		# plate_temperature.append(temperature[index])
		plate_temperature.append(temperature[index] - temperature[-1])
	plt.plot(plate_temperature, temp_difference, 'o',label = 'temperature ramp ' +str(experiment_index))
	experiment_index+=1





plt.xlabel('Temperature difference to room temperature [°C]')
plt.ylabel('Temperature difference across NUC plate [°C]')
plt.legend(loc='best')
plt.pause(0.001)



# It is not really related, but from here I also do something that I need to show in the "Report on initial bench top activities v1"

plt.figure()
type = '_stat.npy'
experiment_index = 2
temp_all=[]
counts_all = []
for experiment_to_check,temperature in [[files2,temperature2],[files3,temperature3],[files4,temperature4],[files5,temperature5],[files6,temperature6]]:
# for experiment_to_check, temperature in [[files2, temperature2], [files6, temperature6]]:
	temp = []
	counts = []
	for index,pathfiles in enumerate(experiment_to_check):
		filenames = coleval.all_file_names(pathfiles, type)[0]
		data = np.array(np.load(os.path.join(pathfiles, filenames))[0])
		counts.append(data[128,160])
		temp.append(temperature[index])
	# plt.plot(counts, temp, 'o',label = 'temperature ramp ' +str(experiment_index))
	plt.plot(counts, temp, 'o')
	experiment_index+=1
	temp_all.append(temp)
	counts_all.append(counts)

temp_all = coleval.flatten_full(temp_all)
counts_all = coleval.flatten_full(counts_all)

for grade in [1,2,3,4]:
	guess=np.ones(grade+1)
	temp1,temp2=curve_fit(coleval.polygen3(grade+1), counts_all,temp_all, p0=guess, maxfev=100000000)
	score=rsquared(temp_all,coleval.polygen3(grade+1)(counts_all,*temp1))
	x=np.sort(counts_all)
	plt.plot(x,coleval.polygen3(grade+1)(x,*temp1),label='polynomial order '+str(grade)+', R2='+str(np.around(score,decimals=3)))

plt.xlabel('Counts [au]')
plt.ylabel('Temperature [°C]')
plt.title("Comparison for pixel horiz=160 vert=128")
plt.legend(loc='best')
plt.pause(0.001)




# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = files13[-1]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '_stat.npy'
# type='csv'

filenames = coleval.all_file_names(pathfiles, type)[0]

# datashort=np.load(os.path.join(pathfiles,filenames))
# data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
# data[:,:,64:128,128:]=datashort
data1 = np.array(np.load(os.path.join(pathfiles, filenames))[0])
aberration = data1 -basecounts*np.mean(data1,axis=(0,1))/np.mean(basecounts,axis=(0,1))

plt.figure()
plt.title('aberration in [177,204]')
plt.plot([177,177],[204-50,204+50],'k')
plt.plot([177-50,177+50],[204,204],'k')
plt.imshow(aberration, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)


# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = files11[-1]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '_stat.npy'
# type='csv'

filenames = coleval.all_file_names(pathfiles, type)[0]

# datashort=np.load(os.path.join(pathfiles,filenames))
# data=np.multiply(3000,np.ones((1,np.shape(datashort)[1],256,320)))
# data[:,:,64:128,128:]=datashort
data2 = np.array(np.load(os.path.join(pathfiles, filenames))[0])
aberration = data2 -basecounts*np.mean(data2,axis=(0,1))/np.mean(basecounts,axis=(0,1))

plt.figure()
plt.title('aberration in [177,128]')
plt.plot([177,177],[128-50,128+50],'k')
plt.plot([177-50,177+50],[128,128],'k')
plt.imshow(aberration, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)




aberration_record = []

for index_temp,ref_temp in enumerate(np.array(temperature11)-temperature11[-1]):
	index_search_temp = coleval.find_nearest_index(np.array(temperature2)-temperature2[-1],ref_temp)
	search_temp = (np.array(temperature2)-temperature2[-1])[index_search_temp]

	if np.abs(ref_temp-search_temp)>0.5:
		print('rejected')
		print(ref_temp)
		print(search_temp)
		continue

	# degree of polynomial of choice
	n = 3
	# folder of the parameters path
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles = files2[index_search_temp]
	# framerate of the IR camera in Hz
	framerate = 383
	# integration time of the camera in ms
	inttime = 1
	# filestype
	type = '_stat.npy'
	# type='csv'

	filenames = coleval.all_file_names(pathfiles, type)[0]
	ref = np.array(np.load(os.path.join(pathfiles, filenames))[0])

	# degree of polynomial of choice
	n = 3
	# folder of the parameters path
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles = files11[index_temp]
	# framerate of the IR camera in Hz
	framerate = 383
	# integration time of the camera in ms
	inttime = 1
	# filestype
	type = '_stat.npy'
	# type='csv'

	filenames = coleval.all_file_names(pathfiles, type)[0]
	data = np.array(np.load(os.path.join(pathfiles, filenames))[0])


	temp1=[]
	temp2=[]
	for i in range(np.shape(ref)[0]):
		for j in range(np.shape(deformed)[1]):
			if ((((i - 130) ** 2 + (j - 177) ** 2) > 10000 ** 2) or (((i - 130) ** 2 + (j - 177) ** 2) < 70 ** 2)):
				continue
			temp1.append(ref[i,j])
			temp2.append(data[i,j])
	temp1=np.mean(temp1)
	temp2 = np.mean(temp2)
	aberration = data - ref * temp2 / temp1

	# aberration = data -ref*np.mean(data,axis=(0,1))/np.mean(ref,axis=(0,1))

	aberration_record.append(aberration)

aberration_mean1 = np.mean(aberration_record,axis=(0))
aberration_std1 = np.std(aberration_record,axis=(0))

plt.figure()
plt.title('aberration in [177,130] file11 temp diff '+str(ref_temp)+' file2 temp diff '+str(search_temp))
plt.plot([177,177],[130-50,130+50],'k')
plt.plot([177-50,177+50],[130,130],'k')
plt.imshow(coleval.average_frame(aberration_mean1, 4), 'rainbow', origin='lower',vmax=5)
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)

plt.figure()
plt.title('aberration in [177,130] file11 temp diff '+str(ref_temp)+' file2 temp diff '+str(search_temp))
plt.plot([177,177],[130-50,130+50],'k')
plt.plot([177-50,177+50],[130,130],'k')
plt.imshow(aberration_std1, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)







aberration_record = []

for index_temp,ref_temp in enumerate(np.array(temperature13[10:])-temperature13[-1]):
	index_search_temp = coleval.find_nearest_index(np.array(temperature2)-temperature2[-1],ref_temp)
	search_temp = (np.array(temperature2)-temperature2[-1])[index_search_temp]

	if np.abs(ref_temp-search_temp)>0.5:
		print('rejected')
		print(ref_temp)
		print(search_temp)
		continue

	# degree of polynomial of choice
	n = 3
	# folder of the parameters path
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles = files2[index_search_temp]
	# framerate of the IR camera in Hz
	framerate = 383
	# integration time of the camera in ms
	inttime = 1
	# filestype
	type = '_stat.npy'
	# type='csv'

	filenames = coleval.all_file_names(pathfiles, type)[0]
	ref = np.array(np.load(os.path.join(pathfiles, filenames))[0])

	# degree of polynomial of choice
	n = 3
	# folder of the parameters path
	pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
	# folder to read
	pathfiles = files13[10:][index_temp]
	# framerate of the IR camera in Hz
	framerate = 383
	# integration time of the camera in ms
	inttime = 1
	# filestype
	type = '_stat.npy'
	# type='csv'

	filenames = coleval.all_file_names(pathfiles, type)[0]
	data = np.array(np.load(os.path.join(pathfiles, filenames))[0])

	temp1=[]
	temp2=[]
	for i in range(np.shape(ref)[0]):
		for j in range(np.shape(deformed)[1]):
			if ((((i - 204) ** 2 + (j - 177) ** 2) > 10000 ** 2) or (((i - 204) ** 2 + (j - 177) ** 2) < 70 ** 2)):
				continue
			temp1.append(ref[i,j])
			temp2.append(data[i,j])
	temp1=np.mean(temp1)
	temp2 = np.mean(temp2)
	aberration = data - ref * temp2 / temp1

	# aberration = data -ref*np.mean(data,axis=(0,1))/np.mean(ref,axis=(0,1))
	aberration_record.append(aberration)

aberration_mean2 = np.mean(aberration_record,axis=(0))
aberration_std2 = np.std(aberration_record,axis=(0))

plt.figure()
plt.title('aberration in [177,204] file13 temp diff '+str(ref_temp)+' file2 temp diff '+str(search_temp))
plt.plot([177,177],[204-50,204+50],'k')
plt.plot([177-50,177+50],[204,204],'k')
plt.imshow(coleval.average_frame(aberration_mean2, 4), 'rainbow', origin='lower',vmax=5)
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)

plt.figure()
plt.title('aberration in [177,204] file13 temp diff '+str(ref_temp)+' file2 temp diff '+str(search_temp))
plt.plot([177,177],[204-50,204+50],'k')
plt.plot([177-50,177+50],[204,204],'k')
plt.imshow(aberration_std2, 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)





v_shift = 204-130-5
h_shift = 177-177



deformed = copy.deepcopy(aberration_mean1)
for i in range(np.shape(deformed)[0]):
	if (i + v_shift > np.shape(deformed)[0] - 1 or i + v_shift < 0):
		continue
	for j in range(np.shape(deformed)[1]):
		if (j + h_shift > np.shape(deformed)[1] - 1 or j + h_shift < 0):
			continue
		# if (aberration_mean2[i + v_shift, j + h_shift])>20:
		# 	continue
		if (((i-130)**2+(j-177)**2)>110**2):
			continue
		deformed[i, j] -= aberration_mean2[i + v_shift, j + h_shift]
plt.figure()
# plt.plot([177,177],[130-50,130+50],'k')
# plt.plot([177-50,177+50],[130,130],'k')
# plt.plot([177, 177], [204 - 50, 204 + 50], 'k')
# plt.plot([177 - 50, 177 + 50], [204, 204], 'k')
plt.imshow(coleval.average_frame(deformed, 4), 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)


v_shift = -v_shift

deformed = copy.deepcopy(aberration_mean2)
for i in range(np.shape(deformed)[0]):
	if (i + v_shift > np.shape(deformed)[0] - 1 or i + v_shift < 0):
		continue
	for j in range(np.shape(deformed)[1]):
		if (j + h_shift > np.shape(deformed)[1] - 1 or j + h_shift < 0):
			continue
		# if (aberration_mean1[i + v_shift, j + h_shift])>20:
		# 	continue
		if (((i - 204) ** 2 + (j - 177) ** 2) > 110 ** 2):
			continue
		deformed[i, j] -= aberration_mean1[i + v_shift, j + h_shift]
plt.figure()
# plt.plot([177,177],[130-50,130+50],'k')
# plt.plot([177-50,177+50],[130,130],'k')
# plt.plot([177, 177], [204 - 50, 204 + 50], 'k')
# plt.plot([177 - 50, 177 + 50], [204, 204], 'k')
plt.imshow(coleval.average_frame(deformed, 4), 'rainbow', origin='lower')
plt.colorbar().set_label('Counts [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)







# This leads nowere
# I replot what I was showing before but clearer


n = 3
# integration time of the camera in ms
inttime = 1
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-06-10_multiple_search_for_parameters/1ms50Hz/averageREDhhhh-c'
pathparams1 = 'No window, averaged parameters'
# framerate of the IR camera in Hz
framerate = 383
# filestype
type = 'npy'
# type='csv'


params_1 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_1 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos1 = [177, 128]
pathfiles = '/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000018'
type = 'npy'
filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))
figure = 1
plt.figure(figure)
plt.imshow(data[0, 0], 'rainbow', origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Counts')
plt.plot(featurepos1[0], featurepos1[1], '+r', markersize=80)
plt.title('Position of the feature , ' + pathparams1)

n = 3
# integration time of the camera in ms
inttime = 1
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
pathparams2 = 'Window, central aberration, averaged parameters'
# framerate of the IR camera in Hz
framerate = 50
# filestype
type = 'npy'
# type='csv'

params_2 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_2 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos2 = [177, 128]
pathfiles = '/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000011'
type = 'npy'
filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))
figure += 1
plt.figure(figure)
plt.imshow(data[0, 0], 'rainbow', origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Counts')
plt.plot(featurepos2[0], featurepos2[1], '+r', markersize=80)
plt.title('Position of the feature , ' + pathparams2 + ' \n aberration in ' + str(featurepos2))

pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/1-1'
pathparams4 = 'Window, central aberration, 1-1 parameters'
params_4 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_4 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos4 = [177, 128]

pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/1-2'
pathparams5 = 'Window, central aberration, 1-2 parameters'
params_5 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_5 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos5 = [177, 128]

pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/2-1'
pathparams6 = 'Window, central aberration, 2-1 parameters'
params_6 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_6 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos6 = [177, 128]

pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/2-2'
pathparams7 = 'Window, central aberration, 2-2 parameters'
params_7 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_7 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos7 = [177, 128]

n = 3
# integration time of the camera in ms
inttime = 1
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/Calibration_Jun14_2018_set3_1ms'
pathparams3 = 'Window, off-center aberration, only 1 hot>room'
# framerate of the IR camera in Hz
framerate = 50
# filestype
type = 'npy'
# type='csv'

params_3 = np.load(os.path.join(pathparams, 'coeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
errparams_3 = np.load(os.path.join(pathparams, 'errcoeffpolydeg' + str(n) + 'int' + str(inttime) + 'ms.npy'))
featurepos3 = [177, 211]
pathfiles = '/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000042'
type = 'npy'
filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))
figure += 1
plt.figure(figure)
plt.imshow(data[0, 0], 'rainbow', origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Counts')
plt.plot(featurepos3[0], featurepos3[1], '+r', markersize=80)
plt.title('Position of the feature , ' + pathparams3 + ' \n aberration in ' + str(featurepos3))
plt.show()

max = np.max((np.max(params_1[:, :, 0]), np.max(params_2[:, :, 0]), np.max(params_3[:, :, 0])))
min = np.min((np.min(params_1[:, :, 0]), np.min(params_2[:, :, 0]), np.max(params_3[:, :, 0])))
figure += 1
plt.figure(figure)
plt.imshow(params_1[:, :, 0], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.plot((featurepos1[0], featurepos1[0]), (0, len(params_1[:, 0, 0])), 'k')
plt.plot((0, len(params_1[0, :, 0])), (featurepos1[1], featurepos1[1]), 'k')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Constant coefficient [K]')
plt.title('C0 , ' + pathparams1)
plt.plot((featurepos1[0], featurepos1[0]), (0, len(params_1[:, 0, 0])), 'k')
plt.plot((0, len(params_1[0, :, 0])), (featurepos1[1], featurepos1[1]), 'k')
figure += 1
plt.figure(figure)
plt.imshow(params_2[:, :, 0], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Constant coefficient [K]')
plt.title('C0 , ' + pathparams2)
plt.plot((featurepos2[0], featurepos2[0]), (0, len(params_2[:, 0, 0])), 'k')
plt.plot((0, len(params_2[0, :, 0])), (featurepos2[1], featurepos2[1]), 'k')
figure += 1
plt.figure(figure)
plt.imshow(params_3[:, :, 0], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Constant coefficient [K]')
plt.title('C0 , ' + pathparams3)
plt.plot((featurepos3[0], featurepos3[0]), (0, len(params_3[:, 0, 0])), 'k')
plt.plot((0, len(params_3[0, :, 0])), (featurepos3[1], featurepos3[1]), 'k')

max = np.max((np.max(params_1[:, :, 1]), np.max(params_2[:, :, 1])))
min = np.min((np.min(params_1[:, :, 1]), np.min(params_2[:, :, 1])))
figure += 1
plt.figure(figure)
plt.imshow(params_1[:, :, 1], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Linear coefficient [K/Counts]')
plt.title('C1 , ' + pathparams1)
plt.plot((featurepos1[0], featurepos1[0]), (0, len(params_1[:, 0, 0])), 'k')
plt.plot((0, len(params_1[0, :, 0])), (featurepos1[1], featurepos1[1]), 'k')
figure += 1
plt.figure(figure)
plt.imshow(params_2[:, :, 1], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Linear coefficient [K/Counts]')
plt.title('C1 , ' + pathparams2)
plt.plot((featurepos2[0], featurepos2[0]), (0, len(params_2[:, 0, 0])), 'k')
plt.plot((0, len(params_2[0, :, 0])), (featurepos2[1], featurepos2[1]), 'k')
figure += 1
plt.figure(figure)
plt.imshow(params_3[:, :, 1], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Linear coefficient [K/Counts]')
plt.title('C1 , ' + pathparams3)
plt.plot((featurepos3[0], featurepos3[0]), (0, len(params_3[:, 0, 0])), 'k')
plt.plot((0, len(params_3[0, :, 0])), (featurepos3[1], featurepos3[1]), 'k')

max = np.max((np.max(params_1[:, :, 2]), np.max(params_2[:, :, 2])))
min = np.min((np.min(params_1[:, :, 2]), np.min(params_2[:, :, 2])))
figure += 1
plt.figure(figure)
plt.imshow(params_1[:, :, 2], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
plt.title('C2 , ' + pathparams1)
plt.plot((featurepos1[0], featurepos1[0]), (0, len(params_1[:, 0, 0])), 'k')
plt.plot((0, len(params_1[0, :, 0])), (featurepos1[1], featurepos1[1]), 'k')
figure += 1
plt.figure(figure)
plt.imshow(params_2[:, :, 2], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
plt.title('C2 , ' + pathparams2)
plt.plot((featurepos2[0], featurepos2[0]), (0, len(params_2[:, 0, 0])), 'k')
plt.plot((0, len(params_2[0, :, 0])), (featurepos2[1], featurepos2[1]), 'k')
figure += 1
plt.figure(figure)
plt.imshow(params_3[:, :, 2], 'rainbow', vmin=min, vmax=max, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.colorbar().set_label('Quadratic coefficient [K/Counts^2]')
plt.title('C2 , ' + pathparams3)
plt.plot((featurepos3[0], featurepos3[0]), (0, len(params_3[:, 0, 0])), 'k')
plt.plot((0, len(params_3[0, :, 0])), (featurepos3[1], featurepos3[1]), 'k')

C0horiz1 = params_1[featurepos1[1], :, 0]
errC0horiz1 = errparams_1[featurepos1[1], :, 0]
C1horiz1 = params_1[featurepos1[1], :, 1]
errC1horiz1 = errparams_1[featurepos1[1], :, 1]
C2horiz1 = params_1[featurepos1[1], :, 2]
errC2horiz1 = errparams_1[featurepos1[1], :, 2]
C0vert1 = params_1[:, featurepos1[0], 0]
errC0vert1 = errparams_1[:, featurepos1[0], 0]
C1vert1 = params_1[:, featurepos1[0], 1]
errC1vert1 = errparams_1[:, featurepos1[0], 1]
C2vert1 = params_1[:, featurepos1[0], 2]
errC2vert1 = errparams_1[:, featurepos1[0], 2]

C0horiz2 = params_2[featurepos2[1], :, 0]
errC0horiz2 = errparams_2[featurepos2[1], :, 0]
C1horiz2 = params_2[featurepos2[1], :, 1]
errC1horiz2 = errparams_2[featurepos2[1], :, 1]
C2horiz2 = params_2[featurepos2[1], :, 2]
errC2horiz2 = errparams_2[featurepos2[1], :, 2]
C0vert2 = params_2[:, featurepos2[0], 0]
errC0vert2 = errparams_2[:, featurepos2[0], 0]
C1vert2 = params_2[:, featurepos2[0], 1]
errC1vert2 = errparams_2[:, featurepos2[0], 1]
C2vert2 = params_2[:, featurepos2[0], 2]
errC2vert2 = errparams_2[:, featurepos2[0], 2]

C0horiz3 = params_3[featurepos3[1], :, 0]
errC0horiz3 = errparams_3[featurepos3[1], :, 0]
C1horiz3 = params_3[featurepos3[1], :, 1]
errC1horiz3 = errparams_3[featurepos3[1], :, 1]
C2horiz3 = params_3[featurepos3[1], :, 2]
errC2horiz3 = errparams_3[featurepos3[1], :, 2]
C0vert3 = params_3[:, featurepos3[0], 0]
errC0vert3 = errparams_3[:, featurepos3[0], 0]
C1vert3 = params_3[:, featurepos3[0], 1]
errC1vert3 = errparams_3[:, featurepos3[0], 1]
C2vert3 = params_3[:, featurepos3[0], 2]
errC2vert3 = errparams_3[:, featurepos3[0], 2]

C0horiz4 = params_4[featurepos4[1], :, 0]
errC0horiz4 = errparams_4[featurepos4[1], :, 0]
C1horiz4 = params_4[featurepos4[1], :, 1]
errC1horiz4 = errparams_4[featurepos4[1], :, 1]
C2horiz4 = params_4[featurepos4[1], :, 2]
errC2horiz4 = errparams_4[featurepos4[1], :, 2]
C0vert4 = params_4[:, featurepos4[0], 0]
errC0vert4 = errparams_4[:, featurepos4[0], 0]
C1vert4 = params_4[:, featurepos4[0], 1]
errC1vert4 = errparams_4[:, featurepos4[0], 1]
C2vert4 = params_4[:, featurepos4[0], 2]
errC2vert4 = errparams_4[:, featurepos4[0], 2]

C0horiz5 = params_5[featurepos5[1], :, 0]
errC0horiz5 = errparams_5[featurepos5[1], :, 0]
C1horiz5 = params_5[featurepos5[1], :, 1]
errC1horiz5 = errparams_5[featurepos5[1], :, 1]
C2horiz5 = params_5[featurepos5[1], :, 2]
errC2horiz5 = errparams_5[featurepos5[1], :, 2]
C0vert5 = params_5[:, featurepos5[0], 0]
errC0vert5 = errparams_5[:, featurepos5[0], 0]
C1vert5 = params_5[:, featurepos5[0], 1]
errC1vert5 = errparams_5[:, featurepos5[0], 1]
C2vert5 = params_5[:, featurepos5[0], 2]
errC2vert5 = errparams_5[:, featurepos5[0], 2]

C0horiz6 = params_6[featurepos6[1], :, 0]
errC0horiz6 = errparams_6[featurepos6[1], :, 0]
C1horiz6 = params_6[featurepos6[1], :, 1]
errC1horiz6 = errparams_6[featurepos6[1], :, 1]
C2horiz6 = params_6[featurepos6[1], :, 2]
errC2horiz6 = errparams_6[featurepos6[1], :, 2]
C0vert6 = params_6[:, featurepos6[0], 0]
errC0vert6 = errparams_6[:, featurepos6[0], 0]
C1vert6 = params_6[:, featurepos6[0], 1]
errC1vert6 = errparams_6[:, featurepos6[0], 1]
C2vert6 = params_6[:, featurepos6[0], 2]
errC2vert6 = errparams_6[:, featurepos6[0], 2]

C0horiz7 = params_7[featurepos7[1], :, 0]
errC0horiz7 = errparams_7[featurepos7[1], :, 0]
C1horiz7 = params_7[featurepos7[1], :, 1]
errC1horiz7 = errparams_7[featurepos7[1], :, 1]
C2horiz7 = params_7[featurepos7[1], :, 2]
errC2horiz7 = errparams_7[featurepos7[1], :, 2]
C0vert7 = params_7[:, featurepos7[0], 0]
errC0vert7 = errparams_7[:, featurepos7[0], 0]
C1vert7 = params_7[:, featurepos7[0], 1]
errC1vert7 = errparams_7[:, featurepos7[0], 1]
C2vert7 = params_7[:, featurepos7[0], 2]
errC2vert7 = errparams_7[:, featurepos7[0], 2]

shape = np.shape(params_1[:, :, 0])
vertaxis1 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis1 = np.add(vertaxis1, -featurepos1[1])
horizaxis1 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis1 = np.add(horizaxis1, -featurepos1[0])

vertaxis2 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis2 = np.add(vertaxis2, -featurepos2[1])
horizaxis2 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis2 = np.add(horizaxis2, -featurepos2[0])

vertaxis3 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis3 = np.add(vertaxis3, -featurepos3[1])
horizaxis3 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis3 = np.add(horizaxis3, -featurepos3[0])

vertaxis4 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis4 = np.add(vertaxis4, -featurepos4[1])
horizaxis4 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis4 = np.add(horizaxis4, -featurepos4[0])

vertaxis5 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis5 = np.add(vertaxis5, -featurepos5[1])
horizaxis5 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis5 = np.add(horizaxis5, -featurepos5[0])

vertaxis6 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis6 = np.add(vertaxis6, -featurepos6[1])
horizaxis6 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis6 = np.add(horizaxis6, -featurepos6[0])

vertaxis7 = np.linspace(0, shape[0] - 1, shape[0])
vertaxis7 = np.add(vertaxis7, -featurepos7[1])
horizaxis7 = np.linspace(0, shape[1] - 1, shape[1])
horizaxis7 = np.add(horizaxis7, -featurepos7[0])

# C0horiz4=params_2[featurepos2[1]+25,:,0]
# C0horiz5=params_2[featurepos2[1]+50,:,0]
# C0horiz6=params_2[featurepos2[1]-25,:,0]
# C0horiz7=params_2[featurepos2[1]-50,:,0]
figure += 1
plt.figure(figure)
# plt.errorbar(horizaxis1,C0horiz1,yerr=errC0horiz1,fmt='b',label='C0 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
# plt.errorbar(horizaxis2,C0horiz2,yerr=errC0horiz2,fmt='r',label='C0 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
# plt.errorbar(horizaxis3,C0horiz3,yerr=errC0horiz3,fmt='y',label='C0 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
plt.plot(horizaxis1, C0horiz1, 'b', label='C0 horizontal \n ' + pathparams1)
# guess=[1,1]
# fit1,errfit1=curve_fit(coleval.polygen3(2), horizaxis2, C0horiz2, p0=guess, sigma=errC0horiz2, maxfev=100000000)
# plt.plot(horizaxis2,coleval.polygen3(2)(horizaxis2,*fit1),'k')
# plt.plot(horizaxis2,np.add(-18,(C0horiz2-coleval.polygen3(2)(horizaxis2,*fit1))),label='1')
plt.plot(horizaxis2, C0horiz2, 'r', label='C0 horizontal \n ' + pathparams2)
# guess=[1,1]
# fit2,errfit2=curve_fit(coleval.polygen3(2), horizaxis3, C0horiz3, p0=guess, sigma=errC0horiz3, maxfev=100000000)
# plt.plot(horizaxis3,coleval.polygen3(2)(horizaxis3,*fit2),'k')
# plt.plot(horizaxis3,np.add(-18,(C0horiz3-coleval.polygen3(2)(horizaxis3,*fit2))),label='2')
plt.plot(horizaxis3, C0horiz3, 'y', label='C0 horizontal \n ' + pathparams3)
# plt.plot(horizaxis4,C0horiz4,label='C0 horizontal \n '+pathparams4)
# plt.plot(horizaxis5,C0horiz5,label='C0 horizontal \n '+pathparams5)
# plt.plot(horizaxis6,C0horiz6,label='C0 horizontal \n '+pathparams6)
# plt.plot(horizaxis7,C0horiz7,label='C0 horizontal \n '+pathparams7)
max = np.max((np.max(C0horiz1), np.max(C0horiz2), np.max(C0horiz3)))
min = np.min((np.min(C0horiz1), np.min(C0horiz2), np.max(C0horiz3)))
plt.plot((0, 0), (min, max), 'k')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Constant coefficient [K]')
plt.title('Comparison of C0 horizontal dependance around feature')
plt.legend()
plt.legend(loc='best')

figure += 1
plt.figure(figure)
# plt.errorbar(horizaxis1,C1horiz1,yerr=errC1horiz1,fmt='b',label='C1 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
# plt.errorbar(horizaxis2,C1horiz2,yerr=errC1horiz2,fmt='r',label='C1 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
# plt.errorbar(horizaxis3,C1horiz3,yerr=errC1horiz3,fmt='y',label='C1 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
plt.plot(horizaxis1, C1horiz1, 'b', label='C1 horizontal \n ' + pathparams1)
# mean=np.mean(C1horiz1)
# guess=[1,1,1,1]
# fit1,errfit1=curve_fit(coleval.polygen3(2), horizaxis2, C1horiz2, p0=guess, sigma=errC1horiz2, maxfev=100000000)
# plt.plot(horizaxis2,coleval.polygen3(2)(horizaxis2,*fit1),'k')
# plt.plot(horizaxis2,np.add(mean,(C1horiz2-coleval.polygen3(2)(horizaxis2,*fit1))),label='1')
plt.plot(horizaxis2, C1horiz2, 'r', label='C1 horizontal \n ' + pathparams2)
# guess=[1,1,1,1]
# fit2,errfit2=curve_fit(coleval.polygen3(2), horizaxis3, C1horiz3, p0=guess, sigma=errC1horiz3, maxfev=100000000)
# plt.plot(horizaxis3,coleval.polygen3(2)(horizaxis3,*fit2),'k')
# plt.plot(horizaxis3,np.add(mean,(C1horiz3-coleval.polygen3(2)(horizaxis3,*fit2))),label='2')
plt.plot(horizaxis3, C1horiz3, 'y', label='C1 horizontal \n ' + pathparams3)
# plt.plot(horizaxis4,C1horiz4,label='C1 horizontal \n '+pathparams4)
# plt.plot(horizaxis5,C1horiz5,label='C1 horizontal \n '+pathparams5)
# plt.plot(horizaxis6,C1horiz6,label='C1 horizontal \n '+pathparams6)
# plt.plot(horizaxis7,C1horiz7,label='C1 horizontal \n '+pathparams7)
max = np.max((np.max(C1horiz1), np.max(C1horiz2), np.max(C1horiz3)))
min = np.min((np.min(C1horiz1), np.min(C1horiz2), np.min(C1horiz3)))
plt.plot((0, 0), (min, max), 'k')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Linear coefficient [K/Counts]')
plt.title('Comparison of C1 horizontal dependance around feature')
plt.legend()
plt.legend(loc='best')

figure += 1
plt.figure(figure)
# plt.errorbar(horizaxis1,C2horiz1,yerr=errC2horiz1,fmt='b',label='C2 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
# plt.errorbar(horizaxis2,C2horiz2,yerr=errC2horiz2,fmt='r',label='C2 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
# plt.errorbar(horizaxis3,C2horiz3,yerr=errC2horiz3,fmt='y',label='C2 horizontal in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
plt.plot(horizaxis1, C2horiz1, 'b', label='C2 horizontal \n ' + pathparams1)
plt.plot(horizaxis2, C2horiz2, 'r', label='C2 horizontal \n ' + pathparams2)
plt.plot(horizaxis3, C2horiz3, 'y', label='C2 horizontal \n ' + pathparams3)
# plt.plot(horizaxis4,C2horiz4,label='C2 horizontal \n '+pathparams4)
# plt.plot(horizaxis5,C2horiz5,label='C2 horizontal \n '+pathparams5)
# plt.plot(horizaxis6,C2horiz6,label='C2 horizontal \n '+pathparams6)
# plt.plot(horizaxis7,C2horiz7,label='C2 horizontal \n '+pathparams7)
max = np.max((np.max(C2horiz1), np.max(C2horiz2), np.max(C2horiz3)))
min = np.min((np.min(C2horiz1), np.min(C2horiz2), np.max(C2horiz3)))
plt.plot((0, 0), (min, max), 'k')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Quadratic coefficient [K/Counts^2]')
plt.title('Comparison of C2 horizontal dependance around feature')
plt.legend()
plt.legend(loc='best')

figure += 1
plt.figure(figure)
# plt.errorbar(vertaxis1,C0vert1,yerr=errC0vert1,fmt='b--',label='C0 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
# plt.errorbar(vertaxis2,C0vert2,yerr=errC0vert2,fmt='r--',label='C0 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
# plt.errorbar(vertaxis3,C0vert3,yerr=errC0vert3,fmt='y--',label='C0 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
plt.plot(vertaxis1, C0vert1, 'b', label='C0 vertical \n ' + pathparams1)
plt.plot(vertaxis2, C0vert2, 'r', label='C0 vertical \n ' + pathparams2)
plt.plot(vertaxis3, C0vert3, 'y', label='C0 vertical \n ' + pathparams3)
max = np.max((np.max(C0vert1), np.max(C0vert2), np.max(C0vert3)))
min = np.min((np.min(C0vert1), np.min(C0vert2), np.max(C0vert3)))
plt.plot((0, 0), (min, max), 'k')
plt.xlabel('Vertical axis [pixles]')
plt.ylabel('Constant coefficient [K]')
plt.title('Comparison of C0 dependance around centre of aberration')
plt.legend()
plt.legend(loc='best', prop={'size': 8})

figure += 1
plt.figure(figure)
# plt.errorbar(vertaxis1,C1vert1,yerr=errC1vert1,fmt='b--',label='C1 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
# plt.errorbar(vertaxis2,C1vert2,yerr=errC1vert2,fmt='r--',label='C1 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
# plt.errorbar(vertaxis3,C1vert3,yerr=errC1vert3,fmt='y--',label='C1 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
plt.plot(vertaxis1, C1vert1, 'b', label='C1 vertical \n ' + pathparams1)
plt.plot(vertaxis2, C1vert2, 'r', label='C1 vertical \n ' + pathparams2)
plt.plot(vertaxis3, C1vert3, 'y', label='C1 vertical \n ' + pathparams3)
max = np.max((np.max(C1vert1), np.max(C1vert2), np.max(C1vert3)))
min = np.min((np.min(C1vert1), np.min(C1vert2), np.max(C1vert3)))
plt.plot((0, 0), (min, max), 'k')
plt.xlabel('Vertical axis [pixles]')
plt.ylabel('Constant coefficient [K]')
plt.title('Comparison of C1 dependance around centre of aberration')
plt.legend()
plt.legend(loc='best', prop={'size': 10})

figure += 1
plt.figure(figure)
# plt.errorbar(vertaxis1,C2vert1,yerr=errC2vert1,fmt='b--',label='C2 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams1)
# plt.errorbar(vertaxis2,C2vert2,yerr=errC2vert2,fmt='r--',label='C2 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams2)
# plt.errorbar(vertaxis3,C2vert3,yerr=errC2vert3,fmt='y--',label='C2 vertical in coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy in \n '+pathparams3)
plt.plot(vertaxis1, C2vert1, 'b', label='C2 vertical \n ' + pathparams1)
plt.plot(vertaxis2, C2vert2, 'r', label='C2 vertical \n ' + pathparams2)
plt.plot(vertaxis3, C2vert3, 'y', label='C2 vertical \n ' + pathparams3)
max = np.max((np.max(C2vert1), np.max(C2vert2), np.min(C2vert3)))
min = np.min((np.min(C2vert1), np.min(C2vert2), np.min(C2vert3)))
plt.plot((0, 0), (min, max), 'k')
plt.xlabel('Vertical axis [pixles]')
plt.ylabel('Constant coefficient [K]')
plt.title('Comparison of C2 dependance around centre of aberration')
plt.legend()
plt.legend(loc='best', prop={'size': 10})

plt.show()




print('J O B   D O N E !')
