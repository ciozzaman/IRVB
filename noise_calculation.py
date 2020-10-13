# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())










# 2018-10-10 I want to explore better the oscillation in the counts











pathfiles = vacuum7[-1]
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '.npy'
filenames = coleval.all_file_names(pathfiles, type)[0]

data = np.load(os.path.join(pathfiles, filenames))



data2=coleval.clear_oscillation_central2(data,framerate,plot_conparison=False)
data2=np.array(data2)

mean = np.mean(data2[0],axis=(0))
std = np.std(data2[0],axis=(0))

plt.figure()
plt.imshow(std, origin='lower')
plt.colorbar().set_label('Amplitude [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Counts std')
# plt.show()
plt.pause(0.0001)

plt.figure()
treshold = 6
plt.imshow(std, origin='lower',vmin=treshold, vmax=treshold+0.001)
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Pixels with std>'+str(treshold))
plt.pause(0.0001)

positions=[[38,80],[132,191],[199,102],[71,271],[198,103]]
plt.figure()
for pos in positions:
	plt.plot(data2[0,:,pos[0],pos[1]]-mean[pos[0],pos[1]],label='counts of '+str(pos)+' with std '+str(std[pos[0],pos[1]]))
plt.title('Counts for different pixels')
plt.legend()
plt.grid()
plt.pause(0.0001)# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/preamble_indexing.py").read())











plt.figure()
for pos in positions:
	spectra = np.fft.fft(data2[0,:,pos[0],pos[1]])
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

	y = magnitude
	y = np.array([y for _, y in sorted(zip(freq, y))])
	x = np.sort(freq)
	plt.figure(0)
	plt.plot(x, y,label=str(pos)+' with std '+str(std[pos[0],pos[1]]))
plt.title('Amplitued from fast Fourier transform')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.grid()
plt.semilogy()
plt.legend()
plt.pause(0.0001)



plt.figure()
for pos in positions:
	means=[]
	for i in range(len(data2[0])):
		means.append(np.mean(data2[0,:i,pos[0],pos[1]])-mean[pos[0],pos[1]])
	means=np.array(means)
	plt.plot(means, label=str(pos) + ' with std ' + str(std[pos[0], pos[1]]))
plt.legend()
plt.grid()
plt.pause(0.0001)




plt.figure()
# pos=positions[0]
pos=[81,203]
for i in [-1,0,1]:
	for j in [-1,0,1]:
		if i==j==0:
			plt.plot((data2[0, :-1, pos[0] + i, pos[1] + j]+data2[0, 1:, pos[0] + i, pos[1] + j])/2,label=str([pos[0]+i, pos[1]+j]) + ' with std ' + str(std[pos[0] + i, pos[1] + j])+' moving average of 2 frames')
		else:
			plt.plot(data2[0,:,pos[0]+i,pos[1]+j],label=str([pos[0]+i, pos[1]+j]) + ' with std ' + str(std[pos[0]+i, pos[1]+j]))
plt.legend()
plt.grid()
plt.ylabel('Counts [au]')
plt.xlabel('Frames')
plt.pause(0.0001)



plt.figure()
std_flat=coleval.flatten_full(std)# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/preamble_indexing.py").read())








for treshold in [2,4,6,7,10,12,14,16,18]:
	index = coleval.find_nearest_index(std_flat,treshold)
	pos0=int(index//np.shape(std)[1])
	pos1=int(index%np.shape(std)[1])
	plt.plot(data2[0,:, pos0, pos1]-mean[pos0, pos1], label=str([pos0, pos1]) + ' with std ' + str(std[pos0,pos1]))
plt.legend()
plt.grid()
plt.ylabel('Counts [au]')
plt.xlabel('Frames')
plt.pause(0.0001)





import collections
plt.figure()
std_flat=coleval.flatten_full(std)
for treshold in [3,4,5,6,7,8,12,20,25,30,70]:
	index = coleval.find_nearest_index(std_flat,treshold)
	pos0=int(index//np.shape(std)[1])
	pos1=int(index%np.shape(std)[1])
	counter = collections.Counter((data2[0, :, pos0, pos1]).astype(int) - int(mean[pos0, pos1]))
	x=list(counter.keys())
	y=np.array(list(counter.values()))/len(data2[0, :, pos0, pos1])
	y = np.array([y for _, y in sorted(zip(x, y))])
	x = np.sort(x)
	plt.plot(x,y, label=str([pos0, pos1]) + ' with std ' + str(np.around(std[pos0,pos1],decimals=2)))
plt.legend()
plt.grid()
plt.title('Frequency of occurrence of a number of counts around the mean')
plt.ylabel('Frequency [%]')
plt.xlabel('Value [au]')
plt.pause(0.0001)


plt.figure()
counter = collections.Counter((std_flat).astype(int))
x = list(counter.keys())
y = np.array(list(counter.values()))
y = np.array([y for _, y in sorted(zip(x, y))])
x = np.sort(x)
plt.plot(x, y, 'o')
plt.plot(x, y)

plt.legend()
plt.grid()
plt.title('Number of pixels with a given std')
plt.ylabel('Number [#]')
plt.xlabel('std [au]')
# plt.semilogy()
plt.semilogx()
plt.pause(0.0001)






flag=np.ones(np.shape(std))
treshold_for_bad_std = 10	# defined 20/02/2019
for i in range(np.shape(std)[0]):
	for j in range(np.shape(std)[1]):
		if std[i,j]>treshold_for_bad_std:
			flag[i,j]=0
plt.figure()
plt.imshow(flag, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Pixels with std>'+str(treshold_for_bad_std))
plt.pause(0.0001)

# is this consistent for different records?
std_all=[]
mean_all=[]
for pathfiles in vacuum7[11:]:	#at least 1 hour after I started taking measurements
# pathfiles = vacuum7[-1]
	framerate = 383
	# integration time of the camera in ms
	inttime = 1
	# filestype
	type = '.npy'
	filenames = coleval.all_file_names(pathfiles, type)[0]
	data = np.load(os.path.join(pathfiles, filenames))
	data2=coleval.clear_oscillation_central2(data,framerate,plot_conparison=False)
	data2=np.array(data2)
	mean = np.mean(data2[0],axis=(0))
	mean_all.append(mean)
	std = np.std(data2[0],axis=(0))
	std_all.append(std)
std_all = np.array(std_all)
mean_all = np.array(mean_all)

treshold_list=[5,6,7,8,9,10,11,12,13,14]
flag_all=[]
for treshold_for_bad_std in treshold_list:
	flag=np.zeros(np.shape(std_all[0]))
	for index in range(len(std_all)):
		for i in range(np.shape(std)[0]):
			for j in range(np.shape(std)[1]):
				if std_all[index,i,j]>treshold_for_bad_std:
					flag[i,j]+=1
	flag_all.append(flag)
flag_all = np.array(flag_all)

index=0
plt.figure()
plt.imshow(flag_all[index], origin='lower',)
plt.colorbar()
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Pixels with std>'+str(treshold_list[index]))
plt.pause(0.0001)

unsure=[]
plt.figure()
for index,tre in enumerate(treshold_list):
	counter = collections.Counter(coleval.flatten_full(flag_all[index]))
	x = list(counter.keys())
	y = np.array(list(counter.values()))
	y = np.array([y for _, y in sorted(zip(x, y))])
	x = np.sort(x)
	plt.plot(x[1:],y[1:],label='Treshold of std = '+str(tre))
plt.xlabel('Nuber of records where std was greater than treshold')
plt.ylabel('Number of pixels')
plt.legend(loc='best')
plt.title('Background records from \n'+vacuum7[11]+' to \n'+vacuum7[-1])
plt.pause(0.0001)

# mode 1
plt.figure()
average_difference_all=[]
std_difference_all=[]
for index,mean in enumerate(mean_all):
	differences_between_mean = np.zeros(np.shape(std))
	for i in range(1,np.shape(std)[0]-1):
		for j in range(1,np.shape(std)[1]-1):
			if std_all[index,i,j]>treshold_for_bad_std:
				continue
			temp=(mean[i-1,j-1:j+2]*flag[i-1,j-1:j+2]).tolist()+[(mean[i,j-1]*flag[i,j-1]).tolist()]+[(mean[i,j+1]*flag[i,j+1]).tolist()]+(mean[i+1,j-1:j+2]*flag[i+1,j-1:j+2]).tolist()
			temp2 = [x for x in temp if x != 0]
			differences_between_mean[i,j] = np.mean(np.abs(temp2-mean[i,j]))
	average_difference = np.mean(differences_between_mean[1:-1,1:-1],axis=(0,1))
	std_difference = np.std(differences_between_mean[1:-1, 1:-1], axis=(0, 1))
	average_difference_all.append(average_difference)
	std_difference_all.append(std_difference)


	differences_between_mean = np.zeros(np.shape(std))
	for i in range(1,np.shape(std)[0]-1):
		for j in range(1,np.shape(std)[1]-1):
			if flag[i,j]==0:
				continue
			temp=(mean[i-1,j-1:j+2]*flag[i-1,j-1:j+2]).tolist()+[(mean[i,j-1]*flag[i,j-1]).tolist()]+[(mean[i,j+1]*flag[i,j+1]).tolist()]+(mean[i+1,j-1:j+2]*flag[i+1,j-1:j+2]).tolist()
			temp2 = [x for x in temp if x != 0]
			# differences_between_mean[i,j] = int(np.mean(np.abs(temp2-mean[i,j])))
			differences_between_mean[i, j] = int(np.mean(np.abs(temp2 - mean[i, j])))

	temp = coleval.flatten_full(differences_between_mean)
	temp2 = [x for x in temp if x != 0]
	counter = collections.Counter(temp2)
	x = list(counter.keys())
	y = np.array(list(counter.values()))
	y = np.array([y for _, y in sorted(zip(x, y))])
	x = np.sort(x)
	# plt.plot(x, y, 'o')
	plt.plot(x, y,label='Minutes from first sample = '+str(vacuumtime7[index+11]-vacuumtime7[0]) )

plt.legend()
plt.grid()
plt.title('distribution of the mean count difference between neighbouring pixels')
plt.ylabel('Number [#]')
plt.xlabel('Counts [au]')
plt.semilogy()
# plt.semilogx()
plt.pause(0.0001)

plt.figure()
plt.plot(np.array(vacuumtime7[11:])-vacuumtime7[0],average_difference_all)
plt.title('Average of the mean difference between neighbouring pixels')
plt.ylabel('Counts [au]')
plt.xlabel('Minutes from first sample [min]')
plt.pause(0.0001)

plt.figure()
plt.plot(np.array(vacuumtime7[11:])-vacuumtime7[0],average_std_all)
plt.title('Average of the std of the mean difference between neighbouring pixels')
plt.ylabel('Counts [au]')
plt.xlabel('Minutes from first sample [min]')
plt.pause(0.0001)

average_difference_all=[]
number_bad_pixels=[]
average_difference_all_std_fail=[]
number_bad_pixels_std_fail=[]
for index in range(len(x)):
	differences_between_mean = np.zeros(np.shape(std))
	differences_between_mean_std_fail = np.zeros(np.shape(std))
	# sum=np.sum(y[index:])
	# new_mean.append(np.multiply(x[index:],y[index:])/np.sum(x[index:]))
	treshold_for_bad_difference =x[index]
	flag_differences = np.ones(np.shape(std))
	flag_differences_std_fail = np.ones(np.shape(std))
	for i in range(1, np.shape(std)[0] - 1):
		for j in range(1, np.shape(std)[1] - 1):
			if flag[i,j]==0:
				temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
					(mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
							   mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
				temp2 = [x for x in temp if x != 0]
				if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
						temp2) - treshold_for_bad_difference):
				# if (np.abs(np.mean(temp2-mean[i,j]))>treshold_for_bad_difference):
					# print(i, j)
					# print(temp)
					# print('mean')
					# print(mean[i, j])
					# print('std')
					# print(std[i, j])
					flag_differences_std_fail[i, j] = 0
				else:
					differences_between_mean_std_fail[i, j] = int(np.mean(np.abs(temp2 - mean[i, j])))
			else:
				temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
					(mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
							   mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
				temp2 = [x for x in temp if x != 0]
				if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
						temp2) - treshold_for_bad_difference):
				# if (np.abs(np.mean(temp2-mean[i,j]))>treshold_for_bad_difference):
					# print(i, j)
					# print(temp)
					# print('mean')
					# print(mean[i, j])
					# print('std')
					# print(std[i, j])
					flag_differences[i, j] = 0
				else:
					differences_between_mean[i, j] = int(np.mean(np.abs(temp2 - mean[i, j])))

	average_difference_all.append(differences_between_mean)
	temp = coleval.flatten_full(flag_differences)
	number_bad_pixels.append(len([x for x in temp if x != 1]))
	average_difference_all_std_fail.append(differences_between_mean_std_fail)
	temp = coleval.flatten_full(flag_differences_std_fail)
	number_bad_pixels_std_fail.append(len([x for x in temp if x != 1]))
average_difference_all=np.array(average_difference_all)
average_difference_all_std_fail=np.array(average_difference_all_std_fail)
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Number of bad pixels solely by mean treshold [#]')
ax1.set_ylabel('Average of the count mean difference between\nneighbouring pixels after filtering, counts [au]', color=color)
ax1.semilogx()
# ax1.semilogy()
ax1.plot(number_bad_pixels, np.mean(average_difference_all[:,1:-1,1:-1],axis=(-1,-2)), color=color,label='excluding std>10 pixels')
ax1.plot(np.array(number_bad_pixels)+np.array(number_bad_pixels_std_fail), np.mean(average_difference_all[:,1:-1,1:-1]+average_difference_all_std_fail[:,1:-1,1:-1],axis=(-1,-2)),'--', color=color,label='all pixels')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Treshold for how much a pixel count mean is allowed\nto be out of the neighbouring pixels mean, counts [au]', color=color)  # we already handled the x-label with ax1
ax2.plot(number_bad_pixels, x, color=color,label='excluding std>10 pixels')
ax2.plot(np.array(number_bad_pixels)+np.array(number_bad_pixels_std_fail), x,'--', color=color,label='all pixels')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid()
ax2.semilogx()
plt.legend(loc='best')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.pause(0.0001)

mean=mean_all[-1]
std=std_all[-1]
treshold_for_bad_difference=13	#defined 25/02/2019
pixel_visualisation = np.zeros(np.shape(std))
# plot all of them
for i in range(1, np.shape(std)[0] - 1):
	for j in range(1, np.shape(std)[1] - 1):
		if flag[i, j] == 0:
			pixel_visualisation[i, j] =6
		temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
			(mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
					   mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
		temp2 = [x for x in temp if x != 0]
		if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
				temp2) - treshold_for_bad_difference):
			# if (np.abs(np.mean(temp2-mean[i,j]))>treshold_for_bad_difference):
			# print(i, j)
			# print(temp)
			# print('mean')
			# print(mean[i, j])
			# print('std')
			# print(std[i, j])
			pixel_visualisation[i, j] += 3
plt.figure()
cmap = plt.cm.rainbow
cmap.set_under(color='white')
plt.imshow(pixel_visualisation,cmap=cmap, origin='lower',vmin=2)
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Dead pixels, std tresh='+str(treshold_for_bad_std)+', mean diff tresh='+str(treshold_for_bad_difference))
plt.pause(0.0001)


pixel_visualisation_check=[]
for i in range(len(mean_all)-1):
	mean=mean_all[i]
	std=std_all[i]
	flag_ceck=np.ones(np.shape(std))
	for i in range(np.shape(std)[0]):
		for j in range(np.shape(std)[1]):
			if std[i,j]>treshold_for_bad_std:
				flag_ceck[i,j]=0
	difference = np.zeros(np.shape(std))
	# plot all of them
	for i in range(1, np.shape(std)[0] - 1):
		for j in range(1, np.shape(std)[1] - 1):
			if flag_ceck[i, j] == 0:
				if pixel_visualisation[i, j] ==0:
					difference[i, j] =6
			else:
				if (pixel_visualisation[i, j] == 6 or pixel_visualisation[i, j] == 9):
					difference[i, j] = -6
			temp = (mean[i - 1, j - 1:j + 2] * flag_ceck[i - 1, j - 1:j + 2]).tolist() + [
				(mean[i, j - 1] * flag_ceck[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag_ceck[i, j + 1]).tolist()] + (
						   mean[i + 1, j - 1:j + 2] * flag_ceck[i + 1, j - 1:j + 2]).tolist()
			temp2 = [x for x in temp if x != 0]
			if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
					temp2) - treshold_for_bad_difference):
				# if (np.abs(np.mean(temp2-mean[i,j]))>treshold_for_bad_difference):
				# print(i, j)
				# print(temp)
				# print('mean')
				# print(mean[i, j])
				# print('std')
				# print(std[i, j])
				if pixel_visualisation[i, j] ==0:
					difference[i, j] += 3
			else:
				if (pixel_visualisation[i, j] == 3 or pixel_visualisation[i, j] == 9):
					difference[i, j] -= 3
	pixel_visualisation_check.append(difference)
pixel_visualisation_check=np.array(pixel_visualisation_check)
threes_all=[]
sixes_all=[]
nines_all=[]
neg_threes_all=[]
neg_sixes_all=[]
neg_nines_all=[]
for i in range(len(mean_all)-1):
	temp=coleval.flatten_full(pixel_visualisation_check[i])
	threes_all.append(len([x for x in temp if x == 3]))
	sixes_all.append(len([x for x in temp if x == 6]))
	nines_all.append(len([x for x in temp if x == 9]))
	neg_threes_all.append(len([x for x in temp if x == -3]))
	neg_sixes_all.append(len([x for x in temp if x == -6]))
	neg_nines_all.append(len([x for x in temp if x == -9]))
plt.figure()
# plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],threes_all,label='was good and became mean difference higher than '+str(treshold_for_bad_difference))
# plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],sixes_all,label='was good and became std higher than '+str(treshold_for_bad_std))
# plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],nines_all,label='was good and became bad for both reasons')
# plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],neg_threes_all,label='was bad for mean difference higher than '+str(treshold_for_bad_difference)+' and become good')
# plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],neg_sixes_all,label='was bad for std higher than '+str(treshold_for_bad_std)+' and become good')
# plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],neg_nines_all,label='was bad for both reasons and became good')
plt.title('Dead pixels variation, std tresh='+str(treshold_for_bad_std)+', mean diff tresh='+str(treshold_for_bad_difference))
plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],np.array(neg_threes_all)+np.array(neg_sixes_all)+np.array(neg_nines_all),label='was bad for and became good')
plt.plot(np.array(vacuumtime7[11:-1])-vacuumtime7[-1],np.array(threes_all)+np.array(sixes_all)+np.array(nines_all),label='was good for and became bad')
plt.legend(loc='best')
plt.xlabel('Minutes before the reference record [min]')
plt.ylabel('Number of pixels [#]')
plt.pause(0.0001)


plt.figure()
cmap = plt.cm.rainbow
cmap.set_under(color='white')
plt.imshow(pixel_visualisation_check[-2],cmap=cmap, origin='lower')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Dead pixels in vacuum7[-5]')
plt.pause(0.0001)




differences_between_mean = np.zeros(np.shape(std))
std_per_pixel = np.zeros(np.shape(std))
for i in range(1, np.shape(std)[0] - 1):
	for j in range(1, np.shape(std)[1] - 1):
		if pixel_visualisation[i, j]!=0:
			continue
		temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
			(mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
					   mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
		temp2 = [x for x in temp if x != 0]
		if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
				temp2) - treshold_for_bad_difference):
		# if (np.abs(np.mean(temp2-mean[i,j]))>treshold_for_bad_difference):
			# print(i, j)
			# print(temp)
			# print('mean')
			# print(mean[i, j])
			# print('std')
			# print(std[i, j])
			# flag_differences[i, j] = 0
			sgna=0
		else:
			differences_between_mean[i, j] = np.mean(np.abs(temp2 - mean[i, j]))
			std_per_pixel[i, j]=np.std(data2[0,:,i,j])

temp=coleval.flatten_full(differences_between_mean)
temp2=([x for x in temp if x != 0])
differences_between_mean_final = np.mean(temp2)

temp=coleval.flatten_full(std_per_pixel)
temp2=([x for x in temp if x != 0])
final_std = np.mean(temp2)



# To calculate the mean difference between pixels from the center to only the 4 touching neighbours
differences_between_mean = np.zeros(np.shape(std))
std_per_pixel = np.zeros(np.shape(std))
for i in range(1, np.shape(std)[0] - 1):
	for j in range(1, np.shape(std)[1] - 1):
		if pixel_visualisation[i, j]!=0:
			continue
		temp = [(mean[i - 1, j ] * flag[i - 1, j ]) ,
			(mean[i, j - 1] * flag[i, j - 1]) , (mean[i, j + 1] * flag[i, j + 1]) , (
					   mean[i + 1, j ] * flag[i + 1, j])]
		temp2 = [x for x in temp if x != 0]
		differences_between_mean[i, j] = np.mean((temp2 - mean[i, j]))
		std_per_pixel[i, j]=np.std(data2[0,:,i,j])

temp=coleval.flatten_full(differences_between_mean)
temp2=([x for x in temp if x != 0])
differences_between_mean_final = np.mean(temp2)

temp=coleval.flatten_full(std_per_pixel)
temp2=([x for x in temp if x != 0])
final_std = np.mean(temp2)





# numerical evaluation of this noises on power measurements
temperature_mean = 273.15+25	# K
count_mean_differences_between_pixels = 0
count_std_differences_between_pixels = 0
count_noise_per_pixel = 3.9


mean_differences_between_pixels = count_mean_differences_between_pixels/100
std_differences_between_pixels = count_std_differences_between_pixels/100
noise_per_pixel = count_noise_per_pixel/100
framerate = 383
spatial_resolution = 0.07/187	# m/pixel
diffusivity_multiplier = 0.4
thickness_multiplier = 1
emissivity_multiplier = 1

def neighbouring_pixel(temperature_mean,mean_differences_between_pixels,std_differences_between_pixels,noise_per_pixel):
	import numpy as np
	return np.random.normal(temperature_mean, noise_per_pixel)+np.random.normal(0, std_differences_between_pixels)

power_all=[]
temporal_difference_all=[]
laplacian_all=[]
black_body_all=[]
for i in range(1000000):
	present_pixel_temp=np.random.normal(temperature_mean, noise_per_pixel)
	temporal_difference = Ptthermalconductivity*thickness_multiplier*2.5E-06 *(1/(diffusivity_multiplier*Ptthermaldiffusivity))*(np.random.normal(temperature_mean, noise_per_pixel)-np.random.normal(temperature_mean, noise_per_pixel))/(2/framerate)
	laplacian = - Ptthermalconductivity*thickness_multiplier*2.5E-06 * ( ( neighbouring_pixel(temperature_mean,mean_differences_between_pixels,std_differences_between_pixels,noise_per_pixel)+neighbouring_pixel(temperature_mean,mean_differences_between_pixels,std_differences_between_pixels,noise_per_pixel) -2*present_pixel_temp ) + ( neighbouring_pixel(temperature_mean,mean_differences_between_pixels,std_differences_between_pixels,noise_per_pixel)+neighbouring_pixel(temperature_mean,mean_differences_between_pixels,std_differences_between_pixels,noise_per_pixel)-2*present_pixel_temp ) )/(spatial_resolution**2)
	black_body = 2*emissivity_multiplier*sigmaSB*(present_pixel_temp**4-temperature_mean**4)
	power=temporal_difference + laplacian + black_body
	power_all.append(power)
	temporal_difference_all.append(temporal_difference)
	laplacian_all.append(laplacian)
	black_body_all.append(black_body)
print('power average='+str(np.mean(power_all)))
print('power std='+str(np.std(power_all)))
print('temporal difference std='+str(np.std(temporal_difference_all)))
print('laplacian std='+str(np.std(laplacian_all)))
print('black body std='+str(np.std(black_body_all)))

power_all=np.array(power_all)
temp=np.around(power_all)
counter = collections.Counter(temp)
x = list(counter.keys())
y = np.array(list(counter.values()))
y = np.array([y for _, y in sorted(zip(x, y))])
x = np.sort(x)
plt.figure()
plt.plot(x,y,label=power)
plt.pause(0.0001)

for i in [2,3,4,5,6,7,8,9,10]:
	print('averaging on '+str(i**2)+' pixels residual std is '+str(np.std(np.convolve(power_all, np.ones((i**2)) / i**2, mode='valid'))))















# Global example, let's check the power oshillation on a real record



n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles=files7[2]
# framerate of the IR camera in Hz
framerate=383
# integration time of the camera in ms
inttime=1
#filestype
type='.npy'
# type='csv'

fullpathparams=os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy')
params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))
data2=coleval.clear_oscillation_central2(data,framerate,plot_conparison=False)
data2=np.array(data2)
flag=coleval.find_dead_pixels(data2)
data3=coleval.replace_dead_pixels(data2,flag)

basecounts = np.mean(data3[0],axis=(0))
rotangle=0
datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
basetemp = coleval.record_rotation(datatemp,rotangle)


datatemp, errdatatemp = coleval.count_to_temp_poly2(data3, params, errparams)
datatempcrop = coleval.record_rotation(datatemp,rotangle)
errdatatempcrop = coleval.record_rotation(errdatatemp,rotangle)


mean = np.mean(data3[0],axis=(0))

testrot = mean
rotangle = -1.5  # in degrees
foilrot = rotangle * 2 * np.pi / 360
foilrotdeg = rotangle
foilcenter = [160, 133]
foilhorizw = 0.09
foilvertw = 0.07
foilhorizwpixel = 240
foilvertwpixel = np.int((foilhorizwpixel * foilvertw) // foilhorizw)
r = ((foilhorizwpixel ** 2 + foilvertwpixel ** 2) ** 0.5) / 2  # HALF DIAGONAL
a = foilvertwpixel / np.cos(foilrot)
tgalpha = np.tan(foilrot)
delta = -(a ** 2) / 4 + (1 + tgalpha ** 2) * (r ** 2)
foilx = np.add(foilcenter[0], [(-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
							   (-0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
							   (0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
							   (0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
							   (-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2)])
foily = np.add(foilcenter[1] - tgalpha * foilcenter[0],
			   [tgalpha * foilx[0] + a / 2, tgalpha * foilx[1] + a / 2, tgalpha * foilx[2] - a / 2,
				tgalpha * foilx[3] - a / 2, tgalpha * foilx[0] + a / 2])
foilxint = (np.rint(foilx)).astype(int)
foilyint = (np.rint(foily)).astype(int)

precisionincrease = 10
dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
foilcenterrot = (
	np.rint([np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
	int)
foilxrot = (np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
					 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
					 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
foilyrot = (np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
					 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
					 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

foillx = min(foilxrot)
foilrx = max(foilxrot)
foilhorizwpixel = foilrx - foillx
foildw = min(foilyrot)
foilup = max(foilyrot)
foilvertwpixel = foilup - foildw

basecounts = mean

datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
# basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
basetemp = np.mean(datatempcrop[0], axis=0)


datatemp, errdatatemp = coleval.count_to_temp_poly2(data3, params, errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]


spatial_averaging =1
time_averaging = 1
if not ('datatempcrop_full_res' in dir()):
	datatempcrop_full_res = copy.deepcopy(datatempcrop)
datatempcrop = coleval.average_multiple_frames(datatempcrop_full_res,spatial_averaging,timemean=time_averaging)
print('ping')

if not ('basetemp_full_res' in dir()):
	basetemp_full_res = copy.deepcopy(basetemp)
basetemp=resize(basetemp_full_res,np.shape(datatempcrop[0,0]),order=1)
print('ping')

foilemissivityscaled_orig = 1 * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
conductivityscaled = Ptthermalconductivity * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
reciprdiffusivityscaled_orig = (1 / (0.4*Ptthermaldiffusivity)) * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
flat_properties = True


dt=time_averaging*1/framerate
dx=foilhorizw/(np.shape(datatempcrop)[-1])
dy=foilvertw/(np.shape(datatempcrop)[-2])
relative_temp = np.add(datatempcrop,-basetemp)
dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
d2Tdxy=np.add(d2Tdx2,d2Tdy2)
negd2Tdxy=np.multiply(-1,d2Tdxy)
T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
T04=np.power(np.add(zeroC,basetemp[1:-1, 1:-1]),4)

print('ping')
reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

BBrad=[]
diffusion=[]
timevariation=[]
ktf=np.multiply(conductivityscaled,foilthicknessscaled)
for i in range(len(datatempcrop[:,0,0,0])):
	BBrad.append([])
	diffusion.append([])
	timevariation.append([])
	for j in range(len(datatempcrop[0,1:-1,0,0])):
		BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
		BBrad[i].append(BBradtemp)
		diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
		diffusion[i].append(diffusiontemp)
		timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
		timevariation[i].append(timevariationtemp)
BBrad=np.array(BBrad)
diffusion=np.array(diffusion)
timevariation=np.array(timevariation)
print('ping')
BBradnoback=np.add(BBrad,0)
diffusionnoback=np.add(diffusion,0)
timevariationnoback=np.add(timevariation,0)

powernoback=np.add(np.add(diffusionnoback,timevariationnoback),BBradnoback)
print('ping')

# test= coleval.flatten_full(timevariationnoback[0,:,150:153,150:153])
# # test= coleval.flatten_full(timevariationnoback[0,:,150:152,150:152])
# counter = collections.Counter(np.around(test))
# x = list(counter.keys())
# y = np.array(list(counter.values()))
# y = np.array([y for _, y in sorted(zip(x, y))])
# x = np.sort(x)
# plt.figure()
# plt.plot(x,y)
# plt.pause(0.0001)

print('spatial_averaging of '+str(spatial_averaging))
print('time_averaging of '+str(time_averaging))
print(np.std(coleval.flatten_full(BBradnoback[0])))
print(np.std(coleval.flatten_full(diffusionnoback[0])))
print(np.std(coleval.flatten_full(timevariationnoback[0])))
print(np.std(coleval.flatten_full(powernoback[0])))


print(np.mean(np.std(timevariationnoback[0],axis=(0)),axis=(-1,-2)))
print(np.std(np.mean(timevariationnoback[0],axis=(0)),axis=(-1,-2)))
print(np.std(timevariationnoback[0,:,150:151,150:151],axis=(0,-1,-2)))



plt.figure()
to_test = BBradnoback
plt.title('plot ot black body radiation term')

for pos in [50,100,150]:
	x=[]
	y=[]
	a=coleval.flatten_full(to_test[0,:,pos,pos])
	for index in range(1,100,5):
		sum=0
		for  index2 in range(index):
			sum+=np.array(a[index2:index2+len(a)-index])/index
		sum=coleval.flatten_full(sum)
		print(index)
		print(np.std(sum))
		x.append(index)
		y.append(np.std(sum))
	plt.plot(x,y,'--',label='all times for pixel '+str([pos,pos]))
plt.plot(x,y[0]*1/np.sqrt(x),'k',label='reference')
plt.pause(0.001)

for time in [10,1000,2000]:
	x=[]
	y=[]
	a=coleval.flatten_full(to_test[0,time])
	for index in range(1,100,5):
		sum=0
		for  index2 in range(index):
			sum+=np.array(a[index2:index2+len(a)-index])/index
		sum=coleval.flatten_full(sum)
		print(index)
		print(np.std(sum))
		x.append(index)
		y.append(np.std(sum))
	plt.plot(x, y, '-', label='all foil for time step ' + str(time))
plt.plot(x,y[0]*1/np.sqrt(x),'k',label='reference')
plt.legend(loc='best')
plt.xlabel('number of pixels averaged')
plt.ylabel('std of the averaged pixels')
plt.pause(0.001)


plt.figure(1)
x=[1,2,3,4,5,6,10]
plt.plot(x,[108,17,11.4,4.84,3.7,1.95,0.9],label='spatial variation term')
plt.plot(x,108*1/np.array(x),'k--')

plt.plot(x,[106.8,62.9,44.3,35.1,29.5,25.7,17.7],label='temporal variation term')
plt.plot(x,106.8*1/np.array(x),'k--')

plt.plot(x,[0.26,0.16,0.23,0.18,0.20,0.15,0.22],label='black body rad term')
plt.plot(x,0.26*1/np.array(x),'k--',label='reference')

plt.plot(x,[152,65,45.7,35.4,29.8,25.8,17.7],label='total power')
plt.plot(x,152*1/np.array(x),'k--')

plt.xlabel('number of pixels for spatial temperature avering')
plt.ylabel('std [W/m2]')
plt.legend(loc='best')
plt.title('no temporal averaging')
plt.grid()
plt.pause(0.001)


plt.figure(2)
x=[1,2,3,4,5,6,8,10]
plt.plot(x,[108.6,77,64,44,50,46,40,36.6],label='spatial variation term')
plt.plot(x,108.6*1/np.sqrt(x),'k--')

plt.plot(x,[107,37.9,20.7,13.5,9.7,7.4,4.8,3.6],label='temporal variation term')
plt.plot(x,107*1/np.sqrt(x),'k--')

plt.plot(x,[0.26,0.19,0.15,0.13,0.12,0.11,0.1,0.09],label='black body rad term')
plt.plot(x,0.26*1/np.sqrt(x),'k--',label='reference')

plt.plot(x,[152,86,67,57,51,46.7,40.8,36.8],label='total power')
plt.plot(x,152*1/np.sqrt(x),'k--')

plt.xlabel('number of pixels for temporal temperature avering')
plt.ylabel('std [W/m2]')
plt.legend(loc='best')
plt.title('temporal averaging for full resolution')
plt.grid()
plt.pause(0.001)


spectra = np.fft.fft(dTdt[0,:,10,10], axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

plt.figure()
plt.plot(freq, magnitude)
plt.plot(freq, magnitude, '+')
# plt.title(
plt.semilogy()
plt.title(
	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.pause(0.001)


plt.figure()
plt.plot(totalpower,label='totalpower')
plt.plot(totalBBrad,label='totalBBrad')
plt.plot(totaldiffusion,label='totaldiffusion')
plt.plot(totaltimevariation,label='totaltimevariation')
plt.title('Sum of the power over all the foil')
plt.xlabel('Frame number')
plt.ylabel('Power sum [W]')
plt.legend()
plt.legend(loc='best')
# plt.savefig(os.path.join(pathfiles,filenames[:-4]+'power_sum_all_foil'+'.eps'))
plt.grid()
plt.pause(0.0001)

plt.figure()
plt.imshow(powernoback[0][10],'rainbow', origin='lower')
plt.colorbar().set_label('Amplitude [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.title('Counts std')
# plt.show()
plt.pause(0.0001)














# 22/04/2019 Little addition to put a bit better in numbers why we need to eliminate the oscillation

# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = files7[2]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '_stat.npy'
# type='csv'

filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))


def fit_sin(x, *params):
	import numpy as np
	c1 = params[0]
	c2 = params[1]
	c3 = params[2]
	c4 = params[3]
	c5 = params[4]
	#c5=0
	return c1+c5*x+c2 * np.sin(x/c3+c4)


plt.figure(figsize=(20,10))
framerate=383
maxstep=99
minstep=0
steps=maxstep-minstep
poscentred=[[70,70],[70,200],[160,133],[250,200],[250,70]]
color=['b','r','m','y','g','c']
index=0
for pos in poscentred:
	for a in [10]:
		datasample=np.mean(data[0,:,pos[1]-a:pos[1]+1+a,pos[0]-a:pos[0]+1+a],axis=(-1,-2))
		plt.plot(datasample,color[index],label='averaged counts in [H,V] '+str(pos)+' +/- '+str(a))
		# index+=1
		guess = [np.mean(datasample), 2, 2, 0,0]
		temp1,temp2=curve_fit(fit_sin,  np.linspace(minstep,maxstep-1,steps),datasample[minstep:maxstep], p0=guess, maxfev=100000000)
		plt.plot(np.linspace(minstep+1,maxstep,steps),fit_sin(np.linspace(minstep+1,maxstep,steps),*temp1),color[index]+'--',label='fitting in [H,V] '+str(pos)+' +/- '+str(a)+' with amplitude and frequency '+str(int(abs(temp1[1])*100)/100)+'[au], '+str(int(100*framerate/(abs(temp1[2])*2*np.pi))/100)+'Hz',linewidth=3)
		index+=1
plt.legend(loc='best')
plt.grid()
plt.title('Averaged counts in different locations fitted with a sinusoidal curve to extract frequency and amplitude  \n from '+pathfiles)
plt.xlabel('Frames')
plt.ylabel('Counts [au]')
plt.show()









# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles = vacuum2[-12]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '.npy'
# type='csv'

filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))



spectra = np.fft.fft(data[0], axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

plt.plot(freq,np.mean(magnitude,axis=(1,2)))
plt.semilogy()
plt.show()

samplefreq = 587
averaging = 1
plt.title('Normalised amplitude of ' + str(np.around(freq[samplefreq],decimals=2))  + 'Hz oscillation from fast Fourier transform in counts in \n ' + pathfiles + ' FR ' + str(
	framerate) + 'Hz, int. time ' + str(inttime) + 'ms',fontsize=9)
plt.imshow(coleval.average_frame(magnitude[samplefreq],averaging), 'rainbow', origin='lower')
plt.colorbar().set_label('Amplitude [au]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
# plt.show()
plt.figure()
plt.title('Phase of ' +  str(np.around(freq[samplefreq],decimals=2))  + 'Hz oscillation from fast Fourier transform in counts in \n ' + pathfiles + ' FR ' + str(
	framerate) + 'Hz, int. time ' + str(inttime) + 'ms',fontsize=9)
plt.imshow(coleval.average_frame(phase[samplefreq],averaging), 'rainbow', origin='lower')
plt.colorbar().set_label('Phase [rad]')
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.show()


data2=coleval.clear_oscillation_central2(data,framerate,plot_conparison=False)
flag=coleval.find_dead_pixels(data2)
data3=coleval.replace_dead_pixels(data2,flag)
data=np.array(data)
data3=np.array(data3)

plt.figure()
averaging_all = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,50,60,100]
std_all_orig = []
std_all_clean = []
shape = (np.array(np.shape(data))/2).astype(int)
# shape=[0,0,100,100]
ref_orif=np.mean(data[0],axis=(0))
ref_clean=np.mean(data3[0],axis=(0))
for averaging in averaging_all:
	average = np.mean(data[0,:,shape[2]:shape[2]+averaging,shape[3]:shape[3]+averaging]-ref_orif[shape[2]:shape[2]+averaging,shape[3]:shape[3]+averaging],axis=(-1,-2))
	# average = coleval.average_multiple_frames(data,averaging)
	# shape = (np.array(np.shape(average))/2).astype(int)
	std = np.std(average)
	std = np.mean((average[:-1]-average[1:])**2)
	std_all_orig.append(std)
	average = np.mean(data3[0,:,shape[2]:shape[2]+averaging,shape[3]:shape[3]+averaging]-ref_clean[shape[2]:shape[2]+averaging,shape[3]:shape[3]+averaging],axis=(-1,-2))
	# average = coleval.average_multiple_frames(data,averaging)
	# shape = (np.array(np.shape(average))/2).astype(int)
	std = np.std(average)
	std = np.mean((average[:-1] - average[1:]) ** 2)
	std_all_clean.append(std)

plt.plot(averaging_all,std_all_orig,'o')
plt.plot(averaging_all,std_all_orig[0]/np.array(averaging_all),'--k')
plt.plot(averaging_all,std_all_clean,'x')
plt.plot(averaging_all,std_all_clean[0]/np.array(averaging_all),'--k')
plt.semilogy()
plt.pause(0.001)






averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,40]
std_all_orig = []
std_all_clean = []
params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))

try:
	del datatempcrop_full_res
	del basetemp_full_res
except:
	1+1

basecounts = np.mean(data[0],axis=(0))
rotangle=0
datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
basetemp = coleval.record_rotation(datatemp,rotangle)


datatemp, errdatatemp = coleval.count_to_temp_poly2(data, params, errparams)
datatempcrop = coleval.record_rotation(datatemp,rotangle)
errdatatempcrop = coleval.record_rotation(errdatatemp,rotangle)


mean = np.mean(data[0],axis=(0))

testrot = mean
rotangle = -1.5  # in degrees
foilrot = rotangle * 2 * np.pi / 360
foilrotdeg = rotangle
foilcenter = [160, 133]
foilhorizw = 0.09
foilvertw = 0.07
foilhorizwpixel = 240
foilvertwpixel = np.int((foilhorizwpixel * foilvertw) // foilhorizw)
r = ((foilhorizwpixel ** 2 + foilvertwpixel ** 2) ** 0.5) / 2  # HALF DIAGONAL
a = foilvertwpixel / np.cos(foilrot)
tgalpha = np.tan(foilrot)
delta = -(a ** 2) / 4 + (1 + tgalpha ** 2) * (r ** 2)
foilx = np.add(foilcenter[0], [(-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
							   (-0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
							   (0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
							   (0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
							   (-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2)])
foily = np.add(foilcenter[1] - tgalpha * foilcenter[0],
			   [tgalpha * foilx[0] + a / 2, tgalpha * foilx[1] + a / 2, tgalpha * foilx[2] - a / 2,
				tgalpha * foilx[3] - a / 2, tgalpha * foilx[0] + a / 2])
foilxint = (np.rint(foilx)).astype(int)
foilyint = (np.rint(foily)).astype(int)

precisionincrease = 10
dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
foilcenterrot = (
	np.rint([np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
	int)
foilxrot = (np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
					 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
					 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
foilyrot = (np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
					 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
					 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

foillx = min(foilxrot)
foilrx = max(foilxrot)
foilhorizwpixel = foilrx - foillx
foildw = min(foilyrot)
foilup = max(foilyrot)
foilvertwpixel = foilup - foildw

basecounts = mean

datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
# basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
basetemp = np.mean(datatempcrop[0], axis=0)


datatemp, errdatatemp = coleval.count_to_temp_poly2(data, params, errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]



for spatial_averaging in averaging_all:

	time_averaging = 1
	if not ('datatempcrop_full_res' in dir()):
		datatempcrop_full_res = copy.deepcopy(datatempcrop)
	datatempcrop = coleval.average_multiple_frames(datatempcrop_full_res,spatial_averaging,timemean=time_averaging)
	print('ping')

	if not ('basetemp_full_res' in dir()):
		basetemp_full_res = copy.deepcopy(basetemp)
	basetemp=resize(basetemp_full_res,np.shape(datatempcrop[0,0]),order=1)
	print('ping')

	foilemissivityscaled_orig = 1 * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	conductivityscaled = Ptthermalconductivity * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	reciprdiffusivityscaled_orig = (1 / (0.4*Ptthermaldiffusivity)) * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	flat_properties = True


	dt=time_averaging*1/framerate
	dx=foilhorizw/(np.shape(datatempcrop)[-1])
	dy=foilvertw/(np.shape(datatempcrop)[-2])
	relative_temp = np.add(datatempcrop,-basetemp)
	dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
	d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
	d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
	# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
	# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
	# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
	d2Tdxy=np.add(d2Tdx2,d2Tdy2)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
	T04=np.power(np.add(zeroC,basetemp[1:-1, 1:-1]),4)

	print('ping')
	reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
	foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
	foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

	BBrad=[]
	diffusion=[]
	timevariation=[]
	ktf=np.multiply(conductivityscaled,foilthicknessscaled)
	for i in range(len(datatempcrop[:,0,0,0])):
		BBrad.append([])
		diffusion.append([])
		timevariation.append([])
		for j in range(len(datatempcrop[0,1:-1,0,0])):
			BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
			BBrad[i].append(BBradtemp)
			diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
			diffusion[i].append(diffusiontemp)
			timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
			timevariation[i].append(timevariationtemp)
	BBrad=np.array(BBrad)
	diffusion=np.array(diffusion)
	timevariation=np.array(timevariation)
	print('ping')
	BBradnoback=np.add(BBrad,0)
	diffusionnoback=np.add(diffusion,0)
	timevariationnoback_orig=np.add(timevariation,0)

	powernoback=np.add(np.add(diffusionnoback,timevariationnoback_orig),BBradnoback)
	print('ping')

	std_all_orig.append(np.std(coleval.flatten_full(timevariationnoback_orig[0])))




try:
	del datatempcrop_full_res
	del basetemp_full_res
except:
	1+1

basecounts = np.mean(data3[0],axis=(0))
rotangle=0
datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
basetemp = coleval.record_rotation(datatemp,rotangle)


datatemp, errdatatemp = coleval.count_to_temp_poly2(data3, params, errparams)
datatempcrop = coleval.record_rotation(datatemp,rotangle)
errdatatempcrop = coleval.record_rotation(errdatatemp,rotangle)


mean = np.mean(data3[0],axis=(0))

testrot = mean
rotangle = -1.5  # in degrees
foilrot = rotangle * 2 * np.pi / 360
foilrotdeg = rotangle
foilcenter = [160, 133]
foilhorizw = 0.09
foilvertw = 0.07
foilhorizwpixel = 240
foilvertwpixel = np.int((foilhorizwpixel * foilvertw) // foilhorizw)
r = ((foilhorizwpixel ** 2 + foilvertwpixel ** 2) ** 0.5) / 2  # HALF DIAGONAL
a = foilvertwpixel / np.cos(foilrot)
tgalpha = np.tan(foilrot)
delta = -(a ** 2) / 4 + (1 + tgalpha ** 2) * (r ** 2)
foilx = np.add(foilcenter[0], [(-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
							   (-0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
							   (0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
							   (0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
							   (-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2)])
foily = np.add(foilcenter[1] - tgalpha * foilcenter[0],
			   [tgalpha * foilx[0] + a / 2, tgalpha * foilx[1] + a / 2, tgalpha * foilx[2] - a / 2,
				tgalpha * foilx[3] - a / 2, tgalpha * foilx[0] + a / 2])
foilxint = (np.rint(foilx)).astype(int)
foilyint = (np.rint(foily)).astype(int)

precisionincrease = 10
dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
foilcenterrot = (
	np.rint([np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
	int)
foilxrot = (np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
					 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
					 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
foilyrot = (np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
					 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
					 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

foillx = min(foilxrot)
foilrx = max(foilxrot)
foilhorizwpixel = foilrx - foillx
foildw = min(foilyrot)
foilup = max(foilyrot)
foilvertwpixel = foilup - foildw

basecounts = mean

datatemp, errdatatemp = coleval.count_to_temp_poly2([[basecounts]], params, errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
# basetemp = np.mean(datatempcrop[0, :, 1:-1, 1:-1], axis=0)
basetemp = np.mean(datatempcrop[0], axis=0)


datatemp, errdatatemp = coleval.count_to_temp_poly2(data3, params, errparams)
datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]



for spatial_averaging in averaging_all:

	time_averaging = 1
	if not ('datatempcrop_full_res' in dir()):
		datatempcrop_full_res = copy.deepcopy(datatempcrop)
	datatempcrop = coleval.average_multiple_frames(datatempcrop_full_res,spatial_averaging,timemean=time_averaging)
	print('ping')

	if not ('basetemp_full_res' in dir()):
		basetemp_full_res = copy.deepcopy(basetemp)
	basetemp=resize(basetemp_full_res,np.shape(datatempcrop[0,0]),order=1)
	print('ping')

	foilemissivityscaled_orig = 1 * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	conductivityscaled = Ptthermalconductivity * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	reciprdiffusivityscaled_orig = (1 / (0.4*Ptthermaldiffusivity)) * np.ones((np.shape(datatempcrop[0,0])[0]-2,np.shape(datatempcrop[0,0])[1]-2))
	flat_properties = True


	dt=time_averaging*1/framerate
	dx=foilhorizw/(np.shape(datatempcrop)[-1])
	dy=foilvertw/(np.shape(datatempcrop)[-2])
	relative_temp = np.add(datatempcrop,-basetemp)
	dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
	d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
	d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
	# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
	# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
	# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
	d2Tdxy=np.add(d2Tdx2,d2Tdy2)
	negd2Tdxy=np.multiply(-1,d2Tdxy)
	T4=np.power(np.add(zeroC,datatempcrop[:,1:-1,1:-1,1:-1]),4)
	T04=np.power(np.add(zeroC,basetemp[1:-1, 1:-1]),4)

	print('ping')
	reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
	foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
	foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

	BBrad=[]
	diffusion=[]
	timevariation=[]
	ktf=np.multiply(conductivityscaled,foilthicknessscaled)
	for i in range(len(datatempcrop[:,0,0,0])):
		BBrad.append([])
		diffusion.append([])
		timevariation.append([])
		for j in range(len(datatempcrop[0,1:-1,0,0])):
			BBradtemp=np.multiply(np.multiply(2*sigmaSB,foilemissivityscaled),np.add(T4[i,j],np.negative(T04)))
			BBrad[i].append(BBradtemp)
			diffusiontemp=np.multiply(ktf,negd2Tdxy[i,j])
			diffusion[i].append(diffusiontemp)
			timevariationtemp=np.multiply(ktf,np.multiply(reciprdiffusivityscaled,dTdt[i,j]))
			timevariation[i].append(timevariationtemp)
	BBrad=np.array(BBrad)
	diffusion=np.array(diffusion)
	timevariation=np.array(timevariation)
	print('ping')
	BBradnoback=np.add(BBrad,0)
	diffusionnoback=np.add(diffusion,0)
	timevariationnoback_clean=np.add(timevariation,0)

	std_all_clean.append(np.std(coleval.flatten_full(timevariationnoback_clean[0])))













plt.figure()
averaging_all = [1,2,3,4,5,6,7,8,9,10,20, 30, 40,80,150,250]
std_all_orig = []
std_all_clean = []
shape = (np.array(np.shape(timevariationnoback_clean))/2).astype(int)

for averaging in averaging_all:
	average = np.mean(timevariationnoback_orig[0,:,shape[2]:shape[2]+averaging,shape[3]:shape[3]+averaging],axis=(-1,-2))
	# average = coleval.average_multiple_frames(data,averaging)
	# shape = (np.array(np.shape(average))/2).astype(int)
	std = np.std(average)
	# std = np.mean((average[:-1]-average[1:])**2)
	std_all_orig.append(std)
	average = np.mean(timevariationnoback_clean[0,:,shape[2]:shape[2]+averaging,shape[3]:shape[3]+averaging],axis=(-1,-2))
	# average = coleval.average_multiple_frames(data,averaging)
	# shape = (np.array(np.shape(average))/2).astype(int)
	std = np.std(average)
	# std = np.mean((average[:-1] - average[1:]) ** 2)
	std_all_clean.append(std)

plt.figure()
plt.plot(np.array(averaging_all)**2,std_all_orig,'o',label='unfiltered')
plt.plot(np.array(averaging_all)**2,std_all_clean,'x',label='filtered')
plt.plot(np.array(averaging_all)**2,std_all_orig[0]/np.array(averaging_all),'--k',label='reference')
# plt.plot(averaging_all,std_all_clean[0]/np.array(averaging_all),'--k')
plt.semilogx()
plt.title('Time variation term of power density in \n ' + pathfiles,fontsize=9)
plt.xlabel('Number of foil pixels averaged')
plt.ylabel('Deviation of power time variation term [W/m2]')
plt.grid()
plt.legend(loc='best')
plt.pause(0.001)




plt.figure()
spectra = np.fft.fft(data[0], axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

plt.figure()
plt.plot(freq, np.mean(magnitude,axis=(-1,-2)),label='unfiltered data')
plt.plot(freq, np.mean(magnitude,axis=(-1,-2)), '+')



spectra = np.fft.fft(data2[0], axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
plt.plot(freq, np.mean(magnitude,axis=(-1,-2)),label='filtered data')
plt.plot(freq, np.mean(magnitude,axis=(-1,-2)), '+')
# plt.title(
plt.semilogy()
plt.title(
	'Amplitued from fast Fourier transform averaged over all pixels in counts in \n ' + pathfiles + ' framerate ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [au]')
plt.legend(loc='best')
plt.pause(0.001)