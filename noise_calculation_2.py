# Created 13/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())





# degree of polynomial of choice
n = 3
# folder of the parameters path
pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
if False:
	pathfiles = files20[-1]
	horiz = [0,-1]
	vert = [0,-1]
elif True:
	pathfiles = vacuum7[-1]
	horiz = [50,277]
	vert = [42,215]
# framerate of the IR camera in Hz
framerate = 383
# integration time of the camera in ms
inttime = 1
# filestype
type = '.npy'
# type='csv'
index_image_to_save = 0

filenames = coleval.all_file_names(pathfiles, type)[0]
data = np.load(os.path.join(pathfiles, filenames))



spectra = np.fft.fft(data[0], axis=0)
# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
magnitude = 2 * np.abs(spectra) / len(spectra)
phase = np.angle(spectra)
freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

if False:
	plt.figure(figsize=(20,10))
	plt.plot(freq,np.mean(magnitude,axis=(1,2)))
	plt.semilogy()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)

	# samplefreq = 587
	samplefreq = np.abs(freq-29.36*1).argmin()
	averaging = 1
	plt.figure(figsize=(20,10))
	plt.title('Normalised amplitude of ' + str(np.around(freq[samplefreq],decimals=2))  + 'Hz oscillation from fast Fourier transform in counts in \n ' + pathfiles + ' FR ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms',fontsize=9)
	plt.imshow(coleval.average_frame(magnitude[samplefreq],averaging), 'rainbow', origin='lower')
	plt.colorbar().set_label('Amplitude [au]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.show()
	plt.figure(figsize=(20,10))
	plt.title('Phase of ' +  str(np.around(freq[samplefreq],decimals=2))  + 'Hz oscillation from fast Fourier transform in counts in \n ' + pathfiles + ' FR ' + str(
		framerate) + 'Hz, int. time ' + str(inttime) + 'ms',fontsize=9)
	plt.imshow(coleval.average_frame(phase[samplefreq],averaging), 'rainbow', origin='lower')
	plt.colorbar().set_label('Phase [rad]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
else:
	index_image_to_save += 3

if False:
	time_averaging_all = [1,2,3,4,5,6,7,8,9,10,12,14,15,20]
	# treshold_for_bad_std_all = np.linspace(4,30,9)
	treshold_for_bad_std_all = np.logspace(np.log10(4), np.log10(30), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_1 = []
	data_to_check = data[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_std in enumerate(treshold_for_bad_std_all):
		# flag = coleval.find_dead_pixels(datatempcrop, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std,treshold_for_bad_difference=80)
		record_bad_pixels_1.append(np.sum(flag!=0))
		to_print = []
		for time_averaging in time_averaging_all:
			pixel_t_new = np.floor(pixel_t / time_averaging).astype('int')
			print('foil will be binned to (t) ' + str(pixel_t_new) + ' time pixels')
			binning_t = np.abs(np.linspace(0, pixel_t - 1, pixel_t) // time_averaging).astype('int')
			binning_t[binning_t > pixel_t_new - 1] = pixel_t_new - 1
			pixels_to_bin = binning_t.astype('int')

			# temp = np.zeros((len(datatempcrop[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros(( np.max(pixels_to_bin) + 1,np.sum(flag==0)))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[bin_index] = np.mean(data_to_check[0,pixels_to_bin == bin_index][:,flag==0], axis=0)
			# datatempcrop_not_binned = copy.deepcopy(datatempcrop)
			data_to_check_binned = cp.deepcopy(temp)
			to_print.append(np.mean(np.std(data_to_check_binned,axis=0)))
		plt.plot(time_averaging_all,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_std = '+str('%.3g' % treshold_for_bad_std)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(time_averaging_all,to_print[0]*(1/(np.array(time_averaging_all)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in time UNFILTERED DATA \n(deviation from the gaussian distribution of values)\non the mean time std')
	plt.xlabel('temporal averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [counts]')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	# plt.figure(figsize=(20,10))
	# plt.plot(treshold_for_bad_std_all, record_bad_pixels)
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('treshold_for_bad_std_all [counts]')
	# plt.ylabel('number of bad pixels [pixels]')
	# plt.title('Unfiltered data')
	# index_image_to_save+=1
	# plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	# plt.close()
	# # plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(1,15,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(1), np.log10(15), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_2 = []
	data_to_check = data[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=50, treshold_for_bad_difference=treshold_for_bad_difference)
		record_bad_pixels_2.append(np.sum(flag!=0))
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanmean(np.nanstd([temp[0:-2,1:-1],temp[1:-1,1:-1],temp[2:,1:-1],temp[1:-1,0:-2],temp[1:-1,2:]],axis=0)))
			to_print.append(np.nanmean(np.nanstd([data_to_check_binned[:,0:-2, 1:-1], data_to_check_binned[:,1:-1, 1:-1], data_to_check_binned[:,2:, 1:-1], data_to_check_binned[:,1:-1, 0:-2], data_to_check_binned[:,1:-1, 2:]], axis=0)))
			# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space UNFILTERED DATA \n(deviation from the max and min value in the neighbouring pixels)\non the mean of the local spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [counts]')
	plt.semilogx()
	index_image_to_save += 1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps',bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	# plt.figure(figsize=(20,10))
	# plt.plot(treshold_for_bad_difference_all, record_bad_pixels)
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('treshold_for_bad_difference_all [counts]')
	# plt.ylabel('number of bad pixels [pixles]')
	# plt.title('Unfiltered data')
	# index_image_to_save+=1
	# plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	# plt.close()
	# # plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(1,15,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(1), np.log10(15), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = data[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=50, treshold_for_bad_difference=treshold_for_bad_difference)
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanstd(temp))
			to_print.append(np.mean(np.nanstd(data_to_check_binned, axis=(-1, -2))))
			# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space UNFILTERED DATA \n(deviation from the max and min value in the neighbouring pixels)\non the global spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [counts]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
else:
	index_image_to_save += 3


data2=coleval.clear_oscillation_central2(data,framerate,plot_conparison=False)
# data2=coleval.clear_oscillation_central2(data,framerate,oscillation_search_window_end=0.25,plot_conparison=True,min_frequency_to_erase=22,max_frequency_to_erase=32)

if False:
	time_averaging_all = [1,2,3,4,5,6,7,8,9,10,12,14,15,20]
	# treshold_for_bad_std_all = np.linspace(4,30,9)
	treshold_for_bad_std_all = np.logspace(np.log10(4), np.log10(30), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_3 = []
	data_to_check = data2[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t,pixel_h,pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_std in enumerate(treshold_for_bad_std_all):
		# flag = coleval.find_dead_pixels(datatempcrop, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std,treshold_for_bad_difference=80)
		record_bad_pixels_3.append(np.sum(flag!=0))
		to_print = []
		for time_averaging in time_averaging_all:
			pixel_t_new = np.floor(pixel_t / time_averaging).astype('int')
			print('foil will be binned to (t) ' + str(pixel_t_new) + ' time pixels')
			binning_t = np.abs(np.linspace(0, pixel_t - 1, pixel_t) // time_averaging).astype('int')
			binning_t[binning_t > pixel_t_new - 1] = pixel_t_new - 1
			pixels_to_bin = binning_t.astype('int')

			# temp = np.zeros((len(datatempcrop[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros(( np.max(pixels_to_bin) + 1,np.sum(flag==0)))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[bin_index] = np.mean(data_to_check[0,pixels_to_bin == bin_index][:,flag==0], axis=0)
			# datatempcrop_not_binned = copy.deepcopy(datatempcrop)
			data_to_check_binned = cp.deepcopy(temp)
			to_print.append(np.mean(np.std(data_to_check_binned,axis=0)))
		plt.plot(time_averaging_all,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_std = '+str('%.3g' % treshold_for_bad_std)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(time_averaging_all,to_print[0]*(1/(np.array(time_averaging_all)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in time FILTERED\n(deviation from the gaussian distribution of values)\non the mean time std')
	plt.xlabel('temporal averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [counts]')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	# plt.figure(figsize=(20,10))
	# plt.plot(treshold_for_bad_std_all, record_bad_pixels)
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('treshold_for_bad_std_all [counts]')
	# plt.ylabel('number of bad pixels [pixles]')
	# plt.title('Filtered data')
	# index_image_to_save+=1
	# plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	# plt.close()
	# # plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(1,15,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(1), np.log10(15), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_4 = []
	data_to_check = data2[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=50, treshold_for_bad_difference=treshold_for_bad_difference)
		record_bad_pixels_4.append(np.sum(flag!=0))
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanmean(np.nanstd([temp[0:-2,1:-1],temp[1:-1,1:-1],temp[2:,1:-1],temp[1:-1,0:-2],temp[1:-1,2:]],axis=0)))
			to_print.append(np.nanmean(np.nanstd([data_to_check_binned[:, 0:-2, 1:-1], data_to_check_binned[:, 1:-1, 1:-1],data_to_check_binned[:, 2:, 1:-1], data_to_check_binned[:, 1:-1, 0:-2],data_to_check_binned[:, 1:-1, 2:]], axis=0)))
			# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space FILTERED\n(deviation from the max and min value in the neighbouring pixels)\non the mean of the local spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [counts]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	plt.figure(figsize=(20,10))
	plt.plot(treshold_for_bad_std_all, record_bad_pixels_1,label='temporal bad pixels, non filtered')
	plt.plot(treshold_for_bad_std_all, record_bad_pixels_3,'--',label='temporal bad pixels, filtered')
	plt.plot(treshold_for_bad_difference_all, record_bad_pixels_2,label='spatial bad pixels, non filtered')
	plt.plot(treshold_for_bad_difference_all, record_bad_pixels_4,'--',label='spatial bad pixels, filtered')
	plt.semilogx()
	plt.semilogy()
	plt.xlabel('treshold value [counts]')
	plt.ylabel('number of bad pixels [pixles]')
	plt.title('Cropped data')
	plt.legend(loc='best')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(1,15,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(1), np.log10(15), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = data2[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=50, treshold_for_bad_difference=treshold_for_bad_difference)
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanstd(temp))
			to_print.append(np.mean(np.nanstd(data_to_check_binned, axis=(-1, -2))))
			# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space FILTERED\n(deviation from the max and min value in the neighbouring pixels)\non the global spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [counts]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
else:
	index_image_to_save += 4



params=np.load(os.path.join(pathparams,'coeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
errparams=np.load(os.path.join(pathparams,'errcoeffpolydeg'+str(n)+'int'+str(inttime)+'ms.npy'))
datatemp, errdatatemp = coleval.count_to_temp_poly2(data2, params, errparams)


if False:
	time_averaging_all = [1,2,3,4,5,6,7,8,9,10,12,14,15,20]
	# treshold_for_bad_std_all = np.linspace(0.025,0.2,9)
	treshold_for_bad_std_all = np.logspace(np.log10(0.025), np.log10(0.2), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_5 = []
	data_to_check = datatemp[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t,pixel_h,pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_std in enumerate(treshold_for_bad_std_all):
		# flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std,treshold_for_bad_difference=13 / 100)
		record_bad_pixels_5.append(np.sum(flag!=0))
		to_print = []
		for time_averaging in time_averaging_all:
			pixel_t_new = np.floor(pixel_t / time_averaging).astype('int')
			print('foil will be binned to (t) ' + str(pixel_t_new) + ' time pixels')
			binning_t = np.abs(np.linspace(0, pixel_t - 1, pixel_t) // time_averaging).astype('int')
			binning_t[binning_t > pixel_t_new - 1] = pixel_t_new - 1
			pixels_to_bin = binning_t.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros(( np.max(pixels_to_bin) + 1,np.sum(flag==0)))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[bin_index] = np.mean(data_to_check[0,pixels_to_bin == bin_index][:,flag==0], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = cp.deepcopy(temp)
			to_print.append(np.mean(np.std(data_to_check_binned,axis=0)))
		plt.plot(time_averaging_all,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_std = '+str('%.3g' % treshold_for_bad_std)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(time_averaging_all,to_print[0]*(1/(np.array(time_averaging_all)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in time\n(deviation from the gaussian distribution of values)\non the mean of time std')
	plt.xlabel('temporal averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [K]')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	# plt.figure(figsize=(20,10))
	# plt.plot(treshold_for_bad_std_all, record_bad_pixels)
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('treshold_for_bad_std_all [K]')
	# plt.ylabel('number of bad pixels [pixles]')
	# plt.title('Filtered data')
	# index_image_to_save+=1
	# plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	# plt.close()
	# # plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(0.003,0.03,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(0.006), np.log10(0.06), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_6 = []
	data_to_check = datatemp[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=0.5, treshold_for_bad_difference=treshold_for_bad_difference)
		record_bad_pixels_6.append(np.sum(flag!=0))
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanmean(np.nanstd([temp[0:-2,1:-1],temp[1:-1,1:-1],temp[2:,1:-1],temp[1:-1,0:-2],temp[1:-1,2:]],axis=0)))
			to_print.append(np.nanmean(np.nanstd([data_to_check_binned[:, 0:-2, 1:-1], data_to_check_binned[:, 1:-1, 1:-1],data_to_check_binned[:, 2:, 1:-1], data_to_check_binned[:, 1:-1, 0:-2],data_to_check_binned[:, 1:-1, 2:]], axis=0)))
			# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space\n(deviation from the max and min value in the neighbouring pixels)\non the mean of the local spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [K]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	# plt.figure(figsize=(20,10))
	# plt.plot(treshold_for_bad_difference_all, record_bad_pixels)
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('treshold_for_bad_difference_all [K]')
	# plt.ylabel('number of bad pixels [pixles]')
	# plt.title('Filtered data')
	# index_image_to_save+=1
	# plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	# plt.close()
	# # plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(0.003,0.03,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(0.006), np.log10(0.06), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = datatemp[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=0.5, treshold_for_bad_difference=treshold_for_bad_difference)
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanstd(temp))
			to_print.append(np.mean(np.nanstd(data_to_check_binned, axis=(-1, -2))))
		# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space\n(deviation from the max and min value in the neighbouring pixels)\non the global spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [K]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
else:
	index_image_to_save += 3

if False:
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
	dummy = np.ones(np.multiply(np.shape(data[0,0]), precisionincrease))
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

	datatemprot = rotate(datatemp, foilrotdeg, axes=(-1, -2))
	datatempcrop = datatemprot[:, :, foildw:foilup, foillx:foilrx]
	errdatatemprot = rotate(errdatatemp, foilrotdeg, axes=(-1, -2))
	errdatatempcrop = errdatatemprot[:, :, foildw:foilup, foillx:foilrx]

	horiz = [0,-1]
	vert = [0,-1]

elif True:
	rotangle = -1.5  # in degrees
	foilrot = rotangle * 2 * np.pi / 360
	foilrotdeg = rotangle
	foilcenter = [160, 133]
	foilhorizw = 0.09
	foilvertw = 0.07
	foilhorizwpixel = 240

	datatempcrop = coleval.record_rotation(datatemp,rotangle, foilcenter= foilcenter,	foilhorizwpixel = foilhorizwpixel)
	errdatatempcrop = coleval.record_rotation(errdatatemp,rotangle, foilcenter= foilcenter,	foilhorizwpixel = foilhorizwpixel)

	horiz = [0,-1]
	vert = [0,-1]


if False:
	time_averaging_all = [1,2,3,4,5,6,7,8,9,10,12,14,15,20]
	# treshold_for_bad_std_all = np.linspace(0.025,0.2,9)
	treshold_for_bad_std_all = np.logspace(np.log10(0.025), np.log10(0.2), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_7 = []
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t,pixel_h,pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_std in enumerate(treshold_for_bad_std_all):
		# flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std,treshold_for_bad_difference=13 / 100)
		record_bad_pixels_7.append(np.sum(flag!=0))
		to_print = []
		for time_averaging in time_averaging_all:
			pixel_t_new = np.floor(pixel_t / time_averaging).astype('int')
			print('foil will be binned to (t) ' + str(pixel_t_new) + ' time pixels')
			binning_t = np.abs(np.linspace(0, pixel_t - 1, pixel_t) // time_averaging).astype('int')
			binning_t[binning_t > pixel_t_new - 1] = pixel_t_new - 1
			pixels_to_bin = binning_t.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros(( np.max(pixels_to_bin) + 1,np.sum(flag==0)))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[bin_index] = np.mean(data_to_check[0,pixels_to_bin == bin_index][:,flag==0], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = cp.deepcopy(temp)
			to_print.append(np.mean(np.std(data_to_check_binned,axis=0)))
		plt.plot(time_averaging_all,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_std = '+str('%.3g' % treshold_for_bad_std)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(time_averaging_all,to_print[0]*(1/(np.array(time_averaging_all)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in time ROTATED AND CROPPED\n(deviation from the gaussian distribution of values)\non the mean of time std')
	plt.xlabel('temporal averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [K]')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
	# plt.figure(figsize=(20,10))
	# plt.plot(treshold_for_bad_std_all, record_bad_pixels)
	# plt.semilogx()
	# plt.semilogy()
	# plt.xlabel('treshold_for_bad_std_all [K]')
	# plt.ylabel('number of bad pixels [pixles]')
	# plt.title('Filtered data ROTATED AND CROPPED')
	# index_image_to_save+=1
	# plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	# plt.close()
	# # plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(0.003,0.03,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(0.006), np.log10(0.06), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	record_bad_pixels_8 = []
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=0.5, treshold_for_bad_difference=treshold_for_bad_difference)
		record_bad_pixels_8.append(np.sum(flag!=0))
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanmean(np.nanstd([temp[0:-2,1:-1],temp[1:-1,1:-1],temp[2:,1:-1],temp[1:-1,0:-2],temp[1:-1,2:]],axis=0)))
			to_print.append(np.nanmean(np.nanstd([data_to_check_binned[:, 0:-2, 1:-1], data_to_check_binned[:, 1:-1, 1:-1],data_to_check_binned[:, 2:, 1:-1], data_to_check_binned[:, 1:-1, 0:-2],data_to_check_binned[:, 1:-1, 2:]], axis=0)))
			# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space ROTATED AND CROPPED\n(deviation from the max and min value in the neighbouring pixels)\non the mean of the local spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [K]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)


	plt.figure(figsize=(20,10))
	plt.plot(treshold_for_bad_std_all, record_bad_pixels_5,label='temporal bad pixels, non rotated')
	plt.plot(treshold_for_bad_std_all, record_bad_pixels_7,'--',label='temporal bad pixels, rotated')
	plt.plot(treshold_for_bad_difference_all, record_bad_pixels_6,label='spatial bad pixels, non rotated')
	plt.plot(treshold_for_bad_difference_all, record_bad_pixels_8,'--',label='spatial bad pixels, rotated')
	plt.semilogx()
	plt.semilogy()
	plt.xlabel('treshold value [K]')
	plt.ylabel('number of bad pixels [pixles]')
	plt.title('Filtered, cropped and temperature converted data')
	plt.legend(loc='best')
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)


	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50,80,100]
	# treshold_for_bad_difference_all = np.linspace(0.003,0.03,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(0.006), np.log10(0.06), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=0.5, treshold_for_bad_difference=treshold_for_bad_difference)
		to_print = []
		for spatial_averaging in spatial_averaging_all:
			pixel_h_new = np.floor(pixel_h / spatial_averaging).astype('int')
			pixel_v_new = np.floor(pixel_v / spatial_averaging).astype('int')
			print('foil will be binned to (hxv) ' + str(pixel_h_new) + 'x' + str(pixel_v_new) + ' pixels')
			binning_h = np.abs(np.linspace(0, pixel_h - 1, pixel_h) // spatial_averaging).astype('int')
			binning_h[binning_h > pixel_h_new - 1] = pixel_h_new - 1
			binning_v = np.abs(np.linspace(0, pixel_v - 1, pixel_v) // spatial_averaging).astype('int')
			binning_v[binning_v > pixel_v_new - 1] = pixel_v_new - 1
			# pixels_to_bin = np.array([[(((np.ones((pixel_v, pixel_h)) * binning_h).T + binning_v * pixel_h_new).astype('int')).tolist()]*len(data_to_check[0])])
			pixels_to_bin = ((np.ones((pixel_h, pixel_v)) * binning_v).T + binning_h * pixel_v_new).T.astype('int')

			# temp = np.zeros((len(data_to_check[0]),np.max(binning_h) + 1, np.max(binning_v) + 1))
			temp = np.zeros((len(data_to_check[0]), np.max(pixels_to_bin) + 1))
			for bin_index in range(np.max(pixels_to_bin) + 1):
				temp[:,bin_index] = np.mean(data_to_check[0,:,np.logical_and(pixels_to_bin == bin_index,flag==0)], axis=0)
			# data_to_check_not_binned = copy.deepcopy(data_to_check)
			data_to_check_binned = temp.reshape((len(temp), pixel_h_new,pixel_v_new))
			# temp = np.mean(data_to_check_binned,axis=0)
			# to_print.append(np.nanstd(temp))
			to_print.append(np.mean(np.nanstd(data_to_check_binned, axis=(-1, -2))))
		# to_print.append(np.std(np.mean(data_to_check_binned,axis=0)))
		plt.plot(np.array(spatial_averaging_all)**2,to_print,'C'+str(1+index_tresh),label='treshold_for_bad_difference = '+str('%.3g' % treshold_for_bad_difference)+', '+str('%.3g' % np.sum(flag!=0))+' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2,to_print[0]*(1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C'+str(1+index_tresh)+'--')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space ROTATED AND CROPPED\n(deviation from the max and min value in the neighbouring pixels)\non the global spatial std')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel std [K]')
	plt.semilogx()
	index_image_to_save+=1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
else:
	index_image_to_save+=4


basetemp = np.mean(datatempcrop[0],axis=(0))[vert[0]:vert[1]:,horiz[0]:horiz[1]]
def expected_noise(framerate,IR_framerate,camera_pixels,IR_pixels):
	framerate = np.array(framerate)
	IR_framerate = np.array(IR_framerate)
	camera_pixels = np.array(camera_pixels)
	IR_pixels = np.array(IR_pixels)
	noise_eq_temperature = 0.0225	# K
	return noise_eq_temperature*((10**0.5)*Ptthermalconductivity*(2.5 / 1000000)/((framerate*camera_pixels)**0.5))*(( ((IR_pixels**3)*IR_framerate/(foilhorizw*foilvertw)) + ((IR_framerate**3)*IR_pixels/(5*((0.4*Ptthermaldiffusivity)**2))) )**0.5)

if True:
	time_averaging_all = [1,2,3,4,5,6,7,8,9,10,12,14,15,20]
	# treshold_for_bad_std_all = np.linspace(0.025,0.2,9)
	treshold_for_bad_std_all = np.logspace(np.log10(0.025), np.log10(0.2), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t,pixel_h,pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_std in enumerate(treshold_for_bad_std_all):
		# flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std,treshold_for_bad_difference=13 / 100)
		to_print = []
		for time_averaging in time_averaging_all:

			datatempcrop_binned = coleval.record_binning(data_to_check,time_averaging,1,1,flag)
			datatempcrop_binned = coleval.replace_dead_pixels_2(datatempcrop_binned, flag)

			basetemp_binned = coleval.record_binning(np.array([[basetemp]]),1,1,1,flag)
			basetemp_binned = coleval.replace_dead_pixels_2(basetemp_binned, flag)[0,0]
			basetemp_binned = np.ones_like(basetemp_binned)*np.mean(basetemp_binned)

			foilemissivityscaled_orig = 1 * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			conductivityscaled = Ptthermalconductivity * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			reciprdiffusivityscaled_orig = (1 / (0.4 * Ptthermaldiffusivity)) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			flat_properties = True



			dt=time_averaging*1/framerate
			dx=foilhorizw/(np.shape(datatempcrop_binned)[-1])
			dy=foilvertw/(np.shape(datatempcrop_binned)[-2])
			relative_temp = np.add(datatempcrop_binned,-basetemp_binned)
			dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
			d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
			d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
			# dTdt=np.divide(datatempcrop_binned[:,2:,1:-1,1:-1]-datatempcrop_binned[:,:-2,1:-1,1:-1],2*dt)
			# d2Tdx2=np.divide(datatempcrop_binned[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop_binned[:,1:-1,1:-1,1:-1])+datatempcrop_binned[:,1:-1,1:-1,:-2],dx**2)
			# d2Tdy2=np.divide(datatempcrop_binned[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop_binned[:,1:-1,1:-1,1:-1])+datatempcrop_binned[:,1:-1,:-2,1:-1],dx**2)
			d2Tdxy=np.add(d2Tdx2,d2Tdy2)
			negd2Tdxy=np.multiply(-1,d2Tdxy)
			T4=np.power(np.add(zeroC,datatempcrop_binned[:,1:-1,1:-1,1:-1]),4)
			T04=np.power(np.add(zeroC,basetemp_binned[1:-1, 1:-1]),4)

			print('ping')
			reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
			foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
			foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

			BBrad=[]
			diffusion=[]
			timevariation=[]
			ktf=np.multiply(conductivityscaled,foilthicknessscaled)
			for i in range(len(datatempcrop_binned[:,0,0,0])):
				BBrad.append([])
				diffusion.append([])
				timevariation.append([])
				for j in range(len(datatempcrop_binned[0,1:-1,0,0])):
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

			print('np.shape(powernoback) '+str(np.shape(powernoback)))

			# to_print.append(np.mean(np.std(powernoback[0], axis=0)))
			to_print.append(np.std(powernoback[0]))
		plt.plot(time_averaging_all, to_print, 'C' + str(1 + index_tresh),label='treshold_for_bad_std = ' + str('%.3g' % treshold_for_bad_std) + ', ' + str('%.3g' % np.sum(flag != 0)) + ' pixels excluded')
		plt.plot(time_averaging_all, to_print[0] * (1 / (np.array(time_averaging_all) ** 0.5)),'C' + str(1 + index_tresh) + '--')
		plt.plot(time_averaging_all,expected_noise(framerate,framerate/np.array(time_averaging_all),pixel_h*pixel_v,pixel_h*pixel_v),'C' + str(1 + index_tresh) + 'x')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space IN POWER, NOT background substracted\n(deviation from the gaussian distribution of values)\non the global std\nmarked with cross the expected value (from error propagation)')
	plt.xlabel('temporal averaging performed [pixles]')
	plt.ylabel('mean of each pixel power std [W/m2]')
	index_image_to_save += 1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps',bbox_inches='tight')
	plt.close()
else:
	index_image_to_save += 1

if True:
	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50]
	# treshold_for_bad_difference_all = np.linspace(0.003,0.03,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(0.006), np.log10(0.06), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	time_averaging = 1
	for index_tresh, treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		# flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=0.5, treshold_for_bad_difference=treshold_for_bad_difference)
		to_print = []
		for spatial_averaging in spatial_averaging_all:

			datatempcrop_binned = coleval.record_binning(data_to_check, 1, spatial_averaging, spatial_averaging, flag)
			datatempcrop_binned = coleval.replace_dead_pixels_2(datatempcrop_binned, flag)

			basetemp_binned = coleval.record_binning(np.array([[basetemp]]),1,spatial_averaging,spatial_averaging,flag)
			basetemp_binned = coleval.replace_dead_pixels_2(basetemp_binned, flag)[0,0]
			basetemp_binned = np.ones_like(basetemp_binned) * np.mean(basetemp_binned)

			foilemissivityscaled_orig = 1 * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			conductivityscaled = Ptthermalconductivity * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			reciprdiffusivityscaled_orig = (1 / (0.4 * Ptthermaldiffusivity)) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			flat_properties = True



			dt=time_averaging*1/framerate
			dx=foilhorizw/(np.shape(datatempcrop_binned)[-1])
			dy=foilvertw/(np.shape(datatempcrop_binned)[-2])
			relative_temp = np.add(datatempcrop_binned,-basetemp_binned)
			dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
			d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
			d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
			# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
			# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
			# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
			d2Tdxy=np.add(d2Tdx2,d2Tdy2)
			negd2Tdxy=np.multiply(-1,d2Tdxy)
			T4=np.power(np.add(zeroC,datatempcrop_binned[:,1:-1,1:-1,1:-1]),4)
			T04=np.power(np.add(zeroC,basetemp_binned[1:-1, 1:-1]),4)

			print('ping')
			reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
			foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
			foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

			BBrad=[]
			diffusion=[]
			timevariation=[]
			ktf=np.multiply(conductivityscaled,foilthicknessscaled)
			for i in range(len(datatempcrop_binned[:,0,0,0])):
				BBrad.append([])
				diffusion.append([])
				timevariation.append([])
				for j in range(len(datatempcrop_binned[0,1:-1,0,0])):
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

			print('np.shape(powernoback) ' + str(np.shape(powernoback)))

			# to_print.append(np.mean(np.std(powernoback[0], axis=(-1,-2))))
			to_print.append(np.std(powernoback[0]))
		plt.plot(np.array(spatial_averaging_all)**2, to_print, 'C' + str(1 + index_tresh),label='treshold_for_bad_difference = ' + str('%.3g' % treshold_for_bad_difference) + ', ' + str('%.3g' % np.sum(flag != 0)) + ' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2, to_print[0] * (1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C' + str(1 + index_tresh) + '--')
		plt.plot(np.array(spatial_averaging_all)**2,expected_noise(framerate,framerate,pixel_h*pixel_v,pixel_h*pixel_v/np.array(spatial_averaging_all)),'C' + str(1 + index_tresh) + 'x')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space IN POWER, NOT background substracted\n(deviation from the max and min value in the neighbouring pixels)\non the global std\nmarked with cross the expected value (from error propagation)')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel power std [W/m2]')
	plt.semilogx()
	index_image_to_save += 1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps',bbox_inches='tight')
	plt.close()
else:
	index_image_to_save += 1


if True:
	time_averaging_all = [1,2,3,4,5,6,7,8,9,10,12,14,15,20]
	# treshold_for_bad_std_all = np.linspace(0.025,0.2,9)
	treshold_for_bad_std_all = np.logspace(np.log10(0.025), np.log10(0.2), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t,pixel_h,pixel_v = np.shape(data_to_check[0])
	for index_tresh,treshold_for_bad_std in enumerate(treshold_for_bad_std_all):
		# flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std,treshold_for_bad_difference=13 / 100)
		to_print = []
		for time_averaging in time_averaging_all:

			datatempcrop_binned = coleval.record_binning(data_to_check,time_averaging,1,1,flag)
			datatempcrop_binned = coleval.replace_dead_pixels_2(datatempcrop_binned, flag)

			basetemp_binned = coleval.record_binning(np.array([[basetemp]]),1,1,1,flag)
			basetemp_binned = coleval.replace_dead_pixels_2(basetemp_binned, flag)[0,0]

			foilemissivityscaled_orig = 1 * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			conductivityscaled = Ptthermalconductivity * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			reciprdiffusivityscaled_orig = (1 / (0.4 * Ptthermaldiffusivity)) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			flat_properties = True



			dt=time_averaging*1/framerate
			dx=foilhorizw/(np.shape(datatempcrop_binned)[-1])
			dy=foilvertw/(np.shape(datatempcrop_binned)[-2])
			relative_temp = np.add(datatempcrop_binned,-basetemp_binned)
			dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
			d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
			d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
			# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
			# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
			# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
			d2Tdxy=np.add(d2Tdx2,d2Tdy2)
			negd2Tdxy=np.multiply(-1,d2Tdxy)
			T4=np.power(np.add(zeroC,datatempcrop_binned[:,1:-1,1:-1,1:-1]),4)
			T04=np.power(np.add(zeroC,basetemp_binned[1:-1, 1:-1]),4)

			print('ping')
			reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
			foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
			foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

			BBrad=[]
			diffusion=[]
			timevariation=[]
			ktf=np.multiply(conductivityscaled,foilthicknessscaled)
			for i in range(len(datatempcrop_binned[:,0,0,0])):
				BBrad.append([])
				diffusion.append([])
				timevariation.append([])
				for j in range(len(datatempcrop_binned[0,1:-1,0,0])):
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

			print('np.shape(powernoback) '+str(np.shape(powernoback)))

			# to_print.append(np.mean(np.std(powernoback[0], axis=0)))
			to_print.append(np.std(powernoback[0]))
		plt.plot(time_averaging_all, to_print, 'C' + str(1 + index_tresh),label='treshold_for_bad_std = ' + str('%.3g' % treshold_for_bad_std) + ', ' + str('%.3g' % np.sum(flag != 0)) + ' pixels excluded')
		plt.plot(time_averaging_all, to_print[0] * (1 / (np.array(time_averaging_all) ** 0.5)),'C' + str(1 + index_tresh) + '--')
		plt.plot(time_averaging_all,expected_noise(framerate,framerate/np.array(time_averaging_all),pixel_h*pixel_v,pixel_h*pixel_v),'C' + str(1 + index_tresh) + 'x')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space IN POWER, background substracted\n(deviation from the gaussian distribution of values)\non the global std\nmarked with cross the expected value (from error propagation)')
	plt.xlabel('temporal averaging performed [pixles]')
	plt.ylabel('mean of each pixel power std [W/m2]')
	index_image_to_save += 1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps',bbox_inches='tight')
	plt.close()
else:
	index_image_to_save += 1

if True:
	spatial_averaging_all = [1,2,3,4,5,6,7,8,9,10,20,30,50]
	# treshold_for_bad_difference_all = np.linspace(0.003,0.03,9)
	treshold_for_bad_difference_all = np.logspace(np.log10(0.006), np.log10(0.06), 9, endpoint=True)
	plt.figure(figsize=(20,10))
	data_to_check = datatempcrop[:,:,vert[0]:vert[1]:,horiz[0]:horiz[1]]
	pixel_t, pixel_h, pixel_v = np.shape(data_to_check[0])
	time_averaging = 1
	for index_tresh, treshold_for_bad_difference in enumerate(treshold_for_bad_difference_all):
		# flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=treshold_for_bad_std, treshold_for_bad_difference=13/100)
		flag = coleval.find_dead_pixels(data_to_check, treshold_for_bad_std=0.5, treshold_for_bad_difference=treshold_for_bad_difference)
		to_print = []
		for spatial_averaging in spatial_averaging_all:

			datatempcrop_binned = coleval.record_binning(data_to_check, 1, spatial_averaging, spatial_averaging, flag)
			datatempcrop_binned = coleval.replace_dead_pixels_2(datatempcrop_binned, flag)

			basetemp_binned = coleval.record_binning(np.array([[basetemp]]),1,spatial_averaging,spatial_averaging,flag)
			basetemp_binned = coleval.replace_dead_pixels_2(basetemp_binned, flag)[0,0]

			foilemissivityscaled_orig = 1 * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			foilthicknessscaled_orig = (2.5 / 1000000) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			conductivityscaled = Ptthermalconductivity * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			reciprdiffusivityscaled_orig = (1 / (0.4 * Ptthermaldiffusivity)) * np.ones((np.shape(basetemp_binned)[0] - 2, np.shape(basetemp_binned)[1] - 2))
			flat_properties = True



			dt=time_averaging*1/framerate
			dx=foilhorizw/(np.shape(datatempcrop_binned)[-1])
			dy=foilvertw/(np.shape(datatempcrop_binned)[-2])
			relative_temp = np.add(datatempcrop_binned,-basetemp_binned)
			dTdt=np.divide(relative_temp[:,2:,1:-1,1:-1]-relative_temp[:,:-2,1:-1,1:-1],2*dt)
			d2Tdx2=np.divide(relative_temp[:,1:-1,1:-1,2:]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,1:-1,:-2],dx**2)
			d2Tdy2=np.divide(relative_temp[:,1:-1,2:,1:-1]-np.multiply(2,relative_temp[:,1:-1,1:-1,1:-1])+relative_temp[:,1:-1,:-2,1:-1],dy**2)
			# dTdt=np.divide(datatempcrop[:,2:,1:-1,1:-1]-datatempcrop[:,:-2,1:-1,1:-1],2*dt)
			# d2Tdx2=np.divide(datatempcrop[:,1:-1,1:-1,2:]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,1:-1,:-2],dx**2)
			# d2Tdy2=np.divide(datatempcrop[:,1:-1,2:,1:-1]-np.multiply(2,datatempcrop[:,1:-1,1:-1,1:-1])+datatempcrop[:,1:-1,:-2,1:-1],dx**2)
			d2Tdxy=np.add(d2Tdx2,d2Tdy2)
			negd2Tdxy=np.multiply(-1,d2Tdxy)
			T4=np.power(np.add(zeroC,datatempcrop_binned[:,1:-1,1:-1,1:-1]),4)
			T04=np.power(np.add(zeroC,basetemp_binned[1:-1, 1:-1]),4)

			print('ping')
			reciprdiffusivityscaled=np.multiply(1/1,reciprdiffusivityscaled_orig)
			foilemissivityscaled=np.multiply(1,foilemissivityscaled_orig)
			foilthicknessscaled=np.multiply(1,foilthicknessscaled_orig)

			BBrad=[]
			diffusion=[]
			timevariation=[]
			ktf=np.multiply(conductivityscaled,foilthicknessscaled)
			for i in range(len(datatempcrop_binned[:,0,0,0])):
				BBrad.append([])
				diffusion.append([])
				timevariation.append([])
				for j in range(len(datatempcrop_binned[0,1:-1,0,0])):
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

			print('np.shape(powernoback) ' + str(np.shape(powernoback)))

			# to_print.append(np.mean(np.std(powernoback[0], axis=(-1,-2))))
			to_print.append(np.std(powernoback[0]))
		plt.plot(np.array(spatial_averaging_all)**2, to_print, 'C' + str(1 + index_tresh),label='treshold_for_bad_difference = ' + str('%.3g' % treshold_for_bad_difference) + ', ' + str('%.3g' % np.sum(flag != 0)) + ' pixels excluded')
		plt.plot(np.array(spatial_averaging_all)**2, to_print[0] * (1/(np.array(np.array(spatial_averaging_all)**2)**0.5)),'C' + str(1 + index_tresh) + '--')
		plt.plot(np.array(spatial_averaging_all)**2,expected_noise(framerate,framerate,pixel_h*pixel_v,pixel_h*pixel_v/np.array(spatial_averaging_all)),'C' + str(1 + index_tresh) + 'x')
	plt.legend(loc='best')
	plt.title('Influence of the treshold for bad pixels in space IN POWER, background substracted\n(deviation from the max and min value in the neighbouring pixels)\non the global std\nmarked with cross the expected value (from error propagation)')
	plt.xlabel('spatial averaging performed [pixles]')
	plt.ylabel('mean of each pixel power std [W/m2]')
	plt.semilogx()
	index_image_to_save += 1
	plt.savefig('/home/ffederic/work/TOPRINT' + '/noise_analysis_long_record_img_' + str(index_image_to_save) + '.eps',bbox_inches='tight')
	plt.close()
else:
	index_image_to_save += 1
