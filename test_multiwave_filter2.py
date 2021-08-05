def real_mean_filter_agent(datasection,freq_to_erase_frames):
	filter_agent = cp.deepcopy(datasection)
	padded_by = int(np.ceil(freq_to_erase_frames/2))+1
	temp = np.pad(filter_agent, ((padded_by,padded_by),(0,0),(0,0)), mode='reflect')
	for ii in range(-int(np.floor((freq_to_erase_frames-1)/2)),int(np.floor((freq_to_erase_frames-1)/2))+1):
		if ii!=0:
			filter_agent += temp[padded_by+ii:-padded_by+ii]
	filter_agent += temp[padded_by-int(np.floor((freq_to_erase_frames-1)/2))-1:-padded_by-int(np.floor((freq_to_erase_frames-1)/2))-1]*((freq_to_erase_frames-1)/2-int(np.floor((freq_to_erase_frames-1)/2)))
	filter_agent += temp[padded_by+int(np.floor((freq_to_erase_frames-1)/2))+1:-padded_by+int(np.floor((freq_to_erase_frames-1)/2))+1]*((freq_to_erase_frames-1)/2-int(np.floor((freq_to_erase_frames-1)/2)))
	filter_agent/=freq_to_erase_frames
	return filter_agent

def clear_oscillation_central5(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',min_frequency_to_erase=20,max_frequency_to_erase=34,plot_conparison=False,which_plot=[1,2,3],ROI='auto',window=2,force_poscentre='auto',output_noise=False,multiple_frequencies_cleaned=1):
	from scipy.signal import find_peaks, peak_prominences as get_proms

	# Created 04/08/2021
	# Function created starting from clear_oscillation_central4, the mean filter is applied before each frequency subtraction

	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')
	# figure_index = plt.gcf().number

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)*framerate):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin*framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)/framerate) or oscillation_search_window_end<=force_start/framerate):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)


	central_freq_for_search = (max_frequency_to_erase-min_frequency_to_erase)/2+min_frequency_to_erase
	if (framerate<2*central_freq_for_search):
		print('There is a problem. The framerate is too low to try to extract the oscillation')
		print('The minimum framrate for doing it is 2*oscillation frequency to detect. Therefore '+str(np.around(2*central_freq_for_search,decimals=1))+'Hz, in this case.')
		print('See http://www.skillbank.co.uk/SignalConversion/rate.htm')
		exit()

	#window = 2	# Previously found that as long as the fft is averaged over at least 4 pixels the peak shape and location does not change
	datasection = data

	if plot_conparison==True:
		# plt.figure()
		# plt.pause(0.01)
		plt.figure()
		figure_index = plt.gcf().number
		if 1 in which_plot:
			data_shape = np.shape(data)
			poscentred = [[int(data_shape[1]*1/5), int(data_shape[2]*1/5)], [int(data_shape[1]*1/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/5)], [int(data_shape[1]*4/5), int(data_shape[2]*1/2)], [int(data_shape[1]*4/5), int(data_shape[2]*4/5)], [int(data_shape[1]*1/2), int(data_shape[2]*1/2)]]

			# spectra_orig = np.fft.fft(data, axis=0)
			# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
			# phase = np.angle(spectra_orig)
			# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

			color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra_orig = np.fft.fft(np.mean(data[:, pos[0] - window:pos[0] + window+1, pos[1] - window:pos[1] + window+1],axis=(-1,-2)), axis=0)
				magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
				freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
				# y = np.mean(magnitude, axis=(-1, -2))
				y = magnitude
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y*100, color[i], label='original data at the point ' + str(pos) + ' x100')
			# plt.title()


			plt.title('Amplitued from fast Fourier transform in the whole time interval\nfor different groups of ' + str(window * 2+1) + 'x' + str(
				window * 2+1) + ' pixels, framerate %.3gHz' %(framerate) )
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Amplitude [au]')
			plt.grid()
			plt.semilogy()
			plt.legend(loc='best',fontsize='xx-small')
		else:
			plt.close(figure_index)
	# plt.show()



	frames_for_oscillation = framerate//central_freq_for_search

	number_of_waves = 3
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	if fft_window_move<10:
		number_of_waves=10//frames_for_oscillation	# I want to scan for at least 10 frame shifts
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	step = 1
	while int(fft_window_move/step)>80:
		step+=1		#if framerate is too high i will skip some of the shifts to limit the number of Fourier transforms to 100


	# I restrict the window over which I search for the oscillation
	datarestricted = data[force_start:force_end]#
	len_data_restricted = len(datarestricted)
	if force_poscentre == 'auto':
		poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	else:
		poscentre = force_poscentre

	if (oscillation_search_window_begin=='auto' and oscillation_search_window_end=='auto'):
		while fft_window_move>(len_data_restricted/5):
			fft_window_move-=1		# I want that the majority of the data I analyse remains the same

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			# max_start = int(sections // 2)
			max_start = int(len(datarestricted) // 2)
		else:
			max_start = int(5*framerate)  # I use 5 seconds of record
	else:
		max_start = len(datarestricted)	# this is actually ineffective, as datarestricted is already limited to force_start:force_end

	if oscillation_search_window_begin == 'auto':
		min_start = 0
	else:
		# min_start = force_start	# alreay enforced through force_start
		min_start = 0

	section_frames = max_start - min_start-fft_window_move

	if ROI=='auto':
		# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
		datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))


	spectra = np.fft.fft(datarestricted2, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	freq = np.fft.fftfreq(len(magnitude_space_averaged), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	peaks_1 = find_peaks(y)[0]
	peaks = peaks_1[np.logical_and(x[peaks_1]<max_frequency_to_erase+15,x[peaks_1]>max_frequency_to_erase)]
	if len(peaks)==0:
		peaks = np.array(peaks_1[np.logical_and(x[peaks_1]>min_frequency_to_erase-15,x[peaks_1]<min_frequency_to_erase)])
		fit = [0,0,np.mean(np.log(y[peaks]))]	# I do this because I want to avoid that the fit only on the left goes too loo and I filter too much
	else:
		peaks = np.concatenate([peaks_1[np.logical_and(x[peaks_1]>min_frequency_to_erase-15,x[peaks_1]<min_frequency_to_erase)] , peaks_1[np.logical_and(x[peaks_1]<max_frequency_to_erase+15,x[peaks_1]>max_frequency_to_erase)]])
		fit = np.polyfit(x[peaks],np.log(y[peaks]),2)
	# poly2deg = lambda freq,a2,a1,a0 : a0 + a1*freq + a2*(freq**2)
	# guess = [min(fit[0],0),min(fit[1],0),fit[2]]
	# bds = [[-np.inf,-np.inf,-np.inf],[0,0,np.inf]]
	# fit = curve_fit(poly2deg, x[peaks],np.log(y[peaks]), p0=guess, bounds=bds, maxfev=100000000)[0]
	y_reference = np.exp(np.polyval(fit,x[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]))
	if min_frequency_to_erase>35:	# this is in place mainly only for the filter around 90Hz
		min_of_fit = x[np.logical_and(x>min_frequency_to_erase-15,x<min_frequency_to_erase)]
		min_of_fit = np.exp(np.polyval(fit,min_of_fit)).min()
		y_reference[y_reference > min_of_fit] = min_of_fit	# this is to avoid that when the parabola is fitted only on the left the fit goes up in the area of interest. the reference is limited to the minimum value of the fit outside of it
	y_reference = (3*np.std(y[peaks]/np.exp(np.polyval(fit,x[peaks])))+1)*y_reference

	y_test = y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--r',label='original')
		plt.grid()
		plt.semilogy()
		plt.plot(x[peaks_1],y[peaks_1],'x')
		plt.axvline(x=min_frequency_to_erase,color='k',linestyle='--')
		plt.axvline(x=max_frequency_to_erase,color='k',linestyle='--')
		plt.plot(x[peaks],y[peaks],'o')
		plt.plot(x,np.exp(np.polyval(fit,x)))
		plt.plot(x[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)],y_reference,'--')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.xlim(left=min_frequency_to_erase-5,right=max_frequency_to_erase+5)
		plt.ylim(bottom=np.median(np.sort(y)[:int(len(y)/4)])*1e-1,top = np.max(y[peaks])*2)

	frequencies_removed_all = []
	first_pass = True
	data2 = cp.deepcopy(data)
	while np.sum((y_test-y_reference)>0)>0:

		record_magnitude = []
		record_phase = []
		record_freq = []
		peak_freq_record = []
		peak_value_record = []
		peak_index_record = []
		shift_record = []

		for i in range(int(fft_window_move/step)):
			shift=i*step
			datasection = datarestricted2[min_start:max_start-fft_window_move+shift]
			spectra = np.fft.fft(datasection, axis=0)
			magnitude = 2 * np.abs(spectra) / len(spectra)
			record_magnitude.append(magnitude[0:len(magnitude) // 2])
			phase = np.angle(spectra)
			record_phase.append(phase[0:len(magnitude) // 2])
			freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
			record_freq.append(freq[0:len(magnitude) // 2])
			# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
			y = np.array([value for _, value in sorted(zip(freq, magnitude))])
			x = np.sort(freq)

			index_min_freq = np.abs(x-min_frequency_to_erase).argmin()  # I restric the window over which I do the peak search
			index_max_freq = np.abs(x-max_frequency_to_erase).argmin()
			index_7 = np.abs(x-7).argmin()
			index_n7 = np.abs(x-(-7)).argmin()
			index_min_freq_n = np.abs(x-(-min_frequency_to_erase)).argmin()
			index_max_freq_n = np.abs(x-(-max_frequency_to_erase)).argmin()
			index_0 = np.abs(x-0).argmin()
			# noise = np.mean(np.array(
			# 	y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[
			# 																									index_max_freq:-3].tolist()),
			# 				axis=(-1))
			# temp = peakutils.indexes(y[index_min_freq:index_max_freq], thres=noise + np.abs(magnitude.min()),
			# 						 min_dist=(index_max_freq - index_min_freq) // 2)
			if len(y[index_min_freq:index_max_freq])==0:
				continue
			# if plot_conparison == True and (2 in which_plot):
			# 	plt.figure(figure_index+1)
			# 	plt.plot(x, y, label='Applied shift of ' + str(shift))
			temp = y[index_min_freq:index_max_freq].argmax()
			# if len(temp) == 1:
			peak_index = index_min_freq + int(temp)
			peak_freq_record.append(x[peak_index])
			peak_value = float(y[peak_index])
			peak_value_record.append(peak_value)
			peak_index_record.append(peak_index)
			shift_record.append(shift)
		record_magnitude = np.array(record_magnitude)
		record_phase = np.array(record_phase)
		record_freq = np.array(record_freq)
		peak_freq_record = np.array(peak_freq_record)
		peak_value_record = np.array(peak_value_record)
		peak_index_record = np.array(peak_index_record)
		shift_record = np.array(shift_record)
		# index = np.array(peak_value_record).argmax()	# this is wrong in the case that the baseline noise is strongly decreasing
		index = (np.array(peak_value_record)/np.exp(np.polyval(fit,peak_freq_record))).argmax()	# here I look at the strongest deviation from the noise baseline
		shift = index * step
		freq_to_erase = peak_freq_record[index]
		datasection = datarestricted[min_start:max_start-fft_window_move+shift]
		print('filtering '+str([freq_to_erase,framerate/freq_to_erase,index]))
		filter_agent = real_mean_filter_agent(datasection,framerate/freq_to_erase)	# added to make sure to dynamically remove only the wanted frequency
		spectra = np.fft.fft(datasection-filter_agent, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude = 2 * np.abs(spectra) / len(spectra)
		phase = np.angle(spectra)
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		freq_to_erase_index = np.abs(freq-freq_to_erase).argmin()
		frequencies_removed_all.append(freq[freq_to_erase_index])
		freq_to_erase_index_multiple = np.arange(-(multiple_frequencies_cleaned-1)//2,(multiple_frequencies_cleaned-1)//2+1) + freq_to_erase_index
		print(freq_to_erase_index_multiple)

		if plot_conparison==True and (2 in which_plot):
			plt.figure(figure_index+1)
			# plt.plot([freq_to_erase]*2,[peak_value_record.min(),peak_value_record.max()],':k')
			# plt.plot([freq[freq_to_erase_index]]*2,[peak_value_record.min(),peak_value_record.max()],'--k')
			# plt.axvline(x=freq_to_erase,color='k',linestyle=':')
			plt.plot(freq[freq_to_erase_index],record_magnitude[index][np.abs(record_freq[index]-freq[freq_to_erase_index]).argmin()],'rx',markersize=10)
			if len(freq_to_erase_index_multiple)>1:
				for i_freq in freq_to_erase_index_multiple:
					if i_freq!=freq_to_erase_index:
						plt.plot(freq[i_freq],record_magnitude[index][np.abs(record_freq[index]-freq[i_freq]).argmin()],'yx',markersize=10)
			if first_pass:
				plt.ylim(top=peak_value_record.max()*2)
		if plot_conparison==True and (1 in which_plot):
			plt.figure(figure_index)
			plt.plot(freq[freq_to_erase_index],100*record_magnitude[index][np.abs(record_freq[index]-freq[freq_to_erase_index]).argmin()],'rx',markersize=10)
			if len(freq_to_erase_index_multiple)>1:
				for i_freq in freq_to_erase_index_multiple:
					if i_freq!=freq_to_erase_index:
						plt.plot(freq[i_freq],100*record_magnitude[index][np.abs(record_freq[index]-freq[i_freq]).argmin()],'yx',markersize=10)
		framenumber = np.linspace(0, len(data) - 1, len(data)) -force_start- min_start
		for i_freq in freq_to_erase_index_multiple:
			# print(i_freq)
			data2 -= np.multiply(magnitude[i_freq], np.cos(np.repeat(np.expand_dims(phase[i_freq], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq[i_freq] * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))
		# data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

		datarestricted = data2[force_start:force_end]#
		if ROI=='auto':
			# datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(1,2))
			datarestricted2 = np.mean(datarestricted[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
		else:
			horizontal_coord = np.arange(np.shape(datarestricted)[2])
			vertical_coord = np.arange(np.shape(datarestricted)[1])
			horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
			select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
			datarestricted2 = np.mean(datarestricted[:, select],axis=(-1))

		spectra = np.fft.fft(datarestricted2, axis=0)
		magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
		freq = np.fft.fftfreq(len(magnitude_space_averaged), d=1 / framerate)
		# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		y_test = y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)]
		first_pass = False

	frequencies_removed_all = np.round(np.array(frequencies_removed_all)*100)/100
	# added to visualize the goodness of the result
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.plot(x, y,'--k',label='subtracted')
		# plt.grid()
		plt.legend(loc='best',fontsize='xx-small')
		if ROI=='auto':
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged in a window of ' %(force_start/framerate,(force_start+max_start)/framerate)+ str([window * 2+1,window * 2+1]) + ' pixels around ' + str(poscentre) + ', framerate %.3gHz' %(framerate)+'\nremoved freq: '+str(frequencies_removed_all)+' Hz')
		else:
			plt.title('Amplitude from fast Fourier transform from %.3gs to %.3gs\naveraged ouside the ROI ' %(force_start/framerate,(force_start+max_start)/framerate)+ str(ROI) + ' pixels around ' + str(ROI) + ', framerate %.3gHz' %(framerate)+'\nremoved freq: '+str(frequencies_removed_all)+' Hz')


	# section only for stats
	# datasection = data[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	index_min_freq = np.abs(x-(min_frequency_to_erase)).argmin()	# I restric the window over which I do the peak search
	index_max_freq = np.abs(x-(max_frequency_to_erase)).argmin()
	index_7 = np.abs(x-(7)).argmin()
	index_n7 = np.abs(x-(-7)).argmin()
	index_min_freq_n = np.abs(x-(-min_frequency_to_erase)).argmin()
	index_max_freq_n = np.abs(x-(-max_frequency_to_erase)).argmin()
	index_0 = np.abs(x-(0)).argmin()
	noise = (np.array(y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[index_max_freq:-3].tolist()))
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_pre_filter = float(y[peak_index])

	# datasection = data2[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	if ROI=='auto':
		datasection = np.mean(data2[:, poscentre[0] -window :poscentre[0] + window+1, poscentre[1] - window:poscentre[1] + window+1],axis=(-1,-2))
	else:
		horizontal_coord = np.arange(np.shape(datarestricted)[2])
		vertical_coord = np.arange(np.shape(datarestricted)[1])
		horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
		select = np.logical_or(np.logical_or(vertical_coord<ROI[0],vertical_coord>ROI[1]),np.logical_or(horizontal_coord<ROI[2],horizontal_coord>ROI[3]))
		datasection = np.mean(data2[:, select],axis=(-1))
	spectra = np.fft.fft(datasection, axis=0)
	magnitude_space_averaged = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	# magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array([magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	# temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	# peak_index = index_min_freq + int(temp)
	peak_index = index_min_freq + y[index_min_freq:index_max_freq].argmax()
	peak_value_post_filter = float(y[peak_index])
	if plot_conparison==True and (2 in which_plot):
		plt.figure(figure_index+1)
		plt.axhline(y=np.max(noise),linestyle='--',color='k',label='max noise')
		plt.axhline(y=np.median(noise),linestyle='--',color='b',label='median noise')
		plt.axhline(y=peak_value_pre_filter,linestyle='--',color='r',label='peak pre')
		plt.axhline(y=peak_value_post_filter,linestyle='--',color='g',label='peak post')
		plt.legend(loc='best',fontsize='xx-small')

	if plot_conparison==True:
		if 1 in which_plot:
			plt.figure(figure_index)
			datasection2 = data2
			# spectra = np.fft.fft(datasection2, axis=0)
			# # magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
			# magnitude2 = 2 * np.abs(spectra) / len(spectra)
			# phase2 = np.angle(spectra)
			# freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
			for i in range(len(poscentred)):
				pos = poscentred[i]
				spectra = np.fft.fft(np.mean(datasection2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window],axis=(-1,-2)), axis=0)
				magnitude2 = 2 * np.abs(spectra) / len(spectra)
				freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
				# y = np.mean(magnitude2, axis=(-1, -2))
				y = magnitude2
				# y=magnitude[:,pos[0],pos[1]]
				y = np.array([y for _, y in sorted(zip(freq, y))])
				x = np.sort(freq)
				plt.plot(x, y, color[i] + '--',label='data at the point ' + str(pos) + ', oscillation substracted')
			# plt.title()


			# plt.grid()
			plt.semilogy()
			plt.xlim(left=0)
			plt.ylim(top=y[np.logical_and(x>min_frequency_to_erase,x<max_frequency_to_erase)].max()*2e3)
			plt.legend(loc='best',fontsize='xx-small')
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(np.around(section_frames/framerate,decimals=5))+'s of '+str(len(data)/framerate)+'s of record')
	try:
		print('found oscillation of frequency '+str(frequencies_removed_all)+'Hz')
	except:
		print('no frequency needed removing')
	print('On the ROI oscillation magnitude reduced from %.5g[au] to %.5g[au]' %(peak_value_pre_filter,peak_value_post_filter)+'\nwith an approximate maximum noise of %.5g[au] and median of %.5g[au]' %(np.max(noise),np.median(noise)))

	if output_noise:
		return np.array([data2]),peak_value_pre_filter,peak_value_post_filter,np.max(noise),np.median(noise)
	else:
		return np.array([data2])
