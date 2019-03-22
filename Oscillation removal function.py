import numpy as np
from scipy.optimize import curve_fit
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import statistics as s
import csv
import os,sys
from astropy.io import fits
import matplotlib.animation as animation
import pandas
import scipy.stats
import peakutils



####################################################################################################

def find_nearest_index(array,value):

	# 14/08/2018 This function returns the index of the closer value to "value" inside an array

	array_shape=np.shape(array)

	index = np.abs(np.add(array,-value)).argmin()
	residual_index=index
	cycle=1
	done=0
	position_min=np.zeros(len(array_shape),dtype=int)
	while done!=1:
		length=array_shape[-cycle]
		if residual_index<length:
			position_min[-cycle]=residual_index
			done=1
		else:
			position_min[-cycle]=round(((residual_index/length) %1) *length +0.000000000000001)
			residual_index=residual_index//length
			cycle+=1

	return position_min

####################################################################################################

def clear_oscillation_central(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',plot_conparison=False):

	# Created 01/11/2018
	# This function take the raw counts. analyse the fast fourier transform in a selectable interval IN THE CENTER OF THE FRAME.
	# Then search for the peak between 20Hz and 34Hz  and substract the oscillation found to the counts

	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin/framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)*framerate) or oscillation_search_window_end<=(force_start*framerate)):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)

	window = 10
	datasection = data

	if plot_conparison==True:
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]

		spectra_orig = np.fft.fft(data, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		phase = np.angle(spectra_orig)
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

		color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
		for i in range(len(poscentred)):
			pos = poscentred[i]
			y = np.mean(magnitude[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i], label='original data at the point ' + str(pos))
		# plt.title()

		plt.figure(1)
		plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
			window * 2) + ' pixels, framerate ' + str(framerate)+'Hz' )
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()
	# plt.show()


	sections = 31  # number found with practice, no specific mathematical reasons
	max_time = 5  # seconds of record that I can use to filter the signal. I assume to start from zero
	poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	record_magnitude = []
	record_phase = []
	record_freq = []
	peak_freq_record = []
	peak_value_record = []
	section_frames_record = []

	# I restrict the window over which I search for the oscillation
	datarestricted=data[force_start:force_end]

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			max_start = int(sections // 2)
		else:

			max_start = min(int(1 + sections / 2), int(
				max_time * framerate / (len(datarestricted) / sections)))  # I can use only a part of the record to filter the signal
	else:
		max_start = int(oscillation_search_window_end * framerate / (len(datarestricted) / sections))

	if (len(datarestricted) / framerate) <= 1:
		min_start = max(1, int(0.2 * framerate / (len(datarestricted) / sections)))


	else:
		extra = 0
		while ((max_start - int(max_start / (5 / 2.5)) + extra) < 7):	# 7 is just a try. this is in a way to have enough fitting to compare
			extra+=1
		min_start = max(1,int(max_start / (5 / 2.5)) - extra)
		# min_start = max(1, int(max_start / (5 / 2.5)) )  # with too little intervals it can interpret noise for signal

	for i in range(min_start, max_start):
		section_frames = (i) * (len(datarestricted) // sections)
		section_frames_record.append(section_frames)
		datasection = datarestricted[0:section_frames, poscentre[0] - window:poscentre[0] + window, poscentre[1] - window:poscentre[1] + window]
		spectra = np.fft.fft(datasection, axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		record_magnitude.append(magnitude[0:len(magnitude) // 2])
		phase = np.angle(spectra)
		record_phase.append(phase[0:len(magnitude) // 2])
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		record_freq.append(freq[0:len(magnitude) // 2])
		magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array(
			[magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		if plot_conparison == True:
			plt.figure(2)
			plt.plot(x, y, label='size of the analysed window ' + str(section_frames / framerate))
		index_20 = int(find_nearest_index(x, 20))  # I restric the window over which I do the peak search
		index_34 = int(find_nearest_index(x, 34))
		index_7 = int(find_nearest_index(x, 7))
		index_n7 = int(find_nearest_index(x, -7))
		index_n20 = int(find_nearest_index(x, -20))
		index_n34 = int(find_nearest_index(x, -34))
		index_0 = int(find_nearest_index(x, 0))
		noise = np.mean(np.array(
			y[3:index_n34].tolist() + y[index_n20:index_n7].tolist() + y[index_7:index_20].tolist() + y[
																									  index_34:-3].tolist()),
						axis=(-1))
		temp = peakutils.indexes(y[index_20:index_34], thres=noise + np.abs(magnitude.min()),
								 min_dist=(index_34 - index_20) // 2)
		if len(temp) == 1:
			peak_index = index_20 + int(temp)
			peak_freq_record.append(x[peak_index])
			peak_value = float(y[peak_index])
			peak_value_record.append(peak_value)
	record_magnitude = np.array(record_magnitude)
	record_phase = np.array(record_phase)
	record_freq = np.array(record_freq)
	peak_freq_record = np.array(peak_freq_record)
	peak_value_record = np.array(peak_value_record)
	section_frames_record = np.array(section_frames_record)
	if plot_conparison==True:
		plt.figure(2)
		plt.title('Amplitued from fast Fourier transform averaged in a wondow of ' + str(window) + 'pixels around ' + str(
			poscentre) + ', framerate ' + str(framerate) + 'Hz')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()


	# I find the highest peak and that will be the one I use
	index = int(find_nearest_index(peak_value_record, max(peak_value_record)+1))
	section_frames = section_frames_record[index]
	datasection = datarestricted[0:section_frames]
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	freq_to_erase = peak_freq_record[index]
	freq_to_erase_index = int(find_nearest_index(freq, freq_to_erase))
	framenumber = np.linspace(0, len(data) - 1, len(data)) - force_start
	data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

	if plot_conparison==True:
		plt.figure(1)
		datasection2 = data2
		spectra = np.fft.fft(datasection2, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude2 = 2 * np.abs(spectra) / len(spectra)
		phase2 = np.angle(spectra)
		freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
		for i in range(len(poscentred)):
			pos = poscentred[i]
			y = np.mean(magnitude2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i] + '--',
					 label='data at the point ' + str(pos) + ', ' + str(freq_to_erase) + 'Hz oscillation substracted')
		# plt.title()


		plt.grid()
		plt.semilogy()
		plt.legend()
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(section_frames_record[index]/framerate)+'s of '+str(len(data)/framerate)+'s of record')
	print('found oscillation of frequency '+str(freq_to_erase)+'Hz')

	return [data2]
