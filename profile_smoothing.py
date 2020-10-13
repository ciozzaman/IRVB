def smoothing_function(a,window_size_first_pass,window_size_second_pass='auto',extra_window='auto',window_size_decrements='auto',prominence_treshold='auto',make_plots=False):

	import scipy.signal
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.signal import find_peaks, peak_prominences as get_proms
	import copy as cp

	# window_size_second_pass = 'auto'; extra_window = 'auto'; window_size_decrements = 'auto'; prominence_treshold = 'auto'; make_plots = True
	# a=binned_data_no_mask[-1]
	# window_size_first_pass = 250
	if window_size_second_pass=='auto':
		window_size_second_pass = 2*int(window_size_first_pass/8*2)
	if extra_window=='auto':
		extra_window = int(window_size_first_pass/5)
	if window_size_decrements=='auto':
		window_size_decrements = int(window_size_first_pass/15)
	# final_polishing_smoothing_window = 2*int(window_size_decrements*1.3/2)
	final_polishing_smoothing_window = 2*int(window_size_decrements/3)
	averaging_to_fing_peaks = int(10*final_polishing_smoothing_window)


	window_size_original = cp.deepcopy(window_size_first_pass)
	yhat = scipy.signal.savgol_filter(a, window_size_original*2+1, 2)
	if prominence_treshold=='auto':
		prominence_treshold = 4 * np.median(np.abs(yhat - a))
		# prominence_treshold = 0.5*np.median(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'))


	print('window_size_second_pass')
	print(window_size_second_pass)
	print('extra_window')
	print(extra_window)
	print('window_size_decrements')
	print(window_size_decrements)
	print('final_polishing_smoothing_window')
	print(final_polishing_smoothing_window)
	print('prominence_treshold')
	print(prominence_treshold)



	# test = scipy.signal.savgol_filter(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks) / averaging_to_fing_peaks, mode='same'),1 + 2 * int(averaging_to_fing_peaks / 2), 2)
	if make_plots==True:
		plt.figure()
		plt.plot(a)
		plt.plot(yhat)
		plt.plot(np.abs(yhat - a))
		# test = scipy.signal.savgol_filter(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'),1+2*int(averaging_to_fing_peaks/2),2)
		# plt.plot(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'))
		# plt.plot(test)
		plt.pause(0.01)

	# while np.std(yhat-a)>np.median(np.convolve(np.abs(yhat - a), np.ones(final_polishing_smoothing_window)/final_polishing_smoothing_window , mode='same')):
	peaks = find_peaks(np.abs(yhat - a),distance=averaging_to_fing_peaks)[0]
	proms = get_proms(np.abs(yhat - a), peaks)[0]
	# peaks = find_peaks(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'),distance=averaging_to_fing_peaks)[0]
	# proms = get_proms(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'), peaks)[0]
	# peaks = find_peaks(test,distance=averaging_to_fing_peaks)[0]
	# proms = get_proms(test, peaks)[0]

	# peaks = np.array([to_be_sorted for used_to_sort, to_be_sorted in sorted(zip(proms, peaks))])
	# peaks = np.flip(peaks,axis=0)
	# proms = np.sort(proms)
	# proms = np.flip(proms, axis=0)
	all_peaks_to_be_fixed = peaks[proms>prominence_treshold]	#0.5 arbitrary treshold
	print('all_peaks_to_be_fixed')
	print(all_peaks_to_be_fixed)
	if len(all_peaks_to_be_fixed)==0:
		return(yhat)
	else:
		# all_proms_to_be_fixed = proms[proms>0.5]	#0.5 arbitrary treshold
		all_peaks_to_be_fixed = np.sort(all_peaks_to_be_fixed)
		all_start_window = np.max([all_peaks_to_be_fixed - window_size_original-extra_window, np.zeros_like(all_peaks_to_be_fixed)],axis=0)
		all_end_window = np.min([all_peaks_to_be_fixed + window_size_original+extra_window, len(yhat)*np.ones_like(all_peaks_to_be_fixed)],axis=0)
		for i in range(len(all_start_window)-1):
			if all_start_window[i+1]<all_end_window[i]:
				median = int((all_peaks_to_be_fixed[i]+all_peaks_to_be_fixed[i+1])/2)
				all_start_window[i + 1] = median+1
				all_end_window[i] = median
		for j,peak_to_fix in enumerate(all_peaks_to_be_fixed):
			# peak_to_fix = all_peaks_to_be_fixed[0]
			window_size = cp.deepcopy(window_size_original)
			if make_plots == True:
				# plt.plot(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'))
				plt.plot([peak_to_fix,peak_to_fix],[np.min(yhat),np.max(yhat)],'k--')
				plt.pause(0.01)
			window_for_peak_prominence = np.linspace(peak_to_fix - int(averaging_to_fing_peaks / 5), peak_to_fix + int(averaging_to_fing_peaks / 5),int(averaging_to_fing_peaks / 5) * 2 + 1).astype('int')
			window_for_peak_prominence = window_for_peak_prominence[window_for_peak_prominence>1]
			window_for_peak_prominence = window_for_peak_prominence[window_for_peak_prominence<len(yhat)-1]

			# 22/06/2020 not sure if this is needed, added to avoid error in get_proms(np.abs(yhat - a), window_for_peak_prominence)
			window_for_peak_prominence = window_for_peak_prominence[np.logical_and(np.abs(yhat - a)[window_for_peak_prominence-1]<np.abs(yhat - a)[window_for_peak_prominence],np.abs(yhat - a)[window_for_peak_prominence+1]<np.abs(yhat - a)[window_for_peak_prominence])]

			proms_test = np.max(get_proms(np.abs(yhat - a), window_for_peak_prominence)[0])
			# proms_test = get_proms(scipy.signal.savgol_filter(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks) / averaging_to_fing_peaks, mode='same'),1 + 2 * int(averaging_to_fing_peaks / 2), 2),[peak_to_fix])[0]
			while proms_test>prominence_treshold/10 and window_size>2*window_size_decrements:	#from np.mean(np.sort(proms))
				start_window = max(peak_to_fix-window_size-extra_window,all_start_window[j])
				end_window = min(peak_to_fix+window_size+1+extra_window,all_end_window[j])
				window_size -= window_size_decrements
				# yhat[start_window:end_window] = scipy.signal.savgol_filter(yhat[:max(start_window+window_size_decrements,0)].tolist()+a[max(start_window+window_size_decrements,0):min(end_window-window_size_decrements,len(yhat))].tolist()+yhat[min(end_window-window_size_decrements,len(yhat)):].tolist(), window_size*2+1, 2)[start_window:end_window]
				yhat[start_window:end_window] = scipy.signal.savgol_filter(a, window_size * 2 + 1, 2)[start_window:end_window]
				if make_plots == True:
					plt.plot(yhat)
					# plt.plot(scipy.signal.savgol_filter(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks) / averaging_to_fing_peaks, mode='same'),1 + 2 * int(averaging_to_fing_peaks / 2), 2))
					plt.plot(np.abs(yhat - a))
					plt.plot([start_window, start_window], [np.min(yhat), np.max(yhat)], 'g--')
					plt.plot([end_window, end_window], [np.min(yhat), np.max(yhat)], 'g--')
					plt.pause(0.01)

				# 22/06/2020 not sure if this is needed, added to avoid error in get_proms(np.abs(yhat - a), window_for_peak_prominence)
				window_for_peak_prominence = window_for_peak_prominence[np.logical_and(np.abs(yhat - a)[window_for_peak_prominence-1]<np.abs(yhat - a)[window_for_peak_prominence],np.abs(yhat - a)[window_for_peak_prominence+1]<np.abs(yhat - a)[window_for_peak_prominence])]

				proms_test = np.max(get_proms(np.abs(yhat - a), window_for_peak_prominence)[0])
				# proms_test = get_proms(scipy.signal.savgol_filter(np.convolve(np.abs(yhat - a), np.ones(averaging_to_fing_peaks) / averaging_to_fing_peaks, mode='same'),1 + 2 * int(averaging_to_fing_peaks / 2), 2), [peak_to_fix])[0]
				print('peak')
				print(peak_to_fix)
				print('proms_test>prominence_treshold/10')
				print(proms_test)
				print(prominence_treshold/10)
				print('window_size')
				print(window_size)

		yhat2 = scipy.signal.savgol_filter(yhat, final_polishing_smoothing_window+1, 2)

		if make_plots == True:
			plt.figure()
			plt.plot(a)
			plt.plot(yhat)
			plt.plot(yhat2)
			plt.plot(np.abs(yhat2 - a))
			plt.pause(0.01)


		window_size_original = cp.deepcopy(window_size_second_pass)
		# peaks = find_peaks(np.convolve(np.abs(yhat2 - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'),distance=window_size_original/2)[0]
		# proms = get_proms(np.convolve(np.abs(yhat2 - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'), peaks)[0]
		peaks = find_peaks(np.abs(yhat2 - a), distance=averaging_to_fing_peaks)[0]
		proms = get_proms(np.abs(yhat2 - a), peaks)[0]
		# peaks = np.array([to_be_sorted for used_to_sort, to_be_sorted in sorted(zip(proms, peaks))])
		# peaks = np.flip(peaks,axis=0)
		# proms = np.sort(proms)
		# proms = np.flip(proms, axis=0)
		if np.max(proms)>prominence_treshold:
			# all_peaks_to_be_fixed = [peaks[0]]
			temp = scipy.signal.savgol_filter(a, window_size_original*2+1, 3)
			# peaks = find_peaks(np.convolve(np.abs(temp - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'), distance=window_size_original / 2)[0]
			# proms = get_proms(np.convolve(np.abs(temp - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'), peaks)[0]
			peaks = find_peaks(np.abs(temp - a), distance=averaging_to_fing_peaks)[0]
			proms = get_proms(np.abs(temp - a), peaks)[0]
			# peaks = np.array([to_be_sorted for used_to_sort, to_be_sorted in sorted(zip(proms, peaks))])
			# peaks = np.flip(peaks, axis=0)
			# proms = np.sort(proms)
			# proms = np.flip(proms, axis=0)
			# peaks = peaks[np.abs(peaks-all_peaks_to_be_fixed[0])<window_size_original]
			all_peaks_to_be_fixed = peaks[proms>prominence_treshold*2] # [peaks[0]]
			all_peaks_to_be_fixed = np.sort(all_peaks_to_be_fixed)
			all_start_window = np.max([all_peaks_to_be_fixed - window_size_original-extra_window, np.zeros_like(all_peaks_to_be_fixed)],axis=0)
			all_end_window = np.min([all_peaks_to_be_fixed + window_size_original+extra_window, len(yhat2)*np.ones_like(all_peaks_to_be_fixed)],axis=0)
			for i in range(len(all_start_window)-1):
				if all_start_window[i+1]<all_end_window[i]:
					median = int((all_peaks_to_be_fixed[i]+all_peaks_to_be_fixed[i+1])/2)
					all_start_window[i + 1] = median+1
					all_end_window[i] = median
			for j,peak_to_fix in enumerate(all_peaks_to_be_fixed):
				# peak_to_fix = all_peaks_to_be_fixed[0]
				window_size = cp.deepcopy(window_size_original)
				if make_plots == True:
					# plt.plot(np.abs(yhat2-a))
					plt.plot([peak_to_fix,peak_to_fix],[np.min(yhat2),np.max(yhat2)],'k--')
					plt.pause(0.01)
				amount_of_error = np.inf
				initial_amount_of_error = 0
				window_for_peak_prominence = np.linspace(peak_to_fix - int(averaging_to_fing_peaks / 5),peak_to_fix + int(averaging_to_fing_peaks / 5),int(averaging_to_fing_peaks / 5) * 2 + 1).astype('int')
				window_for_peak_prominence = window_for_peak_prominence[window_for_peak_prominence > 1]
				window_for_peak_prominence = window_for_peak_prominence[window_for_peak_prominence < len(yhat)-1]

				# 22/06/2020 not sure if this is needed, added to avoid error in get_proms(np.abs(temp - a), window_for_peak_prominence)
				window_for_peak_prominence = window_for_peak_prominence[np.logical_and(np.abs(temp - a)[window_for_peak_prominence-1]<np.abs(temp - a)[window_for_peak_prominence],np.abs(temp - a)[window_for_peak_prominence+1]<np.abs(temp - a)[window_for_peak_prominence])]

				proms_test = np.max(get_proms(np.abs(temp - a), window_for_peak_prominence)[0])
				# proms_test = get_proms(np.convolve(np.abs(temp - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'), [peak_to_fix])[0]
				while (proms_test>prominence_treshold/10 or amount_of_error>2*initial_amount_of_error) and window_size>window_size_decrements:	#from np.mean(np.sort(proms))
					start_window = max(peak_to_fix-window_size-extra_window,all_start_window[j])
					end_window = min(peak_to_fix+window_size+1+extra_window,all_end_window[j])
					window_size -= window_size_decrements
					yhat2[start_window:end_window] = scipy.signal.savgol_filter(yhat2[:max(start_window+final_polishing_smoothing_window,0)].tolist()+a[max(start_window+final_polishing_smoothing_window,0):min(end_window-final_polishing_smoothing_window,len(yhat))].tolist()+yhat2[min(end_window-final_polishing_smoothing_window,len(yhat)):].tolist(), window_size*2+1, 3)[start_window:end_window]
					if make_plots == True:
						plt.plot(yhat2)
						# plt.plot(np.convolve(np.abs(yhat2 - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'))
						plt.plot(np.abs(yhat2 - a))
						plt.plot([start_window, start_window], [np.min(yhat2), np.max(yhat2)], 'g--')
						plt.plot([end_window, end_window], [np.min(yhat2), np.max(yhat2)], 'g--')
						plt.pause(0.01)
					amount_of_error = np.sum(np.abs(yhat2 - a)[start_window:end_window])
					initial_amount_of_error = np.sum(np.convolve(np.abs(yhat - a), np.ones(final_polishing_smoothing_window)/final_polishing_smoothing_window , mode='same')[start_window:end_window])

					# 22/06/2020 not sure if this is needed, added to avoid error in get_proms(np.abs(temp - a), window_for_peak_prominence)
					window_for_peak_prominence = window_for_peak_prominence[np.logical_and(np.abs(yhat2 - a)[window_for_peak_prominence-1]<np.abs(yhat2 - a)[window_for_peak_prominence],np.abs(yhat2 - a)[window_for_peak_prominence+1]<np.abs(yhat2 - a)[window_for_peak_prominence])]

					proms_test = np.max(get_proms(np.abs(yhat2 - a), window_for_peak_prominence)[0])
					# proms_test = get_proms(np.convolve(np.abs(yhat2 - a), np.ones(averaging_to_fing_peaks)/averaging_to_fing_peaks , mode='same'), [peak_to_fix])[0]
			yhat3 = scipy.signal.savgol_filter(yhat2, final_polishing_smoothing_window+1, 2)
			print('yes second pass')
		else:
			print('no second pass')
			print('np.max(proms)>prominence_treshold')
			print(np.max(proms))
			print(prominence_treshold)
			yhat3 = cp.deepcopy(yhat2)

		if make_plots == True:
			plt.figure()
			plt.plot(a)
			plt.plot(yhat)
			plt.plot(yhat2)
			plt.plot(yhat3,'--')
			plt.plot(np.abs(yhat3 - a))
			plt.pause(0.01)

		return yhat3
