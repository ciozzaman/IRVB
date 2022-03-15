# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

# from multiprocessing import Pool,cpu_count
# number_cpu_available = 8	#cpu_count()
# print('Number of cores available: '+str(number_cpu_available))
# import mkl
# mkl.set_num_threads(number_cpu_available)

if False:
	fileshot=np.array([files16[2:],files14[5:]])
	temperaturehot=np.array([temperature16[2:],temperature14[5:]])
	filescold=np.array([files18[4:],files20[2:]])
	temperaturecold=np.array([temperature18[4:],temperature20[2:]])
	inttime=1.0	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
	coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
	coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)


	fileshot=np.array([files15,files17])
	temperaturehot=np.array([temperature15,temperature17])
	filescold=np.array([files19,files21])
	temperaturecold=np.array([temperature19,temperature21])
	inttime=2.0	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
	coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
	coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)

	fileshot=np.array([files22])
	temperaturehot=np.array([temperature22])
	filescold=np.array([files23])
	temperaturecold=np.array([temperature23])
	inttime=1.0	# ms
	framerate=994	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
	coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
	coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)

	fileshot=np.array([files24,files28])
	temperaturehot=np.array([temperature24,temperature28])
	filescold=np.array([files26,files30])
	temperaturecold=np.array([temperature26,temperature30])
	inttime=0.5	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
	coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
	coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)

	fileshot=np.array([files25,files29])
	temperaturehot=np.array([temperature25,temperature29])
	filescold=np.array([files27,files31])
	temperaturecold=np.array([temperature27,temperature31])
	inttime=1.5	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
	coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
	coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)

elif False:
	# added 2021/09/25 to check if there is something wrong in the way I calculate the fir, in particular the covariance matrix
	# IMPORTANT singificant error fixed in coleval.build_poly_coeff_multi_digitizer
	# the estimated error in the fit was 3 orders of magnitude higher than reality
	# additionally, calculating the properties in steps introduce massive errors when I average them. it's much better not do it!

	fileshot=np.array([files16[2:],files14[5:]])
	temperaturehot=np.array([temperature16[2:],temperature14[5:]])
	filescold=np.array([files18[4:],files20[2:]])
	temperaturecold=np.array([temperature18[4:],temperature20[2:]])
	temperature_window = temperaturehot.tolist()+temperaturecold.tolist()
	files_window = fileshot.tolist()+filescold.tolist()
	inttime=1.0	# ms
	framerate=383	# Hz
	fileshot=np.array([files2,files3,files4,files5])
	temperaturehot=np.array([temperature2,temperature3,temperature4,temperature5])
	filescold=np.array([files6])
	temperaturecold=np.array([temperature6])
	temperature_no_window = temperaturehot.tolist()+temperaturecold.tolist()
	files_no_window = fileshot.tolist()+filescold.tolist()
	n=3
	pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperature_window,files_window,inttime,pathparam,n)
	fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
	params_dict=np.load(fullpathparams)
	params_dict.allow_pickle=True
	params1 = params_dict['coeff2']
	errparams1 = params_dict['errcoeff2']
	score2 = params_dict['score2']
	score = params_dict['score']

	fileshot=np.array([files15,files17])
	temperaturehot=np.array([temperature15,temperature17])
	filescold=np.array([files19,files21])
	temperaturecold=np.array([temperature19,temperature21])
	inttime=2.0	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperaturehot.tolist()+temperaturecold.tolist(),fileshot.tolist()+filescold.tolist(),inttime,pathparam,n)

	fileshot=np.array([files22])
	temperaturehot=np.array([temperature22])
	filescold=np.array([files23])
	temperaturecold=np.array([temperature23])
	inttime=1.0	# ms
	framerate=994	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperaturehot.tolist()+temperaturecold.tolist(),fileshot.tolist()+filescold.tolist(),inttime,pathparam,n)

	fileshot=np.array([files24,files28])
	temperaturehot=np.array([temperature24,temperature28])
	filescold=np.array([files26,files30])
	temperaturecold=np.array([temperature26,temperature30])
	inttime=0.5	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperaturehot.tolist()+temperaturecold.tolist(),fileshot.tolist()+filescold.tolist(),inttime,pathparam,n)

	fileshot=np.array([files25,files29])
	temperaturehot=np.array([temperature25,temperature29])
	filescold=np.array([files27,files31])
	temperaturecold=np.array([temperature27,temperature31])
	inttime=1.5	# ms
	framerate=383	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperaturehot.tolist()+temperaturecold.tolist(),fileshot.tolist()+filescold.tolist(),inttime,pathparam,n)

elif False:
	# temperature calibration done 2021/12/07 with the black body source but after the disintegration of the coating on the lens

	files = files42
	temperature = temperature42
	temperature_window = cp.deepcopy(temperature)
	files_window = cp.deepcopy(files)
	inttime=1.0	# ms
	framerate=50	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-12-07_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperature,files,inttime,pathparam,n)
	fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
	params_dict=np.load(fullpathparams)
	params_dict.allow_pickle=True
	params1 = params_dict['coeff2']
	errparams1 = params_dict['errcoeff2']
	score2 = params_dict['score2']
	score = params_dict['score']

	files = files48
	temperature = temperature48
	temperature_no_window = cp.deepcopy(temperature)
	files_no_window = cp.deepcopy(files)
	inttime=1.0	# ms
	framerate=50	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2021-12-07_no_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperature,files,inttime,pathparam,n)
	fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
	params_dict=np.load(fullpathparams)
	params_dict.allow_pickle=True
	params2 = params_dict['coeff2']
	errparams2 = params_dict['errcoeff2']


	fileshot=np.array([files2,files3,files4,files5])
	temperaturehot=np.array([temperature2,temperature3,temperature4,temperature5])
	filescold=np.array([files6])
	temperaturecold=np.array([temperature6])
	inttime=1	# ms
	framerate=50	# Hz
	n=3
	pathparam='/home/ffederic/work/irvb/2018-03-07_no_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
	if not os.path.exists(pathparam):
		os.makedirs(pathparam)
	coleval.build_poly_coeff_multi_digitizer(temperaturehot.tolist()+temperaturecold.tolist(),fileshot.tolist()+filescold.tolist(),inttime,pathparam,n)
	fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
	params_dict=np.load(fullpathparams)
	params_dict.allow_pickle=True
	params2 = params_dict['coeff2']
	errparams2 = params_dict['errcoeff2']





	# without window ~25decC
	file = '/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000170'
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	# with window ~25decC
	file = '/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000083'
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer1,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)

	# without window ~36decC
	file = '/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000188'
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	# with window ~36decC
	file = '/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000134'
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer1,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)

	# without window with NUC
	file = files2[-1]
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	# with window with NUC
	file = files14[-1]
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer1,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)

	# without window with NUC
	file = files3[1]
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)
	# with window with NUC
	file = files14[4]
	full_saved_file_dict=coleval.read_IR_file(file)
	data_per_digitizer1,uniques_digitizer_ID = coleval.separate_data_with_digitizer(full_saved_file_dict)

elif True:
	# his is created to fit simultaneously the curve with and without window

	if False:	# BB source in focus and no window inttime=1.0	# ms
		description = 'BB source in focus and no window inttime=1.0 # ms'
		files = []
		temperature = []
		temperature_window = cp.deepcopy(temperature)
		files_window = cp.deepcopy(files)
		inttime=1.0	# ms
		framerate=50	# Hz
		n=3
		files = files66
		temperature = temperature66
		temperature_no_window = cp.deepcopy(temperature)
		files_no_window = cp.deepcopy(files)
		pathparam='/home/ffederic/work/irvb/2022-02-24_no_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

	if False:	# BB source in focus and no window inttime=2.0	# ms
		description = 'BB source in focus and no window inttime=2.0 # ms'
		files = []
		temperature = []
		temperature_window = cp.deepcopy(temperature)
		files_window = cp.deepcopy(files)
		inttime=2.0	# ms
		framerate=50	# Hz
		n=3
		files = files65
		temperature = temperature65
		temperature_no_window = cp.deepcopy(temperature)
		files_no_window = cp.deepcopy(files)
		pathparam='/home/ffederic/work/irvb/2022-02-24_no_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

	if False:	# BB source as far as possible with window inttime=1.0	# ms
		description = 'BB source as far as possible with window inttime=1.0 # ms'
		files = files58
		temperature = temperature58
		temperature_window = cp.deepcopy(temperature)
		files_window = cp.deepcopy(files)
		inttime=1.0	# ms
		framerate=50	# Hz
		n=3
		files = files63
		temperature = temperature63
		temperature_no_window = cp.deepcopy(temperature)
		files_no_window = cp.deepcopy(files)
		pathparam='/home/ffederic/work/irvb/2022-02-24_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

	if False:	# BB source as far as possible with window inttime=2.0	# ms
		description = 'BB source as far as possible with window inttime=2.0 # ms'
		files = files56
		temperature = temperature56
		temperature_window = cp.deepcopy(temperature)
		files_window = cp.deepcopy(files)
		inttime=2.0	# ms
		framerate=50	# Hz
		n=3
		files = files61
		temperature = temperature61
		temperature_no_window = cp.deepcopy(temperature)
		files_no_window = cp.deepcopy(files)
		pathparam='/home/ffederic/work/irvb/2022-02-24_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

	if False:	# BB source as close as possible with window inttime=1.0	# ms
		description = 'BB source as close as possible with window inttime=1.0 # ms'
		files = files42
		temperature = temperature42
		temperature_window = cp.deepcopy(temperature)
		files_window = cp.deepcopy(files)
		inttime=1.0	# ms
		framerate=50	# Hz
		n=3
		files = files48
		temperature = temperature48
		temperature_no_window = cp.deepcopy(temperature)
		files_no_window = cp.deepcopy(files)
		pathparam='/home/ffederic/work/irvb/2021-12-07_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

		if Flase:	# I think this is wrong. it was based to the fact that data fro 2018 has a3~1, but that dataset is unreliable
			# this will be saved with a3=1 to be used as proper calibration for MU01
			fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
			params_dict=np.load(fullpathparams)
			params_dict.allow_pickle=True
			params_dict['coeff2'][:,:,:,2]=1
			pathparam='/home/ffederic/work/irvb/2021-12-07_MU01_modified_BB_params/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
			if not os.path.exists(pathparam):
				os.makedirs(pathparam)
			fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
			np.savez_compressed(fullpathparams[:-4],**params_dict)


	if False:	# BB source as close as possible with window inttime=2.0	# ms
		description = 'BB source as close as possible with window inttime=2.0 # ms'
		files = files41
		temperature = temperature41
		temperature_window = cp.deepcopy(temperature)
		files_window = cp.deepcopy(files)
		inttime=2.0	# ms
		framerate=50	# Hz
		n=3
		files = files46
		temperature = temperature46
		temperature_no_window = cp.deepcopy(temperature)
		files_no_window = cp.deepcopy(files)
		pathparam='/home/ffederic/work/irvb/2021-12-07_window_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

		if Flase:	# I think this is wrong. it was based to the fact that data fro 2018 has a3~1, but that dataset is unreliable
			# this will be saved with a3=1 to be used as proper calibration for MU01
			fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
			params_dict=np.load(fullpathparams)
			params_dict.allow_pickle=True
			params_dict['coeff2'][:,:,:,2]=1
			pathparam='/home/ffederic/work/irvb/2021-12-07_MU01_modified_BB_params/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
			if not os.path.exists(pathparam):
				os.makedirs(pathparam)
			fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
			np.savez_compressed(fullpathparams[:-4],**params_dict)

	if False:	# NUC plate original scans with window inttime=1.0	# ms
		description = 'NUC plate original scans with window inttime=1.0 # ms'
		fileshot=np.array([files16[2:],files14[5:]])
		temperaturehot=np.array([temperature16[2:],temperature14[5:]])
		filescold=np.array([files18[4:],files20[2:]])
		temperaturecold=np.array([temperature18[4:],temperature20[2:]])
		temperature_window = temperaturehot.tolist()+temperaturecold.tolist()
		files_window = fileshot.tolist()+filescold.tolist()
		inttime=1.0	# ms
		framerate=383	# Hz
		fileshot=np.array([files2,files3,files4,files5])
		temperaturehot=np.array([temperature2,temperature3,temperature4,temperature5])
		filescold=np.array([files6])
		temperaturecold=np.array([temperature6])
		temperature_no_window = temperaturehot.tolist()+temperaturecold.tolist()
		files_no_window = fileshot.tolist()+filescold.tolist()
		n=3
		pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)

	if True:	# NUC plate original scans with window inttime=2.0	# ms
		description = 'NUC plate original scans with window inttime=2.0 # ms'
		fileshot=np.array([files17[2:],files15[5:]])
		temperaturehot=np.array([temperature17[2:],temperature15[5:]])
		filescold=np.array([files19[4:],files21[2:]])
		temperaturecold=np.array([temperature19[4:],temperature21[2:]])
		temperature_window = temperaturehot.tolist()+temperaturecold.tolist()
		files_window = fileshot.tolist()+filescold.tolist()
		inttime=2.0	# ms
		framerate=383	# Hz
		fileshot=np.array([files8])
		temperaturehot=np.array([temperature8])
		filescold=np.array([files7])
		temperaturecold=np.array([temperature7])
		temperature_no_window = temperaturehot.tolist()+temperaturecold.tolist()
		files_no_window = fileshot.tolist()+filescold.tolist()
		n=3
		pathparam='/home/ffederic/work/irvb/2021-09-25_multiple_search_for_parameters/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(n)
		if not os.path.exists(pathparam):
			os.makedirs(pathparam)
		coleval.build_poly_coeff_multi_digitizer_with_no_window_reference(temperature_window,files_window,temperature_no_window,files_no_window,inttime,pathparam,n,wavewlength_top=5,wavelength_bottom=2.5)



	exit()
	if False:	# only to compare results
		fullpathparams=os.path.join(pathparam,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(inttime)+'ms.npz')
		params_dict=np.load(fullpathparams)
		params_dict.allow_pickle=True
		params = params_dict['coeff']
		errparams = params_dict['errcoeff']
		score = params_dict['score']
		params2 = params_dict['coeff2']
		errparams2 = params_dict['errcoeff2']
		score2 = params_dict['score2']

		params3 = params_dict['coeff3']	# pure no window
		errparams3 = params_dict['errcoeff3']
		score3 = params_dict['score3']
		params4 = params_dict['coeff4']	# pure window
		errparams4 = params_dict['errcoeff4']
		score4 = params_dict['score4']

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('score polynomial fit NUC\n'+description)
		else:
			plt.title('score polynomial fit BB source\n'+description)
		plt.imshow(score[0])
		plt.colorbar()
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('score BB curve fit NUC\n'+description)
		else:
			plt.title('score BB curve fit BB source\n'+description)
		plt.imshow(score2[0])
		plt.colorbar()
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('proportional window component BB curve fit NUC\n'+description)
		else:
			plt.title('proportional window component BB curve fit BB source\n'+description)
		to_plot = median_filter(params2[0,:,:,0]*params2[0,:,:,2],[3,3])
		plt.imshow(to_plot,vmin=to_plot[30:170,30:170].min(),vmax=to_plot[30:170,30:170].max())
		plt.colorbar()
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('proportional no window component BB curve fit NUC\n'+description)
		else:
			plt.title('proportional no window component BB curve fit BB source\n'+description)
		to_plot = median_filter(params2[0,:,:,0],[3,3])
		plt.imshow(to_plot)#,vmin=to_plot[30:170,30:170].min(),vmax=to_plot[30:170,30:170].max())
		plt.colorbar().set_label('a1 [counts/photons]')
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('proportional window modifier BB curve fit NUC\n'+description)
		else:
			plt.title('proportional window modifier BB curve fit BB source\n'+description)
		to_plot = median_filter(params2[0,:,:,2],[3,3])
		plt.imshow(to_plot)#,vmin=to_plot[30:170,30:170].min(),vmax=to_plot[30:170,30:170].max())
		plt.colorbar().set_label('a3 [au]')
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('offset no window component BB curve fit NUC\n'+description)
		else:
			plt.title('offset no window component BB curve fit BB source\n'+description)
		plt.imshow(median_filter(params2[0,:,:,1],[3,3]))
		plt.colorbar()
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('offset window modifier BB curve fit NUC\n'+description)
		else:
			plt.title('offset window modifier BB curve fit BB source\n'+description)
		plt.imshow(median_filter(params2[0,:,:,3],[3,3]))
		plt.colorbar()
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('SNR additive factor window component BB curve fit NUC\n'+description)
		else:
			plt.title('SNR additive factor window component BB curve fit BB source\n'+description)
		to_plot = median_filter((params2[0,:,:,1]+params2[0,:,:,3])/((errparams2[0,:,:,1,1]+errparams2[0,:,:,3,3]+2*errparams2[0,:,:,3,1])**0.5),[3,3])
		plt.imshow(to_plot)
		plt.colorbar().set_label('(a2+a4)/sigma(a2+a4) [au]')
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('SNR proportional factor window component BB curve fit NUC\n'+description)
		else:
			plt.title('SNR proportional factor window component BB curve fit BB source\n'+description)
		to_plot = median_filter(1/((errparams2[0,:,:,0,0]**0.5/params2[0,:,:,0])**2 + (errparams2[0,:,:,2,2]**0.5/params2[0,:,:,2])**2 + 2*errparams2[0,:,:,2,0]/(params2[0,:,:,0]*params2[0,:,:,2]))**0.5 ,[3,3])
		plt.imshow(to_plot)
		plt.colorbar().set_label('(a1xa3)/sigma(a1xa3) [au]')
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('proportional pure no window component BB curve fit NUC\n'+description)
		else:
			plt.title('proportional pure no window component BB curve fit BB source\n'+description)
		to_plot = median_filter(params3[0,:,:,0],[3,3])
		plt.imshow(to_plot,vmin=to_plot[30:170,30:170].min(),vmax=to_plot[30:170,30:170].max())
		plt.colorbar()
		plt.pause(0.01)

		plt.figure(figsize=(10, 10))
		if len(temperature_window)<6:
			plt.title('proportional pure window component BB curve fit NUC\n'+description)
		else:
			plt.title('proportional pure window component BB curve fit BB source\n'+description)
		to_plot = median_filter(params4[0,:,:,0],[3,3])
		plt.imshow(to_plot,vmin=to_plot[30:170,30:170].min(),vmax=to_plot[30:170,30:170].max())
		plt.colorbar()
		plt.pause(0.01)
	else:
		pass

	if False:	# only to compare the results from the 2 methods

		wavewlength_top=5
		wavelength_bottom=2.5
		lambda_cam_x = np.linspace(wavelength_bottom,wavewlength_top,10)*1e-6	# m, Range of FLIR SC7500

		def BB_rad_counts_to_delta_temp(trash,T_,lambda_cam_x=lambda_cam_x):
			# lambda_cam_x = np.linspace(1.5,5.1,10)*1e-6	# m, Range of FLIR SC7500
			temp1 = np.trapz(2*scipy.constants.c/(lambda_cam_x**4) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x*scipy.constants.k*T_)) -1) ,x=lambda_cam_x) * inttime/1000
			# temp1 = 2*scipy.constants.h*(scipy.constants.c**2)/(lambda_cam_x.max()**5) * 1/( np.exp(scipy.constants.h*scipy.constants.c/(lambda_cam_x.max()*scipy.constants.k*T_)) -1)
			return temp1

		delta_counts = 100
		base_counts = 11000
		counts = base_counts+delta_counts
		BB_proportional,BB_proportional_std,constant_offset,constant_offset_std,photon_dict = coleval.calc_BB_coefficients_multi_digitizer(params,errparams,digitizer_ID,reference_background,reference_background_std,ref_temperature=ref_temperature,ref_temperature_std=ref_temperature_std,wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)
		photon_dict = coleval.calc_interpolators_BB(wavewlength_top=wavewlength_top,wavelength_bottom=wavelength_bottom,inttime=inttime)
		reverse_photon_flux_interpolator = photon_dict['reverse_photon_flux_interpolator']
		photon_flux_over_temperature_interpolator = photon_dict['photon_flux_over_temperature_interpolator']
		photon_flux_interpolator = photon_dict['photon_flux_interpolator']

		T0 = params[0,100,100,-1] + params[0,100,100,-2] * base_counts + params[0,100,100,-3] * (base_counts**2) + 273.15
		a1a3 = params2[0,100,100,0]*params2[0,100,100,2]
		sigma_a1a3 = a1a3 * ((errparams2[0,100,100,0,0]**0.5/params2[0,100,100,0])**2 + (errparams2[0,100,100,2,2]**0.5/params2[0,100,100,2])**2 + 2*errparams2[0,100,100,0,2]/a1a3)**0.5
		ref = delta_counts/a1a3 + BB_rad_counts_to_delta_temp(1,T0)
		sigma_ref = delta_counts/a1a3*(( (coleval.estimate_counts_std(counts,int_time=inttime)*2/delta_counts)**2 + (sigma_a1a3/(a1a3**2))**2 )**0.5)
		# sigma_ref = (sigma_ref**2 + ((sigma_T_multimpier(300)*0.1)**2))**0.5	# I'm not sure if I should consider this
		check = curve_fit(BB_rad_counts_to_delta_temp,1,ref,sigma=[sigma_ref],absolute_sigma=True,p0=[T0])
		temperature_BB = [check[0][0],check[1].flatten()[0]**0.5]
		print('temperature_BB '+str(temperature_BB))
		temperature = params[0,100,100,-1] + params[0,100,100,-2] * counts + params[0,100,100,-3] * (counts**2) + 273.15
		counts_std = coleval.estimate_counts_std(counts,int_time=inttime)
		temperature_std = (errparams[0,100,100,2,2] + (counts_std**2)*(params[0,100,100,1]**2) + (counts**2+counts_std**2)*errparams[0,100,100,1,1] + (counts_std**2)*(4*counts**2+3*counts_std**2)*(params[0,100,100,0]**2) + (counts**4+6*(counts**2)*(counts_std**2)+3*counts_std**4)*errparams[0,100,100,0,0] + 2*counts*errparams[0,100,100,2,1] + 2*(counts**2+counts_std**2)*errparams[0,100,100,2,0] + 2*(counts**3+counts*(counts_std**2))*errparams[0,100,100,1,0])**0.5
		temperature_POLY = [temperature,temperature_std]
		print('temperature_POLY '+str(temperature_POLY))

		T_array = np.arange(0,40)
		T_array = T_array+273.15
		flux_array = []
		for T in T_array:
			flux_array.append(BB_rad_counts_to_delta_temp(1,T))
		flux_array = np.array(flux_array)



#
