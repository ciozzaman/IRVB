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


fileshot=np.array([files16[2:],files14[5:]])
temperaturehot=np.array([temperature16[2:],temperature14[5:]])
filescold=np.array([files18[4:],files20[2:]])
temperaturecold=np.array([temperature18[4:],temperature20[2:]])
inttime=1	# ms
framerate=383	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)


fileshot=np.array([files15,files17])
temperaturehot=np.array([temperature15,temperature17])
filescold=np.array([files19,files21])
temperaturecold=np.array([temperature19,temperature21])
inttime=2	# ms
framerate=383	# Hz
n=3
pathparam='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
coleval.build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n,function_to_use = coleval.build_poly_coeff_multi_digitizer)
coleval.build_average_poly_coeff_multi_digitizer(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,n)

fileshot=np.array([files22])
temperaturehot=np.array([temperature22])
filescold=np.array([files23])
temperaturecold=np.array([temperature23])
inttime=1	# ms
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
