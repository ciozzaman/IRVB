# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

from multiprocessing import Pool,cpu_count
number_cpu_available = 8	#cpu_count()
print('Number of cores available: '+str(number_cpu_available))
import mkl
mkl.set_num_threads(number_cpu_available)



def function_a(f):
	print(f)
	try:
		# coleval.collect_subfolderfits(f)
		full_saved_file_dict = coleval.ats_to_dict(f+'.ats')
		np.savez_compressed(f,**full_saved_file_dict)
		print(f+' done')
	except:
		print('error in '+f)
	# coleval.save_timestamp(f)


# files=[vacuum5]
# files=coleval.flatten_full(files)
# for file in files:
# 	try:
# 		coleval.collect_subfolderfits(file)
# 		coleval.evaluate_back(file)
# 	except:
# 		print('error in '+file)
# 	coleval.save_timestamp(file)


files1 = [laser41,laser42,laser43,laser44,laser45,laser46,laser47,vacuum7]
files2 = [laser1,laser2,laser10,laser11,laser12,laser13,laser14,vacuum1]
files3 = np.flip([laser15,laser16,laser17,laser18,laser19,vacuum2,vacuumtest1],axis=0)
files4 = [files1,files2,files3,files4,files5,files6,files7,files8,files10,files11,files12,files13,files14,files15,files16,files17,files18,files19,files20,files21,files22,files23,files24,files25,files26,files27,files28,files29,files30,files31]

all_files = [files1,files2,files3,files4]

import concurrent.futures as cf
for files in all_files:
	with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
		files=coleval.flatten_full(files)
		executor.map(function_a,files)
