# Created 03/12/2018
# Fabio Federici


# #this is if working on a pc, use pc printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

#this is if working in batch, use predefined NOT visual printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())



def function_a(f):
	print(f)
	try:
		# coleval.collect_subfolderfits(f)
		coleval.evaluate_back(f)
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


files=[vacuum7]
files=coleval.flatten_full(files)

import concurrent.futures as cf
with cf.ProcessPoolExecutor() as executor:
	executor.map(function_a,files)
