# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())



# degree of polynomial of choice
n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters/1ms383Hz/average'
# folder to read
pathfiles=vacuum2
# framerate of the IR camera in Hz
framerate=383
# integration time of the camera in ms
inttime=1
#filestype
type='_stat.npy'
# type='csv'
