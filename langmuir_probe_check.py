# Created 01/12/2021
# Fabio Federici


from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe, probe_array, compare_shots
import pyuda
client=pyuda.Client()
# import matplotlib.pyplot as plt
import numpy as np

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())

shot=45400
EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction = coleval.mclass(EFIT_path_default+'/epm0'+str(shot)+'.nc',pulse_ID=str(shot))

lp_data = LangmuirProbe(alp_file='/common/uda-scratch/pryan/alp0'+str(shot)+'.nc')
lp_data.contour_plot(colour_lim=[-3,0],trange=[0,1],divertor='lower', sectors=10 ,log_data=True, quantity = 'isat', coordinate='r',tiles=['T2','T3','T4','T5'],show=False)
for __i in range(np.shape(efit_reconstruction.strikepointR)[1]):
	plt.plot(efit_reconstruction.strikepointR[:,__i],efit_reconstruction.time,'--r')
plt.title('log10 Jsat[A],10lower\nshot '+str(shot))
plt.pause(0.01)