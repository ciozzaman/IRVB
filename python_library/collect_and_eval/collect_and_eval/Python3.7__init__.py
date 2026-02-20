import os

# from Kevin 11/05/2022
def uda_transfer(shotnr,tag,savedir=os.getcwd()+'/dum.nc',extra_path = ''):
	import pyuda

	if savedir[-6:] == 'dum.nc':
		if tag=='elp':
			savedir = savedir[:-6] +tag+'0'+ str(shotnr) + '.nc'
		else:
			savedir = savedir[:-6] +tag+ str(shotnr) + '.nc'
	# get efit

	if not isinstance(shotnr,str):
		shotnr = str(shotnr)

	client_int = pyuda.Client()
	try:
		client_int.get_file('$MAST_DATA'+extra_path+'/'+shotnr+'/LATEST/'+tag+'0'+shotnr+'.nc',savedir)
	except:
		savedir = 'failed'
	client_int.reset_connection()
	# reset_connection(client_int)
	# del client_int

	return savedir


def read_LP_data(shot,path = '/home/ffederic/work/irvb/from_pryan_LP',path_alternate='/common/uda-scratch/pryan'):
	from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe
	try:
		tag_cycle='elp'
		try:
			try:
				try:
					lp_data = LangmuirProbe(filename=path+'/'+tag_cycle+'0'+str(shot)+'.nc')
				except:
					lp_data = LangmuirProbe(filename=path_alternate+'/'+tag_cycle+'0'+str(shot)+'.nc')
			except:
				lp_data = LangmuirProbe(shot=shot,tag=tag_cycle)
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'isat', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
		except:
			try:
				lp_data = LangmuirProbe(filename=path+'/'+tag_cycle+'0'+str(shot)+'.nc',version='new')
				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'isat', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
			except:
				lp_data = LangmuirProbe(filename=path+'/'+tag_cycle+'0'+str(shot)+'.nc',version='old')
				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'isat', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
	except:
		tag_cycle='alp'
		try:
			try:
				try:
					lp_data = LangmuirProbe(filename=path+'/'+tag_cycle+'0'+str(shot)+'.nc')
				except:
					lp_data = LangmuirProbe(filename=path_alternate+'/'+tag_cycle+'0'+str(shot)+'.nc')
			except:
				lp_data = LangmuirProbe(shot=shot,tag=tag_cycle)
			output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'isat', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
		except:
			try:
				lp_data = LangmuirProbe(filename=path+'/'+tag_cycle+'0'+str(shot)+'.nc',version='new')
				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'isat', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
			except:
				lp_data = LangmuirProbe(filename=path+'/'+tag_cycle+'0'+str(shot)+'.nc',version='old')
				output_contour1=lp_data.contour_plot(trange=[0,1.5],bad_probes=None,divertor='lower', sectors=10, quantity = 'isat', coordinate='R',tiles=['C5','C6','T2','T3','T4','T5'],show=False)
	return lp_data,output_contour1
