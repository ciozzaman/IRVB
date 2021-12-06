# Fabio Federici

# need to run this code with
#
# source /home/jlovell/venvs/mubol/bin/activate
# python3 /home/ffederic/work/analysis_scripts/scripts/MASTU_resistive_bolometer_post_process.py
# this might be required to be done beforehand to give access to the data
# chmod a+rw -R /home/ffederic/work/irvb/MAST-U

#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)


path = '/home/ffederic/work/irvb/MAST-U/'
# to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25','2021-06-29','2021-06-30','2021-07-01','2021-07-06','2021-07-08','2021-07-09','2021-07-15','2021-07-27','2021-07-28','2021-07-29','2021-08-04','2021-08-05','2021-08-06','2021-08-11','2021-08-12','2021-08-13','2021-08-17','2021-08-18','2021-08-19','2021-08-20','2021-08-24','2021-08-25','2021-08-26','2021-08-27','2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04','2021-10-05','2021-10-06','2021-10-07','2021-10-08','2021-10-11','2021-10-12','2021-10-13','2021-10-14','2021-10-15']
# to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22']
# to_do = ['2021-09-08','2021-06-17','2021-08-05']
to_do = ['2021-10-22']
# to_do = ['2021-09-01','2021-09-08','2021-09-09','2021-09-10','2021-09-16','2021-09-17','2021-09-21','2021-09-22','2021-09-23','2021-09-24','2021-09-28','2021-09-29','2021-09-30','2021-10-01','2021-10-04']
to_do = np.flip(to_do,axis=0)
# to_do = ['2021-06-03']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']


import numpy as np
from cherab.core.math import Interpolate2DLinear
from mubol.phantoms.solps import SOLPSPhantom
import xarray as xr
import netCDF4


f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.append(dirnames)
days_available = f[0]
shot_available = []
for i_day,day in enumerate(to_do):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path+day+'/'):
		f.append(filenames)
	shot_available.append([])
	for name in f[0]:
		if name[-3:]=='ats' or name[-3:]=='ptw':
			shot_available[i_day].append(name)

color = ['b', 'r', 'm', 'y', 'g', 'c', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive', 'royalblue', 'sienna', 'navy']
if True:
	for i_day,day in enumerate(to_do):
	# for i_day,day in enumerate(np.flip(to_do,axis=0)):
		# for name in shot_available[i_day]:
		for name in np.flip(shot_available[i_day],axis=0):
			laser_to_analyse=path+day+'/'+name

			if not os.path.exists(laser_to_analyse[:-4]+'_inverted_baiesian.npz'):
				continue

			print('found inverted data in\n'+laser_to_analyse)
			shot_number = int(laser_to_analyse[-9:-4])
			path_power_output = os.path.split(laser_to_analyse)[0] + '/' + str(shot_number)
			inverted_dict = np.load(laser_to_analyse[:-4]+'_inverted_baiesian.npz')
			inverted_dict.allow_pickle=True
			inverted_dict = dict(inverted_dict)

			all_grid_type = np.unique([int(value) if value!='efit_reconstruction' else 4 for value in list(inverted_dict.keys()) ])
			for grid_resolution in all_grid_type:
				all_shrink_factor_x = np.unique( list(inverted_dict[str(grid_resolution)].all().keys()) ).astype(int)
				for shrink_factor_x in all_shrink_factor_x:
					all_shrink_factor_t = np.unique( list(inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)].keys()) ).astype(int)
					for shrink_factor_t in all_shrink_factor_t:
						binning_type = 'bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)
						print('starting '+binning_type)

						time_full_binned_crop = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop']
						inversion_R = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['R']
						inversion_Z = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['geometry']['Z']
						inverted_data = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['inverted_data']
						inverted_data[np.isnan(inverted_data)] = 0

						class SXDL_translator(SOLPSPhantom):

							def __init__(self,i_t):
								# self.prad_interpolator = rad_interp
								rad_interp = Interpolate2DLinear(inversion_R,inversion_Z,inverted_data[i_t],extrapolate=True)
								data = np.zeros((len(inversion_Z), len(inversion_R)))
								self.sampled_emiss = xr.DataArray(data, coords=[('z', inversion_Z), ('r', inversion_R)])
								super().__init__(prad_interpolator=rad_interp,system='sxdl')

						class Core_translator(SOLPSPhantom):

							def __init__(self,i_t):
								# self.prad_interpolator = rad_interp
								rad_interp = Interpolate2DLinear(inversion_R,inversion_Z,inverted_data[i_t],extrapolate=True)
								data = np.zeros((len(inversion_Z), len(inversion_R)))
								self.sampled_emiss = xr.DataArray(data, coords=[('z', inversion_Z), ('r', inversion_R)])
								super().__init__(prad_interpolator=rad_interp,system='core')

						resistive_bolo_sxdl_from_IRVB = []
						resistive_bolo_core_from_IRVB = []
						resistive_bolo_sxdl_from_IRVB_LOS_index = SXDL_translator(0).channel_numbers
						resistive_bolo_core_from_IRVB_LOS_index = Core_translator(0).channel_numbers
						for i_t in range(len(time_full_binned_crop)):
							print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
							resistive_bolo_sxdl_from_IRVB.append(SXDL_translator(i_t).brightness)
							resistive_bolo_core_from_IRVB.append(Core_translator(i_t).brightness)
						resistive_bolo_sxdl_from_IRVB = np.array(resistive_bolo_sxdl_from_IRVB) *4*np.pi	# the 4pi comes from radiance to brightness in the resistive bolometer geometry matrix
						resistive_bolo_core_from_IRVB = np.array(resistive_bolo_core_from_IRVB) *4*np.pi	# the 4pi comes from radiance to brightness in the resistive bolometer geometry matrix

						fig, ax = plt.subplots( 4,1,figsize=(12, 20), squeeze=False,sharex=True)
						fig.suptitle('Comparison between brightness from IRVB (-)\nand resistive bolometer brigthness (--)')
						for index in range(1,16+1):
							ax[0,0].plot(time_full_binned_crop,resistive_bolo_core_from_IRVB[:,resistive_bolo_core_from_IRVB_LOS_index==index],'-',color=color[index-1],label=str(index))
						ax[0,0].legend(loc='best', fontsize='xx-small')
						ax[0,0].set_title('Core poloidal')
						for index in range(17,32+1):
							ax[1,0].plot(time_full_binned_crop,resistive_bolo_core_from_IRVB[:,resistive_bolo_core_from_IRVB_LOS_index==index],'-',color=color[index-1-16],label=str(index))
						ax[1,0].set_title('Core tangential')
						ax[1,0].legend(loc='best', fontsize='xx-small')
						for index in range(1,16+1):
							ax[2,0].plot(time_full_binned_crop,resistive_bolo_sxdl_from_IRVB[:,resistive_bolo_sxdl_from_IRVB_LOS_index==index],'-',color=color[index-1],label=str(index))
						ax[2,0].set_title('SDX poloidal')
						ax[2,0].legend(loc='best', fontsize='xx-small')
						for index in range(17,32+1):
							ax[3,0].plot(time_full_binned_crop,resistive_bolo_sxdl_from_IRVB[:,resistive_bolo_sxdl_from_IRVB_LOS_index==index],'-',color=color[index-1-16],label=str(index))
						ax[3,0].set_title('SDX tangential')
						ax[3,0].legend(loc='best', fontsize='xx-small')
						ax[3,0].set_xlabel('time [s]')
						ax[0,0].set_ylabel('brightness [W/m2]')
						ax[1,0].set_ylabel('brightness [W/m2]')
						ax[2,0].set_ylabel('brightness [W/m2]')
						ax[3,0].set_ylabel('brightness [W/m2]')
						ax[0,0].grid()
						ax[1,0].grid()
						ax[2,0].grid()
						ax[3,0].grid()
						ax[0,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
						ax[1,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
						ax[2,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())
						ax[3,0].set_xlim(left=time_full_binned_crop.min(),right=time_full_binned_crop.max())

						inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_sxdl_from_IRVB_LOS_index'] = resistive_bolo_sxdl_from_IRVB_LOS_index
						inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_sxdl_from_IRVB'] = resistive_bolo_sxdl_from_IRVB
						inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_core_from_IRVB_LOS_index'] = resistive_bolo_core_from_IRVB_LOS_index
						inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_core_from_IRVB'] = resistive_bolo_core_from_IRVB

						try:
							f = netCDF4.Dataset('/common/uda-scratch/jlovell/abm0'+str(shot_number)+'.nc')
							# core
							good = f['abm']['core']['good'][:].data
							channel = f['abm']['core']['channel'][:].data
							good_channel = channel[good.astype(bool)]
							brightness = f['abm']['core']['brightness'][:].data
							time = f['abm']['core']['time'][:].data
							for index in good_channel[good_channel<=16]:
								ax[0,0].plot(time,brightness[:,channel==index],'--',color=color[index-1],label=str(index))
							for index in good_channel[good_channel>=17]:
								ax[1,0].plot(time,brightness[:,channel==index],'--',color=color[index-1-16],label=str(index))

							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_core_index'] = channel
							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_core_good'] = good
							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_core_brightness'] = brightness
							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_core_time'] = time

							good = f['abm']['sxdl']['good'][:].data
							channel = f['abm']['sxdl']['channel'][:].data
							good_channel = channel[good.astype(bool)]
							brightness = f['abm']['sxdl']['brightness'][:].data
							time = f['abm']['sxdl']['time'][:].data
							for index in good_channel[good_channel<=16]:
								ax[2,0].plot(time,brightness[:,channel==index],'--',color=color[index-1],label=str(index))
							for index in good_channel[good_channel>=17]:
								ax[3,0].plot(time,brightness[:,channel==index],'--',color=color[index-1-16],label=str(index))

							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_sxdl_index'] = channel
							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_sxdl_good'] = good
							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_sxdl_brightness'] = brightness
							inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['resistive_bolo_sxdl_time'] = time
						except:
							pass

						plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_IRVB_res_bolo_comparison.eps')
						plt.close('all')

			np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian',**inverted_dict)

else:
	# i_day,day = 0,'2021-07-29'
	# name='IRVB-MASTU_shot-44578.ptw'
	# i_day,day = 0,'2021-08-13'
	# name='IRVB-MASTU_shot-44677.ptw'
	# i_day,day = 0,'2021-06-24'
	# name='IRVB-MASTU_shot-44308.ptw'
	# i_day,day = 0,'2021-09-01'
	# name='IRVB-MASTU_shot-44863.ptw'
	# i_day,day = 0,'2021-09-28'
	# name='IRVB-MASTU_shot-45071.ptw'
	# i_day,day = 0,'2021-10-05'
	# name='IRVB-MASTU_shot-45156.ptw'
	i_day,day = 0,'2021-10-12'
	name='IRVB-MASTU_shot-45234.ptw'
	laser_to_analyse=path+day+'/'+name
	exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process2.py").read())
	# exec(open("/home/ffederic/work/analysis_scripts/scripts/MASTU_temp_to_power3.py").read())
