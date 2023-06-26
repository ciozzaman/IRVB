# Created 08/12/2020
# Fabio Federici
# this is to analyse the results from the different instances (45239 / 45235) of
# MASTU_second_campaign_compare_geom_standof_45_pinhole_4_radiator_scan2


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 8

# just to import _MASTU_CORE_GRID_POLYGON
calculate_tangency_angle_for_poloidal_section=coleval.calculate_tangency_angle_for_poloidal_section
exec(open("/home/ffederic/work/analysis_scripts/scripts/python_library/collect_and_eval/collect_and_eval/MASTU_structure.py").read())


# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

import pickle
from scipy.ndimage import geometric_transform
# import cherab.mastu.bolometry.grid_construction

from multiprocessing import Pool,cpu_count,set_start_method
# set_start_method('spawn',force=True)
try:
	number_cpu_available = open('/proc/cpuinfo').read().count('processor\t:')
except:
	number_cpu_available = cpu_count()
number_cpu_available = 8	# the previous cheks never work
print('Number of cores available: '+str(number_cpu_available))

from scipy.signal import find_peaks, peak_prominences as get_proms
import time as tm
import pyuda as uda
client = uda.Client()

path = '/home/ffederic/work/irvb/MAST-U/'
to_do = ['2021-05-18','2021-05-19','2021-05-20','2021-05-21','2021-05-25','2021-05-26','2021-05-27','2021-05-28','2021-06-02','2021-06-03','2021-06-04','2021-06-15','2021-06-16','2021-06-17','2021-06-18','2021-06-22','2021-06-23','2021-06-24','2021-06-25','2021-06-29','2021-06-30','2021-07-01','2021-07-06','2021-07-08','2021-07-09','2021-07-15','2021-07-27','2021-07-28','2021-07-29','2021-08-04']
# to_do = ['2021-06-29','2021-07-01']
# to_do = ['2021-08-05']
to_do = np.flip(to_do,axis=0)
# to_do = ['2021-06-03']
# path = '/home/ffederic/work/irvb/MAST-U/preliminaly_shots/'
# to_do = ['2021-05-13','2021-05-12','2021-04-28','2021-04-29','2021-04-30']


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



i_day,day = 0,'2021-10-12'
name_SXD='IRVB-MASTU_shot-45239.ptw'
name_CD='IRVB-MASTU_shot-45235.ptw'
laser_to_analyse_SXD = path+day+'/'+name_SXD
laser_to_analyse_CD = path+day+'/'+name_CD


print('starting '+laser_to_analyse_SXD)
print('starting '+laser_to_analyse_CD)

shot_number_SXD = int(laser_to_analyse_SXD[-9:-4])
shot_number_CD = int(laser_to_analyse_CD[-9:-4])

EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
efit_reconstruction_SXD = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse_SXD[-9:-4]+'.nc',pulse_ID=laser_to_analyse_SXD[-9:-4])
all_time_sep_r_SXD,all_time_sep_z_SXD,r_fine_SXD,z_fine_SXD = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction_SXD)
all_time_separatrix_SXD = coleval.return_all_time_separatrix(efit_reconstruction_SXD,all_time_sep_r_SXD,all_time_sep_z_SXD,r_fine_SXD,z_fine_SXD)
efit_reconstruction_CD = coleval.mclass(EFIT_path_default+'/epm0'+laser_to_analyse_CD[-9:-4]+'.nc',pulse_ID=laser_to_analyse_CD[-9:-4])
all_time_sep_r_CD,all_time_sep_z_CD,r_fine_CD,z_fine_CD = coleval.efit_reconstruction_to_separatrix_on_foil(efit_reconstruction_CD)
all_time_separatrix_CD = coleval.return_all_time_separatrix(efit_reconstruction_CD,all_time_sep_r_CD,all_time_sep_z_CD,r_fine_CD,z_fine_CD)



def load_phantom_scan(phantom_variant,laser_to_analyse):
	# manual bit
	# phantom_variant = '1'
	grid_resolution = 2
	scan_type = 'stand_off_0.045_pinhole_4' + '_'+phantom_variant
	temp_save = np.load(laser_to_analyse[:-4]+'_'+scan_type+'_inverted_baiesian_test_export_radiator_scan.npz')
	temp_save.allow_pickle = True
	temp_save = dict(temp_save)
	temp_save[scan_type] = temp_save[scan_type].all()
	x_optimal_out = temp_save[scan_type]['x_optimal_out']
	x_optimal_input_full_res_all = temp_save[scan_type]['x_optimal_input_full_res_all']
	recompose_voxel_emissivity_out = temp_save[scan_type]['recompose_voxel_emissivity_out']
	recompose_voxel_emissivity_sigma_out = temp_save[scan_type]['recompose_voxel_emissivity_sigma_out']
	covariance_out = temp_save[scan_type]['covariance_out']
	phantom = temp_save[scan_type]['phantom']
	grid_data_masked_crop = temp_save[scan_type]['grid_data_masked_crop']
	regolarisation_coeff = 1e-2
	regolarisation_coeff_all = np.array([regolarisation_coeff]*len(phantom))

	recompose_voxel_emissivity_input_all = []
	recompose_voxel_emissivity_output_all = []
	recompose_voxel_emissivity_difference_all = []
	for i_phantom_int,phantom_int in enumerate(phantom):
		x_optimal_input = np.concatenate([phantom_int,[0],[0]])	# W/m^3/st
		x_optimal_output = x_optimal_out[i_phantom_int]
		x_optimal_difference = x_optimal_out[i_phantom_int] - np.concatenate([phantom_int,[0],[0]])
		recompose_voxel_emissivity_input_all.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),x_optimal_input,np.mean(grid_data_masked_crop,axis=1))[1])
		recompose_voxel_emissivity_output_all.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),x_optimal_output,np.mean(grid_data_masked_crop,axis=1))[1])
		recompose_voxel_emissivity_difference_all.append(coleval.translate_emissivity_profile_with_homo_temp(np.mean(grid_data_masked_crop,axis=1),x_optimal_difference,np.mean(grid_data_masked_crop,axis=1))[1])

	difference_sum = np.nansum(np.abs(recompose_voxel_emissivity_difference_all),axis=(1,2))
	difference_sum_sigma = np.nansum(np.array(recompose_voxel_emissivity_sigma_out)**2,axis=(1,2))**0.5
	# baricenre
	select = np.mean(grid_data_masked_crop,axis=1)[:,1]<-1.1
	radiator_position_r_out = np.nansum((np.array(x_optimal_out)[:,:-2]*(np.mean(grid_data_masked_crop,axis=1)[:,0]))[:,select],axis=(1))/np.nansum(np.array(x_optimal_out)[:,:-2][:,select],axis=(1))
	radiator_position_z_out = np.nansum((np.array(x_optimal_out)[:,:-2]*(np.mean(grid_data_masked_crop,axis=1)[:,1]))[:,select],axis=(1))/np.nansum(np.array(x_optimal_out)[:,:-2][:,select],axis=(1))
	peak_radiator_position_r_out = np.mean(grid_data_masked_crop,axis=1)[:,0][select][np.array(x_optimal_out)[:,:-2][:,select].argmax(axis=1)]
	peak_radiator_position_z_out = np.mean(grid_data_masked_crop,axis=1)[:,1][select][np.array(x_optimal_out)[:,:-2][:,select].argmax(axis=1)]
	radiator_position_r_in = (np.nansum(phantom*(np.mean(grid_data_masked_crop,axis=1)[:,0]),axis=(1))/np.nansum(phantom,axis=(1)))[:len(phantom)-(len(phantom)-len(difference_sum))]
	radiator_position_z_in = (np.nansum(phantom*(np.mean(grid_data_masked_crop,axis=1)[:,1]),axis=(1))/np.nansum(phantom,axis=(1)))[:len(phantom)-(len(phantom)-len(difference_sum))]

	arbitrary_x_point_r = 0.574-0.1
	arbitrary_x_point_z = -1.1+0.1
	select_below_outside_x_point = np.logical_and(np.mean(grid_data_masked_crop,axis=1)[:,1]<=arbitrary_x_point_z,np.mean(grid_data_masked_crop,axis=1)[:,0]>=arbitrary_x_point_r)
	select_peak_dist = ((np.array([np.mean(grid_data_masked_crop,axis=1)[:,0]]*len(peak_radiator_position_r_out)).T-peak_radiator_position_r_out)**2 + (np.array([np.mean(grid_data_masked_crop,axis=1)[:,1]]*len(peak_radiator_position_r_out)).T-peak_radiator_position_z_out)**2).T**0.5 < 0.1

	power_out_peak_dist = np.nansum(np.array(x_optimal_out)[:,:-2]*select_peak_dist*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)*4*np.pi*1e-3
	power_out_below_outside_x_point = np.nansum(np.array(x_optimal_out)[:,:-2]*select_below_outside_x_point*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)*4*np.pi*1e-3
	power_out_all = np.nansum(np.array(x_optimal_out)[:,:-2]*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)*4*np.pi*1e-3
	power_in = np.nansum(phantom*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0],axis=-1)[:len(phantom)-(len(phantom)-len(difference_sum))]*4*np.pi*1e-3



	if False:	# this calculation does not account for covariance and it returns a much higher uncertainty
		r=np.arange(np.round(np.mean(grid_data_masked_crop,axis=1)[:,0].min(),4),np.round(np.mean(grid_data_masked_crop,axis=1)[:,0].max(),4)+grid_resolution*1e-2,grid_resolution*1e-2)
		z=np.arange(np.round(np.mean(grid_data_masked_crop,axis=1)[:,1].min(),4),np.round(np.mean(grid_data_masked_crop,axis=1)[:,1].max(),4)+grid_resolution*1e-2,grid_resolution*1e-2)
		r,z = np.meshgrid(r,z)
		r,z = r.T,z.T
		select_below_outside_x_point = np.logical_and(z<=arbitrary_x_point_z,r>=arbitrary_x_point_r)
		select_peak_dist = ((np.array([r]*len(peak_radiator_position_r_out)).T-peak_radiator_position_r_out)**2 + (np.array([z]*len(peak_radiator_position_r_out)).T-peak_radiator_position_z_out)**2).T**0.5 < 0.1

		power_out_peak_dist_sigma = np.nansum((np.array(recompose_voxel_emissivity_sigma_out)*select_peak_dist*(grid_resolution*1e-2)**2 * 2*np.pi*r)**2,axis=(-1,-2))**0.5*4*np.pi*1e-3
		power_out_below_outside_x_point_sigma = np.nansum((np.array(recompose_voxel_emissivity_sigma_out)*select_below_outside_x_point*(grid_resolution*1e-2)**2 * 2*np.pi*r)**2,axis=(-1,-2))**0.5*4*np.pi*1e-3
		power_out_all_sigma = np.nansum((np.array(recompose_voxel_emissivity_sigma_out)*(grid_resolution*1e-2)**2 * 2*np.pi*r)**2,axis=(-1,-2))**0.5*4*np.pi*1e-3
		# I can calculate it, but I can't really use the uncertainty.
		# the signal level during plasma is "high" because radiation comes from different places, so it can get to a low regularisation. here there is only one source, and it's really unfair to compare the uncertainty everiwhere from a source in a single point
		# of course the uncertainty would be huge compared to the total signal
	else:	# this does
		temp = np.array([select_below_outside_x_point*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*4*np.pi*1e-3]*len(covariance_out))
		power_out_below_outside_x_point_sigma = np.nansum((np.transpose(np.array(covariance_out).T[:-2,:-2]*temp.T,axes=(1,0,2))*temp.T),axis=(0,1))**0.5
		temp = select_peak_dist*(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*4*np.pi*1e-3
		power_out_peak_dist_sigma = np.nansum((np.transpose(np.array(covariance_out).T[:-2,:-2]*temp.T,axes=(1,0,2))*temp.T),axis=(0,1))**0.5
		temp = np.array([(grid_resolution*1e-2)**2 * 2*np.pi*np.mean(grid_data_masked_crop,axis=1)[:,0]*4*np.pi*1e-3]*len(covariance_out))
		power_out_all_sigma = np.nansum((np.transpose(np.array(covariance_out).T[:-2,:-2]*temp.T,axes=(1,0,2))*temp.T),axis=(0,1))**0.5


	return recompose_voxel_emissivity_input_all,recompose_voxel_emissivity_output_all,recompose_voxel_emissivity_difference_all,difference_sum,radiator_position_r_out,radiator_position_z_out,peak_radiator_position_r_out,peak_radiator_position_z_out,radiator_position_r_in,radiator_position_z_in,power_out_peak_dist,power_out_below_outside_x_point,power_out_all,power_in,grid_data_masked_crop,power_out_below_outside_x_point_sigma,power_out_peak_dist_sigma,power_out_all_sigma
grid_resolution = 2e-2

recompose_voxel_emissivity_input_all_1,recompose_voxel_emissivity_output_all_1,recompose_voxel_emissivity_difference_all_1,difference_sum_1,radiator_position_r_out_1,radiator_position_z_out_1,peak_radiator_position_r_out_1,peak_radiator_position_z_out_1,radiator_position_r_in_1,radiator_position_z_in_1,power_out_peak_dist_1,power_out_below_outside_x_point_1,power_out_all_1,power_in_1,grid_data_masked_crop_1,power_out_below_outside_x_point_sigma_1,power_out_peak_dist_sigma_1,power_out_all_sigma_1 = load_phantom_scan('1',laser_to_analyse_SXD)
recompose_voxel_emissivity_input_all_2,recompose_voxel_emissivity_output_all_2,recompose_voxel_emissivity_difference_all_2,difference_sum_2,radiator_position_r_out_2,radiator_position_z_out_2,peak_radiator_position_r_out_2,peak_radiator_position_z_out_2,radiator_position_r_in_2,radiator_position_z_in_2,power_out_peak_dist_2,power_out_below_outside_x_point_2,power_out_all_2,power_in_2,grid_data_masked_crop_2,power_out_below_outside_x_point_sigma_2,power_out_peak_dist_sigma_2,power_out_all_sigma_2 = load_phantom_scan('1',laser_to_analyse_CD)
recompose_voxel_emissivity_input_all_3,recompose_voxel_emissivity_output_all_3,recompose_voxel_emissivity_difference_all_3,difference_sum_3,radiator_position_r_out_3,radiator_position_z_out_3,peak_radiator_position_r_out_3,peak_radiator_position_z_out_3,radiator_position_r_in_3,radiator_position_z_in_3,power_out_peak_dist_3,power_out_below_outside_x_point_3,power_out_all_3,power_in_3,grid_data_masked_crop_3,power_out_below_outside_x_point_sigma_3,power_out_peak_dist_sigma_3,power_out_all_sigma_3 = load_phantom_scan('4',laser_to_analyse_CD)


case=16
plt.figure( figsize=(8, 8))
# im = plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_input_all_1[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop_1[:,:,0].min(),grid_data_masked_crop_1[:,:,0].max(),grid_data_masked_crop_1[:,:,1].min(),grid_data_masked_crop_1[:,:,1].max()],norm=LogNorm(),vmax=np.nanmax(recompose_voxel_emissivity_input_all_1[case])*1e-3,vmin=1e-2*np.nanmax(recompose_voxel_emissivity_input_all_1[case])*1e-3)
im = plt.imshow(np.flip(np.flip(np.flip(np.transpose((recompose_voxel_emissivity_output_all_1)[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop_1[:,:,0].min(),grid_data_masked_crop_1[:,:,0].max(),grid_data_masked_crop_1[:,:,1].min(),grid_data_masked_crop_1[:,:,1].max()],vmin=1e-3*np.nanmax(recompose_voxel_emissivity_output_all_1[case])*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_output_all_1[case])*1e-3)
# im = plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_difference_all_1[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop_1[:,:,0].min(),grid_data_masked_crop_1[:,:,0].max(),grid_data_masked_crop_1[:,:,1].min(),grid_data_masked_crop_1[:,:,1].max()])#,norm=LogNorm(),vmax=np.nanmax(recompose_voxel_emissivity_input_all_1[case])*1e-3,vmin=1e-2*np.nanmax(recompose_voxel_emissivity_input_all_1[case])*1e-3)
# plt.plot(radiator_position_r_in_1[case],radiator_position_z_in_1[case],'+k',markersize=20)
plt.plot([radiator_position_r_in_1[case]-1.5*grid_resolution,radiator_position_r_in_1[case]-1.5*grid_resolution,radiator_position_r_in_1[case]+1.5*grid_resolution,radiator_position_r_in_1[case]+1.5*grid_resolution,radiator_position_r_in_1[case]-1.5*grid_resolution],[radiator_position_z_in_1[case]+1.5*grid_resolution,radiator_position_z_in_1[case]-1.5*grid_resolution,radiator_position_z_in_1[case]-1.5*grid_resolution,radiator_position_z_in_1[case]+1.5*grid_resolution,radiator_position_z_in_1[case]+1.5*grid_resolution],'k',markersize=20)
# im = plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_input_all_2[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop_2[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop_2[:,:,1].min(),grid_data_masked_crop_2[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_input_all_2[case])*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_input_all_2[case])*1e-3)
# plt.plot(radiator_position_r_in_2[case],radiator_position_z_in_2[case],'+k',markersize=20)
# plt.plot(radiator_position_r_in_2, radiator_position_z_in_2, '--k')
cb = plt.colorbar(im,fraction=0.035, pad=0.02).set_label('Emissivity kW/m^3')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(radiator_position_r_in_1, radiator_position_z_in_1, '--k',linewidth=4)
plt.plot(radiator_position_r_in_2, radiator_position_z_in_2, '--g',linewidth=4)
plt.plot(radiator_position_r_in_3, radiator_position_z_in_3, '--m',linewidth=4)
plt.ylim(bottom=-2.1,top=-1.)
plt.xlim(left=0.2,right=1.6)
plt.title('scan 1, case='+str(case)+'\n')
# plt.pause(0.001)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/phantom_example3.png', bbox_inches='tight')
plt.close()

plt.figure( figsize=(12, 12))
im = plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity_output_all_2[case]*1e-3,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()],vmin=np.nanmin(recompose_voxel_emissivity_output_all_2[case])*1e-3,vmax=np.nanmax(recompose_voxel_emissivity_output_all_2[case])*1e-3)
plt.plot(peak_radiator_position_r_out_2[case],peak_radiator_position_z_out_2[case],'+k',markersize=20)
cb = plt.colorbar(im,fraction=0.035, pad=0.02).set_label('Emissivity kW/m^3')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(radiator_position_r_in_2, radiator_position_z_in_2, '--k')
plt.ylim(bottom=-2.1,top=-1.)
plt.xlim(left=0.2,right=1.6)
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/phantom_example2.eps', bbox_inches='tight')
plt.close()


plt.figure(figsize=(10,5))
temp, = plt.plot(peak_radiator_position_r_out_1,((radiator_position_r_in_1-peak_radiator_position_r_out_1)**2+(radiator_position_r_in_1-peak_radiator_position_r_out_1)**2)**0.5,'k',label='SXD')
plt.plot(peak_radiator_position_r_out_1,((radiator_position_r_in_1-peak_radiator_position_r_out_1)**2+(radiator_position_r_in_1-peak_radiator_position_r_out_1)**2)**0.5,'+',color=temp.get_color())
temp, = plt.plot(peak_radiator_position_r_out_2,((radiator_position_r_in_2-peak_radiator_position_r_out_2)**2+(radiator_position_r_in_2-peak_radiator_position_r_out_2)**2)**0.5,'g',label='CD')
plt.plot(peak_radiator_position_r_out_2,((radiator_position_r_in_2-peak_radiator_position_r_out_2)**2+(radiator_position_r_in_2-peak_radiator_position_r_out_2)**2)**0.5,'+',color=temp.get_color())
temp, = plt.plot(peak_radiator_position_r_out_3,((radiator_position_r_in_3-peak_radiator_position_r_out_3)**2+(radiator_position_r_in_3-peak_radiator_position_r_out_3)**2)**0.5,'m',label='surf')
plt.plot(peak_radiator_position_r_out_3,((radiator_position_r_in_3-peak_radiator_position_r_out_3)**2+(radiator_position_r_in_3-peak_radiator_position_r_out_3)**2)**0.5,'+',color=temp.get_color())
# plt.axvline(x=0.79,color='b',linestyle='--',label='acceptable level')
plt.axhline(y=0.02,color='r',linestyle='--',label='grid resolution')
plt.legend(loc='best', fontsize='small')
plt.xlabel('radius of peak of the inverted emissivity [m]')
plt.ylabel('movement of the peak radiation [m]')
plt.grid()
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/position_error.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,10))
# plt.plot(radiator_position_r_out,radiator_position_z_out,'r')
# plt.plot(radiator_position_r_out,radiator_position_z_out,'+r')
temp, = plt.plot(radiator_position_r_in_1,radiator_position_z_in_1,label='SXD')
plt.plot(radiator_position_r_in_1,radiator_position_z_in_1,'+',color=temp.get_color())
temp, = plt.plot(radiator_position_r_in_2,radiator_position_z_in_2,label='CD')
plt.plot(radiator_position_r_in_2,radiator_position_z_in_2,'+',color=temp.get_color())
temp, = plt.plot(radiator_position_r_in_3,radiator_position_z_in_3,label='surf')
plt.plot(radiator_position_r_in_3,radiator_position_z_in_3,'+',color=temp.get_color())
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.ylim(bottom=-2.1,top=-1.1)
plt.xlim(left=0.2,right=1.6)
plt.legend(loc='best', fontsize='medium')
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.grid()
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/phantom_scans.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,10))
# plt.plot(radiator_position_r_out,radiator_position_z_out,'r')
# plt.plot(radiator_position_r_out,radiator_position_z_out,'+r')
temp, = plt.plot(radiator_position_r_in_1,radiator_position_z_in_1,'--k',alpha=0.5)
temp, = plt.plot(radiator_position_r_in_2,radiator_position_z_in_2,'--g',alpha=0.5)
temp, = plt.plot(radiator_position_r_in_3,radiator_position_z_in_3,'--m',alpha=0.5)
# plt.plot(radiator_position_r_in_1,radiator_position_z_in_1,'+b')
# plt.plot(peak_radiator_position_r_out,peak_radiator_position_z_out,'y')
# plt.plot(peak_radiator_position_r_out,peak_radiator_position_z_out,'+y')
# for i in np.arange(len(radiator_position_r_out)):
# 	plt.plot([peak_radiator_position_r_out[i],radiator_position_r_in[i]],[peak_radiator_position_z_out[i],radiator_position_z_in[i]],'--k')
for i in np.arange(len(radiator_position_r_out_1)):
	if ((peak_radiator_position_r_out_1[i]-radiator_position_r_in_1[i])**2+(peak_radiator_position_z_out_1[i]-radiator_position_z_in_1[i])**2)**0.5>0.02:
		plt.plot(radiator_position_r_in_1[i],radiator_position_z_in_1[i],'ok',alpha=1,markersize=7,fillstyle='none')
		plt.plot(peak_radiator_position_r_out_1[i],peak_radiator_position_z_out_1[i],'xk',alpha=1,markersize=7,fillstyle='none')
		temp = ((radiator_position_r_in_1[i]-peak_radiator_position_r_out_1[i])**2 + (radiator_position_z_in_1[i]-peak_radiator_position_z_out_1[i])**2)**0.5
		plt.arrow(radiator_position_r_in_1[i],radiator_position_z_in_1[i],(peak_radiator_position_r_out_1[i]-radiator_position_r_in_1[i]),(peak_radiator_position_z_out_1[i]-radiator_position_z_in_1[i]), head_width=min(0.02,temp*0.9), head_length=min(0.03,temp*1), fc='k', ec='k',ls='-',length_includes_head=True,alpha=0.4,linewidth=1)
for i in np.arange(len(radiator_position_r_out_2)):
	if ((peak_radiator_position_r_out_2[i]-radiator_position_r_in_2[i])**2+(peak_radiator_position_z_out_2[i]-radiator_position_z_in_2[i])**2)**0.5>0.02:
		plt.plot(radiator_position_r_in_2[i],radiator_position_z_in_2[i],'og',alpha=1,markersize=7,fillstyle='none')
		plt.plot(peak_radiator_position_r_out_2[i],peak_radiator_position_z_out_2[i],'xg',alpha=1,markersize=7,fillstyle='none')
		temp = ((radiator_position_r_in_2[i]-peak_radiator_position_r_out_2[i])**2 + (radiator_position_z_in_2[i]-peak_radiator_position_z_out_2[i])**2)**0.5
		plt.arrow(radiator_position_r_in_2[i],radiator_position_z_in_2[i],(peak_radiator_position_r_out_2[i]-radiator_position_r_in_2[i]),(peak_radiator_position_z_out_2[i]-radiator_position_z_in_2[i]), head_width=min(0.02,temp*0.9), head_length=min(0.03,temp*1), fc='k', ec='k',ls='-',length_includes_head=True,alpha=0.4,linewidth=1)
for i in np.arange(len(radiator_position_r_out_3)):
	if ((peak_radiator_position_r_out_3[i]-radiator_position_r_in_3[i])**2+(peak_radiator_position_z_out_3[i]-radiator_position_z_in_3[i])**2)**0.5>0.02:
		plt.plot(radiator_position_r_in_3[i],radiator_position_z_in_3[i],'om',alpha=1,markersize=7,fillstyle='none')
		plt.plot(peak_radiator_position_r_out_3[i],peak_radiator_position_z_out_3[i],'xm',alpha=1,markersize=7,fillstyle='none')
		temp = ((radiator_position_r_in_3[i]-peak_radiator_position_r_out_3[i])**2 + (radiator_position_z_in_3[i]-peak_radiator_position_z_out_3[i])**2)**0.5
		plt.arrow(radiator_position_r_in_3[i],radiator_position_z_in_3[i],(peak_radiator_position_r_out_3[i]-radiator_position_r_in_3[i]),(peak_radiator_position_z_out_3[i]-radiator_position_z_in_3[i]), head_width=min(0.02,temp*0.9), head_length=min(0.03,temp*1), fc='k', ec='k',ls='-',length_includes_head=True,alpha=0.4,linewidth=1)
	# plt.plot([radiator_position_r_out[i],radiator_position_r_in[i]],[radiator_position_z_out[i],radiator_position_z_in[i]],'--r')
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k',label='peak radiation movement')
# temp = np.abs(radiator_position_r_in_1-0.70).argmin()
# plt.plot([0.695,0.695,0.695+0.5],[-1.42-0.5,-1.42,-1.42],'--r',label='limit for power')
# temp = np.abs(radiator_position_r_in_1-0.79).argmin()
# plt.plot([0.795,0.795,0.795+0.5],[-1.625-0.5,-1.625,-1.625],'--k',label='limit for shape')
# plt.plot([0.8,0.8,0.8+0.5],[-1.625-0.5,-1.625,-1.625],'--k',label='limit for shape')
plt.plot([0.76,0.95],[-1.83,-1.55],'--b',label='limit for shape')



ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.ylim(bottom=-2.1,top=-1.1)
plt.xlim(left=0.55,right=1.45)
plt.legend(loc='best', fontsize='small')
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/radiation_movement.png', bbox_inches='tight')
plt.close()


plt.figure()
temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
for i in range(len(all_time_sep_r[temp])):
	plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
temp = np.abs(efit_reconstruction_CD.time-time_full_binned_crop[i_t]).argmin()
for i in range(len(all_time_sep_r_CD[temp])):
	plt.plot(r_fine[all_time_sep_r_CD[temp][i]],z_fine[all_time_sep_z_CD[temp][i]],'--b')
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(radiator_position_r_in-0.70).argmin()
plt.plot([radiator_position_r_in[temp],radiator_position_r_in[temp],radiator_position_r_in[temp]+0.5],[radiator_position_z_in[temp]-0.5,radiator_position_z_in[temp],radiator_position_z_in[temp]],'--r',label='limit for power')
temp = np.abs(radiator_position_r_in-0.76).argmin()
plt.plot([radiator_position_r_in[temp],radiator_position_r_in[temp],radiator_position_r_in[temp]+0.5],[radiator_position_z_in[temp]-0.5,radiator_position_z_in[temp],radiator_position_z_in[temp]],'--k',label='limit for shape')
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.ylim(bottom=-2.1,top=efit_reconstruction.lower_xpoint_z[temp]+0.2)
plt.xlim(left=0.2,right=1.6)
plt.legend(loc='best', fontsize='small')
plt.pause(0.01)


plt.figure(figsize=(10,5))
# plt.plot(radiator_position_r_in,power_out_all,label='total output')
# plt.plot(radiator_position_r_in,power_out_peak_dist,label='output within 0.1m of peak')
# p1, = plt.plot(radiator_position_r_in_1,(power_in_1-power_in_1)/power_in_1,label='input')
# plt.plot(radiator_position_r_in_1,(power_in_1-power_in_1)/power_in_1,'+',color=p1.get_color())
# p1, = plt.plot(radiator_position_r_in_1,100*(power_out_below_outside_x_point_1-power_in_1)/power_in_1,'-k',label='SXD')#,label='output below/outsode x-point')
# plt.plot(radiator_position_r_in_1,100*(power_out_below_outside_x_point_1-power_in_1)/power_in_1,'+',color=p1.get_color())
plt.errorbar(radiator_position_r_in_1,100*(power_out_below_outside_x_point_1-power_in_1)/power_in_1,yerr=100*power_out_below_outside_x_point_sigma_1/power_in_1,color='k',capsize=5,label='SXD')#,label='output below/outsode x-point')

# p1, = plt.plot(radiator_position_r_in_2,(power_in_2-power_in_2)/power_in_2)
# plt.plot(radiator_position_r_in_2,(power_in_2-power_in_2)/power_in_2,'+',color=p1.get_color())
# p1, = plt.plot(radiator_position_r_in_2,100*(power_out_below_outside_x_point_2-power_in_2)/power_in_2,'-g',label='CD')
# plt.plot(radiator_position_r_in_2,100*(power_out_below_outside_x_point_2-power_in_2)/power_in_2,'+',color=p1.get_color())
plt.errorbar(radiator_position_r_in_2,100*(power_out_below_outside_x_point_2-power_in_2)/power_in_2,yerr=100*power_out_below_outside_x_point_sigma_2/power_in_2,color='g',capsize=5,label='CD')

# p1, = plt.plot(radiator_position_r_in_3,(power_in_3-power_in_3)/power_in_3)
# plt.plot(radiator_position_r_in_3,(power_in_3-power_in_3)/power_in_3,'+',color=p1.get_color())
# p1, = plt.plot(radiator_position_r_in_3,100*(power_out_below_outside_x_point_3-power_in_3)/power_in_3,'-m',label='surf')
# plt.plot(radiator_position_r_in_3,100*(power_out_below_outside_x_point_3-power_in_3)/power_in_3,'+',color=p1.get_color())
plt.errorbar(radiator_position_r_in_3,100*(power_out_below_outside_x_point_3-power_in_3)/power_in_3,yerr=100*power_out_below_outside_x_point_sigma_3/power_in_3,color='m',capsize=5,label='surf')

# plt.axvline(x=0.695,color='r',linestyle='--',label='acceptable level')
plt.axhline(y=0,color='k',linestyle='--')
plt.legend(loc='best', fontsize='small')
plt.xlabel('phantom radii [m]')
plt.ylabel('power error [%]')
plt.grid()
# plt.ylim(bottom=0)
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/irvb/0__outputs'+'/power_error.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
# plt.plot(radiator_position_r_in,power_out_all,label='total output')
# plt.plot(radiator_position_r_in,power_out_peak_dist,label='output within 0.1m of peak')
p1, = plt.plot(radiator_position_r_in_1,power_in_1/power_in_1,label='input')
plt.plot(radiator_position_r_in_1,power_in_1/power_in_1,'+',color=p1.get_color())
plt.plot(radiator_position_r_in_1,np.abs((power_out_below_outside_x_point_1-power_in_1)/power_in_1),'--',color=p1.get_color(),label='output below/outsode x-point')
plt.plot(radiator_position_r_in_1,np.abs((power_out_below_outside_x_point_1-power_in_1)/power_in_1),'+',color=p1.get_color(),label='SXD')

p1, = plt.plot(radiator_position_r_in_2,power_in_2/power_in_2)
plt.plot(radiator_position_r_in_2,power_in_2/power_in_2,'+',color=p1.get_color())
plt.plot(radiator_position_r_in_2,np.abs((power_out_below_outside_x_point_2-power_in_2)/power_in_2),'--',color=p1.get_color())
plt.plot(radiator_position_r_in_2,np.abs((power_out_below_outside_x_point_2-power_in_2)/power_in_2),'+',color=p1.get_color(),label='CD')

p1, = plt.plot(radiator_position_r_in_3,power_in_3/power_in_3)
plt.plot(radiator_position_r_in_3,power_in_3/power_in_3,'+',color=p1.get_color())
plt.plot(radiator_position_r_in_3,np.abs((power_out_below_outside_x_point_3-power_in_3)/power_in_3),'--',color=p1.get_color())
plt.plot(radiator_position_r_in_3,np.abs((power_out_below_outside_x_point_3-power_in_3)/power_in_3),'+',color=p1.get_color(),label='surf')

plt.axvline(x=0.695,color='k',linestyle='--',label='acceptable level')
plt.legend(loc='best', fontsize='small')
plt.xlabel('phantom radii [m]')
plt.ylabel('power error [%]')
plt.grid()
plt.ylim(bottom=0)
plt.pause(0.01)

plt.figure()
plt.scatter(radiator_position_r_in_1,radiator_position_z_in_1,c=np.abs(power_out_below_outside_x_point_1-power_in_1)/power_in_1,marker='o',label='SXD')
plt.scatter(radiator_position_r_in_2,radiator_position_z_in_2,c=np.abs(power_out_below_outside_x_point_2-power_in_2)/power_in_2,marker='v',label='CD')
plt.scatter(radiator_position_r_in_3,radiator_position_z_in_3,c=np.abs(power_out_below_outside_x_point_3-power_in_3)/power_in_3,marker='^',label='surf')




grid_resolution_m = grid_resolution*0.01
spatial_coord=np.meshgrid(np.arange(np.shape(recompose_voxel_emissivity_output_all)[2]),np.arange(np.shape(recompose_voxel_emissivity_output_all)[1]))
def gaussian_2D_fitting_plus_gradient(full_output,data):
	def internal(args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		diameter_2 = ((x-args[2])*grid_resolution_m)**2+((y-args[1])*grid_resolution_m)**2
		out = args[0]*np.exp(- diameter_2/(2*(args[3]**2)) )
		full_out = out-data
		grad_a0 = np.nansum( 2*full_out*np.exp(- diameter_2/(2*(args[3]**2)) ) )
		grad_a1 = -np.nansum( 2*full_out*out*(-1/(2*(args[3]**2)))*2*(grid_resolution_m**2)*(y-args[1]))
		grad_a2 = -np.nansum( 2*full_out*out*(-1/(2*(args[3]**2)))*2*(grid_resolution_m**2)*(x-args[2]))
		grad_a3 = np.nansum( 2*full_out*out*( diameter_2/(args[3]**3)) )
		out = full_out**2
		if full_output==True:
			return np.nansum(out),np.array([grad_a0,grad_a1,grad_a2,grad_a3])
		else:
			return np.nansum(out)
	return internal

if False:	# only for testinf the prob_and_gradient function
	temp = recompose_voxel_emissivity_output_all[-2]
	temp[np.isnan(temp)] = 0
	guess = [temp.max(),*np.unravel_index(temp.argmax(),np.shape(temp)),0.01]
	# guess = [temp.max(),20,20,0.03]
	target = 3
	scale = 1e-4
	# guess[target] = 1e5
	temp1 = gaussian_2D_fitting_plus_gradient(True,temp)(guess)
	guess[target] +=scale
	temp2 = gaussian_2D_fitting_plus_gradient(True,temp)(guess)
	guess[target] += -2*scale
	temp3 = gaussian_2D_fitting_plus_gradient(True,temp)(guess)
	guess[target] += scale
	print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))

bds = [[0,np.inf],[-np.inf,np.inf],[-np.inf,np.inf],[0,np.inf]]
x_optimal_output_all = []
fit_output = []
x_optimal_input_all = []
fit_input = []
x = spatial_coord[0]	# horizontal
y = spatial_coord[1]	# vertical
for i_phantom_int,phantom_int in enumerate(phantom):
	temp = recompose_voxel_emissivity_output_all[i_phantom_int]
	temp[np.isnan(temp)] = 0
	guess = [temp.max(),*np.unravel_index(temp.argmax(),np.shape(temp)),0.01]
	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,recompose_voxel_emissivity_output_all[i_phantom_int]), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
	x_optimal_output_all.append(x_optimal)
	diameter_2 = ((x-x_optimal[2])*grid_resolution_m)**2+((y-x_optimal[1])*grid_resolution_m)**2
	out = x_optimal[0]*np.exp(- diameter_2/(2*(x_optimal[3]**2)) )
	fit_output.append(out)

	temp = recompose_voxel_emissivity_input_all[i_phantom_int]
	temp[np.isnan(temp)] = 0
	guess = [temp.max(),*np.unravel_index(temp.argmax(),np.shape(temp)),0.01]
	x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,recompose_voxel_emissivity_input_all[i_phantom_int]), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
	x_optimal_input_all.append(x_optimal)
	diameter_2 = ((x-x_optimal[2])*grid_resolution_m)**2+((y-x_optimal[1])*grid_resolution_m)**2
	out = x_optimal[0]*np.exp(- diameter_2/(2*(x_optimal[3]**2)) )
	fit_input.append(out)
x_optimal_output_all = np.array(x_optimal_output_all)	# A,y,x,sigma
fit_output = np.array(fit_output)
x_optimal_input_all = np.array(x_optimal_input_all)	# A,y,x,sigma
fit_input = np.array(fit_input)

extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
image_extent = [grid_data[:,:,0].min(), grid_data[:,:,0].max(), grid_data[:,:,1].min(), grid_data[:,:,1].max()]
ani = coleval.movie_from_data_radial_profile(np.array([np.flip(np.transpose(np.abs(fit_output--np.array(recompose_voxel_emissivity_output_all)),(0,2,1)),axis=2)]), 1, extent = extent, image_extent=image_extent,timesteps=np.ones(len(fit))*0.5,integration=laser_int_time/1000,barlabel='Emissivity |fit - output| [W/m3]',xlabel='R [m]', ylabel='Z [m]', prelude='shot '  + laser_to_analyse[-9:-4] + '\n'+binning_type+'\n' ,overlay_structure=True,pulse_ID=laser_to_analyse[-9:-4],overlay_x_point=True,overlay_mag_axis=False,overlay_strike_points=True,overlay_separatrix=True,include_EFIT=False)#,extvmin=0,extvmax=4e4)
ani.save('/home/ffederic/work/irvb/MAST-U' + '/' + str(shot_number)+'_'+ binning_type +'_gridres'+str(grid_resolution)+'cm_reconstruct_emissivity_bayesian_radiator_scan3.mp4', fps=5*(30)/383, writer='ffmpeg',codec='mpeg4')
# plt.pause(0.01)
plt.close()

plt.figure()
plt.scatter(np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_output_all[:,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_output_all[:,2].astype(int)],c=x_optimal_output_all[:,3]/x_optimal_input_all[:,3],s=50,marker='s',cmap='rainbow')
plt.colorbar()
for i_phantom_int,phantom_int in enumerate(phantom):
	plt.plot([np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_input_all[i_phantom_int,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_output_all[i_phantom_int,1].astype(int)]],[np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_input_all[i_phantom_int,2].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_output_all[i_phantom_int,2].astype(int)]],'k--')
plt.plot(np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_input_all[:,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_input_all[:,2].astype(int)],'bo')
# plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.gca().set_aspect('equal')
plt.ylim(top=-0.5)
plt.pause(0.01)

plt.figure()
plt.scatter(np.unique(np.mean(grid_data,axis=1)[:,0])[x_optimal_output_all[:,1].astype(int)],np.unique(np.mean(grid_data,axis=1)[:,1])[x_optimal_output_all[:,2].astype(int)],c=np.sum(recompose_voxel_emissivity_output_all,axis=(1,2))/np.max(recompose_voxel_emissivity_output_all,axis=(1,2)),s=50,marker='s',cmap='rainbow')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.gca().set_aspect('equal')
plt.ylim(top=-0.5)
plt.pause(0.01)


regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
x_optimal_all = np.flip(x_optimal_all,axis=0)
recompose_voxel_emissivity_all = np.flip(recompose_voxel_emissivity_all,axis=0)

score_x = np.sum(((np.dot(sensitivities_binned_crop,np.array(x_optimal_all)[:,:-2].T).T  + (np.array([selected_ROI_internal.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-1]).T*homogeneous_scaling + (np.array([select_foil_region_with_plasma.tolist()]*len(x_optimal_all)).T*np.array(x_optimal_all)[:,-2]).T*homogeneous_scaling  - powernoback) ** 2) / (sigma_powernoback**2),axis=1)
score_y = np.sum(((np.dot(grid_laplacian_masked_crop_scaled,np.array(x_optimal_all)[:,:-2].T).T) ** 2) / (sigma_emissivity**2),axis=1)

plt.figure(10)
plt.plot(np.log(score_x),np.log(score_y),'--',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))

regolarisation_coeff_upper_limit = 10**-0.2
score_y,score_x,score_y_record_rel,score_x_record_rel,curvature_range,Lcurve_curvature,recompose_voxel_emissivity,x_optimal,points_removed,regolarisation_coeff,regolarisation_coeff_range,y_opt,opt_info,curvature_range_left_all,curvature_range_right_all,peaks,best_index = coleval.find_optimal_regularisation(score_x,score_y,regolarisation_coeff_range,x_optimal_all,recompose_voxel_emissivity_all,y_opt_all,opt_info_all,regolarisation_coeff_upper_limit=regolarisation_coeff_upper_limit,forward_model_residuals=True)

plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(score_x,score_y,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(score_x[best_index],score_y[best_index],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(score_x[peaks],score_y[peaks],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
plt.xlabel('log ||Gm-d||2')
plt.ylabel('log ||Laplacian(m)||2')
plt.title(csv_file.name[-60:-28])
# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_evolution.eps')
plt.figure(11)
plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(regolarisation_coeff_range[best_index],Lcurve_curvature[best_index-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
plt.plot(regolarisation_coeff_range[peaks],Lcurve_curvature[peaks-curvature_range],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)),fillstyle='none',markersize=10)
plt.axvline(x=regolarisation_coeff_upper_limit,color='r')
plt.semilogx()
plt.xlabel('regularisation coeff')
plt.ylabel('L-curve turvature')
# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution.eps')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

fitted_foil_power = (np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig.shape)
fitted_foil_power_input = (np.dot(sensitivities_binned_crop,x_optimal_input[:-2])+x_optimal_input[-2]*select_foil_region_with_plasma*homogeneous_scaling+x_optimal_input[-1]*selected_ROI_internal*homogeneous_scaling).reshape(powernoback_full_orig.shape)
foil_power = powernoback_full_orig
foil_power_residuals = powernoback_full_orig-fitted_foil_power
foil_power_std = cp.deepcopy(sigma_powernoback_full)
foil_power_std[foil_power_std==1e10]=np.nan

plt.figure(figsize=(12,13))
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
try:
	for i in range(len(all_time_sep_r[temp])):
		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
except:
	pass
plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
plt.colorbar().set_label('emissivity [W/m3]')
plt.ylim(top=0.5)
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(12,13))
# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity-recompose_voxel_emissivity_input,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
try:
	for i in range(len(all_time_sep_r[temp])):
		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
except:
	pass
plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
plt.colorbar().set_label('emissivity inversion error [W/m3]')
plt.ylim(top=0.5)
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)


plt.figure(figsize=(15,12))
plt.imshow(foil_power)
plt.colorbar().set_label('foil_power [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(fitted_foil_power_input)
plt.colorbar().set_label('fitted_foil_power_input [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(fitted_foil_power)
plt.colorbar().set_label('fitted_foil_power [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(foil_power_residuals)
plt.colorbar().set_label('foil_power_residuals [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(foil_power_std)
plt.colorbar().set_label('foil_power_std [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.imshow(foil_power_residuals_simulated)
plt.colorbar().set_label('foil_power_residuals_simulated [W/m2]')
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)

plt.figure(figsize=(15,12))
plt.plot(fitted_foil_power.flatten(),foil_power_residuals_simulated.flatten(),'+')
plt.grid()
plt.title(csv_file.name[-60:-28])
plt.pause(0.01)


 # IMPORTANT - power / std correlation
# plt.figure(figsize=(15,12))
# plt.plot(np.abs(foil_power).flatten(),foil_power_std.flatten(),'+')
# # plt.plot(np.sort(np.abs(foil_power).flatten()),np.polyval(np.polyfit(np.abs(foil_power[np.isfinite(foil_power_std)]).flatten(),foil_power_std[np.isfinite(foil_power_std)].flatten(),1),np.sort(np.abs(foil_power).flatten())),'--')
# plt.plot(np.sort(np.abs(foil_power).flatten()),np.polyval(np.polyfit([0,10,15,20,25],[11.2,11.21,11.24,11.28,11.33],2),np.sort(np.abs(foil_power).flatten())),'--')
# plt.pause(0.01)

temp_save = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test_export.npz')
temp_save.allow_pickle = True
temp_save = dict(temp_save)
temp_save['stand_off_0.045_pinhole_4']['fitted_foil_power'] = fitted_foil_power
temp_save['stand_off_0.045_pinhole_4']['foil_power'] = foil_power
temp_save['stand_off_0.045_pinhole_4']['foil_power_residuals'] = foil_power_residuals
temp_save['stand_off_0.045_pinhole_4']['regolarisation_coeff'] = regolarisation_coeff
temp_save['stand_off_0.045_pinhole_4']['x_optimal'] = x_optimal
temp_save['stand_off_0.045_pinhole_4']['sigma_powernoback'] = sigma_powernoback
np.savez_compressed(laser_to_analyse[:-4]+'_inverted_baiesian_test_export',**temp_save)







if False:	# only visualisation
	plt.figure()
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature)
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+')
	plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o')
	plt.xlabel('regularisation coeff')
	plt.ylabel('L-curve turvature')
	plt.semilogx()
	plt.pause(0.01)

	plt.figure()
	plt.plot(score_x_record_rel,score_y_record_rel)
	plt.plot(score_x_record_rel,score_y_record_rel,'+')
	plt.xlabel('log ||Gm-d||2')
	plt.ylabel('log ||Laplacian(m)||2')
	plt.grid()
	plt.plot(score_x_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],score_y_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],'o')
	plt.pause(0.01)


	plt.figure(figsize=(12,13))
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
	plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
	try:
		for i in range(len(all_time_sep_r[temp])):
			plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
	except:
		pass
	plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
	plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
	plt.colorbar().set_label('emissivity [W/m3]')
	plt.ylim(top=0.5)
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
	plt.pause(0.01)


	# piece of code to plot the traces on the foil of the MASTU geometry and separatrix with changing pinhole location and standoff
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')
	i_time = np.abs(efit_reconstruction.time-0.3).argmin()
	pinhole_offset = np.array([-0.0198,-0.0198])	# toroidal direction parallel to the place surface, z
	# pinhole_offset_extra = np.array([+0.012/(2**0.5),-0.012/(2**0.5)])
	pinhole_offset_extra = np.array([0,0])
	stand_off_length = 0.075	# m
	# Rf=1.54967	# m	radius of the centre of the foil
	Rf=1.48967 + 0.01 + 0.003 + 0.002 + stand_off_length	# m	radius of the centre of the foil
	plane_equation = np.array([1,-1,0,2**0.5 * Rf])	# plane of the foil
	centre_of_foil = np.array([-Rf/(2**0.5), Rf/(2**0.5), -0.7])	# x,y,z
	pinhole_offset += pinhole_offset_extra
	pinhole_location = coleval.locate_pinhole(pinhole_offset=pinhole_offset)
	all_time_separatrix = coleval.return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	all_time_x_point_location = coleval.return_all_time_x_point_location(efit_reconstruction,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	all_time_strike_points_location,all_time_strike_points_location_rot = coleval.return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	fueling_point_location_on_foil = coleval.return_fueling_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	structure_point_location_on_foil = coleval.return_structure_point_location_on_foil(plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	all_time_mag_axis_location = coleval.return_all_time_mag_axis_location(efit_reconstruction,plane_equation=plane_equation,pinhole_location=pinhole_location,centre_of_foil=centre_of_foil)
	cv0 = np.zeros((61,78)).T
	foil_size = [0.07,0.09]
	structure_alpha=0.5
	for i in range(len(fueling_point_location_on_foil)):
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
		plt.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
	for i in range(len(structure_point_location_on_foil)):
		plt.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)
	plt.plot(all_time_x_point_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_x_point_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	plt.plot(all_time_mag_axis_location[i_time][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_mag_axis_location[i_time][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--r')
	for __i in range(len(all_time_strike_points_location_rot[i_time])):
		plt.plot(all_time_strike_points_location_rot[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_strike_points_location_rot[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'-r')
	for __i in range(len(all_time_separatrix[i_time])):
		plt.plot(all_time_separatrix[i_time][__i][:,0]*(np.shape(cv0)[1]-1)/foil_size[0],all_time_separatrix[i_time][__i][:,1]*(np.shape(cv0)[0]-1)/foil_size[1],'--b')
	plt.axhline(y=0,color='k'),plt.axhline(y=np.shape(cv0)[0],color='k'),plt.axvline(x=0,color='k'),plt.axvline(x=np.shape(cv0)[1],color='k')
	plt.title('pinhole additional position [%.3g,%.3g]' %(pinhole_offset_extra[0]*1e3,pinhole_offset_extra[1]*1e3)+'mm\nstand off '+str(stand_off_length*1e3)+'mm')
	plt.pause(0.01)







if False:	# only for testinf the prob_and_gradient function
	target = len(guess)-2
	scale = 1e-3
	# guess[target] = 1e5
	temp1 = prob_and_gradient(guess,*args)
	guess[target] +=scale
	temp2 = prob_and_gradient(guess,*args)
	guess[target] += -2*scale
	temp3 = prob_and_gradient(guess,*args)
	guess[target] += scale
	print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))

	target = 1
	scale = 1e-3
	# guess[target] = 1e5
	temp1 = distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]])(guess)
	guess[target] +=scale
	temp2 = distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]])(guess)
	guess[target] += -2*scale
	temp3 = distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]])(guess)
	guess[target] += scale
	print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))



	voxels_centre = np.mean(grid_data_masked_crop,axis=1)
	dr = np.median(np.diff(np.unique(voxels_centre[:,0])))
	dz = np.median(np.diff(np.unique(voxels_centre[:,1])))
	dist_mean = (dz**2 + dr**2)/2
	recompose_voxel_emissivity = np.zeros((len(np.unique(voxels_centre[:,0])),len(np.unique(voxels_centre[:,1]))))*np.nan
	for i_r,r in enumerate(np.unique(voxels_centre[:,0])):
		for i_z,z in enumerate(np.unique(voxels_centre[:,1])):
			dist = (voxels_centre[:,0]-r)**2 + (voxels_centre[:,1]-z)**2
			if dist.min()<dist_mean/2:
				index = np.abs(dist).argmin()
				# recompose_voxel_emissivity[i_r,i_z] = guess[index]
				# recompose_voxel_emissivity[i_r,i_z] = (x_optimal-guess)[index]
				# recompose_voxel_emissivity[i_r,i_z] = (x_optimal2-x_optimal3)[index]
				recompose_voxel_emissivity[i_r,i_z] = x_optimal[index]
				# recompose_voxel_emissivity[i_r,i_z] = likelihood_emissivity_laplacian[index]
	recompose_voxel_emissivity *= 4*np.pi	# this exist because the sensitivity matrix is built with 1W/str/m^3/ x nm emitters while I use 1W as reference, so I need to multiply the results by 4pi

else:
	pass

temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
xpoint_r = efit_reconstruction.lower_xpoint_r[temp]
xpoint_z = efit_reconstruction.lower_xpoint_z[temp]
z_,r_ = np.meshgrid(np.unique(voxels_centre[:,1]),np.unique(voxels_centre[:,0]))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[z_>xpoint_z] = 0
temp[r_<xpoint_r] = 0
outer_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[z_>xpoint_z] = 0
temp[r_>xpoint_r] = 0
inner_leg_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[z_<xpoint_z] = 0
temp[z_>0] = 0
core_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))
temp = cp.deepcopy(recompose_voxel_emissivity)
temp[((z_-xpoint_z)**2+(r_-xpoint_r)**2)**0.5>0.10] = 0
x_point_tot_rad_power = np.nansum(temp*2*np.pi*r_*((grid_resolution*0.01)**2))

if False:	# just for visualisation

	plt.figure(figsize=(12,13))
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
	plt.imshow(np.flip(np.flip(np.flip(np.transpose(recompose_voxel_emissivity,(1,0)),axis=1),axis=1),axis=0),'rainbow',extent=[grid_data_masked_crop[:,:,0].min(),grid_data_masked_crop[:,:,0].max(),grid_data_masked_crop[:,:,1].min(),grid_data_masked_crop[:,:,1].max()])
	plt.plot(_MASTU_CORE_GRID_POLYGON[:, 0], _MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	plt.plot(FULL_MASTU_CORE_GRID_POLYGON[:, 0], FULL_MASTU_CORE_GRID_POLYGON[:, 1], 'k')
	temp = np.abs(efit_reconstruction.time-time_full_binned_crop[i_t]).argmin()
	for i in range(len(all_time_sep_r[temp])):
		plt.plot(r_fine[all_time_sep_r[temp][i]],z_fine[all_time_sep_z[temp][i]],'--b')
	plt.plot(efit_reconstruction.lower_xpoint_r[temp],efit_reconstruction.lower_xpoint_z[temp],'xr')
	plt.plot(efit_reconstruction.strikepointR[temp],efit_reconstruction.strikepointZ[temp],'xr')
	plt.colorbar().set_label('emissivity [W/m3]')
	plt.ylim(top=0.5)
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example19.eps')
	plt.pause(0.01)



	temp = x_optimal
	temp[1226]=0
	plt.figure(figsize=(6,12))
	plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,s=100,marker='s',cmap='rainbow')
	plt.pause(0.01)


	plt.figure(figsize=(15,12))
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g\nplasma region offset %.3g, whole foil offset %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate,x_optimal[-2],x_optimal[-1]))
	plt.imshow(foil_power_guess.reshape(powernoback_full_orig[i_t].shape))
	plt.colorbar().set_label('power [W/m2]')
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example20.eps')
	plt.pause(0.01)

	plt.figure(figsize=(15,12))
	plt.title('sigma_emissivity %.3g\nregolarisation_coeff %.3g\nregolarisation_coeff_edge %.3g\nregolarisation_coeff_central_border_Z_derivate %.3g\nregolarisation_coeff_central_column_border_R_derivate %.3g' %(sigma_emissivity,regolarisation_coeff,regolarisation_coeff_edge,regolarisation_coeff_central_border_Z_derivate,regolarisation_coeff_central_column_border_R_derivate))
	plt.imshow(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal_no_plasma).reshape(powernoback_full_orig[i_t].shape))
	plt.colorbar().set_label('power error [W/m2]')
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example21.eps')
	plt.pause(0.01)

	plt.figure()
	plt.imshow(powernoback_full_orig[i_t])
	plt.colorbar().set_label('power [W/m2]')
	plt.title('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example0.eps')
	plt.pause(0.01)

	plt.figure()
	plt.imshow(sigma_powernoback_full[i_t])
	plt.colorbar().set_label('power [W/m2]')
	plt.title('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
	# plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_example0.eps')
	plt.pause(0.01)

	plt.figure()
	# plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=likelihood_emissivity_central_border_Z_derivate,marker='s')
	plt.scatter(np.mean(grid_data_masked_crop,axis=1)[:,0],np.mean(grid_data_masked_crop,axis=1)[:,1],c=x_optimal,marker='s')
	plt.colorbar()
	plt.pause(0.01)

	from cherab.core.math import Interpolate2DLinear
	temp = cp.deepcopy(recompose_voxel_emissivity)
	temp[np.isnan(temp)] = 0
	rad_interp = Interpolate2DLinear(np.unique(voxels_centre[:,0]),np.unique(voxels_centre[:,1]),temp)
	gna_dict = dict([])
	gna_dict['recompose_voxel_emissivity'] = temp
	gna_dict['R'] = np.unique(voxels_centre[:,0])
	gna_dict['Z'] = np.unique(voxels_centre[:,1])
	np.savez_compressed(laser_to_analyse[:-4]+'_test',**gna_dict)

	import numpy as np
	from cherab.core.math import Interpolate2DLinear
	gna_dict = np.load('/home/ffederic/work/irvb/MAST-U/2021-09-28/IRVB-MASTU_shot-45071_test.npz')
	rad_interp = Interpolate2DLinear(gna_dict['R'],gna_dict['Z'],gna_dict['recompose_voxel_emissivity'],extrapolate=True)
	from mubol.phantoms.solps import SOLPSPhantom
	import xarray as xr

	class Foo(SOLPSPhantom):

		def __init__(self):
			# self.prad_interpolator = rad_interp
			data = np.zeros((len(gna_dict['Z']), len(gna_dict['R'])))
			self.sampled_emiss = xr.DataArray(data, coords=[('z', gna_dict['Z']), ('r', gna_dict['R'])])
			super().__init__(prad_interpolator=rad_interp,system='sxdl')

	boh = Foo()
	boh.brightness


	inverted_data.append(recompose_voxel_emissivity)
	inverted_data_likelihood.append(y_opt)
	inverted_data_plasma_region_offset.append(x_optimal[-2])
	inverted_data_homogeneous_offset.append(x_optimal[-1])
	inverted_data_info.append(opt_info)
	fitted_foil_power.append((np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
	foil_power.append(powernoback_full_orig[i_t])
	foil_power_residuals.append(powernoback_full_orig[i_t]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape))
	fit_error.append(np.sum(((powernoback_full_orig[i_t][selected_ROI]-(np.dot(sensitivities_binned_crop,x_optimal[:-2])+x_optimal[-2]*select_foil_region_with_plasma+x_optimal[-1]*selected_ROI_internal).reshape(powernoback_full_orig[i_t].shape)[[selected_ROI]]))**2)**0.5/np.sum(selected_ROI))
	chi_square_all.append(chi_square)
	regolarisation_coeff_all.append(regolarisation_coeff)
	outer_leg_tot_rad_power_all.append(outer_leg_tot_rad_power)
	inner_leg_tot_rad_power_all.append(inner_leg_tot_rad_power)
	core_tot_rad_power_all.append(core_tot_rad_power)
	x_point_tot_rad_power_all.append(x_point_tot_rad_power)
	time_per_iteration.append(tm.time()-time_start)
	for value in points_removed:
		Lcurve_curvature = np.concatenate([Lcurve_curvature[:value],[np.nan],Lcurve_curvature[value:]])
	Lcurve_curvature_all.append(Lcurve_curvature)

	plt.figure(10)
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_evolution.eps')
	plt.close()
	plt.figure(11)
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_L_curve_curvature_evolution.eps')
	plt.close()

	inverted_data = np.array(inverted_data)
	inverted_data_likelihood = -np.array(inverted_data_likelihood)
	inverted_data_plasma_region_offset = np.array(inverted_data_plasma_region_offset)
	inverted_data_homogeneous_offset = np.array(inverted_data_homogeneous_offset)
	fit_error = np.array(fit_error)
	chi_square_all = np.array(chi_square_all)
	regolarisation_coeff_all = np.array(regolarisation_coeff_all)
	outer_leg_tot_rad_power_all = np.array(outer_leg_tot_rad_power_all)
	inner_leg_tot_rad_power_all = np.array(inner_leg_tot_rad_power_all)
	core_tot_rad_power_all = np.array(core_tot_rad_power_all)
	x_point_tot_rad_power_all = np.array(x_point_tot_rad_power_all)
	time_per_iteration = np.array(time_per_iteration)
	fitted_foil_power = np.array(fitted_foil_power)
	foil_power = np.array(foil_power)
	foil_power_residuals = np.array(foil_power_residuals)
	score_x_all = np.array(score_x_all)
	score_y_all = np.array(score_y_all)
	regolarisation_coeff_range_all = np.array(regolarisation_coeff_range_all)
	Lcurve_curvature_all = np.array(Lcurve_curvature_all)

	path_for_plots = path_power_output + '/invertions_log/'+binning_type
	if not os.path.exists(path_for_plots):
		os.makedirs(path_for_plots)



	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(time_per_iteration)],time_per_iteration)
	# plt.semilogy()
	plt.title('time spent per iteration')
	plt.xlabel('time [s]')
	plt.ylabel('time [s]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_time_trace_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(inverted_data_likelihood)],inverted_data_likelihood)
	# plt.semilogy()
	plt.title('Fit log likelihood')
	plt.xlabel('time [s]')
	plt.ylabel('log likelihoog [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_likelihood_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(chi_square_all)],chi_square_all)
	# plt.plot(time_full_binned_crop,np.ones_like(time_full_binned_crop)*target_chi_square,'--k')
	# plt.semilogy()
	if False:
		plt.title('chi square obtained vs requested\nfixed regularisation of '+str(regolarisation_coeff))
	else:
		plt.title('chi square obtained vs requested\nflexible regolarisation coefficient')
	plt.xlabel('time [s]')
	plt.ylabel('chi square [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_chi_square_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(regolarisation_coeff_all)],regolarisation_coeff_all)
	# plt.semilogy()
	plt.title('regolarisation coefficient obtained')
	plt.semilogy()
	plt.xlabel('time [s]')
	plt.ylabel('regolarisation coefficient [au]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_regolarisation_coeff_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(fit_error)],fit_error)
	# plt.semilogy()
	plt.title('Fit error ( sum((image-fit)^2)^0.5/num pixels )')
	plt.xlabel('time [s]')
	plt.ylabel('average fit error [W/m2]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_fit_error_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(inverted_data_plasma_region_offset)],inverted_data_plasma_region_offset,label='plasma region')
	plt.plot(time_full_binned_crop[:len(inverted_data_homogeneous_offset)],inverted_data_homogeneous_offset,label='whole foil')
	plt.title('Offsets to match foil power')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('time [s]')
	plt.ylabel('power density [W/m2]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_offsets_bayesian.eps')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(time_full_binned_crop[:len(outer_leg_tot_rad_power_all)],outer_leg_tot_rad_power_all/1e3,label='outer_leg')
	plt.plot(time_full_binned_crop[:len(inner_leg_tot_rad_power_all)],inner_leg_tot_rad_power_all/1e3,label='inner_leg')
	plt.plot(time_full_binned_crop[:len(core_tot_rad_power_all)],core_tot_rad_power_all/1e3,label='core')
	plt.plot(time_full_binned_crop[:len(x_point_tot_rad_power_all)],x_point_tot_rad_power_all/1e3,label='x_point')
	plt.plot(time_full_binned_crop[:len(outer_leg_tot_rad_power_all)],outer_leg_tot_rad_power_all/1e3+inner_leg_tot_rad_power_all/1e3+core_tot_rad_power_all/1e3,label='tot')
	plt.title('radiated power in the lower half of the machine')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('time [s]')
	plt.ylabel('power [kW]')
	plt.grid()
	plt.savefig(path_power_output + '/'+ str(shot_number)+'_'+binning_type+'_gridres'+str(grid_resolution)+'cm_tot_rad_power_bayesian.eps')
	plt.close()


if False:	# I want to take results from the archive
	inverted_dict = np.load(laser_to_analyse[:-4]+'_inverted_baiesian_test.npz')
	inverted_dict.allow_pickle=True
	inverted_dict = dict(inverted_dict)
	score_x_all = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['score_x_all']
	score_y_all = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['score_y_all']
	Lcurve_curvature_all = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['Lcurve_curvature_all']
	time_full_binned_crop = inverted_dict[str(grid_resolution)].all()[str(shrink_factor_x)][str(shrink_factor_t)]['time_full_binned_crop']

	plt.figure(10)
	plt.title('L-curve evolution\nlight=early, dark=late')
	plt.figure(11)
	plt.title('L-curve curvature evolution\nlight=early, dark=late')
	i_t = 18
	for i_t in range(len(time_full_binned_crop)):
		print('starting t=%.4gms' %(time_full_binned_crop[i_t]*1e3))
		regolarisation_coeff_range = 10**np.linspace(2,-4,num=40)
		regolarisation_coeff_range = np.flip(regolarisation_coeff_range,axis=0)
		score_x = score_x_all[i_t]
		score_y = score_y_all[i_t]

		points_removed = []
		counter_score_x = collections.Counter(score_x)
		counter_score_y = collections.Counter(score_y)
		test = np.logical_and( [value in np.array(list(counter_score_x.items()))[:,0][np.array(list(counter_score_x.items()))[:,1]>1] for value in score_x] , [value in np.array(list(counter_score_y.items()))[:,0][np.array(list(counter_score_y.items()))[:,1]>1] for value in score_y] )
		while np.sum(test)>0:
			i__ = test.argmax()
			print(i__)
			regolarisation_coeff_range = np.concatenate([regolarisation_coeff_range[:i__],regolarisation_coeff_range[i__+1:]])
			# x_optimal_all = np.concatenate([x_optimal_all[:i__],x_optimal_all[i__+1:]])
			recompose_voxel_emissivity_all = np.concatenate([recompose_voxel_emissivity_all[:i__],recompose_voxel_emissivity_all[i__+1:]])
			score_x = np.concatenate([score_x[:i__],score_x[i__+1:]])
			score_y = np.concatenate([score_y[:i__],score_y[i__+1:]])
			counter_score_x = collections.Counter(score_x)
			counter_score_y = collections.Counter(score_y)
			test = np.logical_and( [value in np.array(list(counter_score_x.items()))[:,0][np.array(list(counter_score_x.items()))[:,1]>1] for value in score_x] , [value in np.array(list(counter_score_y.items()))[:,0][np.array(list(counter_score_y.items()))[:,1]>1] for value in score_y] )
			points_removed.append(i__)
		points_removed = np.flip(points_removed,axis=0)

		# plt.figure()
		# plt.plot(score_x,score_y)
		# plt.plot(score_x,score_y,'+')
		# plt.xlabel('||Gm-d||2')
		# plt.ylabel('||Laplacian(m)||2')
		# plt.grid()
		# plt.pause(0.01)


		score_y = np.log(score_y)
		score_x = np.log(score_x)

		score_y_record_rel = (score_y-score_y.min())/(score_y.max()-score_y.min())
		score_x_record_rel = (score_x-score_x.min())/(score_x.max()-score_x.min())


		# plt.figure()
		# plt.plot(regolarisation_coeff_range,score_x_record_rel,label='fit error')
		# plt.plot(regolarisation_coeff_range,score_y_record_rel,label='laplacian error')
		# plt.legend()
		# plt.semilogx()
		# plt.pause(0.01)
		#
		# plt.figure()
		# plt.plot(score_x,score_y)
		# plt.plot(score_x,score_y,'+')
		# plt.xlabel('log ||Gm-d||2')
		# plt.ylabel('log ||Laplacian(m)||2')
		# plt.grid()
		# plt.pause(0.01)

		# if I use a fine regularisation range I need to fit the curvature over more points. this takes care of that.
		# curvature_range was originally = 2
		curvature_range = max(1,int(np.ceil(np.abs(-1.8/np.median(np.diff(np.log10(regolarisation_coeff_range)))-1)/2)))
		print('curvature_range = '+str(curvature_range))


		def distance_spread(coord):
			def int(trash,px,py):
				x = coord[0]
				y = coord[1]
				dist = ((x-px)**2 + (y-py)**2)**0.5
				spread = np.sum((dist-np.mean(dist))**2)
				# print(spread)
				return [spread]*5
			return int

		def distance_spread_and_gradient(coord):
			def int(arg):
				x = coord[0]
				y = coord[1]
				px = arg[0]
				py = arg[1]
				dist = ((x-px)**2 + (y-py)**2)**0.5
				spread = np.sum((dist-np.mean(dist))**2)
				temp = (((x-px)**2 + (y-py)**2)**-0.5)
				derivate = np.array([np.sum(2*(dist-np.mean(dist))*( -0.5*temp*2*(x-px) - np.mean(-0.5*temp*2*(x-px)) )) , np.sum(2*(dist-np.mean(dist))*( -0.5*temp*2*(y-py) - np.mean(-0.5*temp*2*(y-py)) ))])
				return spread,derivate
			return int

		# plt.figure()
		# plt.plot(score_x_record_rel,score_y_record_rel)
		curvature_radious = []
		for ii in range(curvature_range,len(score_y_record_rel)-curvature_range):
			# try:
			# 	guess = centre[0]
			# except:
			print(ii)
			try:
				guess = np.max([score_y_record_rel[ii-curvature_range:ii+curvature_range+1]*10,score_x_record_rel[ii-curvature_range:ii+curvature_range+1]*10],axis=1)

				# bds = [[np.min(score_y_record_rel[ii-2:ii+2+1]),np.min(score_x_record_rel[ii-2:ii+2+1])],[np.inf,np.inf]]
				# centre = curve_fit(distance_spread_and_gradient([score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1]]),[0]*5,[0]*5,p0=guess,bounds = bds,maxfev=1e5,gtol=1e-12,verbose=1)

				# bds = [[np.min(score_y_record_rel[ii-curvature_range:ii+curvature_range+1]),np.inf],[np.min(score_x_record_rel[ii-curvature_range:ii+curvature_range+1]),np.inf]]
				bds = [[score_y_record_rel[ii],np.inf],[score_x_record_rel[ii],np.inf]]
				centre, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(distance_spread_and_gradient([score_x_record_rel[ii-curvature_range:ii+curvature_range+1],score_y_record_rel[ii-curvature_range:ii+curvature_range+1]]), x0=guess, bounds = bds, iprint=0, factr=1e8, pgtol=1e-8)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
				centre = [centre]

				dist = ((score_x_record_rel[ii-curvature_range:ii+curvature_range+1]-centre[0][0])**2 + (score_y_record_rel[ii-curvature_range:ii+curvature_range+1]-centre[0][1])**2)**0.5
				radious = np.mean(dist)
				# plt.plot(score_x_record_rel[ii-2:ii+2+1],score_y_record_rel[ii-2:ii+2+1],'+')
				# # plt.plot(centre[0][0],centre[0][1],'o')
				# # plt.plot(np.linspace(centre[0][0]-radious,centre[0][0]+radious),centre[0][1]+(radious**2-np.linspace(-radious,+radious)**2)**0.5)
				# # plt.plot(np.linspace(centre[0][0]-radious,centre[0][0]+radious),centre[0][1]-(radious**2-np.linspace(-radious,+radious)**2)**0.5)
				# plt.axhline(y=np.min(score_y_record_rel[ii-2:ii+2+1]),linestyle='--')
				# plt.axvline(x=np.min(score_x_record_rel[ii-2:ii+2+1]),linestyle='--')
				# plt.pause(0.01)
			except:
				radious = np.inf
			curvature_radious.append(radious)
		# curvature_radious = [np.max(curvature_radious)]+curvature_radious+[np.max(curvature_radious)]
		Lcurve_curvature = 1/np.array(curvature_radious)

		# plt.figure()
		# plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature)
		# plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,'+')
		# plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o')
		# plt.xlabel('regularisation coeff')
		# plt.ylabel('L-curve turvature')
		# plt.semilogx()
		# plt.pause(0.01)
		#
		# plt.figure()
		# plt.plot(score_x_record_rel,score_y_record_rel)
		# plt.plot(score_x_record_rel,score_y_record_rel,'+')
		# plt.xlabel('log ||Gm-d||2')
		# plt.ylabel('log ||Laplacian(m)||2')
		# plt.grid()
		# plt.plot(score_x_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],score_y_record_rel[curvature_range:-curvature_range][Lcurve_curvature.argmax()],'o')
		# plt.pause(0.01)

		plt.figure(10)
		plt.plot(score_x,score_y,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.plot(score_x,score_y,'+',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.plot(score_x[curvature_range:-curvature_range][Lcurve_curvature.argmax()],score_y[curvature_range:-curvature_range][Lcurve_curvature.argmax()],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.xlabel('log ||Gm-d||2')
		plt.ylabel('log ||Laplacian(m)||2')
		plt.figure(11)
		plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range],Lcurve_curvature,color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.plot(regolarisation_coeff_range[curvature_range:-curvature_range][Lcurve_curvature.argmax()],Lcurve_curvature[Lcurve_curvature.argmax()],'o',color=str(0.9-i_t/(len(time_full_binned_crop)/0.9)))
		plt.semilogx()
		plt.xlabel('regularisation coeff')
		plt.ylabel('L-curve turvature')
	plt.figure(10)
	plt.grid()
	plt.figure(11)
	plt.grid()



	recompose_voxel_emissivity = recompose_voxel_emissivity_all[Lcurve_curvature.argmax()+curvature_range]
	regolarisation_coeff = regolarisation_coeff_range[Lcurve_curvature.argmax()+curvature_range]
	x_optimal = x_optimal_all[Lcurve_curvature.argmax()+curvature_range]
