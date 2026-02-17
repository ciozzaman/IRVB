# 2025/07-31 code to read the IR data and plot / safe relevant data


ISP_range = 0.05	# m
OSP_range = 0.2	# m
import netCDF4
try:
	additional_file = netCDF4.Dataset('/common/uda-scratch/ssilburn/ris_for_fabio/ais0'+str(shotnumber)+'.nc')
except:
	try:
		additional_file = netCDF4.Dataset('/common/uda-scratch/ssilburn/ais0'+str(shotnumber)+'.nc')
	except:
		additional_file = None

temp_add = []
temp_add_s = []
temp_add_surfaces = []
if additional_file!= None:
	available = list(additional_file['ais'].groups)
	for surface in available:
		try:
			temp_add = np.concatenate([temp_add,additional_file['ais'][surface]['heatflux'][:].data],axis=1)
			temp_add_s = np.concatenate([temp_add_s,additional_file['ais'][surface]['s_global'][:].data])
		except:
			temp_add = additional_file['ais'][surface]['heatflux'][:].data
			temp_add_s = additional_file['ais'][surface]['s_global'][:].data
		print(surface+' read')
		temp_add_surfaces.append(surface)
		time_additional_file =additional_file['ais'][surface]['t'][:].data

# normal_signal_list = ['T2LT3L_STD','T4L_STD1','T4L_STD2','T5L_STD','T2UT3U_STD','T4U_STD1','T4U_STD2','T5U_STD']
normal_signal_list = ['T2LT3L_STD','T4L_STD2','T5L_STD','T2UT3U_STD','T4U_STD1','T5U_STD']	# T4U_STD2 is fully included into the region of T4U_STD1, T4L_STD1 is fully included into the region of T4L_STD2 (strange, from Scott's presentation it does not seem like, but it is for 52355)
import pyuda
client = pyuda.Client()
temp1 = []
temp1_s = []
temp1_surfaces = []
temp2 = []
temp2_s = []
temp2_surfaces = []
for surface in normal_signal_list:
	try:
		if surface.find('L')!=-1:
			data = client.get('/AIT/'+surface+'/HEATFLUX',shotnumber)
			data_s = client.get('/AIT/'+surface+'/S_GLOBAL',shotnumber)
			try:
				temp1 = np.concatenate([temp1,data.data],axis=1)
				temp1_s = np.concatenate([temp1_s,data_s.data])
			except:
				temp1 = data.data
				temp1_s = data_s.data
			temp1_surfaces.append(surface)
			time_normal1 = data.time.data
		else:
			data = client.get('/AIV/'+surface+'/HEATFLUX',shotnumber)
			data_s = client.get('/AIV/'+surface+'/S_GLOBAL',shotnumber)
			try:
				temp2 = np.concatenate([temp2,data.data],axis=1)
				temp2_s = np.concatenate([temp2_s,data_s.data])
			except:
				temp2 = data.data
				temp2_s = data_s.data
			temp2_surfaces.append(surface)
			time_normal2 = data.time.data
		print(surface+' read')
	except:
		pass
coleval.reset_connection(client)
del client

try:	# what happens if there is only 1 of the 2 cameras running?
	if time_normal1[-1]>=time_normal2[-1]:
		time_normal = cp.deepcopy(time_normal1)
	else:
		time_normal = cp.deepcopy(time_normal2)
except:
	try:
		try:
			print('/AIV/' + ' signal missing')
			time_normal = cp.deepcopy(time_normal1)
		except:
			print('/AIT/' + ' signal missing')
			time_normal = cp.deepcopy(time_normal2)
	except:
		print('/AIT/ and /AIT/' + ' signal missing')
		time_normal = cp.deepcopy(time_additional_file)

# aligning all the time axis
try:
	temp_add = np.array(temp_add)
	temp_add = np.array([np.interp(time_normal,time_additional_file,temp_add[:,i],left=0,right=0) for i in range(np.shape(temp_add)[1])]).T
except:
	temp_add = [[]]
try:
	temp1 = np.array(temp1)
	temp1 = np.array([np.interp(time_normal,time_normal1,temp1[:,i],left=0,right=0) for i in range(np.shape(temp1)[1])]).T
except:
	temp1 = [[]]
try:
	temp2 = np.array(temp2)
	temp2 = np.array([np.interp(time_normal,time_normal2,temp2[:,i],left=0,right=0) for i in range(np.shape(temp2)[1])]).T
except:
	temp2 = [[]]

# merging all data sources together
try:
	temp = np.concatenate([temp1,temp2],axis=1)
	temp_s = np.concatenate([temp1_s,temp2_s])
	temp_surfaces = np.concatenate([temp1_surfaces,temp2_surfaces])
except:
	if len(temp1)>1:
		temp = temp1
		temp_s = temp1_s
		temp_surfaces = temp1_surfaces
	elif len(temp2)>1:
		temp = temp2
		temp_s = temp2_s
		temp_surfaces = temp2_surfaces

if len(temp_add)>1:
	try:
		temp = np.concatenate([temp_add,temp],axis=1)
		temp_s = np.concatenate([temp_add_s,temp_s])
		temp_surfaces = np.concatenate([temp_add_surfaces,temp_surfaces])
	except:
		temp = temp_add
		temp_s = temp_add_s
		temp_surfaces = temp_add_surfaces


order = np.argsort(temp_s)
temp = np.array(temp)[:,order]
temp_s = temp_s[order]

sys.path.append("/home/ffederic/esm-release-v2.1.0")
from esm import read_data
efit_data2 = read_data.get_efit_data(shotnumber)

from esm.calculation import run_esm,limiter_s0
from esm.calc_flux_expansion import convert_RZ_to_s
esm_dataset=run_esm(shotnumber)
limiter,s0 = limiter_s0(shotnumber)
LISP_s = convert_RZ_to_s(limiter,s0,efit_data2['xpl_isp_r'].data,efit_data2['xpl_isp_z'].data)
UISP_s = convert_RZ_to_s(limiter,s0,efit_data2['xpu_isp_r'].data,efit_data2['xpu_isp_z'].data)
LOSP_s = convert_RZ_to_s(limiter,s0,efit_data2['xpl_osp_r'].data,efit_data2['xpl_osp_z'].data)
UOSP_s = convert_RZ_to_s(limiter,s0,efit_data2['xpu_osp_r'].data,efit_data2['xpu_osp_z'].data)
SP_time = esm_dataset['/esm/fluxexp/full']['cos_poloidal_angle_vertical'].time.data

LISP_s = np.interp(time_normal,SP_time,LISP_s,left=0,right=0)
UISP_s = np.interp(time_normal,SP_time,UISP_s,left=0,right=0)
LOSP_s = np.interp(time_normal,SP_time,LOSP_s,left=0,right=0)
UOSP_s = np.interp(time_normal,SP_time,UOSP_s,left=0,right=0)

# Create meshgrid from coordinates
X, Y = np.meshgrid(time_normal,temp_s, indexing="ij")

fig = plt.figure(figsize=(8, 11))
fig.suptitle('shot ' + laser_to_analyse[-9:-4]+' '+scenario+'\nthermography reading with tiles:\n'+str(temp_surfaces))
from matplotlib import gridspec
gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.4)
ax1 = fig.add_subplot(gs[0:2, 0], sharex=None)
pcm = ax1.pcolormesh(X, Y, temp, shading='auto', cmap='rainbow',norm=LogNorm(vmax =np.nanmax(temp)/2, vmin=np.nanmax(temp)*1e-4))
ax1.plot(time_normal,LISP_s,'--k',alpha=0.2)
ax1.plot(time_normal,LISP_s+ISP_range,'--k',alpha=0.1)
ax1.plot(time_normal,LISP_s-ISP_range,'--k',alpha=0.1)
ax1.plot(time_normal,UISP_s,'--k',alpha=0.2)
ax1.plot(time_normal,UISP_s+ISP_range,'--k',alpha=0.1)
ax1.plot(time_normal,UISP_s-ISP_range,'--k',alpha=0.1)
ax1.plot(time_normal,LOSP_s,'--k',alpha=0.2)
ax1.plot(time_normal,LOSP_s+OSP_range,'--k',alpha=0.1)
ax1.plot(time_normal,LOSP_s-OSP_range,'--k',alpha=0.1)
ax1.plot(time_normal,UOSP_s,'--k',alpha=0.2)
ax1.plot(time_normal,UOSP_s+OSP_range,'--k',alpha=0.1)
ax1.plot(time_normal,UOSP_s-OSP_range,'--k',alpha=0.1)
fig.colorbar(pcm,ax=ax1,label='heat flux [MW/m2]')
# ax1.set_xlabel('time [s]')
ax1.set_ylabel('s coord [m]')
# plt.savefig(filename_root+filename_root_add+'_IR_data.eps')
# plt.close()

heatflux_LISP = []
heatflux_UISP = []
heatflux_LOSP = []
heatflux_UOSP = []
for i in range(len(time_normal)):
	if LISP_s[i]==0 or not np.isfinite(LISP_s[i]):
		heatflux_LISP.append(0)
		heatflux_UISP.append(0)
		heatflux_LOSP.append(0)
		heatflux_UOSP.append(0)
	else:
		select=np.abs(temp_s-LISP_s[i])<ISP_range
		# heatflux_LISP.append(np.trapz(temp[i][select]-np.interp(temp_s[select],temp_s[select][[0,-1]],temp[i][select][[0,-1]]),x=temp_s[select]))
		try:
			heatflux_LISP.append(np.trapz(temp[i][select]-np.interp(temp_s[select],temp_s[select][[0,-1]], [np.median(temp[i][np.abs(temp_s-(temp_s[select][0]-0.02))<0.03]),np.median(temp[i][np.abs(temp_s-(temp_s[select][-1]+0.02))<0.03])]),x=temp_s[select]))
		except:
			heatflux_LISP.append(0)
		select=np.abs(temp_s-UISP_s[i])<ISP_range
		try:
			heatflux_UISP.append(np.trapz(temp[i][select]-np.interp(temp_s[select],temp_s[select][[0,-1]], [np.median(temp[i][np.abs(temp_s-(temp_s[select][0]-0.02))<0.03]),np.median(temp[i][np.abs(temp_s-(temp_s[select][-1]+0.02))<0.03])]),x=temp_s[select]))
		except:
			heatflux_UISP.append(0)
		select=np.abs(temp_s-LOSP_s[i])<OSP_range
		try:
			heatflux_LOSP.append(np.trapz(temp[i][select]-np.interp(temp_s[select],temp_s[select][[0,-1]], [np.median(temp[i][np.abs(temp_s-(temp_s[select][0]-0.02))<0.03]),np.median(temp[i][np.abs(temp_s-(temp_s[select][-1]+0.02))<0.03])]),x=temp_s[select]))
		except:
			heatflux_LOSP.append(0)
		select=np.abs(temp_s-UOSP_s[i])<OSP_range
		try:
			heatflux_UOSP.append(np.trapz(temp[i][select]-np.interp(temp_s[select],temp_s[select][[0,-1]], [np.median(temp[i][np.abs(temp_s-(temp_s[select][0]-0.02))<0.03]),np.median(temp[i][np.abs(temp_s-(temp_s[select][-1]+0.02))<0.03])]),x=temp_s[select]))
		except:
			heatflux_UOSP.append(0)

heatflux_LISP = np.array(heatflux_LISP)
heatflux_UISP = np.array(heatflux_UISP)
heatflux_LOSP = np.array(heatflux_LOSP)
heatflux_UOSP = np.array(heatflux_UOSP)

ax2 = fig.add_subplot(gs[2, 0], sharex=None)
ax2.plot(time_normal,heatflux_LISP,label='LISP(+/-'+str(ISP_range*100)+'cm)')
ax2.plot(time_normal,heatflux_UISP,label='UISP(+/-'+str(ISP_range*100)+'cm)')
ax2.legend(loc='best', fontsize='xx-small')
ax2.grid()
ax2.set_ylim(top = max(median_filter(heatflux_LISP,size=[20]).max(),median_filter(heatflux_UISP,size=[20]).max()),bottom=min(median_filter(heatflux_LISP,size=[20]).min(),median_filter(heatflux_UISP,size=[20]).min()))
ax2.tick_params(labelbottom=False)  # Hide x labels
ax2.set_ylabel('heat flux [MW]', fontsize='xx-small')
ax2.set_ylim([None, 0.01])

ax3 = fig.add_subplot(gs[3, 0], sharex=ax2)
ax3.plot(time_normal,heatflux_LOSP,label='LOSP(+/-'+str(OSP_range*100)+'cm)')
ax3.plot(time_normal,heatflux_UOSP,label='UOSP(+/-'+str(OSP_range*100)+'cm)')
ax3.legend(loc='best', fontsize='xx-small')
ax3.grid()
ax3.set_ylim(top = max(median_filter(heatflux_LOSP,size=[20]).max(),median_filter(heatflux_UOSP,size=[20]).max()),bottom=min(median_filter(heatflux_LOSP,size=[20]).min(),median_filter(heatflux_UOSP,size=[20]).min()))
ax3.tick_params(labelbottom=False)  # Hide x labels
ax3.set_ylabel('heat flux [MW]', fontsize='xx-small')

ax4 = fig.add_subplot(gs[4, 0], sharex=ax2)
ax4.plot(time_normal,heatflux_LISP+heatflux_LOSP,label='tot LOW')
ax4.plot(time_normal,heatflux_UISP+heatflux_UOSP,label='tot UP')
ax4.legend(loc='best', fontsize='xx-small')
ax4.grid()
ax4.set_ylim(top = max(median_filter(heatflux_LISP+heatflux_LOSP,size=[20]).max(),median_filter(heatflux_UISP+heatflux_UOSP,size=[20]).max()),bottom=min(median_filter(heatflux_LISP+heatflux_LOSP,size=[20]).min(),median_filter(heatflux_UISP+heatflux_UOSP,size=[20]).min()))
ax4.set_ylabel('heat flux [MW]', fontsize='xx-small')
ax4.set_xlabel('time [s]')

plt.savefig(filename_root+filename_root_add+'_IR_data.png')
plt.close()


multi_instrument_dict['thermography'] = dict([])
multi_instrument_dict['thermography']['raw data'] = temp
multi_instrument_dict['thermography']['global s'] = temp_s
multi_instrument_dict['thermography']['time'] = time_normal
multi_instrument_dict['thermography']['heat flux LISP'] = heatflux_LISP
multi_instrument_dict['thermography']['heat flux UISP'] = heatflux_UISP
multi_instrument_dict['thermography']['heat flux LOSP'] = heatflux_LOSP
multi_instrument_dict['thermography']['heat flux UOSP'] = heatflux_UOSP
print('thermography analysis done')

###
