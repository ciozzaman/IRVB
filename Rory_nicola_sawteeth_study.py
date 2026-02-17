


import pyuda
client=pyuda.Client()
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve,median_filter,generic_filter
from scipy.signal import find_peaks, peak_prominences as get_proms

if False:
	shotnumber = 51512
	time_limits=[0.45,0.8]
elif True:
	shotnumber = 51375
	time_limits=[0.45,0.8]
elif True:
	shotnumber = 51785
	time_limits=[0.3,0.67]
brightness_res_bolo = client.get('/abm/sxdl/brightness', '/projects/codes/MAST-U/ABM/'+str(shotnumber)+'/abm_jlovell_01/abm0'+str(shotnumber)+'.nc').data.T
good_res_bolo = client.get('/abm/sxdl/good', '/projects/codes/MAST-U/ABM/'+str(shotnumber)+'/abm_jlovell_01/abm0'+str(shotnumber)+'.nc').data
time_res_bolo = client.get('/abm/sxdl/time', '/projects/codes/MAST-U/ABM/'+str(shotnumber)+'/abm_jlovell_01/abm0'+str(shotnumber)+'.nc').data
channel_res_bolo = client.get('/abm/sxdl/channel', '/projects/codes/MAST-U/ABM/'+str(shotnumber)+'/abm_jlovell_01/abm0'+str(shotnumber)+'.nc').data
time_resolution = np.median(np.diff(time_res_bolo))
sxdl_prad_res_bolo = client.get('/abm/sxdl/prad', '/projects/codes/MAST-U/ABM/'+str(shotnumber)+'/abm_jlovell_01/abm0'+str(shotnumber)+'.nc').data.T


real_good_channels = [5,6,8,11,13,14,15]
good_data = brightness_res_bolo[[val in real_good_channels for val in channel_res_bolo]]
# plt.figure()
# plt.imshow(to_print)

# plt.figure()
# plt.title(str(shotnumber))
# for i in range(len(channel_res_bolo)):
# 	if channel_res_bolo[i] in real_good_channels:
# 		plt.plot(time_res_bolo,brightness_res_bolo[i],label=str(channel_res_bolo[i]))

good_data_integral = generic_filter(good_data,np.sum,size=[1,round(0.002/time_resolution)])/round(0.002/time_resolution)
sxdl_prad_res_bolo_integral = generic_filter(sxdl_prad_res_bolo,np.sum,size=[round(0.002/time_resolution)])/round(0.002/time_resolution)
peaks_loc = [find_peaks(val,distance=round(0.013/time_resolution))[0] for val in good_data_integral]
peaks_loc = [val[np.logical_and(time_res_bolo[val]>time_limits[0],time_res_bolo[val]<time_limits[1])] for val in peaks_loc]
# plt.figure()
# plt.title(str(shotnumber))
# for i in range(len(real_good_channels)):
# 	# if channel_res_bolo[i] in real_good_channels:
# 	a=plt.plot(time_res_bolo,good_data_integral[i],label=str(real_good_channels[i]))
# 	for i_ in peaks_loc[i]:
# 		plt.axvline(x=time_res_bolo[i_],color=a[0].get_color(),linestyle='--')
# plt.legend()

temp = []
for i in range(len(real_good_channels)):
	temp.append(good_data_integral[i][peaks_loc[i]])

peaks_loc = np.array(peaks_loc)
peaks_loc = np.median(peaks_loc,axis=0).astype(int)
temp2 = sxdl_prad_res_bolo_integral[peaks_loc] - (sxdl_prad_res_bolo_integral[peaks_loc-round(0.002*2/time_resolution)] + sxdl_prad_res_bolo_integral[peaks_loc+round(0.002*2/time_resolution)])/2

# plt.figure(figsize=(10,4))
# plt.title(str(shotnumber))
# plt.imshow(temp,'rainbow')
# plt.colorbar().set_label('brightness integrated over 2ms')
# plt.yticks(np.arange(len(real_good_channels)),labels=[str(val) for val in real_good_channels])
# plt.xticks(np.arange(len(peaks_loc)),labels=[str(round(val*1000)) for val in time_res_bolo[peaks_loc]])
# plt.xlabel('time of sawteeth [ms]')
# plt.ylabel('SXDL channel [au]')

plt.figure()
plt.plot(time_res_bolo[peaks_loc]*1000,temp2,'+',label=str(shotnumber))
plt.ylabel('SXDL radiated power integrated over 2ms [W?]\nnote: background removed')
plt.xlabel('time of sawteeth [ms]')
plt.legend()

plt.figure()
plt.plot(time_res_bolo,sxdl_prad_res_bolo)
plt.plot(time_res_bolo,sxdl_prad_res_bolo_integral)




###
