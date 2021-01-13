# Created 03/12/2018
# Fabio Federici


#this is if working on a pc, use pc printer
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

# #this is if working in batch, use predefined NOT visual printer
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())


#this is for importing all the variables names and which are the files
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_indexing.py").read())
number_cpu_available = 14


# degree of polynomial of choice
n=3
# folder of the parameters path
pathparams='/home/ffederic/work/irvb/2021-01-06_multiple_search_for_parameters'
# reference frames
path_reference_frames=vacuum2
# Lares file I want to read
laser_to_analyse = laser19[0]
# framerate of the IR camera in Hz
framerate=383
# integration time of the camera in ms
int_time=1
#filestype
type='_stat.npy'
# type='csv'

# Load data
pathparams = pathparams+'/'+str(int_time)+'ms'+str(framerate)+'Hz/'+'numcoeff'+str(n)+'/average'
fullpathparams=os.path.join(pathparams,'coeff_polynomial_deg'+str(n-1)+'int_time'+str(int_time)+'ms.npz')
params_dict=np.load(fullpathparams)
params = params_dict['coeff']
errparams = params_dict['errcoeff']

background_timestamps = [(np.mean(np.load(file+'.npz')['time_of_measurement'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
background_temperatures = [(np.mean(np.load(file+'.npz')['SensorTemp_0'])) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
background_images = [(np.load(file+'.npz')['data_time_avg_counts']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
background_images_std = [(np.load(file+'.npz')['data_time_avg_counts_std']) for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]
relevant_background_files = [file for file in path_reference_frames if np.logical_and(np.abs(np.load(file+'.npz')['FrameRate']-framerate)<framerate/100,np.abs(np.load(file+'.npz')['IntegrationTime']/1000-int_time)<int_time/100)]

laser_dict = np.load(laser_to_analyse+'.npz')
laser_counts, laser_digitizer_ID = coleval.separate_data_with_digitizer(laser_dict)
time_of_experiment = laser_dict['time_of_measurement']
mean_time_of_experiment = np.mean(time_of_experiment)
laser_digitizer = laser_dict['digitizer_ID']
time_of_experiment_digitizer_ID, laser_digitizer_ID = coleval.generic_separate_with_digitizer(time_of_experiment,laser_digitizer)
laser_framerate = laser_dict['FrameRate']


# Calculating the background image
temp = np.sort(np.abs(background_timestamps-mean_time_of_experiment))[1]
ref = np.arange(len(background_timestamps))[np.abs(background_timestamps-mean_time_of_experiment)<=temp]
reference_background = (background_images[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_images[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
reference_background_std = ((background_images_std[ref[0]]**2)*(background_timestamps[ref[1]]-mean_time_of_experiment) + (background_images_std[ref[1]]**2)*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
reference_background_flat = np.mean(reference_background,axis=(1,2))
reference_background_camera_temperature = (background_temperatures[ref[0]]*(background_timestamps[ref[1]]-mean_time_of_experiment) + background_temperatures[ref[1]]*(mean_time_of_experiment-background_timestamps[ref[0]]))/(background_timestamps[ref[1]]-background_timestamps[ref[0]])
reference_background_full_1,background_digitizer_ID = coleval.separate_data_with_digitizer(np.load(relevant_background_files[ref[0]]+'.npz'))
reference_background_full_2,background_digitizer_ID = coleval.separate_data_with_digitizer(np.load(relevant_background_files[ref[1]]+'.npz'))
flag_1 = [coleval.find_dead_pixels([data]) for data in reference_background_full_1]
flag_2 = [coleval.find_dead_pixels([data]) for data in reference_background_full_2]
bad_pixels_flag = np.add(flag_1,flag_2)



if False:	# I tried here to have an automatic recognition of the foil position and shape. I can't have a good enough contrast,m so I abandon this route
	import cv2
	c=np.load(vacuum2[0]+'.npz')['data_time_avg_counts'][0]
	c-=c.min()
	c = c/c.max() * 255
	plt.figure();plt.imshow(c);plt.colorbar(),plt.pause(0.01)
	canny=cv2.Canny(np.uint8(c),20+c.min(),120+c.min())

	# cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	#
	# for c in cnts:
	#	 cv2.drawContours(image,[c], 0, (0,255,0), 3)


	# d=cv2.Canny(np.uint8(c),30+c.min(),180+c.min())
	# plt.figure();plt.imshow(d);plt.colorbar()
	plt.figure();plt.imshow(canny);plt.colorbar(),plt.pause(0.01)

	# contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	# cntrRect = []
	# for i in contours:
	# 		epsilon = 0.05*cv2.arcLength(i,True)
	# 		approx = cv2.approxPolyDP(i,epsilon,True)
	# 		# if len(approx) == 4:
	# 			# cv2.drawContours(roi,cntrRect,-1,(0,255,0),2)
	# 			# cv2.imshow('Roi Rect ONLY',roi)
	# 		cntrRect.append(approx)
	# 		plt.plot(approx[:,0],'--k',linewidth=4)

	e=cv2.HoughLinesP(canny, 1, np.pi/180, 70, np.array([]), 40, 10)
	for value in e:
		plt.plot(value[0][0::2],value[0][1::2],'--k',linewidth=4)
	plt.pause(0.01)
else:	# I'm going to use the reference frames for foil position
	testrot=reference_background[0]
	rotangle=-1.5 #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter=[160,133]
	foilhorizw=0.09
	foilvertw=0.07
	foilhorizwpixel=240
	foilvertwpixel=np.int((foilhorizwpixel*foilvertw)//foilhorizw)
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype('int')
	foilyint=(np.rint(foily)).astype('int')
	plt.figure()
	plt.imshow(testrot,'rainbow',origin='lower')
	# plt.imshow(testrot,'rainbow',vmin=26., vmax=27.,origin='lower')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.grid()
	plt.pause(0.01)

	plt.figure()
	plt.title('Foil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testrot), vmax=np.max(testrot))
	plt.colorbar().set_label('counts [au]')
	plt.plot(foilxint,foilyint,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.figure()
	testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	plt.imshow(testrotback,'rainbow',origin='lower')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.clim(vmin=np.min(testrot), vmax=np.max(testrot)) #this set the color limits
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[np.int(foily[0]*precisionincrease),np.int(foilx[0]*precisionincrease)]=3
	dummy[np.int(foily[1]*precisionincrease),np.int(foilx[1]*precisionincrease)]=4
	dummy[np.int(foily[2]*precisionincrease),np.int(foilx[2]*precisionincrease)]=5
	dummy[np.int(foily[3]*precisionincrease),np.int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype('int')
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype('int')
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype('int')
	plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	plt.plot(foilxrot,foilyrot,'r')
	plt.title('Foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	plt.colorbar().set_label('counts [au]')
	plt.pause(0.01)

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw


# before of the temperature conversion I'm better to remove the oscillation
# I try to use all the length of the record at first
laser_counts_filtered = [coleval.clear_oscillation_central2([counts],laser_framerate/len(laser_digitizer_ID),oscillation_search_window_end=(len(counts)-1)/(laser_framerate/len(laser_digitizer_ID)),plot_conparison=False)[0] for counts in laser_counts]


from uncertainties import correlated_values,ufloat
from uncertainties.unumpy import nominal_values,std_devs,uarray
def function_a(arg):
	out = np.sum(np.power(np.array([arg[0].tolist()]*arg[2]).T,np.arange(arg[2]-1,-1,-1))*arg[1],axis=1)
	out1 = nominal_values(out)
	out2 = std_devs(out)
	return [out1,out2]

def function_b(arg):
	out = np.sum(np.power(np.array([arg[0].tolist()]*arg[2]).T,np.arange(arg[2]-1,-1,-1))*arg[1],axis=0)
	out1 = nominal_values(out)
	out2 = std_devs(out)
	return [out1,out2]

import time as tm
import concurrent.futures as cf
laser_temperature = []
laser_temperature_std = []
with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
	# executor = cf.ProcessPoolExecutor()#max_workers=number_cpu_available)
	for i in range(len(laser_digitizer_ID)):
		laser_counts_temp = np.array(laser_counts_filtered[i])
		temp1 = []
		temp2 = []
		for j in range(laser_counts_temp.shape[1]):
			start_time = tm.time()

			arg = []
			for k in range(laser_counts_temp.shape[2]):
				arg.append([laser_counts_temp[:,j,k]-ufloat(reference_background[i,j,k],reference_background_std[i,j,k])+reference_background_flat[i],correlated_values(params[i,j,k],errparams[i,j,k]),n])
			print(str(j) + ' , %.3gs' %(tm.time()-start_time))
			out = list(executor.map(function_a,arg))

			# print(str(j) + ' , %.3gs' %(tm.time()-start_time))
			# out = []
			# for k in range(laser_counts_temp.shape[2]):
			# 	out.append(function_a([laser_counts_temp[:,j,k],correlated_values(params[i,j,k],errparams[i,j,k]),n]))

			print(str(j) + ' , %.3gs' %(tm.time()-start_time))
			temp1.append([x for x,y in out])
			temp2.append([y for x,y in out])
			print(str(j) + ' , %.3gs' %(tm.time()-start_time))
		laser_temperature.append(np.transpose(temp1,(2,0,1)))
		laser_temperature_std.append(np.transpose(temp2,(2,0,1)))

reference_background_temperature = []
reference_background_temperature_std = []
with cf.ProcessPoolExecutor(max_workers=number_cpu_available) as executor:
	# executor = cf.ProcessPoolExecutor()#max_workers=number_cpu_available)
	for i in range(len(laser_digitizer_ID)):
		reference_background_temp = np.array(reference_background[i])
		reference_background_std_temp = np.array(reference_background_std[i])
		temp1 = []
		temp2 = []
		for j in range(reference_background_temp.shape[0]):
			start_time = tm.time()

			arg = []
			for k in range(reference_background_temp.shape[1]):
				arg.append([uarray(reference_background_temp[j,k],reference_background_std_temp[j,k]),correlated_values(params[i,j,k],errparams[i,j,k]),n])
			print(str(j) + ' , %.3gs' %(tm.time()-start_time))
			out = list(executor.map(function_b,arg))

			# print(str(j) + ' , %.3gs' %(tm.time()-start_time))
			# out = []
			# for k in range(laser_counts_temp.shape[2]):
			# 	out.append(function_a([laser_counts_temp[:,j,k],correlated_values(params[i,j,k],errparams[i,j,k]),n]))

			print(str(j) + ' , %.3gs' %(tm.time()-start_time))
			temp1.append([x for x,y in out])
			temp2.append([y for x,y in out])
			print(str(j) + ' , %.3gs' %(tm.time()-start_time))
		reference_background_temperature.append(np.array(temp1))
		reference_background_temperature_std.append(np.array(temp2))

if False:	# I subtract the background to the counts, so this bit is not necessary
	# I want to subtract the reference temperature and add back 300K. in this way I whould remove all the disuniformities.
	# I should add a reference real temperature of the plate at the time of the reference frame recording, but I don't have it
	laser_temperature_relative = [(laser_temperature[i]-reference_background_temperature[i]+300) for i in np.arange(len(laser_digitizer_ID))]
	laser_temperature_std_relative = [(laser_temperature_std[i]**2+reference_background_temperature_std[i]**2)**0.5 for i in np.arange(len(laser_digitizer_ID))]
	# NO!! doing it like that the uncertainty is huge, I need to do it before to convert to temperature
	# I subtract the background to the counts and add back the average of the background counts



# I can replace the dead pixels even after the transformation to temperature, given the flag comes from the background data
laser_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature)]
laser_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,laser_temperature_std)]
reference_background_temperature_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,reference_background_temperature)]
reference_background_temperature_std_no_dead_pixels = [coleval.replace_dead_pixels([data],flag)[0] for flag,data in zip(bad_pixels_flag,reference_background_temperature_std)]
del laser_temperature,laser_temperature_std,reference_background_temperature,reference_background_temperature_std

# I need to put together the data from the two digitizers
laser_temperature_full = [(laser_temperature_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_temperature_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)]
laser_temperature_std_full = [(laser_temperature_std_no_dead_pixels[0][np.abs(time_of_experiment_digitizer_ID[0]-time).argmin()]) if ID==laser_digitizer_ID[0] else (laser_temperature_std_no_dead_pixels[1][np.abs(time_of_experiment_digitizer_ID[1]-time).argmin()]) for time,ID in zip(time_of_experiment,laser_digitizer)]
del laser_temperature_no_dead_pixels,laser_temperature_std_no_dead_pixels

# rotation
laser_temperature_rot=rotate(laser_temperature_full,foilrotdeg,axes=(-1,-2))
laser_temperature_crop=laser_temperature_rot[:,foildw:foilup,foillx:foilrx];del laser_temperature_rot
laser_temperature_std_rot=rotate(laser_temperature_std_full,foilrotdeg,axes=(-1,-2))
laser_temperature_std_crop=laser_temperature_std_rot[:,foildw:foilup,foillx:foilrx];del laser_temperature_std_rot
reference_background_temperature_rot=rotate(reference_background_temperature,foilrotdeg,axes=(-1,-2))
reference_background_temperature_crop=reference_background_temperature_rot[:,foildw:foilup,foillx:foilrx];del reference_background_temperature_rot
reference_background_temperature_std_rot=rotate(reference_background_temperature_std,foilrotdeg,axes=(-1,-2))
reference_background_temperature_std_crop=reference_background_temperature_std_rot[:,foildw:foilup,foillx:foilrx];del reference_background_temperature_std_rot


plt.figure()
plt.imshow(laser_temperature_crop[0],'rainbow',origin='lower')
plt.colorbar().set_label('Temp [°C]')
plt.title('Only foil in frame '+str(0)+' in '+laser_to_analyse+'\n foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=10)
plt.xlabel('Horizontal axis [pixles]')
plt.ylabel('Vertical axis [pixles]')
plt.pause(0.01)

# FOIL PROPERTY ADJUSTMENT
if True:	# spatially resolved foil properties from Japanese producer
	foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
	conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
	reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
elif False:	# homogeneous foil properties from foil experiments
	foilemissivityscaled=1*np.ones((foilvertwpixel,foilhorizwpixel))
	foilthicknessscaled=(2.5/1000000)*np.ones((foilvertwpixel,foilhorizwpixel))
	conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel,foilhorizwpixel))
	reciprdiffusivityscaled=(1/Ptthermaldiffusivity)*np.ones((foilvertwpixel,foilhorizwpixel))

plt.figure()
plt.title('Foil emissivity scaled to camera pixels')
plt.imshow(foilemissivityscaled,'rainbow',origin='lower')
plt.xlabel('Foil reference axis [pixles]')
plt.ylabel('Foil reference axis [pixles]')
plt.colorbar().set_label('Emissivity [adimensional]')

plt.figure()
plt.title('Foil thickness scaled to camera pixels')
plt.imshow(1000000*foilthicknessscaled,'rainbow',origin='lower')
plt.xlabel('Foil reference axis [pixles]')
plt.ylabel('Foil reference axis [pixles]')
plt.colorbar().set_label('Thickness [micrometer]')
plt.pause(0.01)


# basetemp=np.mean(datatempcrop[0,frame-7:frame+7,1:-1,1:-1],axis=0)
dt=1/laser_framerate
dx=foilhorizw/(foilhorizwpixel-1)
dTdt=np.divide(laser_temperature_crop[2:,1:-1,1:-1]-laser_temperature_crop[:-2,1:-1,1:-1],2*dt)
dTdt_std=np.divide((laser_temperature_std_crop[2:,1:-1,1:-1]**2 + laser_temperature_std_crop[:-2,1:-1,1:-1]**2)**0.5,2*dt)
d2Tdx2=np.divide(laser_temperature_crop[1:-1,1:-1,2:]-np.multiply(2,laser_temperature_crop[1:-1,1:-1,1:-1])+laser_temperature_crop[1:-1,1:-1,:-2],dx**2)
d2Tdx2_std=np.divide((laser_temperature_std_crop[1:-1,1:-1,2:]**2+np.multiply(2,laser_temperature_std_crop[1:-1,1:-1,1:-1])**2+laser_temperature_std_crop[1:-1,1:-1,:-2]**2)**0.5,dx**2)
d2Tdy2=np.divide(laser_temperature_crop[1:-1,2:,1:-1]-np.multiply(2,laser_temperature_crop[1:-1,1:-1,1:-1])+laser_temperature_crop[1:-1,:-2,1:-1],dx**2)
d2Tdy2_std=np.divide((laser_temperature_std_crop[1:-1,2:,1:-1]**2+np.multiply(2,laser_temperature_std_crop[1:-1,1:-1,1:-1])**2+laser_temperature_std_crop[1:-1,:-2,1:-1]**2)**0.5,dx**2)
d2Tdxy=np.add(d2Tdx2,d2Tdy2)
d2Tdxy_std=np.add(d2Tdx2_std**2,d2Tdy2_std**2)**0.5
negd2Tdxy=np.multiply(-1,d2Tdxy)
negd2Tdxy_std=d2Tdxy_std
T4=np.power(laser_temperature_crop[1:-1,1:-1,1:-1],4)
T4_std=np.power(laser_temperature_crop[1:-1,1:-1,1:-1]*4,3/2)*laser_temperature_std_crop[1:-1,1:-1,1:-1]
T04=np.power(np.mean(reference_background_temperature_crop)*np.ones_like(laser_temperature_crop[1:-1,1:-1,1:-1]),4)
T04_std=np.power(np.mean(reference_background_temperature_crop)*np.ones_like(laser_temperature_crop[1:-1,1:-1,1:-1])*4,3/2)*np.std(reference_background_temperature_crop)



ktf=np.multiply(conductivityscaled,foilthicknessscaled)
BBrad = 2*sigmaSB*foilemissivityscaled*(T4-T04)
BBrad_std = 2*sigmaSB*foilemissivityscaled*((T4_std**2+T04_std**2)**0.5)
diffusion = ktf*negd2Tdxy
diffusion_std = ktf*negd2Tdxy_std
timevariation = ktf*reciprdiffusivityscaled*dTdt
timevariation_std = ktf*reciprdiffusivityscaled*dTdt_std
del dTdt,dTdt_std,d2Tdx2,d2Tdx2_std,d2Tdy2,d2Tdy2_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4,T4_std,T04,T04_std

powernoback = diffusion + timevariation + BBrad
powernoback_std = diffusion_std + timevariation_std + BBrad_std

ani = coleval.movie_from_data(np.array([powernoback]), framerate, integration=int_time,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]')
plt.show()



#
