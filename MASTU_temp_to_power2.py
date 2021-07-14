from PIL import Image
MASTU_wireframe = Image.open("/home/ffederic/work/irvb/MAST-U/Calcam_theoretical_view.png")
MASTU_wireframe_resize = np.array(np.asarray(MASTU_wireframe)[:,:,3].tolist()).astype(np.float64)
MASTU_wireframe_resize[MASTU_wireframe_resize>0] = 1
MASTU_wireframe_resize = np.flip(MASTU_wireframe_resize,axis=0)
masked = np.ma.masked_where(MASTU_wireframe_resize == 0, MASTU_wireframe_resize)

foil_position_dict = dict([('angle',0.5),('foilcenter',[158,136]),('foilhorizwpixel',241)])	# fixed orientation, for now, this is from 2021-06-04/44168
flat_foil_properties = dict([])
flat_foil_properties['thickness'] = 2.093616658223934e-06
flat_foil_properties['emissivity'] = 1
flat_foil_properties['diffusivity'] = 1.03*1e-5


print('starting power analysis' + laser_to_analyse)

laser_dict = np.load(laser_to_analyse[:-4]+'.npz')
laser_dict.allow_pickle=True
# full_saved_file_dict = dict(laser_dict)

time_full = laser_dict['only_pulse_data'].all()['time_full']
laser_framerate = 1/np.mean(np.sort(np.diff(time_full))[2:-2])
laser_int_time = laser_dict['IntegrationTime']

saved_file_dict_short = np.load(laser_to_analyse[:-4]+'_short.npz')
saved_file_dict_short.allow_pickle=True
saved_file_dict_short = dict(saved_file_dict_short)

try:
	try:
		foilemissivityscaled = saved_file_dict_short['power_data'].all()['foil_properties']['emissivity']['dfsgdfg']	# I want them all
		foilthicknessscaled = saved_file_dict_short['power_data'].all()['foil_properties']['thickness']
		reciprdiffusivityscaled = saved_file_dict_short['power_data'].all()['foil_properties']['diffusivity']
		nan_ROI_mask = saved_file_dict_short['other'].all()['nan_ROI_mask']
		dx = full_saved_file_dict['power_data'].all()['dx']

		temp_BBrad = laser_dict['power_data'].all()['temp_BBrad']
		temp_BBrad_std = laser_dict['power_data'].all()['temp_BBrad_std']
		temp_diffusion = laser_dict['power_data'].all()['temp_diffusion']
		temp_diffusion_std = laser_dict['power_data'].all()['temp_diffusion_std']
		temp_timevariation = laser_dict['power_data'].all()['temp_timevariation']
		temp_timevariation_std = laser_dict['power_data'].all()['temp_timevariation_std']

		BBrad = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		BBrad_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		diffusion = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		diffusion_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		timevariation = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		timevariation_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)

		powernoback = saved_file_dict_short['power_data'].all()['powernoback']
		powernoback_std = saved_file_dict_short['power_data'].all()['powernoback_std']

		foilhorizw=0.09	# m
		foilvertw=0.07	# m

	except:
		print('missing '+laser_to_analyse[:-4]+'_short.npz'+' file. rigenerated')
		# full_saved_file_dict.allow_pickle=True
		full_saved_file_dict = dict(laser_dict)

		laser_temperature_full = full_saved_file_dict['only_pulse_data'].all()['laser_temperature_full_median'] + full_saved_file_dict['only_pulse_data'].all()['laser_temperature_full_minus_median'].astype(np.float32)
		laser_temperature_std_full = full_saved_file_dict['only_pulse_data'].all()['laser_temperature_std_full_median'] + full_saved_file_dict['only_pulse_data'].all()['laser_temperature_std_full_minus_median'].astype(np.float32)
		laser_temperature_minus_background_full = full_saved_file_dict['only_pulse_data'].all()['laser_temperature_minus_background_full_median'] + full_saved_file_dict['only_pulse_data'].all()['laser_temperature_minus_background_full_minus_median'].astype(np.float32)
		laser_temperature_std_minus_background_full = full_saved_file_dict['only_pulse_data'].all()['laser_temperature_std_minus_background_full_median'] + full_saved_file_dict['only_pulse_data'].all()['laser_temperature_std_minus_background_full_minus_median'].astype(np.float32)
		laser_digitizer_ID = np.unique(full_saved_file_dict['digitizer_ID'])

		reference_background_temperature_no_dead_pixels = []
		reference_background_temperature_std_no_dead_pixels = []
		for i in range(len(laser_digitizer_ID)):
			reference_background_temperature_no_dead_pixels.append(full_saved_file_dict['only_pulse_data'].all()[str(laser_digitizer_ID[i])]['reference_background_temperature_no_dead_pixels'])
			reference_background_temperature_std_no_dead_pixels.append(full_saved_file_dict['only_pulse_data'].all()[str(laser_digitizer_ID[i])]['reference_background_temperature_std_no_dead_pixels'])

		# I'm going to use the reference frames for foil position
		testrot=reference_background_temperature_no_dead_pixels[0]

		rotangle = foil_position_dict['angle'] #in degrees
		foilrot=rotangle*2*np.pi/360
		foilrotdeg=rotangle
		foilcenter = foil_position_dict['foilcenter']
		foilhorizw=0.09	# m
		foilvertw=0.07	# m
		foilhorizwpixel = foil_position_dict['foilhorizwpixel']
		foilvertwpixel=int((foilhorizwpixel*foilvertw)//foilhorizw)
		r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
		a=foilvertwpixel/np.cos(foilrot)
		tgalpha=np.tan(foilrot)
		delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
		foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
		foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
		foilxint=(np.rint(foilx)).astype('int')
		foilyint=(np.rint(foily)).astype('int')
		# rotangle = foil_position_dict['angle'] #in degrees
		# foilrot=rotangle*2*np.pi/360
		# foilrotdeg=rotangle
		# foilcenter = foil_position_dict['foilcenter']
		# foilhorizw=0.09	# m
		# foilvertw=0.07	# m
		# foilhorizwpixel = foil_position_dict['foilhorizwpixel']
		# foilvertwpixel=int((foilhorizwpixel*foilvertw)//foilhorizw)
		# r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
		# a=foilvertwpixel/np.cos(foilrot)
		# tgalpha=np.tan(foilrot)
		# delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
		# foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
		# foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
		# foilxint=(np.rint(foilx)).astype('int')
		# foilyint=(np.rint(foily)).astype('int')

		plt.figure(figsize=(20, 10))
		plt.title('Foil search in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
		plt.imshow(testrot,'rainbow',origin='lower')
		plt.ylabel('Horizontal axis [pixles]')
		plt.xlabel('Vertical axis [pixles]')
		temp = np.sort(testrot[testrot>0])
		plt.clim(vmin=np.nanmean(temp[:max(20,int(len(temp)/20))]), vmax=np.nanmean(temp[-max(20,int(len(temp)/20)):]))
		# plt.clim(vmax=27.1,vmin=26.8)
		# plt.clim(vmin=np.nanmin(testrot[testrot>0]), vmax=np.nanmax(testrot))
		plt.colorbar().set_label('counts [au]')
		plt.plot(foilx,foily,'r')
		plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
		plt.savefig(laser_to_analyse[:-4]+'_foil_fit.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(20, 10))
		temp = np.max(laser_temperature_minus_background_full[:,:,:200],axis=(-1,-2)).argmax()
		plt.title('Foil search in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel\nhottest frame at %.4gsec' %(time_full[temp]))
		plt.imshow(laser_temperature_minus_background_full[temp],'rainbow',origin='lower',vmax=np.max(laser_temperature_minus_background_full[:,:,:200],axis=(-1,-2))[temp])
		plt.ylabel('Horizontal axis [pixles]')
		plt.xlabel('Vertical axis [pixles]')
		plt.clim(vmax=np.max(laser_temperature_minus_background_full[:,:,:200],axis=(-1,-2))[temp])
		plt.colorbar().set_label('temp increase [K]')
		plt.plot(foilx,foily,'r')
		plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
		plt.savefig(laser_to_analyse[:-4]+'_foil_fit_plasma.eps', bbox_inches='tight')
		plt.close('all')
		testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
		precisionincrease=10
		dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
		dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
		dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
		dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
		dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
		dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
		dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
		foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype('int')
		foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype('int')
		foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype('int')
		# plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
		# plt.plot(foilxrot,foilyrot,'r')
		# plt.title('Foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
		# plt.colorbar().set_label('counts [au]')
		# plt.pause(0.01)

		foillx=min(foilxrot)
		foilrx=max(foilxrot)
		foilhorizwpixel=foilrx-foillx
		foildw=min(foilyrot)
		foilup=max(foilyrot)
		foilvertwpixel=foilup-foildw

		out_of_ROI_mask = np.ones_like(testrotback)
		out_of_ROI_mask[testrotback<np.nanmin(testrot[testrot>0])]=np.nan
		out_of_ROI_mask[testrotback>np.nanmax(testrot[testrot>0])]=np.nan
		a = generic_filter((testrotback),np.std,size=[19,19])
		out_of_ROI_mask[a>np.mean(a)]=np.nan

		# rotation and crop
		laser_temperature_minus_background_rot=rotate(laser_temperature_minus_background_full,foilrotdeg,axes=(-1,-2))
		laser_temperature_std_minus_background_rot=rotate(laser_temperature_std_minus_background_full,foilrotdeg,axes=(-1,-2))
		if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
			laser_temperature_std_minus_background_rot*=out_of_ROI_mask
			laser_temperature_std_minus_background_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
			laser_temperature_minus_background_rot*=out_of_ROI_mask
			laser_temperature_minus_background_rot[np.logical_and(laser_temperature_minus_background_rot<np.nanmin(laser_temperature_minus_background_full),laser_temperature_minus_background_rot>np.nanmax(laser_temperature_minus_background_full))]=0
		laser_temperature_minus_background_crop=laser_temperature_minus_background_rot[:,foildw:foilup,foillx:foilrx]
		laser_temperature_std_minus_background_crop=laser_temperature_std_minus_background_rot[:,foildw:foilup,foillx:foilrx]
		nan_ROI_mask = np.isfinite(np.nanmedian(laser_temperature_minus_background_crop[:10],axis=0))

		laser_temperature_rot=rotate(laser_temperature_full,foilrotdeg,axes=(-1,-2))
		laser_temperature_std_rot=rotate(laser_temperature_std_full,foilrotdeg,axes=(-1,-2))
		if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
			laser_temperature_std_rot*=out_of_ROI_mask
			laser_temperature_std_rot[np.logical_and(laser_temperature_rot<np.nanmin(laser_temperature_full),laser_temperature_rot>np.nanmax(laser_temperature_full))]=0
			laser_temperature_rot*=out_of_ROI_mask
			laser_temperature_rot[np.logical_and(laser_temperature_rot<np.nanmin(laser_temperature_full),laser_temperature_rot>np.nanmax(laser_temperature_full))]=0
		laser_temperature_crop=laser_temperature_rot[:,foildw:foilup,foillx:foilrx]
		laser_temperature_std_crop=laser_temperature_std_rot[:,foildw:foilup,foillx:foilrx]

		reference_background_temperature_rot=rotate(reference_background_temperature_no_dead_pixels,foilrotdeg,axes=(-1,-2))
		reference_background_temperature_std_rot=rotate(reference_background_temperature_std_no_dead_pixels,foilrotdeg,axes=(-1,-2))
		if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
			reference_background_temperature_std_rot*=out_of_ROI_mask
			reference_background_temperature_std_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=0
			reference_background_temperature_rot*=out_of_ROI_mask
			reference_background_temperature_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=np.nanmean(reference_background_temperature_no_dead_pixels)
		reference_background_temperature_crop=reference_background_temperature_rot[:,foildw:foilup,foillx:foilrx]
		reference_background_temperature_crop = np.nanmean(reference_background_temperature_crop,axis=0)
		reference_background_temperature_std_crop=reference_background_temperature_std_rot[:,foildw:foilup,foillx:foilrx]
		reference_background_temperature_std_crop = np.nanmean(reference_background_temperature_std_crop,axis=0)

		del laser_temperature_minus_background_rot,laser_temperature_std_minus_background_rot,reference_background_temperature_no_dead_pixels,reference_background_temperature_std_no_dead_pixels,laser_temperature_rot,laser_temperature_std_rot

		# FOIL PROPERTY ADJUSTMENT
		if False:	# spatially resolved foil properties from Japanese producer
			foilemissivityscaled=resize(foilemissivity,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
			foilthicknessscaled=resize(foilthickness,(foilvertwpixel,foilhorizwpixel),order=0)[1:-1,1:-1]
			conductivityscaled=np.multiply(Ptthermalconductivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
			reciprdiffusivityscaled=np.multiply(1/Ptthermaldiffusivity,np.ones((foilvertwpixel-2,foilhorizwpixel-2)))
		elif True:	# homogeneous foil properties from foil experiments
			foilemissivityscaled=flat_foil_properties['emissivity']*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
			foilthicknessscaled=flat_foil_properties['thickness']*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
			conductivityscaled=Ptthermalconductivity*np.ones((foilvertwpixel-2,foilhorizwpixel-2))
			reciprdiffusivityscaled=(1/flat_foil_properties['diffusivity'])*np.ones((foilvertwpixel-2,foilhorizwpixel-2))

		plt.figure(figsize=(20, 10))
		temp = np.max(laser_temperature_minus_background_crop,axis=(-1,-2)).argmax()
		plt.title('Foil search in '+laser_to_analyse+'\nhottest frame at %.4gsec with mask from Calcam' %(time_full[temp]) + '\ngas injection points at R=0.261m, Z=-0.264m, '+r'$theta$'+'=105°(LX), 195°(DX)')
		plt.imshow(np.flip(np.transpose(laser_temperature_minus_background_crop[temp],(1,0)),axis=1),'rainbow',interpolation='none')#,origin='lower',vmax=np.max(laser_temperature_minus_background_crop[:,:,:180],axis=(-1,-2))[temp])
		# MASTU_wireframe_resize = MASTU_wireframe.resize(np.shape(laser_temperature_minus_background_crop[temp]))
		# MASTU_wireframe_resize[MASTU_wireframe_resize==0] = np.nan
		plt.colorbar().set_label('temp increase [K]')
		plt.imshow(masked, 'gray', interpolation='none', alpha=0.4,origin='lower',extent = [0,np.shape(laser_temperature_minus_background_crop[temp])[0]-1,0,np.shape(laser_temperature_minus_background_crop[temp])[1]-1])
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		# plt.clim(vmax=np.max(laser_temperature_minus_background_crop[:,:,:180],axis=(-1,-2))[temp])
		plt.savefig(laser_to_analyse[:-4]+'_temp_overlay.png', bbox_inches='tight')
		plt.close('all')
		plt.figure(figsize=(20, 10))
		plt.title('Foil search in '+laser_to_analyse+'\nhottest frame at %.4gsec' %(time_full[temp]) )
		plt.imshow(np.flip(np.transpose(laser_temperature_minus_background_crop[temp],(1,0)),axis=1),'rainbow',interpolation='none',origin='lower')#,vmax=np.max(laser_temperature_minus_background_crop[:,:,:180],axis=(-1,-2))[temp])
		plt.colorbar().set_label('temp increase [K]')
		plt.xlabel('Horizontal axis [pixles]')
		plt.ylabel('Vertical axis [pixles]')
		# plt.clim(vmax=np.max(laser_temperature_minus_background_crop[:,:,:180],axis=(-1,-2))[temp])
		plt.savefig(laser_to_analyse[:-4]+'_temp.eps', bbox_inches='tight')
		plt.close('all')

		# ani = coleval.movie_from_data(np.array([laser_temperature_minus_background_crop]), laser_framerate, integration=laser_int_time/1000,time_offset=-start_time_of_pulse,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='raw counts [au]')
		# dt=1/laser_framerate
		dt = time_full[2:]-time_full[:-2]
		dx=foilhorizw/foilhorizwpixel
		dTdt=np.divide((laser_temperature_crop[2:,1:-1,1:-1]-laser_temperature_crop[:-2,1:-1,1:-1]).T,2*dt).T.astype(np.float32)
		dTdt_std=np.divide((laser_temperature_std_crop[2:,1:-1,1:-1]**2 + laser_temperature_std_crop[:-2,1:-1,1:-1]**2).T**0.5,2*dt).T.astype(np.float32)
		d2Tdx2=np.divide(laser_temperature_minus_background_crop[1:-1,1:-1,2:]-np.multiply(2,laser_temperature_minus_background_crop[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop[1:-1,1:-1,:-2],dx**2).astype(np.float32)
		d2Tdx2_std=np.divide((laser_temperature_std_minus_background_crop[1:-1,1:-1,2:]**2+np.multiply(2,laser_temperature_std_minus_background_crop[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop[1:-1,1:-1,:-2]**2)**0.5,dx**2).astype(np.float32)
		d2Tdy2=np.divide(laser_temperature_minus_background_crop[1:-1,2:,1:-1]-np.multiply(2,laser_temperature_minus_background_crop[1:-1,1:-1,1:-1])+laser_temperature_minus_background_crop[1:-1,:-2,1:-1],dx**2).astype(np.float32)
		d2Tdy2_std=np.divide((laser_temperature_std_minus_background_crop[1:-1,2:,1:-1]**2+np.multiply(2,laser_temperature_std_minus_background_crop[1:-1,1:-1,1:-1])**2+laser_temperature_std_minus_background_crop[1:-1,:-2,1:-1]**2)**0.5,dx**2).astype(np.float32)
		d2Tdxy = np.ones_like(dTdt).astype(np.float32)*np.nan
		d2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2[:,nan_ROI_mask[1:-1,1:-1]],d2Tdy2[:,nan_ROI_mask[1:-1,1:-1]])
		del d2Tdx2,d2Tdy2
		d2Tdxy_std = np.ones_like(dTdt).astype(np.float32)*np.nan
		d2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]=np.add(d2Tdx2_std[:,nan_ROI_mask[1:-1,1:-1]]**2,d2Tdy2_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5
		del d2Tdx2_std,d2Tdy2_std
		negd2Tdxy=np.multiply(-1,d2Tdxy)
		negd2Tdxy_std=d2Tdxy_std
		T4=(laser_temperature_minus_background_crop[1:-1,1:-1,1:-1]+np.nanmean(reference_background_temperature_crop)+zeroC)**4
		T4_std=T4**(3/4) *4 *laser_temperature_std_minus_background_crop[1:-1,1:-1,1:-1]	# the error resulting from doing the average on the whole ROI is completely negligible
		T04=(np.nanmean(reference_background_temperature_crop)+zeroC)**4 *np.ones_like(laser_temperature_minus_background_crop[1:-1,1:-1,1:-1])
		T04_std=0
		T4_T04 = np.ones_like(dTdt).astype(np.float32)*np.nan
		T4_T04[:,nan_ROI_mask[1:-1,1:-1]] = (T4[:,nan_ROI_mask[1:-1,1:-1]]-T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		T4_T04_std = np.ones_like(dTdt).astype(np.float32)*np.nan
		T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]] = ((T4_std[:,nan_ROI_mask[1:-1,1:-1]]**2+T04_std**2)**0.5).astype(np.float32)

		temp_BBrad = np.ones_like(dTdt).astype(np.float32)*np.nan
		temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		temp_BBrad_std = np.ones_like(dTdt).astype(np.float32)*np.nan
		temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (2*sigmaSB*T4_T04_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		temp_diffusion = np.ones_like(dTdt).astype(np.float32)*np.nan
		temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		temp_diffusion_std = np.ones_like(dTdt).astype(np.float32)*np.nan
		temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*negd2Tdxy_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		temp_timevariation = np.ones_like(dTdt).astype(np.float32)*np.nan
		temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		temp_timevariation_std = np.ones_like(dTdt).astype(np.float32)*np.nan
		temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (Ptthermalconductivity*dTdt_std[:,nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		del dTdt,dTdt_std,d2Tdxy,d2Tdxy_std,negd2Tdxy,negd2Tdxy_std,T4,T4_std,T04,T04_std

		full_saved_file_dict['power_data'] = dict([])
		full_saved_file_dict['power_data']['temp_BBrad'] = np.float32(temp_BBrad)
		full_saved_file_dict['power_data']['temp_BBrad_std'] = np.float32(temp_BBrad_std)
		full_saved_file_dict['power_data']['temp_diffusion'] = np.float32(temp_diffusion)
		full_saved_file_dict['power_data']['temp_diffusion_std'] = np.float32(temp_diffusion_std)
		full_saved_file_dict['power_data']['temp_timevariation'] = np.float32(temp_timevariation)
		full_saved_file_dict['power_data']['temp_timevariation_std'] = np.float32(temp_timevariation_std)
		full_saved_file_dict['power_data']['dx'] = dx
		try:
			del full_saved_file_dict['power_data']['foil_properties']
			del full_saved_file_dict['power_data']['powernoback']
			del full_saved_file_dict['power_data']['powernoback_std']
		except:
			print('no legacy elements to erase')
		np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)

		BBrad = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		BBrad[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad[:,nan_ROI_mask[1:-1,1:-1]] * foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		BBrad_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		BBrad_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]*foilemissivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		diffusion = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		diffusion[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		diffusion_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		diffusion_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		timevariation = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		timevariation[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		timevariation_std = np.ones_like(temp_BBrad).astype(np.float32)*np.nan
		timevariation_std[:,nan_ROI_mask[1:-1,1:-1]] = (temp_timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]*foilthicknessscaled[nan_ROI_mask[1:-1,1:-1]]*reciprdiffusivityscaled[nan_ROI_mask[1:-1,1:-1]]).astype(np.float32)
		del temp_BBrad,temp_BBrad_std,temp_diffusion,temp_diffusion_std,temp_timevariation,temp_timevariation_std
		powernoback = (diffusion + timevariation + BBrad).astype(np.float32)
		powernoback_std = np.ones_like(powernoback)*np.nan
		powernoback_std[:,nan_ROI_mask[1:-1,1:-1]] = ((diffusion_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + timevariation_std[:,nan_ROI_mask[1:-1,1:-1]]**2 + BBrad_std[:,nan_ROI_mask[1:-1,1:-1]]**2)**0.5).astype(np.float32)

		saved_file_dict_short['power_data'] = dict([])
		saved_file_dict_short['power_data']['foil_properties'] = dict([])
		saved_file_dict_short['power_data']['foil_properties']['emissivity'] = np.float32(foilemissivityscaled)
		saved_file_dict_short['power_data']['foil_properties']['thickness'] = np.float32(foilthicknessscaled)
		saved_file_dict_short['power_data']['foil_properties']['diffusivity'] = np.float32(reciprdiffusivityscaled)
		saved_file_dict_short['power_data']['powernoback'] = np.float32(powernoback)
		saved_file_dict_short['power_data']['powernoback_std'] = np.float32(powernoback_std)
		saved_file_dict_short['other'] = dict([])
		saved_file_dict_short['other']['nan_ROI_mask'] = nan_ROI_mask
		np.savez_compressed(laser_to_analyse[:-4]+'_short',**saved_file_dict_short)


	horizontal_coord = np.arange(np.shape(powernoback[0])[1])
	vertical_coord = np.arange(np.shape(powernoback[0])[0])
	horizontal_coord,vertical_coord = np.meshgrid(horizontal_coord,vertical_coord)
	horizontal_coord = (horizontal_coord+1+0.5)*dx	# +1 because in the process of calculating the power I eliminate the first and last pixel in spatial coordinates, +0.5 do be the centre of the pixel
	vertical_coord = (vertical_coord+1+0.5)*dx
	horizontal_coord -= foilhorizw*0.5+0.0198
	vertical_coord -= foilvertw*0.5-0.0198
	distance_from_vertical = (horizontal_coord**2+vertical_coord**2)**0.5
	pinhole_to_foil_vertical = 0.008 + 0.003 + 0.002 + 0.045	# pinhole holder, washer, foil holder, standoff
	pinhole_to_pixel_distance = (pinhole_to_foil_vertical**2 + distance_from_vertical**2)**0.5

	etendue = np.ones_like(powernoback[0]) * (np.pi*(0.002**2)) / (pinhole_to_pixel_distance**2)	# I should include also the area of the pixel, but that is already in the w/m2 power
	etendue *= (pinhole_to_foil_vertical/pinhole_to_pixel_distance)**2	 # cos(a)*cos(b). for pixels not directly under the pinhole bot pinhole and pixel are tilted respect to the vertical, with same angle.

	plt.figure(figsize=(8, 5))
	plt.imshow(np.flip(np.transpose(etendue,(1,0)),axis=1)*(dx**2),'rainbow',origin='lower')
	plt.colorbar().set_label('Etendue [m2]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	plt.title('Etendue map')
	plt.savefig(path+day+'/etendue'+'.eps', bbox_inches='tight')
	plt.close('all')

	temp = np.sort(powernoback[:,:,:180].flatten())
	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(powernoback,(0,2,1)),axis=2)]), laser_framerate,integration=laser_int_time/1000,time_offset=time_full[1], extvmin=0,extvmax=np.mean(temp[-len(temp)//60:]),xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
	ani.save(laser_to_analyse[:-4]+ '_power_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	shrink_factor = 20
	temp = generic_filter(powernoback,np.nanmean,size=[1,shrink_factor,shrink_factor])
	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate,mask=masked, integration=laser_int_time/1000,time_offset=time_full[1],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed spatially '+str([shrink_factor,shrink_factor]) + '\n', include_EFIT=True, pulse_ID=laser_to_analyse[-9:-4], overlay_x_point=True, overlay_mag_axis=True)
	ani.save(laser_to_analyse[:-4]+ '_power_smooth1.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	saved_file_dict_short['power_data']['smooth_'+str(1) + 'x' + str(shrink_factor) + 'x' + str(shrink_factor)] = temp
	temp = generic_filter(4*np.pi*powernoback/etendue,np.nanmean,size=[1,shrink_factor,shrink_factor])
	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate, integration=laser_int_time/1000,time_offset=time_full[1],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed spatially '+str([shrink_factor,shrink_factor]) + '\n')
	ani.save(laser_to_analyse[:-4]+ '_brightness_smooth1.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	shrink_factor_t = 13
	shrink_factor_x = 10
	temp = generic_filter(powernoback,np.nanmean,size=[shrink_factor_t,shrink_factor_x,shrink_factor_x])
	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate,mask=masked, integration=laser_int_time/1000,time_offset=time_full[1],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n', include_EFIT=True, pulse_ID=laser_to_analyse[-9:-4], overlay_x_point=True, overlay_mag_axis=True)
	ani.save(laser_to_analyse[:-4]+ '_power_smooth2.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	saved_file_dict_short['power_data']['smooth_'+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)] = temp

	temp_2 = np.max(temp,axis=(-1,-2)).argmax()
	plt.figure(figsize=(20, 10))
	plt.title('Foil search in '+laser_to_analyse+'\nhighest power frame at %.4gsec with mask from Calcam' %(time_full[temp_2]) +' smooth '+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)+ '\ngas injection points at R=0.261m, Z=-0.264m, '+r'$\theta$'+'=105°(LX), 195°(DX)')
	plt.imshow(np.flip(np.transpose(temp[temp_2],(1,0)),axis=1),'rainbow',interpolation='none')#,origin='lower',vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp_2])
	# MASTU_wireframe_resize = MASTU_wireframe.resize(np.shape(temp[temp_2]))
	# MASTU_wireframe_resize[MASTU_wireframe_resize==0] = np.nan
	plt.colorbar().set_label('Power on foil [W/m2]')
	plt.imshow(masked, 'gray', interpolation='none', alpha=0.4,origin='lower',extent = [0,np.shape(temp[temp_2])[0]-1,0,np.shape(temp[temp_2])[1]-1])
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	# plt.clim(vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp_2])
	plt.savefig(laser_to_analyse[:-4]+'_power_overlay.png', bbox_inches='tight')
	plt.close('all')
	plt.figure(figsize=(20, 10))
	plt.title('Foil search in '+laser_to_analyse+'\nhighest power frame at %.4gsec' %(time_full[temp_2]) +' smooth '+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x))
	plt.imshow(np.flip(np.transpose(temp[temp_2],(1,0)),axis=1),'rainbow',interpolation='none',origin='lower',vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp_2])
	plt.colorbar().set_label('Power on foil [W/m2]')
	plt.xlabel('Horizontal axis [pixles]')
	plt.ylabel('Vertical axis [pixles]')
	# plt.clim(vmax=np.max(temp[:,:,:180],axis=(-1,-2))[temp_2])
	plt.savefig(laser_to_analyse[:-4]+'_power.eps', bbox_inches='tight')
	plt.close('all')

	temp = generic_filter(4*np.pi*powernoback/etendue,np.nanmean,size=[shrink_factor_t,shrink_factor_x,shrink_factor_x])
	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(temp,(0,2,1)),axis=2)]), laser_framerate, integration=laser_int_time/1000,time_offset=time_full[1],extvmax=temp[:,:,180].max(),extvmin=0,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Brightness [W/m2]', prelude='shot ' + laser_to_analyse[-9:-4] + '\nsmoothed '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
	ani.save(laser_to_analyse[:-4]+ '_brightness_smooth2.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close('all')


	def bin_ndarray(ndarray, new_shape, operation='sum'):
	    """
	    Bins an ndarray in all axes based on the target shape, by summing or
	        averaging.

	    Number of output dimensions must match number of input dimensions and
	        new axes must divide old ones.

	    Example
	    -------
	    >>> m = np.arange(0,100,1).reshape((10,10))
	    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
	    >>> print(n)

	    [[ 22  30  38  46  54]
	     [102 110 118 126 134]
	     [182 190 198 206 214]
	     [262 270 278 286 294]
	     [342 350 358 366 374]]

	    """
	    operation = operation.lower()
	    if not operation in ['sum', 'mean']:
	        raise ValueError("Operation not supported.")
	    if ndarray.ndim != len(new_shape):
	        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
	                                                           new_shape))
	    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
	                                                  ndarray.shape)]
	    flattened = [l for p in compression_pairs for l in p]
	    ndarray = ndarray.reshape(flattened)
	    for i in range(len(new_shape)):
	        op = getattr(ndarray, operation)
	        ndarray = op(-1*(i+1))
	    return ndarray

	for shrink_factor_t in [13,7]:
		shrink_factor_x = 8
		new_shape=np.array([np.array(powernoback.shape)[0]//shrink_factor_t,np.array(powernoback.shape)[1]//shrink_factor_x,np.array(powernoback.shape)[2]//shrink_factor_x]).astype(int)
		# new_shape=np.array([int(np.ceil(np.array(powernoback.shape)[0]/shrink_factor_t)),int(np.ceil(np.array(powernoback.shape)[1]/shrink_factor_x)),int(np.ceil(np.array(powernoback.shape)[2]/shrink_factor_x))]).astype(int)
		powernoback_short = powernoback[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
		a=bin_ndarray(powernoback_short, new_shape=new_shape, operation='mean')
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
		# ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power [W/m2]',extvmax=a.max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
		ani.save(laser_to_analyse[:-4]+ '_power_bin_' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
		plt.close('all')
		saved_file_dict_short['power_data']['bin_'+str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x)] = a

		timevariation_short = timevariation[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
		a=bin_ndarray(timevariation_short, new_shape=new_shape, operation='mean')
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
		ani.save(laser_to_analyse[:-4]+ '_dT_dt_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

		diffusion_short = diffusion[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
		a=bin_ndarray(diffusion_short, new_shape=new_shape, operation='mean')
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
		ani.save(laser_to_analyse[:-4]+ '_diffus_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

		BBrad_short = BBrad[:powernoback.shape[0]//shrink_factor_t*shrink_factor_t,:int(powernoback.shape[1]//shrink_factor_x*shrink_factor_x),powernoback.shape[2]-int(powernoback.shape[2]//shrink_factor_x*shrink_factor_x):]
		a=bin_ndarray(BBrad_short, new_shape=new_shape, operation='mean')
		ani = coleval.movie_from_data(np.array([np.flip(np.transpose(a,(0,2,1)),axis=2)]), laser_framerate/shrink_factor_t, integration=laser_int_time/1000,time_offset=time_full[1],xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Power on foil [W/m2]',extvmax=a[:,:,int(180//shrink_factor_x)].max(),extvmin=0, prelude='shot ' + laser_to_analyse[-9:-4] + '\nbinned '+str([shrink_factor_t,shrink_factor_x,shrink_factor_x]) + '\n')
		ani.save(laser_to_analyse[:-4]+ '_BB_bin' + str(shrink_factor_t) + 'x' + str(shrink_factor_x) + 'x' + str(shrink_factor_x) + '.mp4', fps=5/shrink_factor_t, writer='ffmpeg',codec='mpeg4')
		plt.close('all')

	np.savez_compressed(laser_to_analyse[:-4]+'_short',**saved_file_dict_short)
	print('completed ' + laser_to_analyse)

except Exception as e:
	print('FAILED ' + laser_to_analyse)
	logging.exception('with error: ' + str(e))
